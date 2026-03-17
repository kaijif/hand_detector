#!/usr/bin/env python3
"""
hand_detector — BlazePalm server

Reads a USB webcam, runs PalmDetector.predict_on_batch() (includes full
decode + weighted NMS from blazepalm.py), and emits a socket event only
when hand presence changes.

Socket protocol (server — clients connect):
  {"hand_detected": true,  "confidence": 0.934}\n
  {"hand_detected": false}\n

Usage:
  python3 hand_detector.py --weights palmdetector.pth [--camera 0]
                           [--socket /tmp/hand_detector.sock]
                           [--every-n 3] [--display]

Dependencies:
  pip install torch opencv-python
"""

import argparse
import json
import math
import os
import signal
import socket
import sys

import cv2
import numpy as np
import torch

from blazepalm import PalmDetector

# ─── Anchor generation (mirrors genarchors.py) ────────────────────────────────


def _build_anchors() -> np.ndarray:
    MIN_S, MAX_S = 0.1171875, 0.75
    STRIDES = [8, 16, 32, 32, 32]
    N = len(STRIDES)

    def scale(i):
        return MIN_S + (MAX_S - MIN_S) * i / (N - 1)

    rows = []
    i = 0
    while i < N:
        aspect_ratios, scales = [], []
        j = i
        while j < N and STRIDES[j] == STRIDES[i]:
            s = scale(j)
            aspect_ratios.append(1.0)
            scales.append(s)
            s_next = 1.0 if j == N - 1 else scale(j + 1)
            aspect_ratios.append(1.0)
            scales.append(math.sqrt(s * s_next))
            j += 1
        fh = fw = 256 // STRIDES[i]
        for y in range(fh):
            for x in range(fw):
                for _ in range(len(aspect_ratios)):
                    rows.append([(x + 0.5) / fw, (y + 0.5) / fh, 1.0, 1.0])
        i = j

    anchors = np.array(rows, dtype=np.float32)
    assert anchors.shape == (2944, 4), anchors.shape
    return anchors


# ─── Socket server ─────────────────────────────────────────────────────────────


class SocketServer:
    def __init__(self, path: str):
        self.path = path
        self._clients: list[socket.socket] = []
        self._srv: socket.socket | None = None

    def start(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.setblocking(False)
        s.bind(self.path)
        s.listen(8)
        self._srv = s
        print(f"Socket: {self.path}")

    def send(self, msg: str):
        while True:
            try:
                conn, _ = self._srv.accept()  # type: ignore[union-attr]
                conn.setblocking(False)
                self._clients.append(conn)
            except BlockingIOError:
                break
        data = msg.encode()
        dead = []
        for c in self._clients:
            try:
                c.sendall(data)
            except OSError:
                dead.append(c)
        for c in dead:
            c.close()
            self._clients.remove(c)

    def close(self):
        for c in self._clients:
            c.close()
        if self._srv:
            self._srv.close()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass


# ─── Args ─────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", default="palmdetector.pth")
    p.add_argument(
        "--anchors",
        default=None,
        help="Path to anchors.npy (generated automatically if absent)",
    )
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--socket", default="/tmp/hand_detector.sock")
    p.add_argument(
        "--every-n",
        type=int,
        default=3,
        help="Run inference every N frames (default: 3)",
    )
    p.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for detections (default: 0.7)",
    )
    p.add_argument(
        "--min-frames",
        type=int,
        default=1,
        help="Consecutive detections needed to trigger event (default: 1)",
    )
    p.add_argument("--display", action="store_true")
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    running = True

    def _stop(_sig, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    # Model
    model = PalmDetector()
    model.load_weights(args.weights)

    if args.anchors:
        model.load_anchors(args.anchors)
    else:
        model.anchors = torch.tensor(_build_anchors(), dtype=torch.float32)

    print(f"Loaded {args.weights}")

    # Webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera}")
    print(f"Camera {args.camera}  |  inference every {args.every_n} frames")

    # Socket
    server = SocketServer(args.socket)
    server.start()
    print("Running — Ctrl-C to quit\n")

    was_detected = False
    target_state = False
    consecutive_frames = 0
    cached_conf = 0.0
    frame_idx = 0
    # Reusable pre-allocated input array (1, 3, 256, 256) float32
    inp = torch.zeros(1, 3, 256, 256)

    while running:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_idx += 1

        if frame_idx % args.every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Letterbox resizing to 256x256
            h, w = rgb.shape[:2]
            scale = 256.0 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(rgb, (new_w, new_h))

            pad_w = 256 - new_w
            pad_h = 256 - new_h
            top, bottom = pad_h // 2, pad_h - (pad_h // 2)
            left, right = pad_w // 2, pad_w - (pad_w // 2)
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

            # predict_on_batch handles preprocessing internally;
            # pass as (1, 3, H, W) uint8 torch tensor
            inp[0] = torch.from_numpy(padded).permute(2, 0, 1)
            results = model.predict_on_batch(inp, score_threshold=args.score_threshold)

            # results: list (one entry per image) of (N, 19) tensors
            # index 18 is confidence score
            raw_detected = len(results) > 0
            cached_conf = float(results[0][:, 18].max()) if raw_detected else 0.0

            if raw_detected == target_state:
                consecutive_frames += 1
            else:
                target_state = raw_detected
                consecutive_frames = 1

            if consecutive_frames >= args.min_frames:
                detected = target_state
            else:
                detected = was_detected

        else:
            detected = was_detected

        if detected != was_detected:
            if detected:
                msg = (
                    json.dumps(
                        {
                            "hand_detected": True,
                            "confidence": round(cached_conf, 3),
                        }
                    )
                    + "\n"
                )
            else:
                msg = json.dumps({"hand_detected": False}) + "\n"
            server.send(msg)
            print(msg, end="", flush=True)

        was_detected = detected

        if args.display:
            color = (0, 220, 0) if detected else (0, 0, 180)
            label = f"Hand {cached_conf:.2f}" if detected else "No hand"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.imshow("hand_detector", frame)
            if cv2.waitKey(1) == 27:
                break

    print("\nShutting down.")
    cap.release()
    server.close()
    if args.display:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
