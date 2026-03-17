#!/usr/bin/env python3
"""
Minimal client for hand_detector's Unix socket server.
Prints every hand-detection event to stdout.

Usage:
    python scripts/test_socket.py [--socket /tmp/hand_detector.sock]
"""

import argparse
import socket
import sys

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--socket", default="/tmp/hand_detector.sock")
    args = p.parse_args()

    print(f"Connecting to {args.socket} ...")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        try:
            s.connect(args.socket)
        except FileNotFoundError:
            sys.exit(f"Socket not found: {args.socket}\nMake sure hand_detector is running.")

        print("Connected. Waiting for events (Ctrl-C to quit)...\n")
        buf = ""
        while True:
            data = s.recv(4096)
            if not data:
                print("Server closed connection.")
                break
            buf += data.decode()
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                print(line)

if __name__ == "__main__":
    main()
