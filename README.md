# hand_detector — BlazePalm server

Reads a USB webcam, runs PalmDetector.predict_on_batch() (includes full
decode + weighted NMS from blazepalm.py), and emits a socket event only
when hand presence changes.

## Socket protocol (server — clients connect):
```
  {"hand_detected": true,  "confidence": 0.934}
  {"hand_detected": false}
```
## Usage:
```
  python3 hand_detector.py --weights palmdetector.pth [--camera 0]
                           [--socket /tmp/hand_detector.sock]
                           [--every-n 3] [--display]
```
## Dependencies:
```
  pip install requirements.txt
```
