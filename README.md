# Driver Monitoring & Drowsiness Warning (Prototype)

This is a lightweight, **Python + OpenCV + MediaPipe** prototype for the project:
**AI 기반 자율주행 차량용 운전자 상태 모니터링 및 졸음경고 시스템**.

## Features
- Eye Aspect Ratio (EAR) for **drowsiness** detection
- Head pose estimation (PnP) for **forward attention** monitoring
- Simple **phone-use heuristic** via hand proximity to face
- Real-time alerts (beep + on-screen overlay) and CSV event logging

## Firststart
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py

## Quickstart
source .venv311/bin/activatedeactivate
python main.py


Tune thresholds in `config.yaml` for your camera and environment.
