from __future__ import annotations
import cv2, dlib, numpy as np, yaml, time, math, csv
from pathlib import Path
from src.eye_aspect_ratio import compute_ears
from src.head_pose import estimate_head_pose, LANDMARKS_IDXS
from src.utils import RunningStat
import requests
import bz2
import os

# macOS 충돌 방지 설정
cv2.setNumThreads(1)

# ---------------- SOUNDDEVICE (for beep) -----------------
try:
    import sounddevice as sd
except Exception:
    sd = None
    print("---------------------------------------------------------------------")
    print("WARNING: 'sounddevice' library not found. Beep alert is DISABLED.")
    print("         -> To enable sound, run: pip install sounddevice")
    print("---------------------------------------------------------------------")


def beep(freq=880, dur_ms=300):
    """Beep sound generator using sounddevice."""
    if sd is None:
        print("DEBUG: Beep called but sounddevice not available.")
        return
    
    fs = 44100
    duration = dur_ms / 1000.0
    t = np.linspace(0, duration, int(fs * duration), False)
    wave = np.sin(2 * np.pi * freq * t).astype(np.float32)

    try:
        sd.play(wave, fs)
        sd.wait()
    except Exception as e:
        print(f"ERROR during beep playback: {e}")


# ---------------- DLIB MODEL MANAGEMENT ------------------
def download_dlib_model(url, filename):
    """Download and decompress Dlib 68-landmark model automatically."""
    if os.path.exists(filename):
        print(f"INFO: Model '{filename}' already exists.")
        return
    
    compressed_filename = filename + ".bz2"
    if not os.path.exists(compressed_filename):
        print(f"INFO: Downloading Dlib model from {url}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(compressed_filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("INFO: Download complete.")

    print("INFO: Decompressing model...")
    try:
        with bz2.BZ2File(compressed_filename, "rb") as src:
            with open(filename, "wb") as dst:
                dst.write(src.read())
        print("INFO: Decompression complete.")
        os.remove(compressed_filename)
    except Exception as e:
        print(f"ERROR: Failed to decompress model: {e}")
        exit(1)


# ---------------- UTILITY FUNCTIONS ----------------------
def draw_text(img, text, org, color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def dlib_to_numpy(landmarks, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(68):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    return coords


def norm_head_size(pts):
    chin = pts[8]
    eye = pts[36]
    return np.linalg.norm(chin - eye)


# ------------------------------ MAIN ------------------------------
def main():
    print("=== Driver Monitoring Prototype Starting... ===")

    config = yaml.safe_load(open("config.yaml", "r"))
    cam_idx = config.get("camera_index", 0)

    # 🔥 macOS 안정화를 위해 AVFOUNDATION backend 사용
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_AVFOUNDATION)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("frame_width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("frame_height", 720))

    if not cap.isOpened():
        print(f"ERROR: Camera index {cam_idx} could not be opened!")
        print("Try:")
        print("- System Settings → Privacy → Camera 권한 확인")
        print("- Zoom, Safari 같은 카메라 사용하는 앱 종료")
        return

    # ----- Load Dlib model -----
    MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    MODEL = "shape_predictor_68_face_landmarks.dat"
    download_dlib_model(MODEL_URL, MODEL)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(MODEL)

    # ----- Runtime States -----
    fps_stat = RunningStat(30)
    ear_low_frames = 0
    pose_off_frames = 0
    events = []

    events_path = Path(config.get("events_csv_path", "events_log.csv"))
    save_events = config.get("save_events_csv", True)

    last_time = time.time()

    # Face pose correction offsets
    YAW_OFFSET = 170.7
    PITCH_OFFSET = -5.3

    print("INFO: System running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Frame grab failed.")
            break

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        status_msgs = []
        event_triggered = False

        if faces:
            face = faces[0]
            landmarks = predictor(gray, face)
            pts = dlib_to_numpy(landmarks, dtype=np.float32)

            # EAR
            le, re = compute_ears(pts)
            ear = (le + re) / 2.0

            if ear < config["ear_sleep_threshold"]:
                ear_low_frames += 1
            else:
                ear_low_frames = 0

            if ear_low_frames >= config["ear_sleep_frames"]:
                status_msgs.append("DROWSINESS DETECTED")
                event_triggered = True

            # Head pose
            pose = estimate_head_pose(pts, frame.shape)
            if pose:
                adjusted_yaw = pose["yaw"] - YAW_OFFSET
                adjusted_pitch = pose["pitch"] - PITCH_OFFSET

                if abs(adjusted_yaw) > config["yaw_abs_threshold"] or abs(adjusted_pitch) > config["pitch_abs_threshold"]:
                    pose_off_frames += 1
                else:
                    pose_off_frames = 0

                if pose_off_frames >= config["pose_frames"]:
                    status_msgs.append("LOOKING AWAY")
                    event_triggered = True

                draw_text(frame, f"Yaw={adjusted_yaw:.1f} Pitch={adjusted_pitch:.1f}", (20,70), (255,255,0))
                draw_text(frame, "OFFSET ACTIVE", (20, 100), (0,255,0))

            nose = pts[30].astype(int)
            cv2.circle(frame, tuple(nose), 5, (0,255,255), -1)

            for (x,y) in pts:
                cv2.circle(frame, (int(x),int(y)), 1, (0,0,255), -1)

            draw_text(frame, f"EAR={ear:.3f}", (20,40), (0,255,0))

        # Alerts
        if event_triggered and config.get("beep_enabled", True):
            beep(config.get("beep_frequency", 880), config.get("beep_duration_ms", 300))

        if status_msgs:
            draw_text(frame, " | ".join(status_msgs), (20,130), (0,0,255))
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            events.append({"time": ts, "messages": "|".join(status_msgs)})

        # FPS
        now = time.time()
        fps_stat.add(1.0 / max(1e-6, (now - last_time)))
        last_time = now

        draw_text(frame, f"FPS: {fps_stat.mean():.1f}", (20, h-20), (200,200,200))

        cv2.imshow("Driver Monitoring Prototype", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break

    cap.release()
    cv2.destroyAllWindows()

    if save_events and events:
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time","messages"])
            writer.writeheader()
            writer.writerows(events)
        print(f"Saved events to {events_path.resolve()}")


if __name__ == "__main__":
    main()
