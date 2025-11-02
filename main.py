from __future__ import annotations
import cv2, mediapipe as mp, numpy as np, yaml, time, math, csv
from pathlib import Path
from src.eye_aspect_ratio import compute_ears
from src.head_pose import estimate_head_pose, LANDMARKS_IDXS
from src.phone_use import hand_near_face
from src.utils import RunningStat

try:
    import simpleaudio as sa
except Exception:
    sa = None

def beep(freq=880, dur_ms=300):
    if sa is None:
        return
    fs = 44100
    t = np.linspace(0, dur_ms/1000.0, int(fs*dur_ms/1000.0), False)
    note = np.sin(freq * 2 * np.pi * t)
    audio = (note * 32767 / np.max(np.abs(note))).astype(np.int16)
    play_obj = sa.play_buffer(audio, 1, 2, fs)
    play_obj.wait_done()

def draw_text(img, text, org, color=(0,255,0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

def norm_head_size(landmarks):
    chin = landmarks[LANDMARKS_IDXS["chin"]]
    eye = landmarks[LANDMARKS_IDXS["left_eye_corner"]]
    return np.linalg.norm(chin-eye)

def main():
    config = yaml.safe_load(open("config.yaml", "r"))
    cam_idx = config.get("camera_index", 0)
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.get("frame_width", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.get("frame_height", 720))

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    hands = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    fps_stat = RunningStat(30)
    ear_low_frames = 0
    pose_off_frames = 0
    hand_near_frames = 0

    events = []
    save_events = config.get("save_events_csv", True)
    events_path = Path(config.get("events_csv_path", "events_log.csv"))

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Face
        f_res = face_mesh.process(frame_rgb)
        h, w = frame.shape[:2]

        status_msgs = []
        event_triggered = False

        if f_res.multi_face_landmarks:
            face_lms = f_res.multi_face_landmarks[0]
            pts = np.array([(lm.x*w, lm.y*h) for lm in face_lms.landmark], dtype=np.float32)

            # EAR for drowsiness
            le, re = compute_ears(pts)
            ear = (le+re)/2.0
            if ear < config["ear_sleep_threshold"]:
                ear_low_frames += 1
            else:
                ear_low_frames = 0
            if ear_low_frames >= config["ear_sleep_frames"]:
                status_msgs.append("DROWSINESS DETECTED")
                event_triggered = True

            # Head pose
            pose = estimate_head_pose(pts, frame.shape)
            if pose is not None:
                yaw, pitch = abs(pose["yaw"]), abs(pose["pitch"])
                if yaw > config["yaw_abs_threshold"] or pitch > config["pitch_abs_threshold"]:
                    pose_off_frames += 1
                else:
                    pose_off_frames = 0
                if pose_off_frames >= config["pose_frames"]:
                    status_msgs.append("LOOKING AWAY")
                    event_triggered = True

            # Hands near face (possible phone use)
            hand_res = hands.process(frame_rgb)
            head_size = norm_head_size(pts)
            nose = pts[LANDMARKS_IDXS["nose_tip"]]

            if hand_res.multi_hand_landmarks:
                # use first hand
                hlm = hand_res.multi_hand_landmarks[0].landmark
                hpts = np.array([(l.x*w, l.y*h) for l in hlm], dtype=np.float32)
                near = hand_near_face(nose, hpts, head_size, config["hand_face_distance_norm"])
            else:
                near = False

            if near:
                hand_near_frames += 1
            else:
                hand_near_frames = 0
            if hand_near_frames >= config["hand_near_face_frames"]:
                status_msgs.append("POSSIBLE PHONE USE")
                event_triggered = True

            # Overlay
            if config.get("overlay_enabled", True):
                cv2.circle(frame, tuple(nose.astype(int)), 5, (0,255,255), -1)
                draw_text(frame, f"EAR={ear:.3f}", (20,40), (0,255,0))
                if pose is not None:
                    draw_text(frame, f"Yaw={pose['yaw']:.1f} Pitch={pose['pitch']:.1f}", (20,70), (255,255,0))

        # Alerts
        if event_triggered and config.get("beep_enabled", True):
            beep(config.get("beep_frequency", 880), config.get("beep_duration_ms", 300))
        if status_msgs:
            draw_text(frame, " | ".join(status_msgs), (20,110), (0,0,255))
            # Log event
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            events.append({"time": ts, "messages": "|".join(status_msgs)})

        # FPS calc
        now = time.time()
        fps = 1.0 / max(1e-6, (now - last_time))
        last_time = now
        fps_stat.add(fps)
        draw_text(frame, f"FPS: {fps_stat.mean():.1f}", (20, h-20), (200,200,200))

        cv2.imshow("Driver Monitoring Prototype", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if save_events and events:
        with open(events_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["time", "messages"])
            writer.writeheader()
            writer.writerows(events)
        print(f"Saved events to {events_path.resolve()}")

if __name__ == "__main__":
    main()
