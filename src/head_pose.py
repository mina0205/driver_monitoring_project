from __future__ import annotations
import numpy as np
import cv2

# 3D model points (generic face model) in mm; adapted for solvePnP
# Using: nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -63.6, -12.5),         # Chin
    (-43.3, 32.7, -26.0),        # Left eye corner (외측 코너 기준)
    (43.3, 32.7, -26.0),         # Right eye corner (외측 코너 기준)
    (-28.9, -28.9, -24.1),       # Left Mouth corner
    (28.9, -28.9, -24.1)         # Right mouth corner
], dtype=np.float64)

# Corresponding Dlib 68-landmark indices
LANDMARKS_IDXS = {
    "nose_tip": 30,      # Dlib Index 30
    "chin": 8,           # Dlib Index 8
    "left_eye_corner": 36, # Dlib Index 36 (외측)
    "right_eye_corner": 45, # Dlib Index 45 (외측)
    "left_mouth": 48,    # Dlib Index 48
    "right_mouth": 54    # Dlib Index 54
}

def estimate_head_pose(landmarks2d, frame_shape):
    h, w = frame_shape[:2]
    image_points = np.array([
        landmarks2d[LANDMARKS_IDXS["nose_tip"]],
        landmarks2d[LANDMARKS_IDXS["chin"]],
        landmarks2d[LANDMARKS_IDXS["left_eye_corner"]],
        landmarks2d[LANDMARKS_IDXS["right_eye_corner"]],
        landmarks2d[LANDMARKS_IDXS["left_mouth"]],
        landmarks2d[LANDMARKS_IDXS["right_mouth"]],
    ], dtype=np.float64)

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))

    # solvePnP를 사용하여 회전 및 이동 벡터 계산
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    # 회전 벡터(rvec)를 오일러 각(Yaw, Pitch, Roll)로 변환 (기존 로직 유지)
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0,0]*rmat[0,0] + rmat[1,0]*rmat[1,0])
    singular = sy < 1e-6
    if not singular:
        yaw = np.degrees(np.arctan2(rmat[2,1], rmat[2,2]))
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll = np.degrees(np.arctan2(rmat[1,0], rmat[0,0]))
    else:
        yaw = np.degrees(np.arctan2(-rmat[1,2], rmat[1,1]))
        pitch = np.degrees(np.arctan2(-rmat[2,0], sy))
        roll = 0

    return {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll), "rvec": rvec, "tvec": tvec, "camera_matrix": camera_matrix}