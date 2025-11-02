from __future__ import annotations
import numpy as np
import cv2

# 3D model points (generic face model) in mm; adapted for solvePnP
# Using: nose tip, chin, left eye corner, right eye corner, left mouth corner, right mouth corner
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -63.6, -12.5),         # Chin
    (-43.3, 32.7, -26.0),        # Left eye left corner
    (43.3, 32.7, -26.0),         # Right eye right corner
    (-28.9, -28.9, -24.1),       # Left Mouth corner
    (28.9, -28.9, -24.1)         # Right mouth corner
], dtype=np.float64)

# Corresponding FaceMesh landmark indices (approximate mapping)
LANDMARKS_IDXS = {
    "nose_tip": 1,      # nose tip approx
    "chin": 152,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth": 61,
    "right_mouth": 291
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

    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        return None

    # Convert rotation vector to Euler angles (degrees)
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
