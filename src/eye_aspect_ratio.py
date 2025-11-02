from __future__ import annotations
import numpy as np

# Eye aspect ratio using 6 landmarks around the eye (MediaPipe Face Mesh indices)
# We'll use left eye indices and right eye indices suitable for EAR-like computation.
# These sets approximate Dlib's 6-point EAR layout.
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # [p1, p2, p3, p4, p5, p6]
RIGHT_EYE = [263, 387, 385, 362, 380, 373]

def ear_from_points(pts):
    # pts: array of shape (6,2) in image coords
    p1, p2, p3, p4, p5, p6 = pts
    vert = np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)
    horiz = np.linalg.norm(p1-p4) + 1e-6
    return float(vert / (2.0*horiz))

def compute_ears(landmarks):
    # landmarks: np.array shape (468,2) for FaceMesh 2D landmarks
    le = landmarks[LEFT_EYE, :]
    re = landmarks[RIGHT_EYE, :]
    return ear_from_points(le), ear_from_points(re)
