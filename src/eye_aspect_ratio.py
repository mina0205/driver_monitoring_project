from __future__ import annotations
import numpy as np

# Eye aspect ratio (EAR) using 6 landmarks around the eye (Dlib 68-point indices)
# Dlib 인덱스를 사용하여 EAR 공식에 맞는 순서로 재배열:
# P1(36/42), P2(37/43), P3(38/44), P4(39/45), P5(40/46), P6(41/47)
# Dlib 공식: Verticals (P2-P6) + (P3-P5) / Horizontals (P1-P4)
LEFT_EYE = [36, 37, 38, 39, 40, 41]   # P1, P2, P3, P4, P5, P6
RIGHT_EYE = [42, 43, 44, 45, 46, 47]  # P1, P2, P3, P4, P5, P6

def ear_from_points(pts):
    # pts: array of shape (6,2) in image coords
    p1, p2, p3, p4, p5, p6 = pts
    
    # 수직 거리 (Vertical distances)
    vert_dist1 = np.linalg.norm(p2 - p6)  # P2 - P6 (위 눈꺼풀 중간 - 아래 눈꺼풀 중간)
    vert_dist2 = np.linalg.norm(p3 - p5)  # P3 - P5
    vert = vert_dist1 + vert_dist2
    
    # 수평 거리 (Horizontal distance)
    # p1(외측 코너) - p4(내측 코너)
    horiz = np.linalg.norm(p1 - p4) + 1e-6 
    
    # EAR 계산
    return float(vert / (2.0 * horiz))

def compute_ears(landmarks):
    # landmarks: np.array shape (68,2) for Dlib 2D landmarks
    le = landmarks[LEFT_EYE, :]
    re = landmarks[RIGHT_EYE, :]
    return ear_from_points(le), ear_from_points(re)