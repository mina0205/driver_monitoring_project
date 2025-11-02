from __future__ import annotations
from typing import Optional, Dict
import numpy as np
from .utils import l2

# Heuristic: if a hand landmark wrist is near the face center (nose tip) for many frames -> possible phone use.
# We avoid external heavy models; this is a lightweight proxy signal.

def hand_near_face(face_center_xy: np.ndarray, hand_landmarks_xy: Optional[np.ndarray], head_size: float, dist_norm_threshold: float) -> bool:
    if hand_landmarks_xy is None or len(hand_landmarks_xy)==0:
        return False
    # Use wrist (index 0) and index_finger_tip (index 8) from MediaPipe Hands 21-landmarks convention
    wrist = hand_landmarks_xy[0]
    idx_tip = hand_landmarks_xy[8]
    # distance from wrist to face center normalized by head size
    d = l2(wrist, face_center_xy) / (head_size + 1e-6)
    # also require the hand to be somewhat "compact" with phone: wrist to index tip small (holding object)
    compact = l2(wrist, idx_tip) / (head_size + 1e-6) < 0.8
    return d < dist_norm_threshold and compact
