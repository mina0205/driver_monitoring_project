from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, List

# FPS 평균값을 매끄럽게 계산 -> 부드러운 FPS 표시
@dataclass
class RunningStat:
    size: int
    values: List[float]

    def __init__(self, size: int = 30):
        self.size = size
        self.values = []

    def add(self, v: float):
        self.values.append(v)
        if len(self.values) > self.size:
            self.values.pop(0)

    def mean(self) -> float:
        if not self.values:
            return 0.0
        return sum(self.values)/len(self.values)

def l2(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])
