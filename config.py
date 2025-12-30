# config.py
# 相机内参 & 外参配置
# 适用于 RGB-D 点云生成、RANSAC 平面、DBSCAN、圆柱拟合等

from dataclasses import dataclass
from typing import Tuple
import numpy as np


# ======================
# 相机内参
# ======================
@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int

    fx: float
    fy: float
    cx: float
    cy: float

    distortion_model: str
    dist_coeffs: Tuple[float, float, float, float, float]

    depth_scale: float = 0.001  # RealSense 默认 mm -> m
    depth_min: float = 0.1
    depth_max: float = 2.0

    def K(self) -> np.ndarray:
        """3x3 内参矩阵"""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


# ======================
# 外参（Depth -> Color）
# ======================
@dataclass(frozen=True)
class Extrinsics:
    R: np.ndarray  # 3x3 rotation
    t: np.ndarray  # 3x1 translation (meters)

    def matrix(self) -> np.ndarray:
        """4x4 齐次变换矩阵"""
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T


# ======================
# Depth Camera Intrinsics
# ======================
DEPTH_INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=390.499176,
    fy=390.499176,
    cx=319.888245,
    cy=244.028564,
    distortion_model="brown_conrady",
    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    depth_scale=0.001,
    depth_min=0.1,
    depth_max=2.0,
)


# ======================
# Color Camera Intrinsics
# ======================
COLOR_INTRINSICS = CameraIntrinsics(
    width=640,
    height=480,
    fx=606.359314,
    fy=605.712402,
    cx=326.675079,
    cy=256.005157,
    distortion_model="inverse_brown_conrady",
    dist_coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    depth_scale=0.001,  # 对彩色相机一般不用，但保留统一接口
)


# ======================
# Extrinsics: Depth → Color
# ======================
DEPTH_TO_COLOR_EXTRINSICS = Extrinsics(
    R=np.array(
        [
            [0.999954, -0.007836, -0.005475],
            [0.007875,  0.999943,  0.007242],
            [0.005418, -0.007284,  0.999959],
        ],
        dtype=np.float64,
    ),
    t=np.array([0.014667, 0.000022, -0.000068], dtype=np.float64),
)
