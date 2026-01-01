import numpy as np
import cv2
import open3d as o3d

def filter_white_points(pcd: o3d.geometry.PointCloud,
                        v_min=180,      # 亮度阈值(0~255)
                        s_max=50,       # 饱和度上限(0~255)
                        min_keep=200):
    if len(pcd.points) == 0 or len(pcd.colors) == 0:
        return pcd

    # Open3D colors: float [0,1] RGB
    rgb = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    s = hsv[:, 1]
    v = hsv[:, 2]

    mask = (v >= v_min) & (s <= s_max)
    idx = np.where(mask)[0]

    # 防止阈值过严导致全没了
    if idx.size < min_keep:
        return pcd

    return pcd.select_by_index(idx)
