import numpy as np
import open3d as o3d

def fit_cylinder_pca(cluster: o3d.geometry.PointCloud,
                     plane_model=None,
                     inlier_rad_tol=0.01,
                     min_hd_ratio=0.5,      # H/D 下限
                     max_hd_ratio=20.0,     # H/D 上限
                     use_inliers_for_height=True):
    """
    返回 dict 或 None（被过滤时）:
      axis_dir: (3,)
      p0: (3,)
      radius: float
      diameter: float
      height: float
      hd_ratio: float
      base_center: (3,) 若提供 plane_model 才算
      inlier_ratio: float
    """
    # 点云点数判断
    pts = np.asarray(cluster.points)
    if pts.shape[0] < 50:
        return None

    # 1) PCA 轴线方向
    mean = pts.mean(axis=0)
    X = pts - mean
    C = (X.T @ X) / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(C)
    v = eigvecs[:, np.argmax(eigvals)]
    v = v / (np.linalg.norm(v) + 1e-12)

    # 2) 轴线上参考点
    p0 = mean

    # 3) 半径估计
    proj = (X @ v)[:, None] * v[None, :]
    perp = X - proj
    ri = np.linalg.norm(perp, axis=1)
    r = float(np.median(ri))

    # 4) inliers
    inliers = np.abs(ri - r) < inlier_rad_tol
    inlier_ratio = float(inliers.mean())

    # ---------- 高度 & 直径比例过滤 ----------
    s = X @ v  # 轴向坐标

    if use_inliers_for_height and inliers.any():
        s_used = s[inliers]
    else:
        s_used = s

    height = float(s_used.max() - s_used.min())
    diameter = float(2.0 * r)
    hd_ratio = float(height / (diameter + 1e-12))

    if not (min_hd_ratio <= hd_ratio <= max_hd_ratio):
        return None
    # --------------------------------------

    result = {
        "axis_dir": v,
        "p0": p0,
        "radius": r,
        "diameter": diameter,
        "height": height,
        "hd_ratio": hd_ratio,
        "inlier_ratio": inlier_ratio,
    }

    # 5) 底面中心
    if plane_model is not None:
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        nv = float(n @ v)
        if abs(nv) > 1e-6:
            t = -(n @ p0 + d) / nv
            result["base_center"] = p0 + t * v
        else:
            result["base_center"] = None

    return result
