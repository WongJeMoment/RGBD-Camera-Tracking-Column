import numpy as np
import open3d as o3d

def fit_cylinder_pca(cluster: o3d.geometry.PointCloud,
                     plane_model=None,
                     inlier_rad_tol=0.01):
    """
    返回 dict:
      axis_dir: (3,)
      p0: (3,) 轴线上参考点
      radius: float
      base_center: (3,) 若提供 plane_model 才算
      inlier_ratio: float
    """
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

    # 2) 用 mean 作为轴线上点（工程上够用）
    p0 = mean

    # 3) 半径：点到轴线的垂距
    # ri = || (p - p0) - ((p-p0)·v) v ||
    proj = (X @ v)[:, None] * v[None, :]
    perp = X - proj
    ri = np.linalg.norm(perp, axis=1)
    r = float(np.median(ri))

    # 4) 估计 inliers（用于质量评估）
    inliers = np.abs(ri - r) < inlier_rad_tol
    inlier_ratio = float(inliers.mean())

    result = {
        "axis_dir": v,
        "p0": p0,
        "radius": r,
        "inlier_ratio": inlier_ratio,
    }

    # 5) 底面中心（轴线与桌面平面交点）
    if plane_model is not None:
        a, b, c, d = plane_model
        n = np.array([a, b, c], dtype=np.float64)
        nv = float(n @ v)
        if abs(nv) > 1e-6:
            t = -(n @ p0 + d) / nv
            base_center = p0 + t * v
            result["base_center"] = base_center
        else:
            # 轴线几乎平行桌面，交点不稳定
            result["base_center"] = None

    return result
