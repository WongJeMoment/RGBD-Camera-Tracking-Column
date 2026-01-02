import cv2
import numpy as np
import open3d as o3d

from DbscanClusters import dbscan_clusters
from DepthtoPointcloudO3d import depth_to_pointcloud_o3d
from SegmentTablePlane import segment_table_plane
from FitCylinderPca import fit_cylinder_pca
from DrawClusteronDepth import project_points_to_image

_TB_INIT = False
_TB_WIN = "HSV Params"

def filter_white_points_range(pcd, v_min=180, v_max=255, s_min=0, s_max=40, min_keep=50):
    """
    在点云上按颜色筛选（HSV区间）：
      保留满足 v_min<=V<=v_max 且 s_min<=S<=s_max 的点
    pcd.colors: Open3D RGB float [0,1]
    """
    if pcd is None or len(pcd.points) == 0 or len(pcd.colors) == 0:
        return pcd

    rgb = (np.asarray(pcd.colors) * 255.0).clip(0, 255).astype(np.uint8)  # (N,3) RGB
    hsv = cv2.cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)

    s = hsv[:, 1]
    v = hsv[:, 2]

    mask = (v >= v_min) & (v <= v_max) & (s >= s_min) & (s <= s_max)
    idx = np.where(mask)[0]

    # 防止阈值太严导致全没了
    if idx.size < min_keep:
        return pcd

    return pcd.select_by_index(idx)


def process_frame(depth_m, color_bgr, intr_color):
    """
    不弹任何额外窗口（不 cv2.imshow），只返回结果给 main 画。
    返回 dict:
      bbox_uv: (u0,v0,u1,v1) or None
      bbox_xyz: (3,) or None
      cylinder: fit dict or None
    """
    # ====== 参数区 ======
    global _TB_INIT, _TB_WIN

    # ====== 参数区（默认值）======
    default_v_min, default_v_max = 120, 198
    default_s_min, default_s_max = 8, 39
    # ============================

    # ✅ 只初始化一次 trackbar
    if not _TB_INIT:
        cv2.namedWindow(_TB_WIN, cv2.WINDOW_NORMAL)
        cv2.createTrackbar("V_min", _TB_WIN, default_v_min, 255, lambda x: None)
        cv2.createTrackbar("V_max", _TB_WIN, default_v_max, 255, lambda x: None)
        cv2.createTrackbar("S_min", _TB_WIN, default_s_min, 255, lambda x: None)
        cv2.createTrackbar("S_max", _TB_WIN, default_s_max, 255, lambda x: None)
        _TB_INIT = True

    # ✅ 每帧读取当前条的值
    v_min = cv2.getTrackbarPos("V_min", _TB_WIN)
    v_max = cv2.getTrackbarPos("V_max", _TB_WIN)
    s_min = cv2.getTrackbarPos("S_min", _TB_WIN)
    s_max = cv2.getTrackbarPos("S_max", _TB_WIN)

    # 防呆：保证 min <= max
    if v_min > v_max:
        v_min, v_max = v_max, v_min
    if s_min > s_max:
        s_min, s_max = s_max, s_min

    min_hd_ratio, max_hd_ratio = 1, 6
    inlier_rad_tol = 0.01
    # ====================

    # 1) 深度 -> 点云（带颜色）
    pcd = depth_to_pointcloud_o3d(depth_m, color_bgr, intr_color,
                                  depth_min=0.15, depth_max=1.5)

    # 降采样 + 去离群
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 2) 分割桌面
    plane_model, table, remain = segment_table_plane(
        pcd, distance_threshold=0.005, num_iterations=2000
    )

    # 3) 白色/低饱和过滤（点云颜色）
    remain = filter_white_points_range(
        remain, v_min=v_min, v_max=v_max, s_min=s_min, s_max=s_max, min_keep=80
    )

    # ======= ✅ HSV过滤可视化：像素级保留原RGB，其余全黑（最直观） =======
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    s_img = hsv[:, :, 1]
    v_img = hsv[:, :, 2]

    mask = ((v_img >= v_min) & (v_img <= v_max) & (s_img >= s_min) & (s_img <= s_max)).astype(np.uint8) * 255
    hsv_vis = cv2.bitwise_and(color_bgr, color_bgr, mask=mask)

    cv2.namedWindow("HSV Filter Visualization", cv2.WINDOW_NORMAL)
    cv2.imshow("HSV Filter Visualization", hsv_vis)
    cv2.waitKey(1)
    # =====================================================================

    # 4) 聚类
    clusters = dbscan_clusters(remain, eps=0.02, min_points=60, min_cluster_size=150)
    if not clusters:
        return {
            "cylinder": None,
            "target_cluster": None,
            "bbox_uv": None,
            "bbox_center_uv": None,
            "bbox_xyz": None,
            "plane_model": plane_model
        }

    # 5) 遍历簇，找第一个通过 H/D 过滤的圆柱
    best_fit = None
    best_cluster = None

    for c in clusters:
        fit = fit_cylinder_pca(
            c,
            plane_model=plane_model,
            inlier_rad_tol=inlier_rad_tol,
            min_hd_ratio=min_hd_ratio,
            max_hd_ratio=max_hd_ratio,
            use_inliers_for_height=True
        )
        if fit is not None:
            best_fit = fit
            best_cluster = c
            break

    if best_fit is None or best_cluster is None:
        return {
            "cylinder": None,
            "target_cluster": None,
            "bbox_uv": None,
            "bbox_center_uv": None,
            "bbox_xyz": None,
            "plane_model": plane_model
        }

    # 6) 计算 bbox（在 aligned 图像坐标系下）
    uv = project_points_to_image(
        np.asarray(best_cluster.points),
        intr_color=intr_color,
        img_shape_hw=depth_m.shape[:2]
    )

    bbox = None
    bbox_center_uv = None
    if uv is not None and len(uv) > 0:
        u = uv[:, 0].astype(np.int32)
        v = uv[:, 1].astype(np.int32)

        # 用百分位裁剪更稳（比 min/max 不容易被离群点拖大）
        u0, u1 = np.percentile(u, [2, 98]).astype(int)
        v0, v1 = np.percentile(v, [2, 98]).astype(int)

        bbox = (int(u0), int(v0), int(u1), int(v1))
        bbox_center_uv = (int((u0 + u1) * 0.5), int((v0 + v1) * 0.5))

    # 7) bbox 对应的 XYZ（推荐：优先用 base_center；否则用簇中心）
    xyz = None
    if best_fit.get("base_center", None) is not None:
        xyz = np.asarray(best_fit["base_center"], dtype=np.float64).reshape(3)
    else:
        xyz = np.asarray(best_cluster.points).mean(axis=0).astype(np.float64)

    return {
        "cylinder": best_fit,
        "target_cluster": best_cluster,
        "bbox_uv": bbox,
        "bbox_center_uv": bbox_center_uv,
        "bbox_xyz": xyz,
        "plane_model": plane_model
    }