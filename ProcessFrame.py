import cv2
import numpy as np

from DbscanClusters import dbscan_clusters
from DepthtoPointcloudO3d import depth_to_pointcloud_o3d
from SegmentTablePlane import segment_table_plane
from FitCylinderPca import fit_cylinder_pca
from DrawClusteronDepth import *

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


def _debug_show_filtered_rgb(color_bgr,
                             v_min=180, v_max=255,
                             s_min=0,   s_max=40,
                             win="Filtered RGB (white)"):
    """
    显示基于 HSV 阈值过滤后的图像（像素级）：
      v_min<=V<=v_max 且 s_min<=S<=s_max
    左：mask；右：过滤后的彩色图
    """
    if color_bgr is None or color_bgr.size == 0:
        return

    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    mask = ((v >= v_min) & (v <= v_max) & (s >= s_min) & (s <= s_max)).astype(np.uint8) * 255
    filtered_bgr = cv2.bitwise_and(color_bgr, color_bgr, mask=mask)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    vis = np.hstack((mask_bgr, filtered_bgr))

    cv2.putText(vis, f"V:[{v_min},{v_max}]  S:[{s_min},{s_max}]",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(win, vis)
    cv2.waitKey(1)


def process_frame(depth_m, color_bgr, intr_color):
    # ====== 白色阈值：你可以在这里改（最小改动点）======
    # 你原先写的是 v_min=60, s_max=20，这里我给你完整上下限
    v_min, v_max = 120, 180
    s_min, s_max = 0, 40
    # ===================================================

    # ✅ 0) 可视化：过滤后的RGB（像素级）
    _debug_show_filtered_rgb(color_bgr,
                             v_min=v_min, v_max=v_max,
                             s_min=s_min, s_max=s_max)

    # 1) 点云
    pcd = depth_to_pointcloud_o3d(depth_m, color_bgr, intr_color,
                                  depth_min=0.15, depth_max=1.5)

    # 可选：降采样 + 去离群（强烈建议）
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 2) 桌面平面
    plane_model, table, remain = segment_table_plane(
        pcd, distance_threshold=0.005, num_iterations=2000
    )

    # ✅ 2.5) 点云色彩过滤（HSV区间）
    remain = filter_white_points_range(remain,
                                       v_min=v_min, v_max=v_max,
                                       s_min=s_min, s_max=s_max,
                                       min_keep=80)

    # 3) 聚类
    clusters = dbscan_clusters(remain, eps=0.02, min_points=60, min_cluster_size=150)

    # 4) 对最大簇拟合圆柱（你也可以遍历所有簇）
    # 4) 对最大簇拟合圆柱
    if not clusters:
        return None

    target = clusters[0]
    fit = fit_cylinder_pca(target, plane_model=plane_model, inlier_rad_tol=0.01)

    # ✅ 4.5) 把聚类（圆柱体点云）投影回深度图并显示
    # target.points 是相机坐标系下的 xyz（通常 depth_to_pointcloud_o3d 就是用相机坐标生成的）
    uv = project_points_to_image(
        np.asarray(target.points),
        intr_color=intr_color,
        img_shape_hw=depth_m.shape[:2]
    )
    draw_cluster_on_depth(depth_m, uv, win="Depth + Cylinder Cluster")

    return {
        "pcd": pcd,
        "table": table,
        "remain": remain,
        "clusters": clusters,
        "plane_model": plane_model,
        "cylinder": fit
    }
