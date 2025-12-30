from DbscanClusters import dbscan_clusters
from DepthtoPointcloudO3d import depth_to_pointcloud_o3d
from SegmentTablePlane import segment_table_plane
from FitCylinderPca import fit_cylinder_pca
def process_frame(depth_m, color_bgr, intr_color):
    # 1) 点云
    pcd = depth_to_pointcloud_o3d(depth_m, color_bgr, intr_color, depth_min=0.15, depth_max=1.5)

    # 可选：降采样 + 去离群（强烈建议）
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    # 2) 桌面平面
    plane_model, table, remain = segment_table_plane(
        pcd, distance_threshold=0.005, num_iterations=2000
    )

    # 3) 聚类
    clusters = dbscan_clusters(remain, eps=0.02, min_points=60, min_cluster_size=150)

    # 4) 对最大簇拟合圆柱（你也可以遍历所有簇）
    if not clusters:
        return None

    target = clusters[0]
    fit = fit_cylinder_pca(target, plane_model=plane_model, inlier_rad_tol=0.01)

    return {
        "pcd": pcd,
        "table": table,
        "remain": remain,
        "clusters": clusters,
        "plane_model": plane_model,
        "cylinder": fit
    }

if __name__ == '__main__':
    process_frame()