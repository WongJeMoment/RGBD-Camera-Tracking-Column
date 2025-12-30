import open3d as o3d

def segment_table_plane(pcd: o3d.geometry.PointCloud,
                        distance_threshold=0.005,
                        ransac_n=3,
                        num_iterations=2000):
    """
    返回：plane_model(a,b,c,d), table_pcd, remain_pcd
    """
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    table = pcd.select_by_index(inliers)
    remain = pcd.select_by_index(inliers, invert=True)
    return plane_model, table, remain
