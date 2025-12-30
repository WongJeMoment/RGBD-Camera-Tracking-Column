import numpy as np
import open3d as o3d
def dbscan_clusters(pcd: o3d.geometry.PointCloud,
                    eps=0.015,
                    min_points=80,
                    min_cluster_size=200):
    """
    返回 clusters: list[o3d.geometry.PointCloud]
    """
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    clusters = []
    if labels.size == 0:
        return clusters

    max_label = labels.max()
    for k in range(max_label + 1):
        idx = np.where(labels == k)[0]
        if idx.size < min_cluster_size:
            continue
        clusters.append(pcd.select_by_index(idx))

    # 大簇优先
    clusters.sort(key=lambda c: np.asarray(c.points).shape[0], reverse=True)
    return clusters
