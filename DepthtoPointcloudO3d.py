import numpy as np
import open3d as o3d

def depth_to_pointcloud_o3d(depth_m: np.ndarray, color_bgr: np.ndarray, intr, depth_min=0.1, depth_max=2.0):
    """
    depth_m: (H,W) float32 meters, already aligned to color
    color_bgr: (H,W,3) uint8 (OpenCV BGR)
    intr: config.COLOR_INTRINSICS (有 width/height/fx/fy/cx/cy)
    """
    assert depth_m.shape[:2] == color_bgr.shape[:2]
    H, W = depth_m.shape

    # Open3D 要求 RGB
    color_rgb = color_bgr[..., ::-1].copy()

    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))
    color_o3d = o3d.geometry.Image(color_rgb.astype(np.uint8))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1.0,         # 因为 depth_m 已经是米
        depth_trunc=depth_max,
        convert_rgb_to_intensity=False
    )

    pinhole = o3d.camera.PinholeCameraIntrinsic(
        intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy
    )

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, pinhole)

    # 二次深度过滤
    pts = np.asarray(pcd.points)
    z = pts[:, 2]
    mask = (z > depth_min) & (z < depth_max) & np.isfinite(z)
    pcd = pcd.select_by_index(np.where(mask)[0])

    return pcd
