import cv2
import numpy as np

def _get_intrinsics(intr_color):
    """
    兼容几种常见 intr 表示：
    - dict: {'fx':..,'fy':..,'cx':..,'cy':..,'width':..,'height':..}
    - pyrealsense2.intrinsics: 有 fx,fy,ppx,ppy,width,height
    - open3d.camera.PinholeCameraIntrinsic: .intrinsic_matrix, .width, .height
    """
    # dict
    if isinstance(intr_color, dict):
        fx = float(intr_color["fx"])
        fy = float(intr_color["fy"])
        cx = float(intr_color["cx"])
        cy = float(intr_color["cy"])
        w = int(intr_color.get("width", 0))
        h = int(intr_color.get("height", 0))
        return fx, fy, cx, cy, w, h

    # RealSense intrinsics
    if hasattr(intr_color, "fx") and hasattr(intr_color, "fy"):
        fx = float(intr_color.fx)
        fy = float(intr_color.fy)
        # RealSense: ppx/ppy
        cx = float(getattr(intr_color, "ppx", getattr(intr_color, "cx", 0.0)))
        cy = float(getattr(intr_color, "ppy", getattr(intr_color, "cy", 0.0)))
        w = int(getattr(intr_color, "width", 0))
        h = int(getattr(intr_color, "height", 0))
        return fx, fy, cx, cy, w, h

    # Open3D PinholeCameraIntrinsic
    if hasattr(intr_color, "intrinsic_matrix"):
        K = np.asarray(intr_color.intrinsic_matrix, dtype=np.float32)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        w = int(getattr(intr_color, "width", 0))
        h = int(getattr(intr_color, "height", 0))
        return fx, fy, cx, cy, w, h

    raise TypeError(f"Unsupported intr_color type: {type(intr_color)}")


def project_points_to_image(points_xyz, intr_color, img_shape_hw):
    """
    points_xyz: (N,3) 相机坐标系下的点 (X,Y,Z)，单位米
    intr_color: 相机内参
    img_shape_hw: (H,W)
    返回：uv (M,2) int32，只保留落在图像内且Z>0的点
    """
    H, W = img_shape_hw
    fx, fy, cx, cy, _, _ = _get_intrinsics(intr_color)

    xyz = np.asarray(points_xyz, dtype=np.float32)
    z = xyz[:, 2]
    valid = z > 1e-6
    xyz = xyz[valid]
    z = z[valid]

    x = xyz[:, 0]
    y = xyz[:, 1]

    u = (fx * (x / z) + cx)
    v = (fy * (y / z) + cy)

    u = np.round(u).astype(np.int32)
    v = np.round(v).astype(np.int32)

    inside = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    uv = np.stack([u[inside], v[inside]], axis=1)
    return uv

def draw_cluster_on_depth(depth_m, uv, win="Depth + Cylinder Cluster", thickness=2):
    """
    depth_m: (H,W) float 深度(米) 或者 uint16(毫米也行，但显示会不一样)
    uv: (N,2) 像素坐标
    显示：深度伪彩 + 聚类点(红) + bbox(绿) + 轮廓(黄)
    """
    if depth_m is None or depth_m.size == 0:
        return
    H, W = depth_m.shape[:2]
    if uv is None or len(uv) == 0:
        # 只显示深度
        depth_vis = _depth_to_colormap(depth_m)
        cv2.imshow(win, depth_vis)
        cv2.waitKey(1)
        return

    # 1) 深度伪彩
    depth_vis = _depth_to_colormap(depth_m)

    # 2) mask
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[uv[:, 1], uv[:, 0]] = 255

    # 让mask变“连贯”一点（点云投影是稀疏的）
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3) 叠加半透明区域
    overlay = depth_vis.copy()
    overlay[mask > 0] = (0, 0, 255)  # 红色区域
    depth_vis = cv2.addWeighted(depth_vis, 0.75, overlay, 0.25, 0)

    # 4) bbox
    ys, xs = np.where(mask > 0)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    # 5) 轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(depth_vis, contours, -1, (0, 255, 255), thickness)

    cv2.putText(depth_vis, "Cylinder cluster", (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow(win, depth_vis)
    cv2.waitKey(1)


def _depth_to_colormap(depth_m):
    """
    把深度(米)变成可视化伪彩(BGR)
    """
    d = depth_m.astype(np.float32)

    # 把无效深度置0
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d[d < 0] = 0

    # 归一化到 0~255（你可以按场景改最大深度）
    max_d = np.percentile(d[d > 0], 95) if np.any(d > 0) else 1.0
    max_d = max(max_d, 1e-3)

    dn = np.clip(d / max_d, 0, 1)
    dn8 = (dn * 255).astype(np.uint8)

    # 伪彩
    return cv2.applyColorMap(dn8, cv2.COLORMAP_JET)