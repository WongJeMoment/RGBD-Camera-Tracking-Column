import numpy as np
import open3d as o3d
import cv2
def _make_axis_lineset(base_center, axis_dir, length=0.30):
    """画轴线：从 base_center 沿 axis_dir 方向前后各 length/2"""
    base_center = np.asarray(base_center, dtype=np.float64).reshape(3)
    v = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-12)

    p1 = base_center - v * (length * 0.5)
    p2 = base_center + v * (length * 0.5)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([p1, p2]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    # 不指定颜色也能看；如果你想固定颜色可以设置 ls.colors
    return ls

def _make_center_sphere(center, radius=0.01, resolution=20):
    """画底面中心点：一个小球"""
    c = np.asarray(center, dtype=np.float64).reshape(3)
    sph = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sph.translate(c)
    sph.compute_vertex_normals()
    return sph

def _make_radius_ring(center, axis_dir, radius, n=80):
    """
    画一个表示半径的圆环：
    圆环平面垂直于轴线 axis_dir，圆心在 center，半径为 radius
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    v = np.asarray(axis_dir, dtype=np.float64).reshape(3)
    v = v / (np.linalg.norm(v) + 1e-12)

    # 构造两个与 v 正交的单位向量 u, w
    # 先找一个不平行的参考向量
    ref = np.array([1.0, 0.0, 0.0]) if abs(v[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(v, ref)
    u = u / (np.linalg.norm(u) + 1e-12)
    w = np.cross(v, u)
    w = w / (np.linalg.norm(w) + 1e-12)

    theta = np.linspace(0, 2*np.pi, n, endpoint=False)
    pts = c + radius * (np.cos(theta)[:, None] * u[None, :] + np.sin(theta)[:, None] * w[None, :])

    # LineSet 连接成环
    lines = np.stack([np.arange(n), (np.arange(n) + 1) % n], axis=1).astype(np.int32)
    ring = o3d.geometry.LineSet()
    ring.points = o3d.utility.Vector3dVector(pts)
    ring.lines = o3d.utility.Vector2iVector(lines)
    return ring

def visualize_cylinder_result(result, axis_length=0.30):
    """
    result: 你的 process_frame 返回的 dict
      - 推荐包含 result["pcd"]（Open3D 点云）
      - result["cylinder"] 里包含 axis_dir/base_center/radius
    """
    if result is None or result.get("cylinder", None) is None:
        print("[VIS] No cylinder result to visualize.")
        return

    cyl = result["cylinder"]
    axis_dir = cyl["axis_dir"]
    base_center = cyl["base_center"]
    radius = cyl["radius"]

    geoms = []

    # 点云（如果 result 里有）
    pcd = result.get("pcd", None)
    if pcd is not None:
        geoms.append(pcd)

    # 桌面（如果有）
    table = result.get("table", None)
    if table is not None:
        geoms.append(table)

    # 轴线
    geoms.append(_make_axis_lineset(base_center, axis_dir, length=axis_length))

    # 底面中心点
    geoms.append(_make_center_sphere(base_center, radius=max(0.005, radius*0.3)))

    # 半径圆环（可选）
    geoms.append(_make_radius_ring(base_center, axis_dir, radius))

    o3d.visualization.draw_geometries(geoms, window_name="Cylinder Visualization")

def project_point_to_pixel(p_cam, intr):
    """
    p_cam: (3,) in camera coords (meters) [X,Y,Z]
    intr: COLOR_INTRINSICS with fx,fy,cx,cy
    return: (u,v) int or None if invalid
    """
    X, Y, Z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    if Z <= 1e-6 or not np.isfinite(Z):
        return None
    u = intr.fx * (X / Z) + intr.cx
    v = intr.fy * (Y / Z) + intr.cy
    return int(round(u)), int(round(v))

def draw_cylinder_on_rgb(img_bgr, cyl, intr, axis_draw_len_m=0.12):
    """
    img_bgr: OpenCV BGR image (H,W,3)
    cyl: dict with axis_dir, base_center, radius, inlier_ratio(optional)
    intr: COLOR_INTRINSICS
    """
    if cyl is None:
        return img_bgr

    axis_dir = np.asarray(cyl["axis_dir"], dtype=np.float64).reshape(3)
    base_center = np.asarray(cyl["base_center"], dtype=np.float64).reshape(3)
    radius = float(cyl.get("radius", 0.0))
    inlier_ratio = cyl.get("inlier_ratio", None)

    # normalize axis
    axis_dir = axis_dir / (np.linalg.norm(axis_dir) + 1e-12)

    h, w = img_bgr.shape[:2]

    # 1) base_center -> pixel
    uv0 = project_point_to_pixel(base_center, intr)

    # 2) axis arrow endpoint: base_center + axis_dir * L
    p1 = base_center + axis_dir * float(axis_draw_len_m)
    uv1 = project_point_to_pixel(p1, intr)

    overlay = img_bgr  # in-place draw

    # draw base point + arrow
    if uv0 is not None:
        u0, v0 = uv0
        if 0 <= u0 < w and 0 <= v0 < h:
            cv2.circle(overlay, (u0, v0), 6, (0, 255, 255), -1)  # yellow dot
            cv2.putText(overlay, "base", (u0 + 8, v0 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

    if uv0 is not None and uv1 is not None:
        u0, v0 = uv0
        u1, v1 = uv1
        if 0 <= u0 < w and 0 <= v0 < h and 0 <= u1 < w and 0 <= v1 < h:
            cv2.arrowedLine(overlay, (u0, v0), (u1, v1), (255, 0, 255), 2, tipLength=0.2)  # magenta arrow

    # text block (top-left)
    lines = [
        f"[CYL] base XYZ(m): ({base_center[0]:.3f}, {base_center[1]:.3f}, {base_center[2]:.3f})",
        f"[CYL] axis dir:    ({axis_dir[0]:.3f}, {axis_dir[1]:.3f}, {axis_dir[2]:.3f})",
        f"[CYL] radius(m):   {radius:.3f}",
    ]
    if inlier_ratio is not None:
        lines.append(f"[CYL] inlier_ratio:{float(inlier_ratio):.2f}")

    y0 = 25
    for i, t in enumerate(lines):
        cv2.putText(overlay, t, (10, y0 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return overlay
