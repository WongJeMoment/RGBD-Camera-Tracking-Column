import time
import numpy as np
import cv2
import pyrealsense2 as rs

from ProcessFrame import process_frame
from config import COLOR_INTRINSICS
from Vision import *

class RealSenseCylinderRunner:
    def __init__(self,
                 rgb_size=(640, 480),
                 depth_size=(640, 480),
                 fps=30,
                 align_to_color=True,
                 max_disp_m=3.0):
        self.rgb_w, self.rgb_h = rgb_size
        self.dep_w, self.dep_h = depth_size
        self.fps = fps
        self.align_to_color = align_to_color
        self.max_disp_m = max_disp_m

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.rgb_w, self.rgb_h, rs.format.bgr8, self.fps)
        self.config.enable_stream(rs.stream.depth, self.dep_w, self.dep_h, rs.format.z16, self.fps)

        self.profile = self.pipeline.start(self.config)

        # depth scale (z16 -> meters)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        # Align depth to color
        self.align = rs.align(rs.stream.color) if self.align_to_color else None

        # UI / FPS
        self.click_xy = None
        self._last_time = time.time()
        self._fps = 0.0

        # cache last result
        self.last_cyl = None

        print(f"[INFO] Depth scale: {self.depth_scale:.12f} m/LSB")
        print(f"[INFO] Press 'p' to run process_frame() on current frame.")
        print(f"[INFO] Press 'q' or ESC to quit. Press 'c' to clear click.\n")

    def close(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def _update_fps(self):
        now = time.time()
        dt = now - self._last_time
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps > 0 else (1.0 / dt)
        self._last_time = now

    def _on_mouse(self, event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_xy = (x, y)

    def _draw_overlay(self, color_image, depth_frame, depth_m):
        self._update_fps()
        h, w = color_image.shape[:2]

        # Use principal point from config for center query
        cx_cfg = int(round(COLOR_INTRINSICS.cx))
        cy_cfg = int(round(COLOR_INTRINSICS.cy))
        cx_cfg = int(np.clip(cx_cfg, 0, w - 1))
        cy_cfg = int(np.clip(cy_cfg, 0, h - 1))
        center_depth = depth_frame.get_distance(cx_cfg, cy_cfg)


        # Click depth
        if self.click_xy is not None:
            x, y = self.click_xy
            if 0 <= x < w and 0 <= y < h:
                click_depth = depth_frame.get_distance(x, y)
                cv2.drawMarker(color_image, (x, y), (0, 255, 0),
                               markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        # Draw principal point marker
        cv2.drawMarker(color_image, (cx_cfg, cy_cfg), (255, 255, 255),
                       markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)

        # Draw last cylinder result (text only)
        if self.last_cyl is not None:
            v = self.last_cyl.get("axis_dir", None)
            bc = self.last_cyl.get("base_center", None)
            r = self.last_cyl.get("radius", None)
            ir = self.last_cyl.get("inlier_ratio", None)


        # Put text
        y0 = 25
        return color_image

    def run(self):
        cv2.namedWindow("RGB | Depth", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("RGB | Depth", self._on_mouse)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                if self.align is not None:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(depth_frame.get_data())  # uint16 z16
                color_image = np.asanyarray(color_frame.get_data())  # bgr8

                # depth in meters (float32)
                depth_m = depth_image.astype(np.float32) * self.depth_scale

                # Depth colormap for display
                depth_vis = np.clip(depth_m, 0, self.max_disp_m)
                depth_vis_u8 = (depth_vis / self.max_disp_m * 255.0).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(depth_vis_u8, cv2.COLORMAP_JET)

                # overlay
                color_disp = color_image.copy()
                color_disp = self._draw_overlay(color_disp, depth_frame, depth_m)

                if self.last_cyl is not None and self.last_cyl.get("base_center", None) is not None:
                    base_center = np.asarray(self.last_cyl["base_center"], dtype=np.float64).reshape(3)
                    uv = project_point_to_pixel(base_center, COLOR_INTRINSICS)
                    if uv is not None:
                        u, v = uv
                        h, w = color_disp.shape[:2]
                        if 0 <= u < w and 0 <= v < h:
                            cv2.circle(color_disp, (u, v), 7, (0, 255, 255), -1)
                            cv2.putText(color_disp, "C", (u + 8, v - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (0, 255, 255), 2, cv2.LINE_AA)

                if depth_colormap.shape[:2] != color_disp.shape[:2]:
                    depth_colormap = cv2.resize(depth_colormap, (color_disp.shape[1], color_disp.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)

                combined = np.hstack((color_disp, depth_colormap))
                cv2.imshow("RGB | Depth", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break
                elif key == ord('c'):
                    self.click_xy = None
                elif key == ord('s'):
                    ts = int(time.time() * 1000)
                    cv2.imwrite(f"rgb_{ts}.png", color_image)
                    cv2.imwrite(f"depth_raw_{ts}.png", depth_image)
                    cv2.imwrite(f"depth_color_{ts}.png", depth_colormap)
                    print(f"[SAVE] rgb/depth saved with timestamp {ts}")
                elif key == ord('p'):
                    # 只更新一次圆柱结果（中心点），不做任何可视化弹窗/打印
                    result = process_frame(depth_m, color_image, COLOR_INTRINSICS)
                    if result is None or result.get("cylinder", None) is None:
                        self.last_cyl = None
                    else:
                        self.last_cyl = result["cylinder"]

        finally:
            self.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = RealSenseCylinderRunner(
        rgb_size=(640, 480),
        depth_size=(640, 480),
        fps=30,
        align_to_color=True,
        max_disp_m=3.0
    )
    runner.run()
