import time
import numpy as np
import cv2
import pyrealsense2 as rs

from config import COLOR_INTRINSICS
from ProcessFrame import process_frame  # 你刚刚给的 process_frame


def depth_to_colormap(depth_m, max_disp_m=3.0):
    depth_vis = np.clip(depth_m, 0, max_disp_m)
    depth_u8 = (depth_vis / max_disp_m * 255.0).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def draw_bbox_and_xyz(img, bbox, xyz, label="CYL", color=(0, 255, 255)):
    if bbox is None:
        return

    h, w = img.shape[:2]
    u0, v0, u1, v1 = bbox
    u0 = int(np.clip(u0, 0, w - 1))
    u1 = int(np.clip(u1, 0, w - 1))
    v0 = int(np.clip(v0, 0, h - 1))
    v1 = int(np.clip(v1, 0, h - 1))

    cv2.rectangle(img, (u0, v0), (u1, v1), color, 2)
    cv2.putText(img, label, (u0, max(0, v0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    if xyz is not None:
        x, y, z = [float(t) for t in xyz]
        txt = f"XYZ: [{x:.3f}, {y:.3f}, {z:.3f}] m"
        cv2.putText(img, txt, (u0, min(h - 5, v1 + 22)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


class RealSenseCylinderContinuous:
    def __init__(self,
                 rgb_size=(640, 480),
                 depth_size=(640, 480),
                 fps=30,
                 align_to_color=True,
                 max_disp_m=3.0,
                 run_every_n=1):
        """
        run_every_n:
          1 = 每帧都跑 process_frame（最实时，最吃CPU）
          2/3 = 每2/3帧跑一次，其余帧复用上次bbox（更流畅）
        """
        self.rgb_w, self.rgb_h = rgb_size
        self.dep_w, self.dep_h = depth_size
        self.fps = fps
        self.align_to_color = align_to_color
        self.max_disp_m = max_disp_m
        self.run_every_n = max(1, int(run_every_n))

        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.rgb_w, self.rgb_h, rs.format.bgr8, self.fps)
        cfg.enable_stream(rs.stream.depth, self.dep_w, self.dep_h, rs.format.z16, self.fps)
        self.profile = self.pipeline.start(cfg)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())

        self.align = rs.align(rs.stream.color) if self.align_to_color else None

        # cache last detection
        self.last_bbox = None
        self.last_xyz = None
        self.last_fit = None

        # fps
        self._last_time = time.time()
        self._fps = 0.0

        self._frame_idx = 0

        print(f"[INFO] Depth scale: {self.depth_scale:.12f} m/LSB")
        print("[INFO] Continuous output ON (bbox + xyz). Press 'q' or ESC to quit.")

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

    def run(self):
        cv2.namedWindow("RGB | Depth", cv2.WINDOW_NORMAL)

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                if self.align is not None:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                depth_u16 = np.asanyarray(depth_frame.get_data())
                depth_m = depth_u16.astype(np.float32) * self.depth_scale

                # -------- 每 run_every_n 帧跑一次检测，其余帧复用上次结果 --------
                self._frame_idx += 1
                if (self._frame_idx % self.run_every_n) == 0:
                    result = process_frame(depth_m, color_bgr, COLOR_INTRINSICS)

                    if result is not None and result.get("bbox_uv", None) is not None:
                        self.last_bbox = result.get("bbox_uv", None)
                        self.last_xyz = result.get("bbox_xyz", None)
                        self.last_fit = result.get("cylinder", None)
                    else:
                        # 没检测到就清掉（如果你希望“保持上一次框不消失”，把这三行注释掉）
                        self.last_bbox = None
                        self.last_xyz = None
                        self.last_fit = None
                # ----------------------------------------------------------------

                # display images (only RGB + Depth)
                self._update_fps()
                rgb_disp = color_bgr.copy()
                depth_disp = depth_to_colormap(depth_m, max_disp_m=self.max_disp_m)

                # 在两边都画 bbox + xyz（持续输出）
                draw_bbox_and_xyz(rgb_disp, self.last_bbox, self.last_xyz, label="CYL", color=(0, 255, 255))
                draw_bbox_and_xyz(depth_disp, self.last_bbox, self.last_xyz, label="CYL", color=(0, 255, 255))

                # overlay fps / status
                cv2.putText(rgb_disp, f"FPS: {self._fps:.1f}  run_every_n={self.run_every_n}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                if self.last_fit is not None:
                    ir = self.last_fit.get("inlier_ratio", None)
                    hd = self.last_fit.get("hd_ratio", None)
                    if ir is not None:
                        cv2.putText(rgb_disp, f"inlier: {float(ir):.2f}",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2, cv2.LINE_AA)
                    if hd is not None:
                        cv2.putText(rgb_disp, f"H/D: {float(hd):.2f}",
                                    (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2, cv2.LINE_AA)

                # size match then show
                if depth_disp.shape[:2] != rgb_disp.shape[:2]:
                    depth_disp = cv2.resize(depth_disp, (rgb_disp.shape[1], rgb_disp.shape[0]),
                                            interpolation=cv2.INTER_NEAREST)

                combined = np.hstack((rgb_disp, depth_disp))
                cv2.imshow("RGB | Depth", combined)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        finally:
            self.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    runner = RealSenseCylinderContinuous(
        rgb_size=(640, 480),
        depth_size=(640, 480),
        fps=30,
        align_to_color=True,
        max_disp_m=3.0,
        run_every_n=1  # 如果卡顿，把它改成 2 或 3
    )
    runner.run()
