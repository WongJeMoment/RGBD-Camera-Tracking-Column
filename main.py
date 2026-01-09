import time
import numpy as np
import cv2
import pyrealsense2 as rs

# ===== ROS2 publish =====
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped  # 推荐用 PoseStamped

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


class PosePublisher(Node):
    """ROS2 Node: publish PoseStamped (xyz + quat 0,0,0,1)."""

    def __init__(self,
                 topic_name="/target_pose_stamped",
                 frame_id="base_link"):
        super().__init__("realsense_cylinder_pose_pub")
        self.pub = self.create_publisher(PoseStamped, topic_name, 10)
        self.frame_id = frame_id

    def publish_xyz_quat(self, xyz, quat=(0.0, 0.0, 0.0, 1.0)):
        if xyz is None:
            return

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])

        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.pub.publish(msg)


class RealSenseCylinderContinuous:
    def __init__(self,
                 ros_node: PosePublisher,
                 rgb_size=(640, 480),
                 depth_size=(640, 480),
                 fps=30,
                 align_to_color=True,
                 max_disp_m=3.0,
                 run_every_n=1,
                 publish_every_n=1):
        """
        run_every_n:
          1 = 每帧都跑 process_frame
        publish_every_n:
          1 = 每帧都发布pose
          2/3 = 每2/3帧发布一次（减轻DDS/ROS通信压力）
        """
        self.ros_node = ros_node

        self.rgb_w, self.rgb_h = rgb_size
        self.dep_w, self.dep_h = depth_size
        self.fps = fps
        self.align_to_color = align_to_color
        self.max_disp_m = max_disp_m
        self.run_every_n = max(1, int(run_every_n))
        self.publish_every_n = max(1, int(publish_every_n))

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
        print("[INFO] Continuous output ON (bbox + xyz + ROS2 publish). Press 'q' or ESC to quit.")

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
            while rclpy.ok():
                frames = self.pipeline.wait_for_frames()
                if self.align is not None:
                    frames = self.align.process(frames)

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    rclpy.spin_once(self.ros_node, timeout_sec=0.0)
                    continue

                color_bgr = np.asanyarray(color_frame.get_data())
                depth_u16 = np.asanyarray(depth_frame.get_data())
                depth_m = depth_u16.astype(np.float32) * self.depth_scale

                self._frame_idx += 1

                # ---- 检测 ----
                if (self._frame_idx % self.run_every_n) == 0:
                    result = process_frame(depth_m, color_bgr, COLOR_INTRINSICS)

                    if result is not None and result.get("bbox_uv", None) is not None:
                        self.last_bbox = result.get("bbox_uv", None)
                        self.last_xyz = result.get("bbox_xyz", None)
                        self.last_fit = result.get("cylinder", None)
                    else:
                        self.last_bbox = None
                        self.last_xyz = None
                        self.last_fit = None

                # ---- 发布 PoseStamped：xyz + quat(0,0,0,1) ----
                if self.last_xyz is not None and (self._frame_idx % self.publish_every_n) == 0:
                    self.ros_node.publish_xyz_quat(self.last_xyz, quat=(0.0, 0.0, 0.0, 1.0))

                # 让 ROS2 node 处理内部事件（timer / pub buffer）
                rclpy.spin_once(self.ros_node, timeout_sec=0.0)

                # ---- 可视化 ----
                self._update_fps()
                rgb_disp = color_bgr.copy()
                depth_disp = depth_to_colormap(depth_m, max_disp_m=self.max_disp_m)

                draw_bbox_and_xyz(rgb_disp, self.last_bbox, self.last_xyz, label="CYL", color=(0, 255, 255))
                draw_bbox_and_xyz(depth_disp, self.last_bbox, self.last_xyz, label="CYL", color=(0, 255, 255))

                cv2.putText(rgb_disp, f"FPS: {self._fps:.1f}  run_every_n={self.run_every_n}  pub_every_n={self.publish_every_n}",
                            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
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
    # ===== init ROS2 =====
    rclpy.init()

    # 你可以改 topic / frame_id
    node = PosePublisher(topic_name="/target_pose_stamped", frame_id="base_link")

    runner = RealSenseCylinderContinuous(
        ros_node=node,
        rgb_size=(640, 480),
        depth_size=(640, 480),
        fps=30,
        align_to_color=True,
        max_disp_m=3.0,
        run_every_n=1,        # 检测频率
        publish_every_n=1     # 发布频率（卡顿就改 2 或 3）
    )

    try:
        runner.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
