#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import pandas as pd
import os
import sys
import time

class D435PlayerNode(Node):
    def __init__(self, dataset_dir):
        super().__init__('d435_player_node')
        self.dataset_dir = dataset_dir
        self.cam0_dir = os.path.join(dataset_dir, 'mav0', 'cam0')
        self.cam1_dir = os.path.join(dataset_dir, 'mav0', 'cam1')

        self.cam0_csv = os.path.join(self.cam0_dir, 'data.csv')
        self.cam1_csv = os.path.join(self.cam1_dir, 'data.csv')

        if not os.path.exists(self.cam0_csv) or not os.path.exists(self.cam1_csv):
            self.get_logger().error(f"Cannot find cam0/cam1 data.csv in {dataset_dir}")
            sys.exit(1)

        self.df0 = pd.read_csv(self.cam0_csv, comment='#', names=['timestamp', 'filename'])
        self.df1 = pd.read_csv(self.cam1_csv, comment='#', names=['timestamp', 'filename'])

        self.pub_left = self.create_publisher(Image, '/camera/left/image_raw', 10)
        self.pub_right = self.create_publisher(Image, '/camera/right/image_raw', 10)
        self.pub_info_left = self.create_publisher(CameraInfo, '/camera/left/camera_info', 10)
        self.pub_info_right = self.create_publisher(CameraInfo, '/camera/right/camera_info', 10)

        self.bridge = CvBridge()
        self.get_logger().info(f"Loaded D435 EuRoC dataset: {len(self.df0)} left frames, {len(self.df1)} right frames.")

    def play(self):
        min_len = min(len(self.df0), len(self.df1))
        self.get_logger().info(f"Starting playback of {min_len} stereo frames at 30 FPS...")

        for i in range(min_len):
            if not rclpy.ok():
                break

            t0 = time.time()
            row0 = self.df0.iloc[i]
            row1 = self.df1.iloc[i]

            img0_path = os.path.join(self.cam0_dir, 'data', str(row0['filename']).strip())
            img1_path = os.path.join(self.cam1_dir, 'data', str(row1['filename']).strip())

            img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
            img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

            if img0 is None or img1 is None:
                continue

            stamp_ns = int(row0['timestamp'])
            sec = stamp_ns // 1000000000
            nanosec = stamp_ns % 1000000000

            msg0 = self.bridge.cv2_to_imgmsg(img0, encoding='mono8')
            msg0.header.stamp.sec = sec
            msg0.header.stamp.nanosec = nanosec
            msg0.header.frame_id = "camera_0_link"

            msg1 = self.bridge.cv2_to_imgmsg(img1, encoding='mono8')
            msg1.header.stamp.sec = sec
            msg1.header.stamp.nanosec = nanosec
            msg1.header.frame_id = "camera_1_link"

            self.pub_left.publish(msg0)
            self.pub_right.publish(msg1)

            if (i + 1) % 100 == 0 or i == 0:
                self.get_logger().info(f"Published frame {i+1}/{min_len}")

            elapsed = time.time() - t0
            sleep_time = max(0.0, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

        self.get_logger().info("D435 Dataset playback finished.")

def main(args=None):
    rclpy.init(args=args)
    dataset_dir = "/home/elp/picture_resize_recording_NVIDA/d435_vo_20260810_084058"
    if len(sys.argv) > 1:
        dataset_dir = sys.argv[1]

    player = D435PlayerNode(dataset_dir)
    try:
        player.play()
    except KeyboardInterrupt:
        pass
    player.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
