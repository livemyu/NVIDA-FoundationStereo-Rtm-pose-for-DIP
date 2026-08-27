import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage

class LiveDisparityViewer(Node):
    def __init__(self):
        super().__init__('live_disparity_viewer')
        self.sub = self.create_subscription(DisparityImage, '/disparity', self.callback, 10)
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 6), num="SGBM Stereo Disparity Live Window")
        self.im = None
        self.get_logger().info('Live Disparity Window Started! Subscribed to /disparity...')

    def callback(self, msg):
        h, w = msg.image.height, msg.image.width
        disp_data = np.frombuffer(msg.image.data, dtype=np.float32).reshape((h, w))
        
        valid_mask = (disp_data > msg.min_disparity) & (disp_data < msg.max_disparity)
        disp_render = np.copy(disp_data)
        disp_render[~valid_mask] = np.nan
        
        if self.im is None:
            self.im = self.ax.imshow(disp_render, cmap='turbo')
            self.ax.set_title("Live SGBM Stereo Disparity Depth Map (/disparity)", fontsize=12, fontweight='bold')
            self.ax.axis('off')
            self.fig.colorbar(self.im, ax=self.ax, label='Disparity (Pixels)')
        else:
            self.im.set_data(disp_render)
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

def main():
    rclpy.init()
    node = LiveDisparityViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
