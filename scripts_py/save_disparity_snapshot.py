import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage

class DisparitySaver(Node):
    def __init__(self):
        super().__init__('disparity_saver')
        self.sub = self.create_subscription(DisparityImage, '/disparity', self.callback, 10)
        self.get_logger().info('Waiting for /disparity message...')

    def callback(self, msg):
        self.get_logger().info('Received /disparity image, processing...')
        h, w = msg.image.height, msg.image.width
        disp_data = np.frombuffer(msg.image.data, dtype=np.float32).reshape((h, w))
        
        valid_mask = (disp_data > msg.min_disparity) & (disp_data < msg.max_disparity)
        disp_render = np.copy(disp_data)
        disp_render[~valid_mask] = np.nan
        
        out_path = 'output/sgbm_disparity_snapshot.png'
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        im = ax.imshow(disp_render, cmap='turbo')
        ax.set_title("Live SGBM Stereo Disparity Depth Map (/disparity)", fontsize=13, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, label='Disparity (Pixels)')
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        
        self.get_logger().info(f'Saved disparity snapshot to {out_path}')
        rclpy.shutdown()

def main():
    rclpy.init()
    node = DisparitySaver()
    try:
        rclpy.spin(node)
    except Exception:
        pass

if __name__ == '__main__':
    main()
