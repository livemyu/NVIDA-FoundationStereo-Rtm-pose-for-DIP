#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

class FixBagStepNode(Node):
    def __init__(self):
        super().__init__('fix_bag_step_node')
        
        # 使用 BEST_EFFORT QoS 兼容 rosbag2 录制的数据流
        qos_sub = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        qos_pub = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.sub_left = self.create_subscription(Image, '/camera/left/image_raw', self.left_cb, qos_sub)
        self.sub_right = self.create_subscription(Image, '/camera/right/image_raw', self.right_cb, qos_sub)
        self.pub_left = self.create_publisher(Image, '/camera/left/image_fixed', qos_pub)
        self.pub_right = self.create_publisher(Image, '/camera/right/image_fixed', qos_pub)

    def left_cb(self, msg):
        msg.step = msg.width
        self.pub_left.publish(msg)

    def right_cb(self, msg):
        msg.step = msg.width
        self.pub_right.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = FixBagStepNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
