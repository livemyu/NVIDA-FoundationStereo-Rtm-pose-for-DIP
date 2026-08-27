#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np

class OmniFPVViewer(Node):
    def __init__(self):
        super().__init__('omni_fpv_viewer_node')
        
        # Subscribe to left camera image using sensor data QoS profile (BEST_EFFORT)
        self.sub = self.create_subscription(
            Image,
            '/camera/left/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.bridge = CvBridge()
        
        # New Kalibr MEI parameters for left camera
        self.xi = np.array([2.3464045], dtype=np.float64)
        self.D = np.array([-0.06083223, 0.47276886, -0.00012874, 0.00053027], dtype=np.float64)
        self.K = np.array([
            [1203.78775629, 0.0, 943.867371],
            [0.0, 1202.78765827, 605.25993813],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Desired output pinhole parameters (FPV)
        self.out_width = 1920
        self.out_height = 1200
        # f_new around 600-800 gives a good balance of FOV and clarity for 1080p
        f_new = 600.0
        self.K_new = np.array([
            [f_new, 0.0, self.out_width / 2.0],
            [0.0, f_new, self.out_height / 2.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        self.get_logger().info("Omni FPV Viewer initialized. Waiting for /camera/left/image_raw...")
        self.get_logger().info("Please play your rosbag in another terminal: ros2 bag play <bag_dir>")
        self.map1 = None
        self.map2 = None

    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV image
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Initialize maps on the first frame
            if self.map1 is None or self.map2 is None:
                scale_x = cv_img.shape[1] / 1920.0
                scale_y = cv_img.shape[0] / 1200.0
                K_scaled = self.K.copy()
                K_scaled[0, 0] *= scale_x
                K_scaled[0, 2] *= scale_x
                K_scaled[1, 1] *= scale_y
                K_scaled[1, 2] *= scale_y
                
                # Generate high quality mapping (Perspective)
                self.map1, self.map2 = cv2.omnidir.initUndistortRectifyMap(
                    K_scaled, 
                    self.D, 
                    self.xi, 
                    np.eye(3), 
                    self.K_new, 
                    (self.out_width, self.out_height), 
                    cv2.CV_32FC1, 
                    cv2.omnidir.RECTIFY_PERSPECTIVE
                )
            
            # Apply remapping with high-quality cubic interpolation to prevent blurriness
            undistorted = cv2.remap(cv_img, self.map1, self.map2, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
            
            # Show images
            cv2.imshow("Raw Omni Image (Left)", cv2.resize(cv_img, (self.out_width, self.out_height)))
            cv2.imshow("Undistorted FPV View (Left)", undistorted)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = OmniFPVViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
