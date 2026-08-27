import os
import sys
import struct
import numpy as np
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker, MarkerArray

def read_ply_points_and_colors(ply_path):
    with open(ply_path, 'rb') as f:
        header = []
        num_vertices = 0
        is_binary = False
        while True:
            line = f.readline().decode('ascii', errors='ignore').strip()
            header.append(line)
            if line.startswith('element vertex'):
                num_vertices = int(line.split()[2])
            elif 'format binary_little_endian' in line:
                is_binary = True
            elif line == 'end_header':
                break
        
        if is_binary:
            dt = np.dtype([
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                ('curvature', '<f4')
            ])
            data = np.fromfile(f, dtype=dt, count=num_vertices)
            pts = np.vstack([data['x'], data['y'], data['z']]).T
            colors = np.vstack([data['r'], data['g'], data['b']]).T
            return pts, colors
        else:
            pts, colors = [], []
            for _ in range(num_vertices):
                parts = f.readline().decode('ascii').strip().split()
                if len(parts) >= 6:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
            return np.array(pts), np.array(colors)

class RVizGridPublisher(Node):
    def __init__(self):
        super().__init__('rviz_grid_publisher')
        self.cloud_pub = self.create_publisher(PointCloud2, '/cloud_map', 10)
        self.path_pub = self.create_publisher(Path, '/trajectory_path', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/keyframe_markers', 10)
        
        ply_path = 'output/rtabmap_3d_room_cloud.ply'
        csv_path = 'output/vio_loop.csv'
        
        self.get_logger().info(f'Loading PLY point cloud from {ply_path}...')
        self.pts, self.colors = read_ply_points_and_colors(ply_path)
        valid = np.all(np.isfinite(self.pts), axis=1)
        self.pts = self.pts[valid]
        self.colors = self.colors[valid]
        
        # Build PointCloud2 message
        self.pc2_msg = self.build_pointcloud2_msg(self.pts, self.colors)
        
        # Build Path message
        self.get_logger().info(f'Loading CSV trajectory from {csv_path}...')
        self.path_msg, self.marker_array = self.build_trajectory_msg(csv_path)
        
        # Timer to publish continuously
        self.timer = self.create_timer(0.5, self.publish_all)
        self.get_logger().info('RViz Grid Publisher Ready! Publishing at 2 Hz...')

    def build_pointcloud2_msg(self, pts, colors):
        msg = PointCloud2()
        msg.header.frame_id = 'map'
        msg.height = 1
        msg.width = len(pts)
        msg.is_dense = True
        msg.is_bigendian = False
        
        msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]
        msg.point_step = 16
        msg.row_step = msg.point_step * len(pts)
        
        # Pack x, y, z, rgb(uint32)
        r = colors[:, 0].astype(np.uint32)
        g = colors[:, 1].astype(np.uint32)
        b = colors[:, 2].astype(np.uint32)
        rgb_packed = (r << 16) | (g << 8) | b
        
        buffer = bytearray(msg.row_step)
        for i in range(len(pts)):
            offset = i * 16
            struct.pack_into('<fffI', buffer, offset, pts[i,0], pts[i,1], pts[i,2], rgb_packed[i])
            
        msg.data = bytes(buffer)
        return msg

    def build_trajectory_msg(self, csv_path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        
        marker_array = MarkerArray()
        
        if not os.path.exists(csv_path):
            return path_msg, marker_array
            
        rows = []
        with open(csv_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 8:
                    try:
                        vals = [float(p) for p in parts[:8]]
                        rows.append(vals)
                    except ValueError:
                        continue
                        
        if not rows:
            return path_msg, marker_array
            
        data = np.array(rows)
        step = max(1, len(data) // 300)
        
        for idx in range(0, len(data), step):
            row = data[idx]
            px, py, pz = row[1], row[2], row[3]
            qx, qy, qz, qw = row[4], row[5], row[6], row[7]
            
            pose = PoseStamped()
            pose.header.frame_id = 'map'
            pose.pose.position.x = px
            pose.pose.position.y = py
            pose.pose.position.z = pz
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path_msg.poses.append(pose)
            
            if idx % (step * 5) == 0:
                marker = Marker()
                marker.header.frame_id = 'map'
                marker.id = idx
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose = pose.pose
                marker.scale.x = 0.15
                marker.scale.y = 0.15
                marker.scale.z = 0.15
                marker.color.r = 0.1
                marker.color.g = 0.5
                marker.color.b = 1.0
                marker.color.a = 0.9
                marker_array.markers.append(marker)
                
        return path_msg, marker_array

    def publish_all(self):
        now = self.get_clock().now().to_msg()
        self.pc2_msg.header.stamp = now
        self.path_msg.header.stamp = now
        for m in self.marker_array.markers:
            m.header.stamp = now
            
        self.cloud_pub.publish(self.pc2_msg)
        self.path_pub.publish(self.path_msg)
        self.marker_pub.publish(self.marker_array)

def main():
    rclpy.init()
    node = RVizGridPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
