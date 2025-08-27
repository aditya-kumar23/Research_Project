#!/usr/bin/env python3
import math, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

class GTPublisher(Node):
    def __init__(self):
        super().__init__('gt_publisher_s3e')
        self.declare_parameter('agent_name', 'Alpha')
        self.declare_parameter('gt_file', '')
        self.declare_parameter('rate_hz', 20.0)

        self.agent = self.get_parameter('agent_name').value
        gt_file = self.get_parameter('gt_file').value
        rate = float(self.get_parameter('rate_hz').value)
        self.frame = f'{self.agent}_gt'

        self.rows = []
        with open(gt_file, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s[0] == '#':
                    continue
                parts = [p for p in s.replace(',', ' ').split() if p]
                if len(parts) < 5:
                    continue
                t = float(parts[0]); x=float(parts[1]); y=float(parts[2]); z=float(parts[3]); yaw=float(parts[4])
                qx=0.0; qy=0.0; qz=math.sin(yaw*0.5); qw=math.cos(yaw*0.5)
                ps = PoseStamped()
                ps.header.frame_id = self.frame
                ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = x, y, z
                ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx,qy,qz,qw
                self.rows.append(ps)

        self.pub_pose = self.create_publisher(PoseStamped, 'gt_pose', 10)
        self.pub_path = self.create_publisher(Path, 'gt_path', 10)
        self.path = Path(); self.path.header.frame_id = self.frame

        self.idx = 0
        self.timer = self.create_timer(1.0/max(rate,1e-3), self.tick)
        self.get_logger().info(f'[{self.agent}] GT loaded: {len(self.rows)} samples from {gt_file}')

    def tick(self):
        if self.idx >= len(self.rows):
            return
        now = self.get_clock().now().to_msg()
        ps = self.rows[self.idx]
        ps.header.stamp = now
        self.pub_pose.publish(ps)
        self.path.header.stamp = now
        self.path.poses.append(ps)
        self.pub_path.publish(self.path)
        self.idx += 1

def main(args=None):
    rclpy.init(args=args)
    node = GTPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

