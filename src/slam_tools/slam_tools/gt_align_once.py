#!/usr/bin/env python3
import math, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import StaticTransformBroadcaster

def yaw(q):  # yaw from quaternion
    s = 2.0*(q.w*q.z + q.x*q.y)
    c = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(s, c)

class GTAlignOnce(Node):
    def __init__(self):
        super().__init__('gt_align_once')
        self.declare_parameter('agent_name', 'Alpha')
        self.agent = self.get_parameter('agent_name').value
        self.pub = StaticTransformBroadcaster(self)
        self.gt=None; self.slam=None; self.done=False
        self.create_subscription(PoseStamped, f'/{self.agent}/gt_pose', self.cb_gt,   10)
        self.create_subscription(PoseStamped, f'/{self.agent}/icp_odom', self.cb_slam, 10)

    def cb_gt(self, m):   self.gt=m;   self.try_pub()
    def cb_slam(self, m): self.slam=m; self.try_pub()

    def try_pub(self):
        if self.done or self.gt is None or self.slam is None:
            return
        # Compute yaw-only alignment + translation so first GT lines up with first SLAM
        gz = self.gt.pose.orientation; sz = self.slam.pose.orientation
        dyaw = yaw(sz) - yaw(gz)
        cos, sin = math.cos(dyaw), math.sin(dyaw)
        gx,gy,gz_ = self.gt.pose.position.x, self.gt.pose.position.y, self.gt.pose.position.z
        sx,sy,sz_ = self.slam.pose.position.x, self.slam.pose.position.y, self.slam.pose.position.z
        tx = sx - (cos*gx - sin*gy)
        ty = sy - (sin*gx + cos*gy)
        tz = sz_ - gz_

        tf = TransformStamped()
        tf.header.frame_id = 'map'
        tf.child_frame_id  = f'{self.agent}_gt'
        tf.transform.translation.x = tx
        tf.transform.translation.y = ty
        tf.transform.translation.z = tz
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = math.sin(dyaw*0.5)
        tf.transform.rotation.w = math.cos(dyaw*0.5)

        self.pub.sendTransform(tf)
        self.get_logger().info(f'Aligned GT: static TF map -> {self.agent}_gt published.')
        self.done = True

def main(args=None):
    rclpy.init(args=args)
    node = GTAlignOnce()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

