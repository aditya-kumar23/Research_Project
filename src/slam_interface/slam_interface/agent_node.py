#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2, CompressedImage, Image
from cv_bridge import CvBridge

class AgentNode(Node):
    def __init__(self):
        super().__init__('agent_interface')

        # --- parameters with defaults ---
        self.declare_parameter('agent_name', 'Alpha')
        self.declare_parameter('raw_topic_prefix', '')     # default to empty string

        self.agent = self.get_parameter('agent_name').value
        prefix = self.get_parameter('raw_topic_prefix').value
        # if prefix is empty (falsy), fall back to '/<agent>'
        self.raw = prefix if prefix else f'/{self.agent}'

        # --- publishers on clean, relative topics ---
        self.pub_imu  = self.create_publisher(Imu,         'imu/data',      10)
        self.pub_pc   = self.create_publisher(PointCloud2, 'lidar/points',  1)
        #self.pub_limg = self.create_publisher(Image,       'camera/left',   1)
        #self.pub_rimg = self.create_publisher(Image,       'camera/right',  1)

        # --- subscriptions to raw topics under /<agent> ---
        self.create_subscription(
            Imu,
            f'{self.raw}/imu/data',
            self.cb_imu,
            10
        )
        self.create_subscription(
            PointCloud2,
            f'{self.raw}/velodyne_points',
            self.cb_pc,
            1
        )
        #self.create_subscription(
        #    CompressedImage, f'{self.raw}/left_camera/compressed',
        #    lambda msg: self.cb_img(msg, self.pub_limg), 1)
        #self.create_subscription(
        #    CompressedImage, f'{self.raw}/right_camera/compressed',
        #    lambda msg: self.cb_img(msg, self.pub_rimg), 1)

        self.get_logger().info(
            f"[{self.agent}] Agent interface up — remapping "
            f"{self.raw}/imu/data → imu/data and "
            f"{self.raw}/velodyne_points → lidar/points"
        )

    def cb_imu(self, msg: Imu):
        self.pub_imu.publish(msg)

    def cb_pc(self, msg: PointCloud2):
        self.pub_pc.publish(msg)

    #def cb_img(self, cmsg: CompressedImage, pub: Node):
    #    img = self.bridge.compressed_imgmsg_to_imgmsg(cmsg)
    #    pub.publish(img)

def main(args=None):
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

