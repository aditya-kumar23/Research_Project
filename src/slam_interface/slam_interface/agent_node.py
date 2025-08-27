"""Agent interface node for collaborative SLAM.

This node subscribes to raw sensor topics under a given namespace (e.g.
``/Alpha/imu/data`` and ``/Alpha/velodyne_points``) and republishes them
on clean, relative topics (``imu/data`` and ``lidar/points``).  The agent
name can be passed as a ROS 2 parameter, allowing the same executable to
serve multiple robots.
"""

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, PointCloud2


class AgentNode(Node):
    """Republish raw sensor topics under a clean namespace."""

    def __init__(self) -> None:
        super().__init__('agent_interface')

        # Declare configurable parameters
        self.declare_parameter('agent_name', 'Alpha')
        self.declare_parameter('raw_topic_prefix', '')

        self.agent = self.get_parameter('agent_name').value
        prefix = self.get_parameter('raw_topic_prefix').value
        # If no explicit prefix is set, fall back to '/<agent>'
        self.raw = prefix if prefix else f'/{self.agent}'

        # Publishers for clean topics
        self.pub_imu = self.create_publisher(Imu, 'imu/data', 10)
        self.pub_pc = self.create_publisher(PointCloud2, 'lidar/points', 1)

        # Subscriptions to raw topics under the agent namespace
        self.create_subscription(
            Imu,
            f'{self.raw}/imu/data',
            self.cb_imu,
            10,
        )
        self.create_subscription(
            PointCloud2,
            f'{self.raw}/velodyne_points',
            self.cb_pc,
            1,
        )

        self.get_logger().info(
            f"[{self.agent}] Interface ready — republishing "
            f"{self.raw}/imu/data → imu/data and {self.raw}/velodyne_points → lidar/points"
        )

    def cb_imu(self, msg: Imu) -> None:
        """Forward IMU messages directly to the clean topic."""
        self.pub_imu.publish(msg)

    def cb_pc(self, msg: PointCloud2) -> None:
        """Forward point cloud messages directly to the clean topic."""
        self.pub_pc.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AgentNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()