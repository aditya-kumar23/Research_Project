# imu_listener.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import sys

class ImuListener(Node):
    def __init__(self, agent_name):
        super().__init__('imu_listener_' + agent_name)
        topic_name = f'/{agent_name}/imu/data'
        self.subscription = self.create_subscription(
            Imu,
            topic_name,
            self.imu_callback,
            10
        )
        self.get_logger().info(f"Subscribed to: {topic_name}")

    def imu_callback(self, msg):
        self.get_logger().info(f"IMU timestamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

def main(args=None):
    rclpy.init(args=args)
    agent = sys.argv[1] if len(sys.argv) > 1 else 'Bob'
    node = ImuListener(agent)
    rclpy.spin(node)
    rclpy.shutdown()

