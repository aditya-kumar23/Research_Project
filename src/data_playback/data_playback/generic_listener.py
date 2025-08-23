# generic_listener.py
import rclpy
from rclpy.node import Node
import importlib
import sys

class GenericListener(Node):
    def __init__(self, topic_type, topic_name):
        super().__init__('generic_listener_' + topic_name.replace('/', '_'))

        # Dynamically import the message type
        package_name, msg_name = topic_type.split('/')
        msg_module = importlib.import_module(f'{package_name}.msg')
        msg_class = getattr(msg_module, msg_name)

        self.subscription = self.create_subscription(
            msg_class,
            topic_name,
            self.callback,
            10
        )

        self.get_logger().info(f'Subscribed to [{topic_name}] with type [{topic_type}]')

    def callback(self, msg):
        self.get_logger().info(f"Received message with header: {msg.header.stamp.sec}.{msg.header.stamp.nanosec}")

def main(args=None):
    rclpy.init(args=args)
    if len(sys.argv) < 3:
        print("Usage: ros2 run data_playback generic_listener <package/msg> <topic>")
        sys.exit(1)
    
    topic_type = sys.argv[1]  # e.g., sensor_msgs/Imu
    topic_name = sys.argv[2]  # e.g., /Bob/imu/data

    node = GenericListener(topic_type, topic_name)
    rclpy.spin(node)
    rclpy.shutdown()

