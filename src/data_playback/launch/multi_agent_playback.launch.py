from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    bag_path = LaunchConfiguration('bag_path', default=os.path.expanduser('~/datasets/S3E/S3E_Playground_1/S3E_Playground_1.db3'))

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'ros2', 'bag', 'play', bag_path,
                '--clock',                    # Publish /clock for simulated time
                '--rate', '1.0',              # Optional: play slower/faster
                '--remap', '/clock:=/clock'   # Remap clock if needed
            ],
            output='screen'
        )
    ])

