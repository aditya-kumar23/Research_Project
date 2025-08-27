"""Launch file to play back a ROS 2 bag file.

This launch script declares a `bag_path` argument.  If provided, it
invokes `ros2 bag play` on the given file and enables the simulated clock.
The playback rate can be adjusted via additional arguments (e.g. `--rate`).
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value=os.path.expanduser(
            '~/datasets/S3E/S3E_Playground_1/S3E_Playground_1.db3'
        ),
        description='Path to a SQLite3 bag file to play back'
    )

    # Use LaunchConfiguration to substitute the bag path at runtime
    bag_play = ExecuteProcess(
        cmd=[
            'ros2', 'bag', 'play', LaunchConfiguration('bag_path'),
            '--clock', '--rate', '0.8','--read-ahead-queue-size', '10000',
        ],
        output='screen'
    )

    return LaunchDescription([
        bag_path_arg,
        bag_play,
    ])


