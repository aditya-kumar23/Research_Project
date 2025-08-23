#!/usr/bin/env python3
import os

from launch import LaunchDescription
from launch.actions import GroupAction          # <— correct import
from launch_ros.actions import Node, PushRosNamespace

# List of agents to launch
AGENTS = ['Alpha'] #, 'Bob', 'Carol'

def generate_agent_group(agent: str) -> GroupAction:
    return GroupAction([
        # 1) Push into /<agent> namespace
        PushRosNamespace(agent),

        # 2) Agent interface: raw → clean topics
        Node(
            package='slam_interface',
            executable='agent_node',
            name='agent_interface',
            parameters=[{'agent_name': agent}],
            output='screen'
        ),

        # 3) ICP frontend: lidar → icp_odom
        Node(
            package='slam_icp_frontend',
            executable='lidar_icp_node',
            name='icp_frontend',
            output='screen'
        ),

        # 4) Centralized backend: icp_odom + imu → path
        Node(
            package='slam_centralized',
            executable='backend_node',
            name='central_node',
            parameters=[{'agent_name': agent}],
            output='screen'
        ),

    ])

def generate_launch_description() -> LaunchDescription:
    ld = LaunchDescription()
    for agent in AGENTS:
        ld.add_action(generate_agent_group(agent))
    return ld

