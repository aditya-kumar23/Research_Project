"""Launch multiple agents for centralized SLAM (with GT + calibration)."""

from typing import List
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterValue


def _configure_agents(context, *args, **kwargs) -> List[GroupAction]:
    """Create a group of actions for each agent based on the comma-separated list."""
    agents_str = LaunchConfiguration('agents').perform(context)
    agents = [a.strip() for a in agents_str.split(',') if a.strip()]

    # Typed front-end params
    voxel_leaf_size = ParameterValue(LaunchConfiguration('voxel_leaf_size'), value_type=float)
    max_iterations = ParameterValue(LaunchConfiguration('max_iterations'), value_type=int)
    max_corr_dist = ParameterValue(LaunchConfiguration('max_correspondence_distance'), value_type=float)

    # Optional dirs
    gt_dir = LaunchConfiguration('gt_dir').perform(context)
    calib_dir = LaunchConfiguration('calib_dir').perform(context)

    groups: List[GroupAction] = []
    for agent in agents:
        # Files
        gt_file = os.path.join(gt_dir, f"{agent.lower()}_gt.txt") if gt_dir else ""
        tf_file = os.path.join(calib_dir, f"{agent.lower()}_tf.yaml") if calib_dir else ""

        per_agent = [
            # 1) Namespace
            PushRosNamespace(agent),

            # 2) Agent interface: republish raw sensor topics
            Node(
                package='slam_interface',
                executable='agent_node',
                name='agent_interface',
                parameters=[{'agent_name': agent}],
                output='screen',
            ),

            # 3) ICP front-end: LiDAR → relative pose
            Node(
                package='slam_icp_frontend',
                executable='lidar_icp_node',
                name='icp_frontend',
                parameters=[{
                    'voxel_leaf_size': voxel_leaf_size,
                    'max_iterations': max_iterations,
                    'max_correspondence_distance': max_corr_dist,
                }],
                output='screen',
            ),

            # 4) Centralized backend: fuse IMU & ICP → /path (in 'map')
            Node(
                package='slam_centralized',
                executable='backend_node',
                name='central_node',
                parameters=[{'agent_name': agent}],
                output='screen',
            ),
        ]

        # 5) Static TF (map → <agent>_base)
        if tf_file:
            per_agent.append(
                Node(
                    package='slam_tools',
                    executable='static_tf_from_yaml',
                    name='calib_tf',
                    parameters=[{'yaml_file': tf_file}],
                    output='screen',
                )
            )

        # 6) Ground-truth publisher (S3E format)
        if gt_file:
            per_agent.extend([
                Node(
                    package='slam_tools',
                    executable='gt_publisher_s3e',
                    name='gt_pub',
                    parameters=[{
                        'agent_name': agent,
                        'gt_file': gt_file,
                        'rate_hz': 20.0,
                    }],
                    output='screen',
                ),

                # 7) One-shot alignment (TF: map → <agent>_gt)
                Node(
                    package='slam_tools',
                    executable='gt_align_once',
                    name='gt_align',
                    parameters=[{'agent_name': agent}],
                    output='screen',
                ),
            ])

        groups.append(GroupAction(per_agent))

    return groups


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('agents', default_value='Alpha,Bob,Carol',
                              description='Comma-separated list of agent names'),
        DeclareLaunchArgument('voxel_leaf_size', default_value='0.3',
                              description='Leaf size for the voxel grid filter (metres)'),
        DeclareLaunchArgument('max_iterations', default_value='20',
                              description='Maximum number of ICP iterations'),
        DeclareLaunchArgument('max_correspondence_distance', default_value='1.0',
                              description='Maximum correspondence distance for ICP (metres)'),
        DeclareLaunchArgument('gt_dir', default_value='',
                              description='Directory with <agent>_gt.txt files'),
        DeclareLaunchArgument('calib_dir', default_value='',
                              description='Directory with <agent>_tf.yaml files for static TF'),
        OpaqueFunction(function=_configure_agents),
    ])

