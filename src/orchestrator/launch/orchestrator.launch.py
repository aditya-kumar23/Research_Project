"""Top‑level launch file for the collaborative SLAM demo.

This launch script coordinates the multi‑agent SLAM pipeline and optional
data playback.  It accepts two launch arguments:

* ``agents``: a comma‑separated list of agent names to launch.  For each
  agent, a set of nodes (interface, ICP front‑end and centralized
  back‑end) will be created.
* ``bag_path``: path to a recorded ROS 2 bag file (SQLite3).  When
  provided, a ``ros2 bag play`` process will be started.

The SLAM nodes are brought up via ``multi_agent_centralized.launch.py``.
Data playback uses the ``multi_agent_playback.launch.py`` from the
``data_playback`` package.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def generate_launch_description() -> LaunchDescription:
    # Declare launch arguments
    agents_arg = DeclareLaunchArgument(
        'agents',
        default_value='Alpha',
        description='Comma‑separated list of agent names to launch'
    )
    bag_path_arg = DeclareLaunchArgument(
        'bag_path',
        default_value='',
        description='Path to a SQLite3 bag file to play back (empty for none)'
    )

    # Resolve package share directories
    orchestrator_pkg = get_package_share_directory('orchestrator')
    data_playback_pkg = get_package_share_directory('data_playback')

    # Include the multi‑agent SLAM launch file
    central_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [orchestrator_pkg, '/launch/multi_agent_centralized.launch.py']
        ),
        launch_arguments={
            'agents': LaunchConfiguration('agents'),
        }.items(),
    )

    # Include data playback only if bag_path is provided (empty means disabled)
    playback_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [data_playback_pkg, '/launch/multi_agent_playback.launch.py']
        ),
        launch_arguments={
            'bag_path': LaunchConfiguration('bag_path'),
        }.items(),
    )

    return LaunchDescription([
        agents_arg,
        bag_path_arg,
        central_launch,
        playback_launch,
    ])