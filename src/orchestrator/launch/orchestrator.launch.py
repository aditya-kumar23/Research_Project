#!/usr/bin/env python3
from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable

"""
orchestrator.launch.py (GNOME-terminal variant)
------------------------------------------------
Spawns two windows:
  1) Multi-agent centralized SLAM pipeline (Alpha, Bob, Carlo)
  2) S3E dataset playback (all three robots)
"""

def generate_launch_description():
    # 1) Enable DDS statistics globally
    enable_dds_stats = SetEnvironmentVariable(
        name='FASTRTPS_STATISTICS',
        value='1'
    )

    # 2) Helper to open a gnome-terminal window running `cmd`
    def gnome_terminal(cmd: str, title: str):
        return ExecuteProcess(
            cmd=[
                'gnome-terminal', '--', 'bash', '-c',
                # splash a green title, run the command, then keep shell open
                f'echo -e "\\e[1;32m{title}\\e[0m"; {cmd}; exec bash'
            ],
            output='screen'
        )

    return LaunchDescription([
        enable_dds_stats,

        # — Multi-agent SLAM orchestrator —
        gnome_terminal(
            'ros2 launch orchestrator multi_agent_centralized.launch.py',
            'Multi-Agent Centralized SLAM'
        ),

        # — S3E dataset playback (Alpha, Bob, Carlo) —
        gnome_terminal(
            'ros2 launch data_playback multi_agent_playback.launch.py',
            'Dataset Playback'
        ),
    ])

