# Example keyframe message

# Header for timestamp and frame_id
std_msgs/Header header

# The ID of the agent that generated this keyframe
string agent_id

# TIme stamps 
builtin_interfaces/Time stamp

# The pose of the agent at the time of the keyframe
geometry_msgs/Pose pose

