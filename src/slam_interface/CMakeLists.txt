cmake_minimum_required(VERSION 3.8)
project(slam_interface)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Keyframe.msg"
)

find_package(ament_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

ament_package()
