cmake_minimum_required(VERSION 3.5)
project(slam_icp_frontend)

find_package(ament_cmake REQUIRED)
find_package(rclcpp        REQUIRED)
find_package(sensor_msgs   REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)

add_executable(lidar_icp_node src/lidar_icp_node.cpp)
ament_target_dependencies(lidar_icp_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  pcl_conversions
  PCL
)
target_link_libraries(lidar_icp_node ${PCL_LIBRARIES})

install(TARGETS lidar_icp_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()

