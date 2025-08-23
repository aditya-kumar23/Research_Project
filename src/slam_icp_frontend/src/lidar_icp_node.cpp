#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <Eigen/Core>

class LidarIcpNode : public rclcpp::Node {
public:
  LidarIcpNode()
  : Node("lidar_icp_node"),
    has_last_(false),
    last_cloud_(new pcl::PointCloud<pcl::PointXYZ>())
  {
    // Subscribe to the *relative* LiDAR topic under each namespace:
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "lidar/points",  rclcpp::QoS(1),
      std::bind(&LidarIcpNode::cloud_cb, this, std::placeholders::_1));

    // Publish ICP odometry on a clean, relative topic:
    pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "icp_odom", rclcpp::QoS(1));

    // Configure voxel‐grid and ICP
    vg_.setLeafSize(0.3f, 0.3f, 0.3f);
    icp_.setMaximumIterations(20);
    icp_.setMaxCorrespondenceDistance(1.0);

    RCLCPP_INFO(this->get_logger(),
                "LidarIcpNode initialized. Listening on 'lidar/points', publishing on 'icp_odom'.");
  }

private:
  void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1) Convert ROS2 msg → PCL cloud:
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    // 2) Downsample
    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vg_.setInputCloud(cloud);
    vg_.filter(*filtered);

    // 3) On first frame, just store and return
    if (!has_last_) {
      last_cloud_ = filtered;
      has_last_ = true;
      return;
    }

    // 4) Run ICP between filtered and last_cloud_
    auto aligned = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    icp_.setInputSource(filtered);
    icp_.setInputTarget(last_cloud_);
    icp_.align(*aligned);

    // 5) Extract the 4×4 transform
    Eigen::Matrix4f rel = icp_.getFinalTransformation();

    // 6) Build a PoseStamped message
    geometry_msgs::msg::PoseStamped out;
    out.header = msg->header;

    // Translation
    out.pose.position.x = rel(0, 3);
    out.pose.position.y = rel(1, 3);
    out.pose.position.z = rel(2, 3);

    // Rotation (matrix → tf2::Quaternion)
    Eigen::Matrix3f eigR = rel.block<3,3>(0,0);
    tf2::Matrix3x3 R(
      eigR(0,0), eigR(0,1), eigR(0,2),
      eigR(1,0), eigR(1,1), eigR(1,2),
      eigR(2,0), eigR(2,1), eigR(2,2)
    );
    tf2::Quaternion q;
    R.getRotation(q);
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();

    // 7) Publish the ICP result
    pub_->publish(out);

    // 8) Update last_cloud_ for next iteration
    last_cloud_ = filtered;
  }

  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;

  pcl::VoxelGrid<pcl::PointXYZ> vg_;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

  pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud_;
  bool has_last_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarIcpNode>());
  rclcpp::shutdown();
  return 0;
}
