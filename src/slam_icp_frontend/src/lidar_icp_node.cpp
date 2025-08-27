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

/**
 * @brief A simple ICP front‑end node.
 *
 * This node subscribes to a `sensor_msgs/PointCloud2` topic (``lidar/points``),
 * downsamples incoming scans using a voxel grid filter, runs Iterative Closest
 * Point (ICP) against the previous scan and publishes a relative pose as a
 * ``geometry_msgs/PoseStamped`` message.  Parameters controlling the voxel
 * leaf size, maximum iterations and correspondence distance are declared and
 * can be set from the command line or via a YAML file.
 */
class LidarIcpNode : public rclcpp::Node {
public:
  LidarIcpNode()
  : Node("lidar_icp_node"),
    has_last_(false),
    last_cloud_(new pcl::PointCloud<pcl::PointXYZ>())
  {
    // Declare configurable parameters with sane defaults
    this->declare_parameter<double>("voxel_leaf_size", 0.3);
    this->declare_parameter<int>("max_iterations", 20);
    this->declare_parameter<double>("max_correspondence_distance", 1.0);

    // Retrieve parameter values
    const double leaf = this->get_parameter("voxel_leaf_size").as_double();
    const int iterations = this->get_parameter("max_iterations").as_int();
    const double corr = this->get_parameter("max_correspondence_distance").as_double();

    // Configure the voxel grid and ICP based on parameters
    vg_.setLeafSize(static_cast<float>(leaf), static_cast<float>(leaf), static_cast<float>(leaf));
    icp_.setMaximumIterations(iterations);
    icp_.setMaxCorrespondenceDistance(corr);

    // Subscribe to the relative LiDAR topic under each namespace
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "lidar/points", rclcpp::QoS(rclcpp::KeepLast(1)),
      std::bind(&LidarIcpNode::cloud_cb, this, std::placeholders::_1)
    );

    // Publish ICP odometry on a clean, relative topic
    pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>(
      "icp_odom", rclcpp::QoS(rclcpp::KeepLast(1))
    );

    RCLCPP_INFO(
      this->get_logger(),
      "LidarIcpNode initialized. leaf_size=%.2f, max_iterations=%d, max_corr_dist=%.2f",
      leaf, iterations, corr
    );
  }

private:
  void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // Convert ROS2 msg to PCL cloud
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    // Downsample
    auto filtered = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    vg_.setInputCloud(cloud);
    vg_.filter(*filtered);

    // On first frame just store and return
    if (!has_last_) {
      last_cloud_ = filtered;
      has_last_ = true;
      return;
    }

    // Run ICP between filtered and last_cloud_
    auto aligned = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    icp_.setInputSource(filtered);
    icp_.setInputTarget(last_cloud_);
    icp_.align(*aligned);

    // Check convergence before using the result
    if (!icp_.hasConverged()) {
      RCLCPP_WARN(this->get_logger(), "ICP did not converge; skipping this frame");
      // Update last_cloud_ anyway to avoid repeatedly aligning against an old scan
      last_cloud_ = filtered;
      return;
    }

    // Extract the 4×4 relative transform
    const Eigen::Matrix4f rel = icp_.getFinalTransformation();

    // Build a PoseStamped message
    geometry_msgs::msg::PoseStamped out;
    out.header = msg->header;
    // Translation
    out.pose.position.x = rel(0, 3);
    out.pose.position.y = rel(1, 3);
    out.pose.position.z = rel(2, 3);
    // Rotation (matrix → tf2::Quaternion)
    const Eigen::Matrix3f eigR = rel.block<3, 3>(0, 0);
    tf2::Matrix3x3 R(
      eigR(0, 0), eigR(0, 1), eigR(0, 2),
      eigR(1, 0), eigR(1, 1), eigR(1, 2),
      eigR(2, 0), eigR(2, 1), eigR(2, 2)
    );
    tf2::Quaternion q;
    R.getRotation(q);
    out.pose.orientation.x = q.x();
    out.pose.orientation.y = q.y();
    out.pose.orientation.z = q.z();
    out.pose.orientation.w = q.w();

    // Publish the ICP result
    pub_->publish(out);

    // Update last_cloud_ for next iteration
    last_cloud_ = filtered;
  }

  // ROS2 subscription and publisher
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_;

  // PCL filters and ICP implementation
  pcl::VoxelGrid<pcl::PointXYZ> vg_;
  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;

  // State for the previous scan
  pcl::PointCloud<pcl::PointXYZ>::Ptr last_cloud_;
  bool has_last_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LidarIcpNode>());
  rclcpp::shutdown();
  return 0;
}