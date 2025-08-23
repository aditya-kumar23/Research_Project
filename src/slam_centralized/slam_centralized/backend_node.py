#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped, Pose
from nav_msgs.msg import Path
import gtsam
from gtsam import imuBias
from collections import OrderedDict
import numpy as np
from message_filters import Subscriber, ApproximateTimeSynchronizer

def pose_to_gtsam(msg: PoseStamped) -> gtsam.Pose3:
    t = msg.pose.position
    q = msg.pose.orientation
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(q.w, q.x, q.y, q.z),
        gtsam.Point3(t.x, t.y, t.z)
    )

def gtsam_to_msg(pose: gtsam.Pose3) -> Pose:
    """Convert a gtsam.Pose3 into a geometry_msgs/Pose."""
    p = Pose()
    t  = pose.translation()
    p.position.x, p.position.y, p.position.z = float(t[0]), float(t[1]), float(t[2])
    r  = pose.rotation().toQuaternion()
    # GTSAM may return either attributes or array
    try:
        p.orientation.x = r.x()
        p.orientation.y = r.y()
        p.orientation.z = r.z()
        p.orientation.w = r.w()
    except AttributeError:
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (
            float(r[0]), float(r[1]), float(r[2]), float(r[3])
        )
    return p

class CentralBackend(Node):
    def __init__(self):
        super().__init__('central_node')

        # --- Parameterize agent name for frame_id, logging ---
        self.declare_parameter('agent_name', 'Alpha')
        self.declare_parameter('odom_topic', 'icp_odom')
        self.agent = self.get_parameter('agent_name').value
        self.odom_topic = self.get_parameter('odom_topic').value

        # --- ISAM2 setup ---
        self.isam     = gtsam.ISAM2()
        self.graph    = gtsam.NonlinearFactorGraph()
        self.initial  = gtsam.Values()
        self.pose_idx = 0

        # --- Noise models ---
        prior_sigmas = np.ones(6) * 1e-6
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        odom_sigmas  = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.odom_noise  = gtsam.noiseModel.Diagonal.Sigmas(odom_sigmas)
        self.vel_prior_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.bias_prior_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        self.bias_between_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-4)

        # --- State holders ---
        self.current_pose = gtsam.Pose3()        # identity to start
        self.current_vel  = np.zeros(3)
        self.current_bias = imuBias.ConstantBias()
        self.path_msg     = Path()
        self.path_msg.header.frame_id = f"{self.agent}_map"
        
        # loop closure helpers
        self.saved_poses  = OrderedDict()  # history of optimized poses
        self.looped_pairs = set()          # (i,j) pairs already connected
        self.loop_noise   = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 0.02)


        # --- (Optional) IMU pre-integration setup ---
        imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        imu_params.setAccelerometerCovariance(np.eye(3) * (0.1 ** 2))
        imu_params.setGyroscopeCovariance(np.eye(3) * (0.01 ** 2))
        imu_params.setIntegrationCovariance(np.eye(3) * (0.01 ** 2))

        self.imu_bias = imuBias.ConstantBias()       # initial bias guess
        self.preintegrator = gtsam.PreintegratedImuMeasurements(imu_params, self.imu_bias)
        self.prev_imu_time = None

        # --- Subscribers & synchronizer ---
        imu_sub = Subscriber(self, Imu, 'imu/data')
        icp_sub = Subscriber(self, PoseStamped, self.odom_topic)
        ats = ApproximateTimeSynchronizer([imu_sub, icp_sub],
                                         queue_size=10,
                                         slop=0.1)
        ats.registerCallback(self.sync_callback)

        # --- Publisher ---
        self.path_pub = self.create_publisher(Path, 'path', 10)

        self.get_logger().info(
            f"[{self.agent}] CentralBackend ready: "
            f"listening on 'imu/data' & '{self.odom_topic}', publishing on 'path'"
        )

    def sync_callback(self, imu: Imu, icp: PoseStamped):
        # 1) Logging
        self.get_logger().info(
            f"[{self.agent}] Sync IMU @{imu.header.stamp.sec}.{imu.header.stamp.nanosec} "
            f"ICP @{icp.header.stamp.sec}.{icp.header.stamp.nanosec}"
        )

        # 2) First‐pose prior
        if self.pose_idx == 0:
            self.graph.add(gtsam.PriorFactorPose3(
                gtsam.symbol('x', 0),
                gtsam.Pose3(),
                self.prior_noise
            ))
            self.initial.insert(gtsam.symbol('x', 0), gtsam.Pose3())

         # --- IMU factor ---
        t = imu.header.stamp.sec + imu.header.stamp.nanosec * 1e-9
        if self.prev_imu_time is not None:
            dt = t - self.prev_imu_time
            if dt > 0.0:
                accel = np.array([
                    imu.linear_acceleration.x,
                    imu.linear_acceleration.y,
                    imu.linear_acceleration.z,
                ])
                gyro = np.array([
                    imu.angular_velocity.x,
                    imu.angular_velocity.y,
                    imu.angular_velocity.z,
                ])
                self.preintegrator.integrateMeasurement(accel, gyro, dt)
            else:
                self.get_logger().warn(
                    f"[{self.agent}] Skipping IMU sample with non-positive dt={dt:.6f}"
                )
        self.prev_imu_time = t

        if self.pose_idx == 0:
            # additional priors for velocity and bias at the first node
            self.graph.add(
                gtsam.PriorFactorVector(
                    gtsam.symbol('v', 0), self.current_vel, self.vel_prior_noise
                )
            )
            self.graph.add(
                gtsam.PriorFactorConstantBias(
                    gtsam.symbol('b', 0), self.current_bias, self.bias_prior_noise
                )
            )
            self.initial.insert(gtsam.symbol('v', 0), self.current_vel)
            self.initial.insert(gtsam.symbol('b', 0), self.current_bias)
        else:
            dt_sum = self.preintegrator.deltaTij()
            if dt_sum > 0.0:
                self.graph.add(
                    gtsam.ImuFactor(
                        gtsam.symbol('x', self.pose_idx - 1),
                        gtsam.symbol('v', self.pose_idx - 1),
                        gtsam.symbol('x', self.pose_idx),
                        gtsam.symbol('v', self.pose_idx),
                        gtsam.symbol('b', self.pose_idx - 1),
                        self.preintegrator,
                    )
                )
                self.graph.add(
                    gtsam.BetweenFactorConstantBias(
                        gtsam.symbol('b', self.pose_idx - 1),
                        gtsam.symbol('b', self.pose_idx),
                        imuBias.ConstantBias(),
                        self.bias_between_noise,
                    )
                )
            else:
                self.get_logger().warn(
                    f"[{self.agent}] No IMU integration (Δt=0) → adding weak priors on v{self.pose_idx} & b{self.pose_idx}"
                )
                self.graph.add(
                    gtsam.PriorFactorVector(
                        gtsam.symbol('v', self.pose_idx),
                        self.current_vel,
                        self.vel_prior_noise,
                    )
                )
                self.graph.add(
                    gtsam.PriorFactorConstantBias(
                        gtsam.symbol('b', self.pose_idx),
                        self.current_bias,
                        self.bias_prior_noise,
                    )
                )

        # 3) Add BetweenFactor from ICP
        rel = pose_to_gtsam(icp)
        guess = self.current_pose.compose(rel)
        self.graph.add(
            gtsam.BetweenFactorPose3(
                gtsam.symbol('x', self.pose_idx),
                gtsam.symbol('x', self.pose_idx + 1),
                rel,
                self.odom_noise,
            )
        )
        self.initial.insert(gtsam.symbol('x', self.pose_idx + 1), guess)
        self.initial.insert(gtsam.symbol('v', self.pose_idx + 1), self.current_vel)
        self.initial.insert(gtsam.symbol('b', self.pose_idx + 1), self.current_bias)

        # 4) Loop-closure detection
        for past_idx, past_pose in list(self.saved_poses.items()):
            # skip very recent poses
            if self.pose_idx + 1 - past_idx < 20:
                continue
            dist = np.linalg.norm(
                np.array(past_pose.translation()) -
                 np.array(guess.translation())
            )
            if dist < 1.0 and (past_idx, self.pose_idx + 1) not in self.looped_pairs:
                loop_rel = past_pose.between(guess)
                self.graph.add(gtsam.BetweenFactorPose3(
                    gtsam.symbol('x', past_idx),
                    gtsam.symbol('x', self.pose_idx + 1),
                    loop_rel,
                    self.loop_noise
                ))
                self.looped_pairs.add((past_idx, self.pose_idx + 1))
                self.get_logger().info(
                    f"[{self.agent}] Loop closure detected: {past_idx} -> {self.pose_idx + 1}"
                )
                break

        # 5) ISAM2 update & clear
        self.isam.update(self.graph, self.initial)
        self.graph.resize(0)
        self.initial.clear()

        # 6) Extract optimized pose
        estimate = self.isam.calculateEstimate()
        self.current_pose = estimate.atPose3(gtsam.symbol('x', self.pose_idx + 1))
        self.current_vel = estimate.atVector(gtsam.symbol('v', self.pose_idx + 1))
        self.current_bias = estimate.atConstantBias(gtsam.symbol('b', self.pose_idx + 1))

        # reset IMU integrator for next segment
        self.preintegrator.resetIntegrationAndSetBias(self.current_bias)
        self.prev_imu_time = None

        # 7) Append the _optimized_ pose to Path
        stamped = PoseStamped()
        stamped.header = icp.header
        stamped.pose   = gtsam_to_msg(self.current_pose)

        self.path_msg.header.stamp = icp.header.stamp
        self.path_msg.poses.append(stamped)
        self.path_pub.publish(self.path_msg)

        self.get_logger().info(
            f"[{self.agent}] Published optimized path length: "
            f"{len(self.path_msg.poses)}"
        )
        
        # store pose for future loop closures
        self.saved_poses[self.pose_idx + 1] = self.current_pose
        if len(self.saved_poses) > 100:
            self.saved_poses.popitem(last=False)

        # 8) Increment
        self.pose_idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = CentralBackend()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

