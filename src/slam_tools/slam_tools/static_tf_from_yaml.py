#!/usr/bin/env python3
import rclpy, yaml
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

def mat4_to_tf(mat):
    # mat: 4x4 list-of-lists
    x, y, z = mat[0][3], mat[1][3], mat[2][3]
    m = [[mat[i][j] for j in range(3)] for i in range(3)]
    # rotation matrix -> quaternion
    tr = m[0][0]+m[1][1]+m[2][2]
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        w = 0.25*S
        xq = (m[2][1]-m[1][2]) / S
        yq = (m[0][2]-m[2][0]) / S
        zq = (m[1][0]-m[0][1]) / S
    elif (m[0][0] > m[1][1]) and (m[0][0] > m[2][2]):
        S = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])*2
        w = (m[2][1]-m[1][2]) / S
        xq = 0.25*S
        yq = (m[0][1]+m[1][0]) / S
        zq = (m[0][2]+m[2][0]) / S
    elif (m[1][1] > m[2][2]):
        S = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])*2
        w = (m[0][2]-m[2][0]) / S
        xq = (m[0][1]+m[1][0]) / S
        yq = 0.25*S
        zq = (m[1][2]+m[2][1]) / S
    else:
        S = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])*2
        w = (m[1][0]-m[0][1]) / S
        xq = (m[0][2]+m[2][0]) / S
        yq = (m[1][2]+m[2][1]) / S
        zq = 0.25*S
    return (x, y, z, xq, yq, zq, w)

class StaticTFFromYAML(Node):
    def __init__(self):
        super().__init__('static_tf_from_yaml')
        self.declare_parameter('yaml_file', '')
        self.declare_parameter('agent_name', 'Alpha')
        yaml_path = self.get_parameter('yaml_file').value
        agent     = self.get_parameter('agent_name').value

        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        tfs = []
        broadcaster = StaticTransformBroadcaster(self)

        # Frames
        left_cam = f'{agent}_left_cam'
        imu_f    = f'{agent}_imu'
        velo_f   = f'{agent}_velodyne'

        if 'Tic' in data:
            mat = data['Tic']['data']; rows=data['Tic']['rows']; cols=data['Tic']['cols']
            M = [mat[i*cols:(i+1)*cols] for i in range(rows)]
            x,y,z,qx,qy,qz,qw = mat4_to_tf(M)
            t = TransformStamped()
            t.header.frame_id = left_cam
            t.child_frame_id  = imu_f
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx; t.transform.rotation.y = qy
            t.transform.rotation.z = qz; t.transform.rotation.w = qw
            tfs.append(t)

        if 'Tlc' in data:
            mat = data['Tlc']['data']; rows=data['Tlc']['rows']; cols=data['Tlc']['cols']
            M = [mat[i*cols:(i+1)*cols] for i in range(rows)]
            x,y,z,qx,qy,qz,qw = mat4_to_tf(M)
            t = TransformStamped()
            t.header.frame_id = left_cam
            t.child_frame_id  = velo_f
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation.x = qx; t.transform.rotation.y = qy
            t.transform.rotation.z = qz; t.transform.rotation.w = qw
            tfs.append(t)

        broadcaster.sendTransform(tfs)
        self.get_logger().info(f'Published {len(tfs)} static TFs from {yaml_path}')

def main(args=None):
    rclpy.init(args=args)
    node = StaticTFFromYAML()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

