#!/usr/bin/env python3

import pyrealsense2 as rs
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import cv2
import pyrealsense2 as rs2
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
from stream import Stream
# import tf2_ros
import rospy
# import tf2_geometry_msgs

bridge = CvBridge()


# def convert_coordinate(camera_pose):
#     # tf_buffer = tf2_ros.Buffer()  # 管理变换关系的缓冲区
#     # # 等待特定的变换关系在缓冲区中变得可用
#     # tf_buffer.can_transform('base_link', 'camera_color_optical_frame', rospy.Time(), rospy.Duration(4.0))

#     # 表示坐标系之间变换关系的消息类型。它包含了时间戳、源坐标系、目标坐标系、平移和旋转参数等信息
#     transform = TransformStamped()
#     transform.header.stamp = rospy.Time.now()
#     transform.header.frame_id = 'wrist_3_link'
#     transform.child_frame_id = 'camera_color_optical_frame'
#     transform.transform.translation.x = -0.0540329
#     transform.transform.translation.y = 0.202304
#     transform.transform.translation.z = -0.737878
#     transform.transform.rotation.x = 0.00648441
#     transform.transform.rotation.y = -0.00964821
#     transform.transform.rotation.z = 0.999881
#     transform.transform.rotation.w = 0.010114

#     # Perform the transformation
#     base_pose = tf2_geometry_msgs.do_transform_pose(camera_pose, transform)
#     return base_pose


def position2pose(position: list = [0, 0, 0],
                  orientation: list = [0, 0, 0, 1],
                  frame: str = "camera_color_optical_frame") -> PoseStamped:
    target_pose = PoseStamped()
    target_pose.header.frame_id = frame
    target_pose.pose.position.x = position[0]
    target_pose.pose.position.y = position[1]
    target_pose.pose.position.z = position[2]

    target_pose.pose.orientation.x = orientation[0]
    target_pose.pose.orientation.y = orientation[1]
    target_pose.pose.orientation.z = orientation[2]
    target_pose.pose.orientation.w = orientation[3]

    return target_pose


def info2intrinsic(camera_info: CameraInfo) -> rs2.intrinsics:
    intrinsics = rs2.intrinsics()
    intrinsics.width = camera_info.width
    intrinsics.height = camera_info.height
    intrinsics.ppx = camera_info.K[2]
    intrinsics.ppy = camera_info.K[5]
    intrinsics.fx = camera_info.K[0]
    intrinsics.fy = camera_info.K[4]
    intrinsics.coeffs = [i for i in camera_info.D]

    return intrinsics


def intrinsic2info(intrinsics: rs2.intrinsics) -> CameraInfo:
    camera_info = CameraInfo()
    camera_info.width = intrinsics.width
    camera_info.height = intrinsics.height
    camera_info.K = [intrinsics.fx, 0.0, intrinsics.ppx,
                     0.0, intrinsics.fy, intrinsics.ppy,
                     0.0, 0.0, 1.0]
    camera_info.D = intrinsics.coeffs
    camera_info.distortion_model = "plumb_bob"

    return camera_info


def project_point(depth_image: np.ndarray,
                  xy: list,
                  camera_info: CameraInfo) -> list:
    depth = depth_image[xy[0], xy[1]] / 1000
    # convert 2D position to 3D position
    xyz = [depth * (xy[0] - camera_info.K[2]) / camera_info.K[0],
           depth * (xy[1] - camera_info.K[5]) / camera_info.K[5],
           depth]

    return xyz


def img_to_cv2(img):
    return bridge.imgmsg_to_cv2(img, img.encoding)  # numpy数组


def test_rgbintrin_depthintrin(rgb_info: CameraInfo,
                               depth_info: CameraInfo):
    rgb_intrinsics = info2intrinsic(rgb_info)
    depth_intrinsics = info2intrinsic(depth_info)
    print(rgb_intrinsics == depth_intrinsics)


def test_intrinsics(rgb_info: CameraInfo,
                    depth_info: CameraInfo):
    print("rgb width:", rgb_info.width, "depth width:", depth_info.width,
          rgb_info.width == depth_info.width)
    print("rgb height:", rgb_info.height, "depth height:",
          depth_info.height, rgb_info.height == depth_info.height)
    print("rgb ppx:", rgb_info.K[2], "depth ppx:", depth_info.K[2],
          rgb_info.K[2] == depth_info.K[2])
    print("rgb ppy:", rgb_info.K[5], "depth ppy:", depth_info.K[5],
          rgb_info.K[5] == depth_info.K[5])
    print("rgb fx:", rgb_info.K[0], "depth fx:", depth_info.K[0],
          rgb_info.K[0] == depth_info.K[0])
    print("rgb fy:", rgb_info.K[4], "depth fy:", depth_info.K[4],
          rgb_info.K[4] == depth_info.K[4])
    print("rgb coeffs:", rgb_info.D, "depth coeffs:", depth_info.D,
          rgb_info.D == depth_info.D)


def test_project_point(camera_info: CameraInfo):
    depth = 300
    xy = [123, 321]
    xyz = project_point(depth, xy, camera_info)

    _intrinsics = info2intrinsic(camera_info)

    _xyz = rs2.rs2_deproject_pixel_to_point(_intrinsics, xy, depth)

    print(f'xyz: {xyz}\n', f'_xyz: {_xyz}\n', xyz == _xyz)


def gen_command(char, command):
    """Update the command according to the character entered by the user."""

    if char == 'a':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 1
        command.rGTO = 1
        command.rSP = 255
        command.rFR = 150

    if char == 'r':
        command = outputMsg.Robotiq2FGripper_robot_output()
        command.rACT = 0

    if char == 'c':
        command.rPR = 255

    if char == 'o':
        command.rPR = 0

    # If the command entered is a int, assign this value to rPRA
    try:
        command.rPR = int(char)
        if command.rPR > 255:
            command.rPR = 255
        if command.rPR < 0:
            command.rPR = 0
    except ValueError:
        pass

    if char == 'f':
        command.rSP += 25
        if command.rSP > 255:
            command.rSP = 255

    if char == 'l':
        command.rSP -= 25
        if command.rSP < 0:
            command.rSP = 0

    if char == 'i':
        command.rFR += 25
        if command.rFR > 255:
            command.rFR = 255

    if char == 'd':
        command.rFR -= 25
        if command.rFR < 0:
            command.rFR = 0

    return command
