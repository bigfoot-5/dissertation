#!/usr/bin/env python3

from __future__ import print_function
import time
from robotiq_2f_gripper_control.msg import _Robotiq2FGripper_robot_output as outputMsg
import rospy
import roslib
import sys
import argparse
# import random
import moveit_commander
import moveit_msgs.msg
from moveit_msgs.msg import RobotTrajectory, Grasp
from geometry_msgs.msg import PoseStamped
import tf
from utils import gen_command

roslib.load_manifest('robotiq_2f_gripper_control')
place1_pose = [0.7853, -1.5707, 1.5707, 1.5707, 1.5708, -1.5708]
place2_pose = [-1.5707, -1.5707, 1.5707, 1.5707, 1.5707, -1.5708]


class UR5:

    def __init__(self,
                 move_group_name: str = "manipulator",
                 reference_frame: str = "base_link",
                 end_effector_link: str = None,
                 robot_description: str = "robot_description",
                 planning_time: float = 30.0,
                 num_planning_attempts: int = 200,
                 goal_tolerance: float = 0.01,
                 goal_orientation_tolerance: float = 0.01,
                 max_velocity_scaling_factor: float = 0.1,
                 max_acceleration_scaling_factor: float = 0.1,
                 max_pick_attempts: int = 10,
                 go_home: bool = False,
                 gripper_length: float = 0.13):

        #self.gripper_pub = rospy.Publisher('Robotiq2FGripperRobotOutput',
        #                                   outputMsg.Robotiq2FGripper_robot_output, queue_size=1)
        #self.base_orientation = [0.0419532, -0.017782, 0.0330182, 0.9984155]

        # set gripper parameters
        self.gripper_length = gripper_length
        self.pre_grasp_approach = 0.05
        self.post_grasp_retreat = 0.05

        self.max_pick_attempts = max_pick_attempts

        # Initialize the move group for the ur5_arm
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface(synchronous=True)
        self.group = moveit_commander.MoveGroupCommander(move_group_name)
        # for displaying the trajectory in rviz
        self.display_trajectory_publisher = rospy.Publisher(
            '/move_group/display_planned_path',
            moveit_msgs.msg.DisplayTrajectory,
            queue_size=20)

        # set the group parameters
        self.group.set_planning_time(planning_time)
        self.group.set_num_planning_attempts(num_planning_attempts)
        self.group.set_goal_position_tolerance(goal_tolerance)
        self.group.set_goal_orientation_tolerance(goal_orientation_tolerance)
        self.group.set_max_velocity_scaling_factor(max_velocity_scaling_factor)
        self.group.set_max_acceleration_scaling_factor(
            max_acceleration_scaling_factor)
        self.group.set_planner_id("RRTstark")
        self.group.allow_replanning(True)
        self.group.set_pose_reference_frame(reference_frame)

        if end_effector_link is not None:
            self.group.set_end_effector_link(end_effector_link)

        # check all the objects in the scene
        objects = self.scene.get_known_object_names()
        # clear all objects in the scene
        for obj in objects:
            self.scene.remove_world_object(obj)

        # table parameters
        table_height = 0.00
        table_size = [1.5, 1.5, 0.01]
        table_orientation = [0, 0, 0, 1]
        table_position = [0, 0, table_height - table_size[2] / 2 - 0.001]

        # add table to the scene
        table_pose = self.list_to_pose(table_position, table_orientation)
        self.add_object(table_pose, table_size, "table")

        # set the workspace dimensions to the table size
        self.group.set_workspace([-table_size[0] / 2, table_size[0] / 2,
                                  -table_size[1] / 2, table_size[1] / 2,
                                  0.05, 1.5])

        # set table as the support surface
        self.group.set_support_surface_name("table")

        # get end effector pose
        self.end_effector_pose = self.group.get_current_pose()

        # set target position to the named target in the SRDF
        if go_home:
            print("Going home now")
            self.go_home()
            rospy.sleep(2)
        print("Fully initialized")

    def go_home(self):
        self.group.set_named_target("home")
     
        self.group.go()  
        self.group.stop()
        self.group.clear_pose_targets()  
        rospy.sleep(2)

    def get_position(self):
        position = self.group.get_current_pose().pose.position
        orientation = self.group.get_current_pose().pose.orientation
        return (position, orientation)

    def move_to(self, target_pose: PoseStamped):
        self.group.set_start_state_to_current_state()

        self.group.set_pose_target(target_pose)
        plan = self.group.plan()[1]
        while True:
            self.display_plan(plan)
            ans = input("Execute plan? (y/n): ")
            if ans == "y":
                self.group.execute(plan, wait=True)
                break
            elif ans == "n":
                plan = self.group.plan()[1]
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(2)

    # add box to the scene
    def add_object(self,
                   pose: PoseStamped,
                   size: list,
                   object_id: str = "object"):

        print(f"...Adding {object_id}...")
        # pose.header.frame_id = self.group.get_planning_frame()
        self.scene.add_box(object_id, pose, size)
        print(f"...Added {object_id}...")

    # display a plan in Rviz
    def display_plan(self, plan: RobotTrajectory):
        display_trajectory = moveit_msgs.msg.DisplayTrajectory()
        display_trajectory.trajectory_start = self.robot.get_current_state()
        display_trajectory.trajectory.append(plan)
        self.display_trajectory_publisher.publish(display_trajectory)

    # convert from a tuple of position and orientation to PoseStamped
    def list_to_pose(self,
                     position: list,
                     orientation: list):

        pose = PoseStamped()
        pose.header.frame_id = self.group.get_planning_frame()
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]
        pose.pose.orientation.x = orientation[0]
        pose.pose.orientation.y = orientation[1]
        pose.pose.orientation.z = orientation[2]
        pose.pose.orientation.w = orientation[3]

        return pose

    # generate a list of possible grasps given a PoseStamped
    def generate_grasps(self, target_pose: PoseStamped):

        grasps = Grasp()

        # set the grasp position
        grasps.grasp_pose.pose.position = target_pose.pose.position
        # self.end_effector_pose.pose.orientation
        grasps.grasp_pose.pose.orientation = target_pose.pose.orientation

        # set the pre-grasp approach
        grasps.pre_grasp_approach.direction.header.frame_id = self.group.get_planning_frame()
        grasps.pre_grasp_approach.direction.vector.z = 1.0
        grasps.pre_grasp_approach.min_distance = 0.02
        grasps.pre_grasp_approach.desired_distance = 0.05

        # set the post-grasp retreat
        grasps.post_grasp_retreat.direction.header.frame_id = self.group.get_planning_frame()
        grasps.post_grasp_retreat.direction.vector.z = 1.0
        grasps.post_grasp_retreat.min_distance = 0.02
        grasps.post_grasp_retreat.desired_distance = 0.05

        return grasps

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        q = tf.transformations.quaternion_from_euler(roll, pitch, yaw)
        return q

    @staticmethod
    def euler_from_quaternion(q):
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(q)
        return [roll, pitch, yaw]

    def go_to_place(self, place_pose: PoseStamped):
        target = place_pose
        self.group.set_start_state_to_current_state()
        self.group.set_joint_value_target(target)
        plan = self.group.plan()[1]
        while True:
            self.display_plan(plan)
            ans = input("Execute plan? (y/n): ")
            if ans == "y":
                self.group.execute(plan, wait=True)
                break
            elif ans == "n":
                plan = self.group.plan()[1]
        self.group.stop()
        self.group.clear_pose_targets()
        rospy.sleep(2)


# def pose_generator(orientation: list = [0, 0, 0, 1]) -> PoseStamped:
#     pose = PoseStamped()

#     pose.header.frame_id = "camera_color_optical_frame"
#     # pose.pose.position.x = 0.086349 + 0.0540329
#     # pose.pose.position.x = 0.02
#     # pose.pose.position.y = -0.01514 - 0.202304
#     # pose.pose.position.y = 0
#     # pose.pose.position.z = 0.3569 - 0.737878
#     # pose.pose.position.z = 0.3569

#     # pose.pose.orientation.x = orientation[0] - 0.00648441
#     # pose.pose.orientation.y = orientation[1] + 0.00964821
#     # pose.pose.orientation.z = orientation[2] - 0.999881
#     # pose.pose.orientation.w = orientation[3] - 0.010114
#     # x: 0.05384165421128273
#     # y: -0.06403755396604538
#     # z: 0.3499999940395355


#     pose.pose.position.x: 0.05384165421128273 
#     pose.pose.position.y: -0.06403755396604538 
#     pose.pose.position.z: 0.3499999940395355
#     pose.pose.orientation.x = orientation[0] 
#     pose.pose.orientation.y = orientation[1] 
#     pose.pose.orientation.z = orientation[2] 
#     pose.pose.orientation.w = orientation[3] 

#     return pose

def pose_generator(orientation: list = [0.0, 0.0, 0.9998462089314685, -0.017537345448220706]) -> PoseStamped:
    # x = random.uniform(-0.5, 0.5)
    # y = random.uniform(-0.5, 0.5)
    # z = random.uniform(0.1, 0.5)

    x = -0.04714800417423248
    y = -0.0523250475525856
    z = 0.38600000739097595
    pose = PoseStamped()
    pose.header.frame_id = "camera_color_optical_frame"
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = z

    pose.pose.orientation.x = orientation[0]
    pose.pose.orientation.y = orientation[1]
    pose.pose.orientation.z = orientation[2]
    pose.pose.orientation.w = orientation[3]

    return pose


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", type=str, choices=["real", "sim"],
                    required=True, help="Select mode: real or sim")
    ap.add_argument("--home", type=bool,
                    default=False, help="Go to home position")
    args = vars(ap.parse_args())

    rospy.init_node("ur5_moveit", anonymous=True)
    grasp_pub = rospy.Publisher(
        "/ur5/grasp_pose", PoseStamped, queue_size=1)
    target_pub = rospy.Publisher(
        "/ur5/target_pose", PoseStamped, queue_size=1)

    if args["mode"] == "real":
        # ur5 = UR5(reference_frame="camera_color_optical_frame",
        ur5 = UR5(reference_frame="base_link",
                  end_effector_link="wrist_3_link",
                  go_home=args["home"])
    elif args["mode"] == "sim":
        ur5 = UR5(reference_frame="robot_arm_base_link",
                  go_home=args["home"])

    def main(target_pose):
        #command = outputMsg.Robotiq2FGripper_robot_output()
        # target_pose = convert_coordinate(target_pose)
        print("reset the gripper")
        #command = gen_command("r", command)
        #ur5.gripper_pub.publish(command)
        rospy.sleep(3)

        print("active the gripper")
        #command = gen_command("a", command)
        #ur5.gripper_pub.publish(command)
        rospy.sleep(3)

        # pre-grasp Pose
        grasp_pose = target_pose
        grasp_pose.pose.position.z -= ur5.gripper_length

        # grasps = ur5.generate_grasps(grasp_pose)

        # ur5.scene.remove_world_object("object")
        # rospy.sleep(1)
        # ur5.add_object(grasp_pose, [0.04, 0.04, 0.04], "object")
        # rospy.sleep(1)
        # plan = ur5.group.pick("object", grasps, plan_only=True)

        # while True:
        # execute = input("Execute plan?")
        # if execute == "y":
        # ur5.group.execute(plan, wait=True)
        # else:
        # plan = ur5.group.pick("object", grasps, plan_only=True)
        # break

        # pre_grasp_pose = grasp_pose
        # pre_grasp_pose.pose.position.z -= 0.02

        # pre_grasp_pub.publish(pre_grasp_pose)
        grasp_pub.publish(grasp_pose)
        target_pub.publish(target_pose)
        print(target_pose)
        rospy.sleep(2)

        print("go to grasp pose")
        ur5.move_to(target_pose)
        rospy.sleep(2)

        # press enter to continue
        input("Press enter to close gripper")
        print("close the gripper")
        #command = gen_command("c", command)
        #ur5.gripper_pub.publish(command)
        rospy.sleep(2)

        print("go to the second joint_value_target")
        ur5.go_to_place(place2_pose)
        rospy.sleep(2)

        # clear everything
        ur5.group.stop()
        ur5.group.clear_pose_targets()
        rospy.sleep(2)

        drop = input("Drop object? (y/n):")
        if drop == "y":
            #command = gen_command("o", command)
            #ur5.gripper_pub.publish(command)
            print("open the gripper")
            rospy.sleep(2)

        print("go to the first join_value_pose")
        ur5.go_to_place(place1_pose)
        rospy.sleep(2)

    try:
        if args["mode"] == "real":
            print("go to the first join_value_pose")
            ur5.go_to_place(place1_pose)
            rospy.sleep(2)
            rospy.Subscriber("/vision/pose", PoseStamped, main)
            rospy.spin()
            # main(pose_generator())
            print("over")
            sys.exit()
        elif args["mode"] == "sim":
            print("go to the first join_value_pose")
            ur5.go_to_place(place1_pose)
            rospy.sleep(2)
            main(pose_generator())
            print("over")
            sys.exit()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        pass
