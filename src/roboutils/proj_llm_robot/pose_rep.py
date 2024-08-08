import numpy as np
import transformations
from geometry_msgs.msg import Pose, PoseStamped


def T2sevendof(T):
    translation_vector = T[:3, 3]
    quat = transformations.quaternion_from_matrix(T)
    return [*translation_vector, *quat]


def sevenDof2T(pose):
    (position, quaternion) = pose[:3], pose[3:]
    homo_T = np.identity(4, dtype=np.float32)
    homo_T[:3, -1] = position
    homo_T[:3, :3] = transformations.quaternion_matrix(quaternion)[:3, :3]
    return homo_T


def T2pose(T):
    pose_msg = Pose()
    sevenDof_pose = T2sevendof(T)
    pose_msg.position.x = sevenDof_pose[0]
    pose_msg.position.y = sevenDof_pose[1]
    pose_msg.position.z = sevenDof_pose[2]
    pose_msg.orientation.w = sevenDof_pose[3]
    pose_msg.orientation.x = sevenDof_pose[4]
    pose_msg.orientation.y = sevenDof_pose[5]
    pose_msg.orientation.z = sevenDof_pose[6]
    return pose_msg


def pose2T(pose):
    position = np.array([pose.position.x, pose.position.y, pose.position.z])
    quaternion = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
    return sevenDof2T(list(position) + list(quaternion))


def T2posestamped(T, frame_id, stamp=None):
    pose_msg = PoseStamped()
    pose_msg.pose = T2pose(T)
    pose_msg.header.frame_id = frame_id
    if stamp is not None:
        pose_msg.header.stamp = stamp
    return pose_msg


def posestamp2T(pose):
    position = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
    quaternion = np.array([pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z])
    return sevenDof2T(list(position) + list(quaternion))