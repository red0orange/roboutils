#! /usr/bin/env python
import os
import re
import cv2

import tf
import rospy
import numpy as np

from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
import message_filters
from cv_bridge import CvBridge, CvBridgeError


def pose_msg_to_T(pose_msg):
    T = np.identity(4)
    T[:3, :3] = tf.transformations.quaternion_matrix(
        [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w])
    T[:3, 3] = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    return T


def sevenDof2T(pose):
    (position, quaternion) = pose[:3], pose[3:]
    homo_T = np.identity(4, dtype=np.float32)
    homo_T[:3, -1] = position
    homo_T[:3, :3] = tf.transformations.quaternion_matrix(quaternion)[:3, :3]
    return homo_T


def can_convert_to_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


class ManualSaveImage(object):
    def __init__(
        self,
        save_dir,
        rgb_topic_name,
        depth_topic_name,
        camera_topic_name,
        start_index,
    ) -> None:
        ### Image
        self.rgb_topic_name = rgb_topic_name
        self.depth_topic_name = depth_topic_name
        self.save_dir = save_dir if save_dir is not None else os.path.dirname(__file__)

        files = os.listdir(self.save_dir)
        file_names = [i.rsplit(".", 1)[0] for i in files if can_convert_to_int(i.rsplit(".", 1)[0])]
        file_names = sorted(file_names, key=lambda x: int(x))
        if start_index is None:
            self.index = file_names[-1] + 1 if len(file_names) > 0 else 0

        self.bridge = CvBridge()

        rospy.loginfo("Waiting camera_info")
        camera_info = rospy.wait_for_message(camera_topic_name, CameraInfo)
        self.fx, self.fy, self.cx, self.cy = (
            camera_info.K[0],
            camera_info.K[4],
            camera_info.K[2],
            camera_info.K[5],
        )
        self.image_width, self.image_height = camera_info.width, camera_info.height
        rospy.loginfo("Get camera_info")

        rospy.loginfo("Mode: rgbd")
        assert rgb_topic_name is not None
        assert depth_topic_name is not None
        while True:
            rospy.loginfo("================Click to one sample!================")
            cv2.namedWindow("click to one sample", cv2.WINDOW_NORMAL)
            cv2.imshow("click to one sample", np.zeros([100, 100]))
            cv2.waitKey(0)
            # cv2.destroyAllWindows()

            rospy.loginfo("Getting RGBD Image!")
            rgb_msg = rospy.wait_for_message(rgb_topic_name, Image)
            depth_msg = rospy.wait_for_message(depth_topic_name, Image)
            rospy.loginfo("Get RGBD Image!")
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "16UC1")
            rospy.loginfo("Obtaining Pose!")        

            vis_depth_image = depth_image.copy()
            vis_depth_image = vis_depth_image.astype(np.float32) * 1000.0
            vis_depth_image = (vis_depth_image / vis_depth_image.max() * 255).astype(np.uint8)
            vis_depth_image = cv2.applyColorMap(vis_depth_image, cv2.COLORMAP_JET)
            cv2.imshow("color_image", np.concatenate([rgb_image, vis_depth_image], axis=0))
            key = cv2.waitKey(0)
            if key == ord("s"):
                cv2.imwrite(
                    os.path.join(self.save_dir, "images", "{:0>8d}.png".format(self.index)),
                    rgb_image,
                )
                cv2.imwrite(
                    os.path.join(self.save_dir, "depth", "{:0>8d}.png".format(self.index)),
                    depth_image,
                )
                K_txt_path = os.path.join(self.save_dir, "cams", "{:0>8d}_cam.txt".format(self.index))
                intrinsic = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
                np.savetxt(K_txt_path, intrinsic)
                self.index += 1
                pass
        pass


if __name__ == "__main__":
    rospy.init_node("manually_save_rgbd", anonymous=True)

    save_dir = "/home/red0orange/Projects/LocalBundleSDF/test_data"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "cams"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth"), exist_ok=True)

    manually_saver = ManualSaveImage(
        save_dir=save_dir,
        rgb_topic_name="/camera/color/image_raw",
        depth_topic_name="/camera/aligned_depth_to_color/image_raw",
        camera_topic_name="/camera/aligned_depth_to_color/camera_info",
        start_index=None,
    )

    rospy.spin()