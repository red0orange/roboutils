import os
import numpy as np
import torch
import pickle
import transforms3d

from .mano import MANO
from .aIK import joints3d_to_mano_pose


util_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MyMANOAPI(object):
    def __init__(self):
        self.device = "cuda"
        self.mano_model = MANO().to(self.device)
        self.mano_model.layer = self.mano_model.layer.cuda()
        pass

    def get_3d_joints_from_vertices(self, vertices):
        vertices = torch.from_numpy(vertices).to(self.device)
        vertices = vertices[None, ...]
        joints = self.mano_model.get_3d_joints(vertices)
        joints = joints[0].cpu().numpy()
        return joints

    def get_mano_pose_using_AIK(self, joints_3d):
        rot_pose = joints3d_to_mano_pose(joints_3d)  # (16, 3, 3)
        axisang_poses = []
        for i in range(rot_pose.shape[0]):
            axis, angle = transforms3d.axangles.mat2axangle(rot_pose[i])
            axisang_poses.append(axis * angle)
        axisang_pose = np.array(axisang_poses).flatten()

        axisang_poses = np.array(axisang_pose).reshape(-1, 3)
        mats = []
        for i in range(axisang_poses.shape[0]):
            mats.append(transforms3d.axangles.axangle2mat(axisang_poses[i], np.linalg.norm(axisang_poses[i])))
        return np.array(mats)

    def get_demo_mano_hand(self):
        demo_hand_pose_path = os.path.join(util_dir, 'assets', 'mano', 'demo_mano_pose.pkl')
        with open(demo_hand_pose_path, 'rb') as f:
            demo_hand_pose = pickle.load(f)
        
        mano_pose = demo_hand_pose['mano_pose']
        mano_shape = demo_hand_pose['mano_shape']

        mano_output = self.mano_model.layer(
            torch.from_numpy(mano_pose).unsqueeze(0).cuda(), torch.from_numpy(mano_shape).unsqueeze(0).cuda()
        )
        hand_faces = self.mano_model.layer.th_faces.cpu().numpy()
        hand_verts = mano_output[0].squeeze().cpu().numpy()
        hand_verts -= np.mean(hand_verts, axis=0)

        data_dict = {
            'hand_pose': mano_pose,
            'hand_shape': mano_shape,
            'hand_verts': hand_verts,
            'hand_faces': hand_faces
        }
        return data_dict