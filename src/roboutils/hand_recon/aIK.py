# From https://github.com/MengHao666/Minimal-Hand-pytorch/blob/main/utils/AIK.py
# Copyright (c) Hao Meng. All Rights Reserved.
import os
import numpy as np
import torch
import transforms3d
from manopth import manolayer

mano_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "mano")
mano = manolayer.ManoLayer(flat_hand_mean=True,
                            side="right",
                            mano_root=mano_dir,
                            use_pca=False,
                            root_rot_mode='rotmat',
                            joint_rot_mode='rotmat')
SNAP_PARENT = [
    0,  # 0's parent
    0,  # 1's parent
    1,
    2,
    3,
    0,  # 5's parent
    5,
    6,
    7,
    0,  # 9's parent
    9,
    10,
    11,
    0,  # 13's parent
    13,
    14,
    15,
    0,  # 17's parent
    17,
    18,
    19,
]
kinematic_tree = [2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20]
ID2ROT = {
        2: 13, 3: 14, 4: 15,
        6: 1, 7: 2, 8: 3,
        10: 4, 11: 5, 12: 6,
        14: 10, 15: 11, 16: 12,
        18: 7, 19: 8, 20: 9,
    }
angels0 = np.zeros((1, 21))


def to_dict(joints):
    temp_dict = dict()
    for i in range(21):
        temp_dict[i] = joints[:, [i]]
    return temp_dict


def adaptive_IK(T_, P_):
    '''
    Computes pose parameters given template and predictions.
    We think the twist of hand bone could be omitted.

    :param T: template ,21*3
    :param P: target, 21*3
    :return: pose params.
    '''

    T = T_.copy().astype(np.float64)
    P = P_.copy().astype(np.float64)

    P = P.transpose(1, 0)
    T = T.transpose(1, 0)

    # to dict
    P = to_dict(P)
    T = to_dict(T)

    # some globals
    R = {}
    R_pa_k = {}
    q = {}

    q[0] = T[0]  # in fact, q[0] = P[0] = T[0].

    # compute R0, here we think R0 is not only a Orthogonal matrix, but also a Rotation matrix.
    # you can refer to paper "Least-Squares Fitting of Two 3-D Point Sets. K. S. Arun; T. S. Huang; S. D. Blostein"
    # It is slightly different from  https://github.com/Jeff-sjtu/HybrIK/blob/main/hybrik/utils/pose_utils.py#L4, in which R0 is regard as orthogonal matrix only.
    # Using their method might further boost accuracy.
    P_0 = np.concatenate([P[1] - P[0], P[5] - P[0],
                          P[9] - P[0], P[13] - P[0],
                          P[17] - P[0]], axis=-1)
    T_0 = np.concatenate([T[1] - T[0], T[5] - T[0],
                          T[9] - T[0], T[13] - T[0],
                          T[17] - T[0]], axis=-1)
    H = np.matmul(T_0, P_0.T)

    U, S, V_T = np.linalg.svd(H)
    V = V_T.T
    R0 = np.matmul(V, U.T)

    det0 = np.linalg.det(R0)

    if abs(det0 + 1) < 1e-6:
        V_ = V.copy()

        if (abs(S) < 1e-4).sum():
            V_[:, 2] = -V_[:, 2]
            R0 = np.matmul(V_, U.T)

    R[0] = R0

    # the bone from 1,5,9,13,17 to 0 has same rotations
    R[1] = R[0].copy()
    R[5] = R[0].copy()
    R[9] = R[0].copy()
    R[13] = R[0].copy()
    R[17] = R[0].copy()

    # compute rotation along kinematics
    for k in kinematic_tree:
        pa = SNAP_PARENT[k]
        pa_pa = SNAP_PARENT[pa]
        q[pa] = np.matmul(R[pa], (T[pa] - T[pa_pa])) + q[pa_pa]
        delta_p_k = np.matmul(np.linalg.inv(R[pa]), P[k] - q[pa])
        delta_p_k = delta_p_k.reshape((3,))
        delta_t_k = T[k] - T[pa]
        delta_t_k = delta_t_k.reshape((3,))
        temp_axis = np.cross(delta_t_k, delta_p_k)
        axis = temp_axis / (np.linalg.norm(temp_axis, axis=-1) + 1e-8)
        temp = (np.linalg.norm(delta_t_k, axis=0) + 1e-8) * (np.linalg.norm(delta_p_k, axis=0) + 1e-8)
        cos_alpha = np.dot(delta_t_k, delta_p_k) / temp

        alpha = np.arccos(cos_alpha)

        twist = delta_t_k
        D_sw = transforms3d.axangles.axangle2mat(axis=axis, angle=alpha, is_normalized=False)
        D_tw = transforms3d.axangles.axangle2mat(axis=twist, angle=angels0[:, k], is_normalized=False)
        R_pa_k[k] = np.matmul(D_sw, D_tw)
        R[k] = np.matmul(R[pa], R_pa_k[k])

    pose_R = np.zeros((1, 16, 3, 3))
    pose_R[0, 0] = R[0]
    for key in ID2ROT.keys():
        value = ID2ROT[key]
        pose_R[0, value] = R_pa_k[key]

    return pose_R


def joints3d_to_mano_pose(joints3d):
    pose0 = torch.eye(3).repeat(1, 16, 1, 1)
    _, j3d_p0_ops = mano(pose0)
    template = j3d_p0_ops.cpu().numpy().squeeze() / 1000.0  # template, m

    ratio = np.linalg.norm(template[9] - template[0]) / np.linalg.norm(joints3d[9] - joints3d[0])
    j3d_pre_process = joints3d * ratio  # template, m
    j3d_pre_process = j3d_pre_process - j3d_pre_process[0] + template[0]

    pose_R = adaptive_IK(template, j3d_pre_process)
    pose_R = torch.from_numpy(pose_R).float()
    pose_R = pose_R.cpu().numpy().squeeze()

    return pose_R
