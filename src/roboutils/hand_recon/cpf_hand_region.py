import os
import pickle

import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

from .mano_api import MyMANOAPI


util_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cpf_hand_region():
    with open(os.path.join(util_dir, 'assets', 'CPF_anchor', 'mano_verts_mapping.pkl'), 'rb') as f:
        mano_verts_mapping = pickle.load(f)
    return list(mano_verts_mapping.values())


def get_palm_verts_idx():
    """返回手掌部分在 mano_verts 中的 index 列表
    """
    face_vert_idx = np.loadtxt(os.path.join(util_dir, 'assets', 'CPF_anchor', 'face_vertex_idx.txt'))
    face_index = np.arange(face_vert_idx.shape[0]).reshape(-1, 1).repeat(3, axis=1)
    face_index_flatten = face_index.flatten().astype(np.int32)
    with open(os.path.join(util_dir, 'assets', 'CPF_anchor', 'anchor_mapping_path.pkl'), 'rb') as f:
        anchor_verts_mapping = pickle.load(f)
    palm_verts_index = []
    for k, v in anchor_verts_mapping.items():
        if v in [15, 16]:
            palm_verts_index.append(k)
    palm_verts_index = face_vert_idx[palm_verts_index].flatten().astype(np.int32)
    return palm_verts_index


def get_four_finger_tip_verts_idx():
    """返回手掌部分在 mano_verts 中的 index 列表
    """
    face_vert_idx = np.loadtxt(os.path.join(util_dir, 'assets', 'CPF_anchor', 'face_vertex_idx.txt'))
    face_index = np.arange(face_vert_idx.shape[0]).reshape(-1, 1).repeat(3, axis=1)
    face_index_flatten = face_index.flatten().astype(np.int32)
    with open(os.path.join(util_dir, 'assets', 'CPF_anchor', 'anchor_mapping_path.pkl'), 'rb') as f:
        anchor_verts_mapping = pickle.load(f)
    four_finger_verts_index = []
    for k, v in anchor_verts_mapping.items():
        if v not in [0, 1, 2, 15, 16]:
            four_finger_verts_index.append(k)
    four_finger_verts_index = face_vert_idx[four_finger_verts_index].flatten().astype(np.int32)
    return four_finger_verts_index


def vis_cpf_region():
    api = MyMANOAPI()
    demo_data_dict = api.get_demo_mano_hand()
    hand_verts = demo_data_dict['hand_verts']
    hand_region_mapping = cpf_hand_region()
    hand_region = hand_region_mapping
    
    distinct_colors = np.array([
        (158, 83, 170),  # 大拇指
        (101, 150, 200), # 大拇指
        (118, 236, 63),  # 大拇指
        (135, 149, 235),
        (109, 189, 153),
        (73, 207, 171),
        (157, 97, 188),
        (45, 245, 138),
        (110, 255, 8),
        (55, 222, 37),
        (196, 126, 111),
        (198, 168, 145),
        (187, 5, 22),
        (191, 125, 12),
        (186, 179, 90),
        (129, 223, 44),  # 手掌前半部分
        (64, 182, 71)    # 手掌后半部分
    ])
    hand_colors = [distinct_colors[i] / 255.0 for i in hand_region]

    hand_pcd = o3d.geometry.PointCloud()
    hand_pcd.points = o3d.utility.Vector3dVector(hand_verts)
    hand_pcd.colors = o3d.utility.Vector3dVector(hand_colors)
    o3d.visualization.draw_geometries([hand_pcd])
    pass


if __name__ == "__main__":
    vis_cpf_region()


