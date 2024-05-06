import numpy as np


def update_pose(T, translate=[0, 0, 0], rotate=0, rotate_axis='x'):
    # 创建旋转矩阵
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(rotate), -np.sin(rotate), 0],
        [0, np.sin(rotate), np.cos(rotate), 0],
        [0, 0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(rotate), 0, np.sin(rotate), 0],
        [0, 1, 0, 0],
        [-np.sin(rotate), 0, np.cos(rotate), 0],
        [0, 0, 0, 1]
    ])

    Rz = np.array([
        [np.cos(rotate), -np.sin(rotate), 0, 0],
        [np.sin(rotate), np.cos(rotate), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # 旋转矩阵组合
    if rotate_axis == 'x':
        R = Rx
    elif rotate_axis == 'y':
        R = Ry
    elif rotate_axis == 'z':
        R = Rz
    else:
        raise ValueError('Invalid rotate_axis value. Must be one of "x", "y", or "z".')

    # 创建平移矩阵
    T_translate = np.identity(4)
    T_translate[:3, 3] = translate

    # 更新位姿
    T_updated = T @ T_translate @ R

    return T_updated


def calculate_angle_with_xy_plane(homo_matrix):
    # 提取旋转矩阵的z轴方向
    z_axis_vector = homo_matrix[0:3, 2]

    # 计算该向量与 XY 平面法向量的夹角
    xy_plane_normal = np.array([0, 0, 1])
    dot_product = np.dot(xy_plane_normal, z_axis_vector)

    # 使用点积公式计算夹角
    angle = np.arccos(dot_product / (np.linalg.norm(xy_plane_normal) * np.linalg.norm(z_axis_vector)))

    # 将弧度转换为度数，如果你需要的话
    angle_degrees = np.degrees(angle)

    return angle_degrees