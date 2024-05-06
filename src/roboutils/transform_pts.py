import numpy as np


def transform_points(points, T):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    trans_points = np.matmul(T, points.T).T[:, :3]
    return trans_points