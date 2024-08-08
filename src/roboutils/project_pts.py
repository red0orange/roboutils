import numpy as np


def project_pts(pts, K):
    projected = K @ pts.T
    projected /= projected[2, :]
    return projected[:2].T