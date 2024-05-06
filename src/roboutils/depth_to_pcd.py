import numpy as np
import open3d as o3d


def depth2pc(depth, K, rgb=None, mask=None, max_depth=1.0):
    """
    Convert depth and intrinsics to point cloud and optionally point cloud color
    :param depth: hxw depth map in m
    :param K: 3x3 Camera Matrix with intrinsics
    :returns: (Nx3 point cloud, point cloud color)
    """

    if mask is not None:
        mask = np.where((depth < max_depth) & (depth > 0) & (mask != 0))
    else:
        mask = np.where((depth < max_depth) & (depth > 0))
    x,y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32) - K[0,2])
    normalized_y = (y.astype(np.float32) - K[1,2])

    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]

    if rgb is not None:
        rgb = rgb[y,x,:]
        
    pc = np.vstack((world_x, world_y, world_z)).T
    return pc, rgb, np.vstack((x,y)).T


def filter_point_cloud(point_cloud, colors=None, min_neighbors=40, radius=0.01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)

    pcd, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    index = np.full((len(point_cloud),), False)
    index[ind] = True

    if colors is not None:
        return np.asarray(pcd.points), np.asarray(pcd.colors)
    return np.asarray(pcd.points), np.array(index)