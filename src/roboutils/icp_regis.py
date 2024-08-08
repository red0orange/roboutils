import os
import shutil
import open3d as o3d
import numpy as np
from scipy.optimize import minimize
from manopth.manolayer import ManoLayer

mano_dir = os.path.join(os.path.dirname(__file__), "assets", "mano")
mano = ManoLayer(mano_root=mano_dir)


def scale_icp(source, target, initial_scale=1.0):
    """
    自定义的 ICP 实现，优化尺度因子和位移。
    :param source: 源点云
    :param target: 目标点云
    :param initial_scale: 初始尺度因子
    :param initial_translation: 初始位移向量
    :return: 优化后的尺度因子和位移
    """

    def objective_function(params):
        s = params[0]
        T = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])

        # 应用变换
        source_data = np.array(source.points)
        source_temp_data = np.vstack(
            (source_data.T, np.ones((1, source_data.shape[0])))
        )
        source_temp_data = T.dot(source_temp_data).T[:, :3]
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source_temp_data)

        # 计算源点云和目标点云之间的距离
        distance = source_temp.compute_point_cloud_distance(target)
        return np.mean(distance)

    # 使用 SciPy 的优化方法
    options = {
        "maxiter": 5,  # 最大迭代次数
        # 'xtol': 1e-8,    # 解的收敛容忍度
        # 'ftol': 1e-8     # 目标函数值的收敛容忍度
    }
    res = minimize(
        objective_function, [initial_scale], method="Powell", options=options
    )

    # 构建最终的变换矩阵
    s = res.x[0]
    T_optimized = np.array([[s, 0, 0, 0], [0, s, 0, 0], [0, 0, s, 0], [0, 0, 0, 1]])

    return T_optimized, res.fun  # 返回变换矩阵和目标函数值


def trans_icp(source, target, initial_translation=[0, 0, 0]):
    """
    自定义的 ICP 实现，优化尺度因子和位移。
    :param source: 源点云
    :param target: 目标点云
    :param initial_scale: 初始尺度因子
    :param initial_translation: 初始位移向量
    :return: 优化后的尺度因子和位移
    """

    def objective_function(params):
        tx, ty, tz = params
        T = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

        # 应用变换
        source_data = np.array(source.points)
        source_temp_data = np.vstack(
            (source_data.T, np.ones((1, source_data.shape[0])))
        )
        source_temp_data = T.dot(source_temp_data).T[:, :3]
        source_temp = o3d.geometry.PointCloud()
        source_temp.points = o3d.utility.Vector3dVector(source_temp_data)

        # 计算源点云和目标点云之间的距离
        distance = source_temp.compute_point_cloud_distance(target)
        return np.max(distance)

    # 使用 SciPy 的优化方法
    options = {
        "maxiter": 20,  # 最大迭代次数
        # 'xtol': 1e-2,    # 解的收敛容忍度
    }
    res = minimize(
        objective_function, [*initial_translation], method="Powell", options=options
    )

    # 构建最终的变换矩阵
    tx, ty, tz = res.x
    T_optimized = np.array([[1, 0, 0, tx], [0, 1, 0, ty], [0, 0, 1, tz], [0, 0, 0, 1]])

    return T_optimized, res.fun  # 返回变换矩阵和目标函数值


def to_homo(pts):
    """
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    """
    assert len(pts.shape) == 2, f"pts.shape: {pts.shape}"
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def points_to_o3d_pcd(points, colors=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def hand_verts_to_o3d_mesh(points):
    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(points)
    hand_mesh.triangles = o3d.utility.Vector3iVector(mano.th_faces.numpy())
    hand_mesh.compute_vertex_normals()
    return hand_mesh

def o3d_mesh_to_hand_verts(hand_mesh):
    return np.array(hand_mesh.vertices)


def transform_points(points, T):
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    trans_points = np.matmul(T, points.T).T[:, :3]
    return trans_points


def render_mesh_depth(mesh):
    # center_mesh = np.mean(np.array(mesh.vertices), axis=0)
    # depth_data_center = np.mean(depth_pcd, axis=0)
    # mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.vertices) - center_mesh + depth_data_center)

    # 设置相机的姿态
    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # 创建渲染器和渲染选项
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)  # 创建一个不可见的渲染窗口
    vis.add_geometry(mesh)

    # 设置相机参数和视口
    ctr = vis.get_view_control()
    cam_params = ctr.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = camera_pose
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    # 渲染深度图像
    vis.capture_depth_point_cloud(
        "o3d_tmp.ply", do_render=True, convert_to_world_coordinate=True
    )
    vis.destroy_window()

    # 从深度图像生成点云
    pcd = o3d.io.read_point_cloud("o3d_tmp.ply")
    os.remove("o3d_tmp.ply")
    return pcd


def hand_regis(pred_verts, depth_pcd, downsample_num=5000):
    pred_cloud_T = np.eye(4)

    copy_mesh = o3d.geometry.TriangleMesh()
    copy_mesh.vertices = o3d.utility.Vector3dVector(np.array(pred_verts))
    copy_mesh.triangles = o3d.utility.Vector3iVector(mano.th_faces.numpy())

    # translation 粗配准到点云中心
    mesh_points = np.array(copy_mesh.vertices)
    mesh_points_center = np.mean(mesh_points, axis=0)
    depth_data_center = np.mean(depth_pcd, axis=0)
    pred_cloud_T[:3, 3] = depth_data_center - mesh_points_center
    mesh_points = pred_cloud_T.dot(
        np.concatenate([mesh_points, np.ones([mesh_points.shape[0], 1])], axis=1).T
    ).T[:, :3]
    copy_mesh.vertices = o3d.utility.Vector3dVector(mesh_points)

    # 渲染预测点云
    render_pred_cloud = render_mesh_depth(copy_mesh)
    pred_cloud_data = np.array(render_pred_cloud.points)

    # translation 粗配准
    pred_cloud_data_center = np.mean(pred_cloud_data, axis=0)
    depth_data_center = np.mean(depth_pcd, axis=0)
    pred_cloud_data = pred_cloud_data - pred_cloud_data_center + depth_data_center
    pred_cloud_T[:3, 3] = pred_cloud_T[:3, 3] + depth_data_center - pred_cloud_data_center

    # 降采样到同样数量
    pred_cloud = points_to_o3d_pcd(
        pred_cloud_data, colors=np.array([[1.0, 0.0, 0.0]] * pred_cloud_data.shape[0])
    )
    depth_cloud = points_to_o3d_pcd(
        depth_pcd, colors=np.array([[0.0, 1.0, 0.0]] * depth_pcd.shape[0])
    )
    depth_cloud = depth_cloud.random_down_sample(
        min(downsample_num / depth_pcd.shape[0], 1)
    )
    pred_cloud = pred_cloud.random_down_sample(
        min(downsample_num / pred_cloud_data.shape[0], 1)
    )
    # o3d.visualization.draw_geometries([pred_cloud, depth_cloud])

    # # open3d 的版本难以移除旋转，因此使用自定义的 scipy 的 Powell 优化器
    # thres = 0.02
    # distance = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    # distance.with_scaling = True
    # reg_p2p = o3d.pipelines.registration.registration_icp(
    #     pred_cloud,
    #     depth_cloud,
    #     thres,
    #     np.eye(4),
    #     distance,
    # )
    # pred_cloud_data_icp = (reg_p2p.transformation @ to_homo(pred_cloud_data).T).T[:, :3]
    # pred_cloud_icp_cloud = points_to_o3d_pcd(pred_cloud_data_icp)
    # o3d.visualization.draw_geometries([pred_cloud_icp_cloud, depth_cloud, gt_cloud])

    # 优化计算
    translation_T, trans_objective_value = trans_icp(pred_cloud, depth_cloud)
    pred_cloud_T = pred_cloud_T.dot(translation_T)
    pred_cloud.transform(translation_T)
    scale_T, scale_objective_value = scale_icp(pred_cloud, depth_cloud)
    pred_cloud_T = pred_cloud_T.dot(scale_T)
    pred_cloud.transform(scale_T)

    # # 打印和使用最终的变换矩阵
    # print("Trans Objective Function Value:", trans_objective_value)
    # print("Scale Objective Function Value:", scale_objective_value)
    return pred_cloud_T, trans_objective_value, scale_objective_value


if __name__ == "__main__":
    gt_cloud_data = np.load(
        "/home/red0orange/Projects/LocalBundleSDF/tmp_gt_hand.npy"
    )  # gt
    pred_cloud_data = np.load(
        "/home/red0orange/Projects/LocalBundleSDF/tmp_pred_hand.npy"
    )  # pred
    depth_pcd = np.load(
        "/home/red0orange/Projects/LocalBundleSDF/tmp_depth.npy"
    )  # depth
    pred_mesh = o3d.io.read_triangle_mesh(
        "/home/red0orange/Projects/LocalBundleSDF/tmp_hand.obj"
    )
    gt_mesh = o3d.io.read_triangle_mesh(
        "/home/red0orange/Projects/LocalBundleSDF/tmp_gt_hand.obj"
    )

    gt_cloud = points_to_o3d_pcd(
        gt_cloud_data, colors=np.array([[0.0, 0.0, 1.0]] * gt_cloud_data.shape[0])
    )

    # trans
    pred_cloud_T = hand_regis(pred_mesh, depth_pcd)
    pred_mesh.transform(pred_cloud_T)
    o3d.visualization.draw_geometries([pred_mesh, gt_cloud])
