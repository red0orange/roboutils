import numpy as np


def fit_plane(points):
    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    C, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    normal_vector = np.array([C[0], C[1], -1])
    normal_vector = normal_vector / np.linalg.norm(normal_vector)  # 单位化

    C = [C[0], C[1], -1, C[2]]
    return C, normal_vector


def same_side_of_plane(point1, point2, plane_coeffs):
    # plane_coeffs 是平面方程的系数 [a, b, c, d]
    a, b, c, d = plane_coeffs

    # 计算两个点代入平面方程后的值
    value1 = a * point1[0] + b * point1[1] + c * point1[2] + d
    value2 = a * point2[0] + b * point2[1] + c * point2[2] + d

    # 检查两个值的符号是否相同
    return np.sign(value1) == np.sign(value2)

    
def vis_fit_plane(points):
    import open3d as o3d

    C, normal_vector = fit_plane(points)
    normal_vector /= (1 / np.std(points, axis=0))

    start_point = [0, 0, 0] + np.mean(points, axis=0)
    end_point = normal_vector + np.mean(points, axis=0)

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector([start_point, end_point])
    lines.lines = o3d.utility.Vector2iVector([[0, 1]])

    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd, lines])


def project_point_to_plane(point, plane_coeffs):
    a, b, c, d = plane_coeffs
    x0, y0, z0 = point

    # 计算点到平面的距离
    D = np.abs(a * x0 + b * y0 + c * z0 + d) / np.sqrt(a**2 + b**2 + c**2)

    # 确定投影的方向
    sign = np.sign(a * x0 + b * y0 + c * z0 + d)

    # 计算投影点的坐标
    x_prime = x0 - sign * D * a / np.sqrt(a**2 + b**2 + c**2)
    y_prime = y0 - sign * D * b / np.sqrt(a**2 + b**2 + c**2)
    z_prime = z0 - sign * D * c / np.sqrt(a**2 + b**2 + c**2)

    return np.array([x_prime, y_prime, z_prime])


def build_transformation_matrix(x_axis, z_axis, origin):
    # 标准化 x 轴和 z 轴向量
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # 计算 y 轴向量
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # 构建变换矩阵
    transformation_matrix = np.eye(4)
    transformation_matrix[0:3, 0] = x_axis
    transformation_matrix[0:3, 1] = y_axis
    transformation_matrix[0:3, 2] = z_axis
    transformation_matrix[0:3, 3] = origin
    
    return transformation_matrix


if __name__ == "__main__":
    import open3d as o3d

    to_project_p = np.array([-0.01214473, 0.00629553, 1.31399])
    points = np.load("/home/red0orange/Projects/DemoTaskGrasp/hand_Pc.npy")
    C, normal_vector = fit_plane(points)
    
    normal_point = normal_vector + np.mean(points, axis=0)
    if not same_side_of_plane(to_project_p, normal_point, C):
        normal_vector = -normal_vector

    projected_p = project_point_to_plane(to_project_p, C)
    normal_vector /= (1 / np.std(points, axis=0))

    x_axis = projected_p - np.mean(points, axis=0)
    x_axis = x_axis / np.linalg.norm(x_axis)
    z_axis = normal_vector
    z_axis = z_axis / np.linalg.norm(z_axis)
    T = build_transformation_matrix(x_axis, z_axis, np.mean(points, axis=0))

    start_point = [0, 0, 0] + np.mean(points, axis=0)
    end_point = normal_vector + np.mean(points, axis=0)
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector([start_point, end_point])
    lines.lines = o3d.utility.Vector2iVector([[0, 1]])

    # 可视化
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0, 0, 0])

    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
    coordinate_frame.transform(T)

    to_project_pcd = o3d.geometry.PointCloud()
    to_project_pcd.points = o3d.utility.Vector3dVector([to_project_p])
    to_project_pcd.paint_uniform_color([0, 1, 0])
    projected_pcd = o3d.geometry.PointCloud()
    projected_pcd.points = o3d.utility.Vector3dVector([projected_p])
    projected_pcd.paint_uniform_color([1, 0, 0])
    o3d.visualization.draw_geometries([pcd, lines, to_project_pcd, projected_pcd, coordinate_frame])

    pass