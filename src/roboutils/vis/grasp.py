import argparse
import os
import sys
import glob
import time
import numpy as np
import open3d as o3d
import trimesh
import trimesh.transformations as tra
import matplotlib.pyplot as plt
from collections import defaultdict

from copy import deepcopy as copy

from PIL import Image
from tqdm import tqdm

from roboutils.proj_llm_robot.pose_transform import update_pose


def mkdir(dir):
    """
    Creates folder if it doesn't exist
    """
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_gripper_control_points():
    # panda?
    return np.array([
        [-0.10, 0, 0, 1],
        [-0.03, 0, 0, 1],
        [-0.03, 0.07, 0, 1],
        [0.03, 0.07, 0, 1],
        [-0.03, 0.07, 0, 1],
        [-0.03, -0.07, 0, 1],
        [0.03, -0.07, 0, 1]])
    # # robotiq_85
    # return np.array([
    #     [-0.08, 0, 0, 1],
    #     [-0.03, 0, 0, 1],
    #     [-0.03, 0.055, 0, 1],
    #     [0.03, 0.055, 0, 1],
    #     [-0.03, 0.055, 0, 1],
    #     [-0.03, -0.055, 0, 1],
    #     [0.03, -0.055, 0, 1]])
    

def gen_control_points(line_lens, ori_pt=[-0.08, 0]):
    ###############################
    #          24-----3
    #          |
    #          |
    #    0-----1
    #          |
    #          |
    #          5------6
    #
    #  line_lens: 分别是第一条、第二条、第三、四条 line 的长度
    #  ori_pt: 第一条 line 的起点坐标
    #  仿照 gcngrasp、contact-graspnet 的 control_points，生成 7 个点的坐标，仅在 xoy 平面上
    ###############################
    assert len(line_lens) == 3, "line_lens should be a list of length 3"
    assert len(ori_pt) == 2, "ori_pt should be a list of length 2"

    control_points = np.zeros((7, 4))
    control_points[:, 3] = 1
    control_points[0, :2] = ori_pt
    control_points[1, :2] = [ori_pt[0] + line_lens[0], ori_pt[1]]
    control_points[2, :2] = [ori_pt[0] + line_lens[0], ori_pt[1] + (line_lens[1])/2]
    control_points[3, :2] = [ori_pt[0] + line_lens[0] + line_lens[2], ori_pt[1] + (line_lens[1])/2]
    control_points[4, :2] = control_points[2, :2]
    control_points[5, :2] = [ori_pt[0] + line_lens[0], ori_pt[1] - (line_lens[1])/2]
    control_points[6, :2] = [ori_pt[0] + line_lens[0] + line_lens[2], ori_pt[1] - (line_lens[1])/2]

    return control_points


def get_gripper_control_points_o3d(
    grasp,
    control_points,
    color=(0.2, 0.8, 0)):
    """
    Open3D Visualization of parallel-jaw grasp

    grasp: [4, 4] np array
    """
    ###############################
    #          24-----3
    #          |
    #          |
    #    0-----1
    #          |
    #          |
    #          5------6
    ###############################
    cylinder_1_len = control_points[2, 1] - control_points[5, 1]
    cylinder_2_len = control_points[1, 0] - control_points[0, 0]
    cylinder_3_len = control_points[3, 0] - control_points[2, 0]
    cylinder_4_len = control_points[6, 0] - control_points[5, 0]
    

    meshes = []
    align = tra.euler_matrix(np.pi / 2, 0, 0)

    # Cylinder 3,5,6
    cylinder_1 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=cylinder_1_len)
    transform = np.eye(4)
    transform[0, 3] = control_points[1, 0]
    transform[2, 3] = control_points[1, 1]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_1.paint_uniform_color(color)
    cylinder_1.transform(transform)

    # Cylinder 1 and 2
    cylinder_2 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=cylinder_2_len)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = ((control_points[0] + control_points[1]) / 2)[0]
    transform[2, 3] = ((control_points[0] + control_points[1]) / 2)[1]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_2.paint_uniform_color(color)
    cylinder_2.transform(transform)

    # Cylinder 5,4
    cylinder_3 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=cylinder_3_len)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = ((control_points[2] + control_points[3])/2)[0]
    transform[2, 3] = ((control_points[2] + control_points[3])/2)[1]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_3.paint_uniform_color(color)
    cylinder_3.transform(transform)

    # Cylinder 6, 7
    cylinder_4 = o3d.geometry.TriangleMesh.create_cylinder(
        radius=0.005, height=cylinder_4_len)
    transform = tra.euler_matrix(0, np.pi / 2, 0)
    transform[0, 3] = ((control_points[5] + control_points[6])/2)[0]
    transform[2, 3] = ((control_points[5] + control_points[6])/2)[1]
    transform = np.matmul(align, transform)
    transform = np.matmul(grasp, transform)
    cylinder_4.paint_uniform_color(color)
    cylinder_4.transform(transform)

    cylinder_1.compute_vertex_normals()
    cylinder_2.compute_vertex_normals()
    cylinder_3.compute_vertex_normals()
    cylinder_4.compute_vertex_normals()

    meshes.append(cylinder_1)
    meshes.append(cylinder_2)
    meshes.append(cylinder_3)
    meshes.append(cylinder_4)

    return meshes


def downsample_pc(pc, nsample):
    if pc.shape[0] < nsample:
        print(
            'Less points in pc {}, than target dimensionality {}'.format(
                pc.shape[0],
                nsample))
    chosen_one = np.random.choice(
        pc.shape[0], nsample, replace=pc.shape[1] > nsample)
    return pc[chosen_one, :], chosen_one


def get_gripper_keypoints(grasps, z_direction=True, scale=1):
    grasps = np.array(grasps)
    grasps = grasps.copy()
    if z_direction:
        for i in range(len(grasps)):
            grasps[i] = update_pose(grasps[i], rotate=-np.pi / 2, rotate_axis='y')
            grasps[i] = update_pose(grasps[i], rotate=np.pi / 2, rotate_axis='x')
    
    gripper_control_points = get_gripper_control_points()
    gripper_control_points[:, :3] = scale * gripper_control_points[:, :3]

    grasp_keypoints = []
    for gi, grasp in enumerate(grasps):
        grasp_tmp = copy(grasp)

        grasp_pc = gripper_control_points
        grasp_pc = np.matmul(grasp_tmp, grasp_pc.T).T
        grasp_pc = grasp_pc[:, :3]

        grasp_keypoints.append(grasp_pc)
    
    return np.array(grasp_keypoints)


def draw_scene(
        pc=None,
        grasps=None,
        z_direction=False,
        scale=1,
        other_geometry=None,
        gripper_control_points=None,
        subtract_pc_mean=False,
        z_direction=False,
        meshes=None,
        debug_mode=False,
        max_pc_points=15000,
        max_grasps=10,
        use_pc_color=True,
        view_matrix=None,
        save_dir=None,
        grasp_colors=None,
        window_name=""):
    """
    Uses Open3D to plot grasps and point cloud data

    Args:
        save_dir: provide absolute path to save figure instead of visualizing on the GUI
    """
<<<<<<< HEAD
    grasps = np.array(grasps).copy()
    if z_direction:
        for i in range(len(grasps)):
            grasps[i] = update_pose(grasps[i], rotate=-np.pi / 2, rotate_axis='y')
            grasps[i] = update_pose(grasps[i], rotate=np.pi / 2, rotate_axis='x')
    if scale != 1:
        pc = pc * scale
        grasps[:, :3, 3] = grasps[:, :3, 3] * scale
=======
    if z_direction:
        grasps = np.array(grasps).copy()
        for i in range(len(grasps)):
            grasps[i] = update_pose(grasps[i], rotate=-np.pi / 2, rotate_axis='y')
            grasps[i] = update_pose(grasps[i], rotate=np.pi / 2, rotate_axis='x')
>>>>>>> cc591f77094ffa6600aaa3bd23945f66c5076828

    if gripper_control_points is None:
        gripper_control_points = get_gripper_control_points()

    if view_matrix is None:
        view_matrix = np.eye(4)

    if grasps is not None:
        if isinstance(grasps, np.ndarray):
            grasps = list(grasps)
        assert isinstance(grasps, list)

    if pc is None and meshes is None:
        raise InvalidArgumentError(
            "Pass in at least a mesh or the point cloud")

    if grasp_colors is not None:
        assert isinstance(grasp_colors, list)
        if grasps is not None:
            assert len(grasps) == len(grasp_colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)

    if pc is not None:
        if pc.shape[1] == 6:
            color = pc[:, 3:] / 255.0

        if pc.shape[0] > max_pc_points:
            pc, selection = downsample_pc(pc, max_pc_points)

        n_points = pc.shape[0]
        color = None

        if pc.shape[1] == 6:
            color = np.zeros((n_points, 4))
            color[:, :3] = pc[:, 3:]
            color[:, 3] = np.ones(n_points)

        if subtract_pc_mean:
            pc_mean = np.mean(pc[:, :3], 0)
            pc[:, :3] -= pc_mean

        pc_o3d = o3d.geometry.PointCloud()
        pc_o3d.points = o3d.utility.Vector3dVector(pc[:, :3])

        if color is not None and use_pc_color:
            pc_o3d.colors = o3d.utility.Vector3dVector(pc[:, 3:] / 255)
        else:
            pc_o3d.paint_uniform_color([1, 0.706, 0])

        pc_o3d.transform(view_matrix)
        vis.add_geometry(pc_o3d)

    if grasps is not None:
        if len(grasps) > max_grasps:
            assert isinstance(grasps, list)
            selection = np.random.randint(
                low=0, high=len(grasps), size=max_grasps)
            grasps = list(np.array(grasps)[selection])
            if grasp_colors is not None:
                if not isinstance(grasp_colors, np.ndarray):
                    grasp_colors = np.array(grasp_colors)
                grasp_colors = grasp_colors[selection]

        for gi, grasp in enumerate(grasps):
            grasp_tmp = copy(grasp)
            if pc is not None and subtract_pc_mean:
                grasp_tmp[:3, 3] -= pc_mean

            color = (0.2, 0.8, 0)
            if grasp_colors is not None:
                color = grasp_colors[gi]

            for item in get_gripper_control_points_o3d(
                    grasp_tmp, gripper_control_points, color=color):
                item.transform(view_matrix)
                vis.add_geometry(item)

            grasp_pc = gripper_control_points
            grasp_pc = np.matmul(grasp_tmp, grasp_pc.T).T
            # grasp_pc = grasp_pc[:4, :]
            grasp_pc_o3d = o3d.geometry.PointCloud()
            grasp_pc_o3d.points = o3d.utility.Vector3dVector(grasp_pc[:, :3])
            grasp_pc_o3d.paint_uniform_color([0, 0, 1])
            grasp_pc_o3d.transform(view_matrix)
            vis.add_geometry(grasp_pc_o3d)

    if meshes is not None:
        assert isinstance(meshes, list)
        if isinstance(
                meshes[0],
                tuple) and isinstance(
                meshes[0][0],
                trimesh.primitives.Box):

            # Convert from trimesh to open3d
            meshes_o3d = []
            for elem in meshes:
                (voxel, extents, transform) = elem
                voxel_o3d = o3d.geometry.TriangleMesh.create_box(
                    width=extents[0], height=extents[1], depth=extents[2])
                voxel_o3d.compute_vertex_normals()
                voxel_o3d.paint_uniform_color([0.8, 0.2, 0])
                voxel_o3d.transform(transform)
                meshes_o3d.append(voxel_o3d)
            meshes = meshes_o3d

        for mesh in meshes:
            mesh.transform(view_matrix)
            vis.add_geometry(mesh)
    
    if other_geometry is not None:
        for geom in other_geometry:
            vis.add_geometry(geom)

    opt = vis.get_render_option()
    opt.point_size = 10.0

    if save_dir is None:
        vis.run()

    if save_dir is not None:
        mkdir(os.path.dirname(save_dir))
        time.sleep(0.25)
        image = vis.capture_screen_float_buffer(True)
        time.sleep(0.25)
        image = np.asarray(image)
        DELTA_Y = 20
        DELTA_X = 200
        image = crop(image, DELTA_X, DELTA_Y)
        plt.imsave(save_dir, np.asarray(image))
        time.sleep(0.25)

    if save_dir is None:
        vis.destroy_window()


def crop(image, delta_x, delta_y):
    h, w, _ = image.shape
    return image[delta_y:h - delta_y, delta_x:w - delta_x, :]


def visualize_labels(
        obj_path,
        rgb_image_path,
        vis_dir,
        label_filename,
        visualize_labels_blacklist_object,
        visualize_labels_blacklist_task):
    """
    This function is used to visualize labeled grasps

    :param obj_path: path to the /scans folder
    :param vis_dir: where visualizations should be saved
    :param label_filename: file storing labeled grasps. In this file, each grasp can be labeled either by majority vote
                           or accumulated score. Need to set visualize_majority correspondingly.
    :param visualize_labels_blacklist_object:
    :param visualize_labels_blacklist_task:
    :return:
    """

    # visualization from different angles
    all_yaws = [np.pi / 3, -np.pi / 3, 0, 0]
    all_pitchs = [0, 0, np.pi / 3, -np.pi / 3]

    # read results
    # save labels in a dictionary. E.g., {("001_squeezer", "flatten"): [(1,
    # "True"), (2, "Weak True"), (3, "False")]}
    all_labels = defaultdict(list)
    with open(label_filename, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                grasp_id, score = line.split(":")
                score = int(score)
                obj, grasp_num, task = grasp_id.split("-")
                grasp_num = int(grasp_num)
                all_labels[(obj, task)].append((grasp_num, score))

    for obj, task in tqdm(all_labels):
        if obj.find(visualize_labels_blacklist_object) >= 0 and task.find(
                visualize_labels_blacklist_task) >= 0:

            pc_file = os.path.join(obj_path, obj, "fused_pc.npy")
            if not os.path.exists(pc_file):
                raise ValueError(
                    'Unable to find processed point cloud file {}'.format(pc_file))

            rgb_file = os.path.join(rgb_image_path, "{}.png".format(obj))
            if not os.path.exists(rgb_file):
                raise ValueError(
                    'Unable to find rgb image file {}'.format(rgb_file))

            pc = np.load(pc_file)
            pc_mean = pc[:, :3].mean(axis=0)
            pc[:, :3] -= pc_mean

            grasp_colors = []
            grasps = []
            labels = all_labels[(obj, task)]
            labels = sorted(labels, key=lambda x: x[0])
            for grasp_id, label in labels:
                grasp_file = os.path.join(
                    args.obj_path, obj, "grasps", str(grasp_id), "grasp.npy")
                if not os.path.exists(grasp_file):
                    raise ValueError(
                        'Grasp {} not found, have you rendered the grasp?'.format(grasp_file))
                grasp = np.load(grasp_file)
                grasps.append(grasp)

                if label == 1:
                    color = (0, 0.9, 0)
                elif label == 0:
                    color = (0.9, 0.9, 0)
                elif label == -1:
                    color = (0.9, 0, 0)
                elif label == -2:
                    color = (0.9, 0.5, 0)
                else:
                    raise Exception

                grasp_colors.append(color)

            print("Object {}, Task {}".format(obj, task))

            draw_scene(
                pc,
                grasps,
                grasp_colors=grasp_colors,
                window_name=task,
                max_grasps=len(grasps))


def combine_images(
        visualization_files,
        rgb_image_file,
        combined_file,
        single_img_dimesion=[
            1920,
            1080],
        final_img_width=None):
    """
    This function is used to combine point cloud visualizations from different angles and the rgb image

    :param visualization_files: paths to visualizations
    :param rgb_image_file: path to the rgb image
    :param single_img_dimesion: size of each image
    :param final_img_width:
    :return:
    """

    imgs = []
    for img_file in visualization_files:
        img = Image.open(img_file)
        # resize images to fixed size. Images rendered from open3d do not have
        # fixed size
        img = img.resize(single_img_dimesion, Image.ANTIALIAS)
        img = np.asarray(img)[..., :3]
        imgs.append(img)

    rgb_image = Image.open(rgb_image_file)
    width, height = rgb_image.size
    resize_height = imgs[0].shape[0]
    resize_width = int(resize_height * 1.0 / height * width)
    rgb_image = np.asarray(
        rgb_image.resize(
            (resize_width,
             resize_height),
            Image.ANTIALIAS))
    imgs.insert(0, rgb_image)

    imgs_comb = np.hstack([img for img in imgs])
    imgs_comb = Image.fromarray(imgs_comb)
    if final_img_width is not None:
        width, height = imgs_comb.size
        resize_width = final_img_width
        resize_height = int(resize_width * 1.0 / width * height)
        imgs_comb = imgs_comb.resize(
            (resize_width, resize_height), Image.ANTIALIAS)

    imgs_comb.save(combined_file)


def main(args):
    session_dir = os.path.join(args.obj_path, args.obj_name)
    object_files = sorted(glob.glob(os.path.join(session_dir, "*.pkl")))
    obj_dir = os.path.dirname(object_files[0])

    if args.data_and_grasps:

        pc_file = os.path.join(obj_dir, 'fused_pc_clean.npy')
        grasps_file = os.path.join(obj_dir, 'fused_grasps_clean.npy')

        if not os.path.exists(pc_file):
            raise ValueError(
                'Unable to find processed point cloud file {}'.format(pc_file))

        data_and_grasps(pc_file, grasps_file, fps=args.fps)

    elif args.visualize_grasp:

        if args.grasp_id == -1:
            raise ValueError('Please pass in grasp_id in the args')

        pc_file = os.path.join(obj_dir, 'fused_pc_clean.npy')
        grasp_file = os.path.join(
            obj_dir, str("grasps"), str(
                args.grasp_id), "grasp.npy")

        if not os.path.exists(pc_file):
            raise ValueError(
                'Unable to find processed point cloud file {}'.format(pc_file))

        if not os.path.exists(grasp_file):
            raise ValueError(
                'Grasp {} not found, have you rendered the grasp?'.format(
                    args.grasp_id))

        pc = np.load(pc_file)
        # Sampled grasps are already pc mean centered
        grasp = np.load(grasp_file)

        pc_mean = pc[:, :3].mean(axis=0)
        pc[:, :3] -= pc_mean

        draw_scene(pc, [grasp, ])

    elif args.visualize_labels:

        if not os.path.exists(args.label_path):
            raise ValueError('Please pass in grasp label file in the args')

        vis_dir = os.path.join(args.obj_path, "../labeled_grasps")
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        print("Saving visualization of labeled grasps to {} ...".format(vis_dir))

        if not os.path.exists(args.rgb_image_path):
            raise ValueError('Please pass in path to rgb images in the args')

        visualize_labels(
            args.obj_path,
            args.rgb_image_path,
            vis_dir,
            args.label_path,
            args.visualize_labels_blacklist_object,
            args.visualize_labels_blacklist_task)
    else:
        print("Nothing to do :) Please provide the right args ")


def set_seed(seed):
    np.random.seed(seed)


def process_args():
    parser = argparse.ArgumentParser(description="visualize data and stuff")
    parser.add_argument('--obj_name', help='', default='002_strainer')
    parser.add_argument('--obj_path', help='', default='')
    parser.add_argument('--seed', help='', type=int, default=0)
    parser.add_argument('--grasp_id', help='', type=int, default=-1)
    parser.add_argument(
        '--visualize_grasp',
        help='',
        action='store_true',
        default=False)
    parser.add_argument(
        '--visualize_labels',
        help='',
        action='store_true',
        default=False)
    parser.add_argument(
        '--visualize_labels_debug',
        help='Step through grasps one at a time',
        action='store_true',
        default=False)
    parser.add_argument(
        '--visualize_labels_blacklist_task',
        help='',
        type=str,
        default='')
    parser.add_argument(
        '--visualize_labels_blacklist_object',
        help='',
        type=str,
        default='')
    parser.add_argument(
        '--visualize_data_and_grasps',
        help='',
        action='store_true',
        default=False)
    parser.add_argument(
        '--data_and_grasps',
        help='',
        action='store_true',
        default=False)
    parser.add_argument(
        '--fps',
        help='Use farthest point sampling on grasps',
        type=int,
        default=1)
    parser.add_argument('--label_path', help='', default='')
    parser.add_argument('--rgb_image_path', help='', default='')
    args = parser.parse_args()

    if args.obj_path == '':
        args.obj_path = os.path.join(os.getcwd(), 'data/taskgrasp')
    assert os.path.exists(args.obj_path)
    assert os.path.isabs(args.obj_path)

    if args.label_path == '':
        args.label_path = os.path.join(args.obj_path, "task2_results.txt")

    if args.rgb_image_path == '':
        args.rgb_image_path = os.path.join(args.obj_path, "rgb_images")

    args.obj_path = os.path.join(args.obj_path, 'scans')
    assert os.path.exists(args.obj_path)

    set_seed(args.seed)

    return args

if __name__ == '__main__':
    args = process_args()
    main(args)
