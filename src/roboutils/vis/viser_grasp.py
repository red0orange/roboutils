import time

import numpy as np
import open3d as o3d
import viser
import viser.transforms as tf

from roboutils.proj_llm_robot.pose_transform import update_pose


class ViserForGrasp(object):
    def __init__(self):
        self.server = viser.ViserServer()
        self.gui_reset_scene = self.server.gui.add_button("Reset Scene")
        self.gui_break_flag = self.server.gui.add_button("Break Flag")

        self.reset_flag = False
        self.break_flag = False
        @self.gui_reset_scene.on_click
        def _reset(_) -> None:
            """Reset the scene when the reset button is clicked."""
            self.reset_flag = True
            pass
        @self.gui_break_flag.on_click
        def _break(_) -> None:
            """Reset the scene when the reset button is clicked."""
            self.break_flag = True
            pass

        self.pc_index = 0
        self.mesh_index = 0
        self.coord_index = 0
        self.grasp_index = 0
        pass

    def wait_for_reset(self):
        while (not self.reset_flag) and (not self.break_flag) :
            time.sleep(0.1)
        self.server.scene.reset()

        if self.reset_flag:
            self.reset_flag = False
            self.break_flag = False
            return False
        elif self.break_flag:
            self.reset_flag = False
            self.break_flag = False
            return True

    def add_mesh(self, mesh, name=None):
        if name is None:
            name = "mesh_{}".format(self.mesh_index)
            self.mesh_index += 1
        mesh_vertices = np.array(mesh.vertices)
        mesh_faces = np.array(mesh.triangles)
        self.server.scene.add_mesh_simple(name=name, vertices=mesh_vertices, faces=mesh_faces)
        pass

    def add_pcd(self, points, colors=None, name=None, point_size=0.003):
        if name is None:
            name = "pc_{}".format(self.pc_index)
            self.pc_index += 1
        if colors is None:
            colors = np.zeros((points.shape[0], 3))

        self.server.scene.add_point_cloud(name, points, colors=colors, point_size=point_size)
        pass

    def add_crood(self, T, name=None, axes_length=None, axes_radius=None):
        if axes_length is None:
            axes_length = 0.1
        if axes_radius is None:
            axes_radius = 0.006
        if name is None:
            name = "coord_{}".format(self.coord_index)
            self.coord_index += 1
            pass

        tf_se3 = tf.SE3.from_matrix(T)
        wxyz = tf_se3.wxyz_xyz[:4]
        xyz = tf_se3.wxyz_xyz[4:]

        self.server.scene.add_frame(name, wxyz=wxyz, position=xyz, axes_length=axes_length, axes_radius=axes_radius)
        pass

    def add_grasp(self, grasp_T, name=None, z_direction=True):
        if name is None:
            name = "grasp_{}".format(self.grasp_index)
            self.grasp_index += 1

        if z_direction:
            grasp_T = update_pose(grasp_T, rotate=-np.pi / 2, rotate_axis='y')
            grasp_T = update_pose(grasp_T, rotate=np.pi / 2, rotate_axis='x')
            pass

        meshes = self.get_gripper_control_points_o3d(grasp_T)
        for mesh_i, mesh in enumerate(meshes):
            sub_name = name + "/" + str(mesh_i)
            self.server.scene.add_mesh_simple(
                name=sub_name,
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.triangles),
                wxyz=tf.SO3.from_x_radians(0.0).wxyz,
                position=(0.0, 0.0, 0.0),
            )
        pass

    def vis_grasp_scene(self, grasp_Ts, pc=None, mesh=None, max_grasp_num=50, z_direction=True):
        if pc is not None:
            self.add_pcd(pc)
        if mesh is not None:
            self.add_mesh(mesh)
        for grasp_i, grasp_T in enumerate(grasp_Ts):
            if grasp_i >= max_grasp_num:
                break
            self.add_grasp(grasp_T, z_direction=z_direction)
        pass

    @staticmethod
    def get_gripper_control_points_o3d(grasp, color=(0.2, 0.8, 0)):
        # panda?
        control_points =  np.array([
            [-0.10, 0, 0, 1],
            [-0.03, 0, 0, 1],
            [-0.03, 0.07, 0, 1],
            [0.03, 0.07, 0, 1],
            [-0.03, 0.07, 0, 1],
            [-0.03, -0.07, 0, 1],
            [0.03, -0.07, 0, 1]])
        # # robotiq_85
        # control_points =  np.array([
        #     [-0.08, 0, 0, 1],
        #     [-0.03, 0, 0, 1],
        #     [-0.03, 0.055, 0, 1],
        #     [0.03, 0.055, 0, 1],
        #     [-0.03, 0.055, 0, 1],
        #     [-0.03, -0.055, 0, 1],
        #     [0.03, -0.055, 0, 1]])

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
        align = tf.SE3.from_rotation(tf.SO3.from_x_radians(np.pi/2)).as_matrix()

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
        transform = tf.SE3.from_rotation(tf.SO3.from_y_radians(np.pi / 2)).as_matrix()
        transform[0, 3] = ((control_points[0] + control_points[1]) / 2)[0]
        transform[2, 3] = ((control_points[0] + control_points[1]) / 2)[1]
        transform = np.matmul(align, transform)
        transform = np.matmul(grasp, transform)
        cylinder_2.paint_uniform_color(color)
        cylinder_2.transform(transform)

        # Cylinder 5,4
        cylinder_3 = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.005, height=cylinder_3_len)
        transform = tf.SE3.from_rotation(tf.SO3.from_y_radians(np.pi / 2)).as_matrix()
        transform[0, 3] = ((control_points[2] + control_points[3])/2)[0]
        transform[2, 3] = ((control_points[2] + control_points[3])/2)[1]
        transform = np.matmul(align, transform)
        transform = np.matmul(grasp, transform)
        cylinder_3.paint_uniform_color(color)
        cylinder_3.transform(transform)

        # Cylinder 6, 7
        cylinder_4 = o3d.geometry.TriangleMesh.create_cylinder(
            radius=0.005, height=cylinder_4_len)
        transform = tf.SE3.from_rotation(tf.SO3.from_y_radians(np.pi / 2)).as_matrix()
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