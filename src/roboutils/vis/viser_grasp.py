import time

import cv2
import imageio as iio
import numpy as np
import open3d as o3d
import viser
import viser.transforms as tf

from roboutils.proj_llm_robot.pose_transform import update_pose


def ray_intersection_with_xy_plane(origin, direction):
    x0, y0, z0 = origin
    dx, dy, dz = direction
    
    if dz == 0:
        raise ValueError("The ray is parallel to the x-y plane and will not intersect.")
    
    t = -z0 / dz  # Calculate the parameter t for the intersection
    x_intersect = x0 + t * dx
    y_intersect = y0 + t * dy
    
    return (x_intersect, y_intersect, 0)


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

        self.gui_render_diffusion_steps = self.server.gui.add_button("Render Grasp Diffusion Steps")
        self.diffusion_steps_data = None
        self.results_images = None
        @self.gui_render_diffusion_steps.on_click
        def _render_diffusion_steps(event: viser.GuiEvent) -> None:
            if self.diffusion_steps_data is not None:
                client = event.client
                client.scene.reset()

                time_grasp_Ts = self.diffusion_steps_data['grasp_Ts']
                obj_xyz  = self.diffusion_steps_data['obj_xyz']
                images = []
                for time_i in range(time_grasp_Ts.shape[0]):
                    grasp_Ts = time_grasp_Ts[time_i]
                    self.vis_grasp_scene(grasp_Ts, pc=obj_xyz, z_direction=True)
                    time.sleep(0.03)
                    images.append(client.get_render(height=480, width=640, transport_format="jpeg"))
                    client.scene.reset()
                self.results_images = images
                # print("Render grasp diffusion steps to images")
                # client.send_file_download(
                #     "image.gif", iio.imwrite("<bytes>", images, extension="gif", loop=0)
                # )
                # print("Render grasp diffusion steps to image.gif")

        self.pc_index = 0
        self.mesh_index = 0
        self.coord_index = 0
        self.grasp_index = 0
        pass

    def interact_image(self, image, name="image"):
        image = cv2.flip(image, -1)
        image_ori_width = image.shape[1]
        image_ori_height = image.shape[0]
        image_handle = self.add_image(image, name)

        # 设置视角
        self.server.scene.set_up_direction("+x")
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # Set up the camera -- this gives a nice view of the full mesh.
            client.camera.position = np.array([0.0, 0.0, -120.0])
            client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])
            camera_handle = client.camera
            pass

        # 点击事件
        self.tmp_return_cross_point = None
        @image_handle.on_click
        def _image_click(event):
            # print("Image click event:", event)
            wxyz = event.target.wxyz
            xyz = event.target.position
            render_width = image_handle.render_width
            render_height = image_handle.render_height

            ray_origin = event.ray_origin
            ray_direction = event.ray_direction

            x_y_cross_point = ray_intersection_with_xy_plane(ray_origin, ray_direction)
            print("x_y_cross_point:", x_y_cross_point)

            width_ratio = image_ori_width / render_width
            x_y_cross_point = (x_y_cross_point[0] * width_ratio, x_y_cross_point[1] * width_ratio)
            x_y_cross_point = (int(x_y_cross_point[0]), int(x_y_cross_point[1]))

            self.tmp_return_cross_point = np.array(x_y_cross_point)
        
        self.wait_for_reset()

        return_cross_point = self.tmp_return_cross_point
        return_cross_point[0] = -return_cross_point[0]
        return_cross_point[1] = -return_cross_point[1]
        return_cross_point[0] += (image_ori_width // 2)
        return_cross_point[1] += (image_ori_height // 2)
        image_handle.remove_click_callback()
        return return_cross_point
    
    def interact_select_image(self, images):
        assert len(images) > 0
        return_index = None

        self.next_button_handle = self.server.gui.add_button("Next One")
        self.break_button_handle = self.server.gui.add_button("This One")

        # 设置视角
        self.server.scene.set_up_direction("+x")
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            # Set up the camera -- this gives a nice view of the full mesh.
            client.camera.position = np.array([0.0, 0.0, -150.0])
            client.camera.wxyz = np.array([0.0, 0.0, 0.0, 1.0])
            camera_handle = client.camera
            pass

        # 初始化
        cur_index = 0
        cur_image = images[cur_index]
        cur_image_handle = self.add_image(cur_image, name="selected_image")

        @self.next_button_handle.on_click
        def _next_one(_) -> None:
            nonlocal cur_index
            nonlocal cur_image
            nonlocal cur_image_handle
            if cur_index >= len(images) - 1:
                cur_index = 0
            else:
                cur_index += 1
            cur_image_handle.remove()
            cur_image = images[cur_index]
            cur_image_handle = self.add_image(cur_image, name="selected_image")
        
        break_flag = False
        @self.break_button_handle.on_click
        def _this_one(_) -> None:
            nonlocal return_index
            nonlocal cur_index
            nonlocal break_flag
            print("This one")
            return_index = cur_index
            break_flag = True

        print("Waiting for select image...")
        while break_flag == False:
            time.sleep(0.1)

        self.next_button_handle.remove()
        self.break_button_handle.remove()
        return return_index

    def wait_for_reset(self):
        print("Waiting for reset...")
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

    def add_pcd(self, points, colors=None, name=None, point_size=0.002):
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

    def add_grasp(self, grasp_T, name=None, z_direction=True, grasp_color=None):
        if name is None:
            name = "grasp_{}".format(self.grasp_index)
            self.grasp_index += 1

        if z_direction:
            grasp_T = update_pose(grasp_T, rotate=-np.pi / 2, rotate_axis='y')
            grasp_T = update_pose(grasp_T, rotate=np.pi / 2, rotate_axis='x')
            pass

        if grasp_color is None:
            grasp_color = (90, 200, 255)
        meshes = self.get_gripper_control_points_o3d(grasp_T)
        for mesh_i, mesh in enumerate(meshes):
            sub_name = name + "/" + str(mesh_i)
            self.server.scene.add_mesh_simple(
                name=sub_name,
                vertices=np.array(mesh.vertices),
                faces=np.array(mesh.triangles),
                color=grasp_color,
                wxyz=tf.SO3.from_x_radians(0.0).wxyz,
                position=(0.0, 0.0, 0.0),
            )
        pass

    def vis_grasp_scene(self, grasp_Ts, pc=None, mesh=None, grasp_colors=None, max_grasp_num=50, z_direction=True):
        if grasp_colors is None:
            grasp_colors = [None] * len(grasp_Ts)
        if pc is not None:
            self.add_pcd(pc)
        if mesh is not None:
            self.add_mesh(mesh)
        for grasp_i, grasp_T in enumerate(grasp_Ts):
            if grasp_i >= max_grasp_num:
                break
            self.add_grasp(grasp_T, z_direction=z_direction, grasp_color=grasp_colors[grasp_i])
        pass

    def add_image(self, image, name=None, fix_width=320):
        if name is None:
            name = "image"
        render_ratio = fix_width / image.shape[1]
        render_height = int(image.shape[0] * render_ratio)
        image_handle = self.server.scene.add_image(name, image, render_width=fix_width, render_height=render_height)
        return image_handle

    def set_grasp_diffusion_steps(self, time_grasp_Ts, obj_xyz):
        self.diffusion_steps_data = {'grasp_Ts': time_grasp_Ts, 'obj_xyz': obj_xyz}
        pass

    def get_grasp_diffusion_steps_images(self):
        return self.results_images

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