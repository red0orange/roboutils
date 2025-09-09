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
    def __init__(self, port=8080):
        self.server = viser.ViserServer(port=port)

        self.pc_index = 0
        self.mesh_index = 0
        self.coord_index = 0
        self.grasp_index = 0

        self.show_text_handle = self.server.gui.add_markdown(content="")

        self.clients = []
        self.clients_camera_handle = []
        self.camera_pose_update_func = []
        self.camera_dict = {}
        @self.server.on_client_connect
        def connect_handler(client: viser.ClientHandle) -> None:
            self.clients.append(client)
            self.clients_camera_handle.append(client.camera)

            # 当先设置场景，再连接的时候，正常设置视角
            self.set_client_camera_view(self.camera_dict)

            # # 为了手动选择视角，添加监听事件
            # @self.camera_handle.on_update
            # def update_handler(camera_handle) -> None:
            #     print("position: {}".format(camera_handle.position))
            #     print("wxyz: {}".format(camera_handle.wxyz))
            #     print("look at: {}".format(camera_handle.look_at))
            #     print("up: {}".format(camera_handle.up_direction))
            #     pass
            # self.camera_pose_update_func.append(update_handler) # 防止被销毁
        self.on_client_connect_func = connect_handler  # 防止被销毁
        pass

    def set_client_camera_view(self, camera_dict):
        self.camera_dict = camera_dict
        if len(self.clients) == 0:
            # 等待连接后自动设置视角
            return False
        for camera_key, camera_value in self.camera_dict.items():
            # 给每个客户端都更新视角
            for client_handle, camera_handle in zip(self.clients, self.clients_camera_handle):
                if camera_key == "position":
                    camera_handle.position = camera_value
                elif camera_key == "wxyz":
                    camera_handle.wxyz = camera_value
                elif camera_key == "look_at":
                    camera_handle.look_at(camera_value)
                elif camera_key == "up_direction":
                    camera_handle.up_direction = camera_value
        return True
    
    def set_image_visual_view(self):
        # 展示设置固定视角
        self.server.scene.set_up_direction("+x")
        camera_dict = {
            "position": np.array([0.0, 0.0, -180]),
            "wxyz": np.array([0.0, 0.0, 0.0, 1.0]),
        }
        self.set_client_camera_view(camera_dict)
        pass

    def set_grasp_visual_view(self):
        # 展示设置固定视角
        # @TODO
        self.server.scene.set_up_direction("+y")
        camera_dict = {
            "position": np.array([-0.24913841, 0.02283902, 0.52435203]),
            "wxyz": np.array([0.3728618, -0.5706226, 0.61251547, -0.40023585]),
            "up_direction": np.array([0.0, 0.0, 1.0]),
        }
        self.set_client_camera_view(camera_dict)
        pass

    def reset_scene(self):
        self.server.scene.reset()
        pass

    def update_show_text(self, text):
        self.show_text_handle.content = text
        pass

    def wait_for_button_click(self, button_name):
        button_handle = self.server.gui.add_button(button_name)

        clicked_flag = False
        @button_handle.on_click
        def _button_click(event):
            nonlocal clicked_flag
            clicked_flag = True
        
        while clicked_flag == False:
            time.sleep(0.1)
        
        button_handle.remove()
        return True

    def wait_for_button_group_click(self, button_group_name, options):
        assert len(options) > 0

        # @note button_group 有点问题，不灵敏，而且显示不全
        # button_group_handle = self.server.gui.add_button_group(button_group_name, options)
        # return_option = None
        # @button_group_handle.on_click
        # def _button_group_click(event):
        #     nonlocal return_option
        #     option = event.target.value
        #     return_option = option
        # while return_option is None:
        #     time.sleep(0.1)
        # button_group_handle.remove()

        # 添加说明
        markdown_handle = self.server.gui.add_markdown(content="### {}".format(button_group_name))

        button_handles_dict = {}
        clicked_option = None

        # 为每个按钮添加监听事件
        def _button_click(option):
            nonlocal clicked_option
            clicked_option = option

        for option in options:
            button_handle = self.server.gui.add_button(option)
            button_handles_dict[option] = button_handle
            # 设置每个按钮的点击事件
            button_handle.on_click(lambda event, option=option: _button_click(option))
        
        # 等待按钮点击
        while clicked_option is None:
            time.sleep(0.1)

        # 移除所有按钮
        markdown_handle.remove()
        for button_handle in button_handles_dict.values():
            button_handle.remove()

        return clicked_option
    
    def wait_for_text_input(self, text_input_name):
        text_input_handle = self.server.gui.add_text(label=text_input_name, initial_value="")
        self.wait_for_button_click("确认输入")
        value = text_input_handle.value
        text_input_handle.remove()
        return value

    def show_image(self, image, name="image"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, -1)
        image_ori_width = image.shape[1]
        image_ori_height = image.shape[0]
        image_handle = self.add_image(image, name)

        # 设置视角
        self.set_image_visual_view()
        self.wait_for_button_click("Next")
        self.server.scene.reset()
        pass

    def interact_image(self, image, name="image"):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, -1)
        image_ori_width = image.shape[1]
        image_ori_height = image.shape[0]
        image_handle = self.add_image(image, name)

        # 设置视角
        self.set_image_visual_view()

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
        
        option = self.wait_for_button_group_click("确认选择", ["确认选择", "取消选择"])
        if option == "取消选择":
            return_cross_point = None
        else:
            return_cross_point = self.tmp_return_cross_point
            return_cross_point[0] = -return_cross_point[0]
            return_cross_point[1] = -return_cross_point[1]
            return_cross_point[0] += (image_ori_width // 2)
            return_cross_point[1] += (image_ori_height // 2)

        image_handle.remove_click_callback()
        self.server.scene.reset()
        return return_cross_point
    
    def interact_select_image(self, images):
        assert len(images) > 0
        images = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
        images = [cv2.flip(image, -1) for image in images]
        return_index = None

        next_button_handle = self.server.gui.add_button("Next One")
        break_button_handle = self.server.gui.add_button("This One")
        giveup_button_handle = self.server.gui.add_button("Give Up")

        # 设置视角
        self.set_image_visual_view()

        # 初始化
        cur_index = 0
        cur_image = images[cur_index]
        cur_image_handle = self.add_image(cur_image, name="selected_image")

        @next_button_handle.on_click
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
        @break_button_handle.on_click
        def _this_one(_) -> None:
            nonlocal return_index
            nonlocal cur_index
            nonlocal break_flag
            print("This one")
            return_index = cur_index
            break_flag = True
        
        @giveup_button_handle.on_click
        def _give_up(_) -> None:
            nonlocal return_index
            nonlocal break_flag
            return_index = None
            break_flag = True

        print("Waiting for select image...")
        while break_flag == False:
            time.sleep(0.1)

        next_button_handle.remove()
        break_button_handle.remove()
        giveup_button_handle.remove()
        cur_image_handle.remove()
        self.server.scene.reset()
        return return_index

    def wait_for_reset(self):
        gui_reset_scene = self.server.gui.add_button("Reset Scene")
        gui_break_flag = self.server.gui.add_button("Break Flag")

        reset_flag = False
        break_flag = False
        @gui_reset_scene.on_click
        def _reset(_) -> None:
            """Reset the scene when the reset button is clicked."""
            nonlocal reset_flag
            reset_flag = True
            pass
        @gui_break_flag.on_click
        def _break(_) -> None:
            """Reset the scene when the reset button is clicked."""
            nonlocal break_flag
            break_flag = True
            pass

        print("Waiting for reset...")
        while (not reset_flag) and (not break_flag) :
            time.sleep(0.1)

        self.server.scene.reset()
        gui_break_flag.remove()
        gui_reset_scene.remove()

        if break_flag:
            return True
        else:
            return False

    def add_mesh(self, mesh, name=None, color=(90, 200, 255)):
        if name is None:
            name = "mesh_{}".format(self.mesh_index)
            self.mesh_index += 1
        mesh_vertices = np.array(mesh.vertices)
        mesh_faces = np.array(mesh.triangles)
        self.server.scene.add_mesh_simple(name=name, vertices=mesh_vertices, faces=mesh_faces, color=color)
        pass

    def add_pcd(self, points, colors=None, name=None, point_size=0.001):
        if name is None:
            name = "pc_{}".format(self.pc_index)
            self.pc_index += 1
        if colors is None:
            colors = np.zeros((points.shape[0], 3), dtype=np.float16)
            colors[:, 0] = 0.3
            colors[:, 1] = 0.3
            colors[:, 2] = 0.3

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

    def vis_grasp_scene(self, grasp_Ts, pc=None, mesh=None, grasp_colors=None, max_grasp_num=50, z_direction=True, pc_colors=None, set_view=True):
        if set_view:
            self.set_grasp_visual_view()
        
        if grasp_colors is None:
            grasp_colors = [None] * len(grasp_Ts)
        if pc is not None:
            self.add_pcd(pc, colors=pc_colors)
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
        render_height = image.shape[0] * render_ratio
        image_handle = self.server.scene.add_image(name, image, render_width=fix_width, render_height=render_height)
        return image_handle

    def render_diffusion_steps(self, grasp_Ts, obj_xyz):
        gui_render_diffusion_steps = self.server.gui.add_button("Render Grasp Diffusion Steps")
        diffusion_steps_data = {
            "grasp_Ts": grasp_Ts,
            "obj_xyz": obj_xyz,
        }
        results_images = None

        finish_render_flag = False
        @gui_render_diffusion_steps.on_click
        def _render_diffusion_steps(event: viser.GuiEvent) -> None:
            nonlocal finish_render_flag
            nonlocal results_images
            nonlocal diffusion_steps_data
            if diffusion_steps_data is not None:
                client = event.client
                client.scene.reset()

                time_grasp_Ts = diffusion_steps_data['grasp_Ts']
                obj_xyz  = diffusion_steps_data['obj_xyz']
                images = []
                print(1)
                for time_i in range(time_grasp_Ts.shape[0]):
                    grasp_Ts = time_grasp_Ts[time_i]
                    grasp_colors = [(210, 210, 210)] * len(grasp_Ts)
                    print(grasp_Ts.shape)
                    self.vis_grasp_scene(grasp_Ts, z_direction=True, set_view=False, grasp_colors=grasp_colors)
                    obj_colors = [(90, 200, 255)] * len(obj_xyz)
                    self.add_pcd(obj_xyz, colors=obj_colors)
                    print(2)
                    time.sleep(0.03)
                    images.append(client.get_render(height=480, width=640, transport_format="jpeg"))
                    client.scene.reset()
                results_images = images
                finish_render_flag = True
                # print("Render grasp diffusion steps to images")
                # client.send_file_download(
                #     "image.gif", iio.imwrite("<bytes>", images, extension="gif", loop=0)
                # )
                # print("Render grasp diffusion steps to image.gif")
        
        while finish_render_flag == False:
            time.sleep(0.1)

        gui_render_diffusion_steps.remove()
        return results_images

    def render_hand_diffusion_steps(self, grasp_Ts, obj_xyz, hand_mesh, contact_center, palm_arrow_mesh):
        gui_render_diffusion_steps = self.server.gui.add_button("Render Grasp Diffusion Steps")
        diffusion_steps_data = {
            "grasp_Ts": grasp_Ts,
            "obj_xyz": obj_xyz,
        }
        results_images = None

        finish_render_flag = False
        @gui_render_diffusion_steps.on_click
        def _render_diffusion_steps(event: viser.GuiEvent) -> None:
            nonlocal finish_render_flag
            nonlocal results_images
            nonlocal diffusion_steps_data
            if diffusion_steps_data is not None:
                client = event.client
                client.scene.reset()

                time_grasp_Ts = diffusion_steps_data['grasp_Ts']
                obj_xyz  = diffusion_steps_data['obj_xyz']
                images = []
                for time_i in range(time_grasp_Ts.shape[0]):
                    grasp_Ts = time_grasp_Ts[time_i]
                    grasp_colors = [(210, 210, 210)] * len(grasp_Ts)
                    self.vis_grasp_scene(grasp_Ts, z_direction=True, set_view=False, grasp_colors=grasp_colors)
                    obj_colors = [(90, 200, 255)] * len(obj_xyz)
                    self.add_pcd(obj_xyz, colors=obj_colors)
                    self.add_mesh(hand_mesh, color=(225, 184, 155), opacity=0.5)
                    self.add_mesh(palm_arrow_mesh, color=(0, 126, 60), opacity=0.8)
                    self.add_big_pcd([contact_center], colors=[(0, 126, 60)], point_size=0.01, opacity=0.8)
                    time.sleep(0.03)
                    images.append(client.get_render(height=480, width=640, transport_format="jpeg"))
                    client.scene.reset()
                results_images = images
                finish_render_flag = True
                # print("Render grasp diffusion steps to images")
                # client.send_file_download(
                #     "image.gif", iio.imwrite("<bytes>", images, extension="gif", loop=0)
                # )
                # print("Render grasp diffusion steps to image.gif")
        
        while finish_render_flag == False:
            time.sleep(0.1)

        gui_render_diffusion_steps.remove()
        return results_images

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