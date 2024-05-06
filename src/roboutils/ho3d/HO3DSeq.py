import os
import cv2
import numpy as np
import trimesh

import torch
from liegroups import SO3
from manopth.manolayer import ManoLayer

from my_utils.ho3d.eval import EvalUtil, calculate_fscore, align_sc_tr, align_w_scale, align_by_trafo
from my_utils.mask_to_bbox import masks_to_boxes


glcam_in_cvcam = np.array([[1,0,0,0],
                          [0,-1,0,0],
                          [0,0,-1,0],
                          [0,0,0,1]])


def transform_joint_to_other_db(src_joint, src_name, dst_name):
    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]), dtype=np.float32)
    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint


mano_joints_name = ('Wrist', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_1', 'Index_2', 'Index_3', 'Index_4', 'Middle_1', 'Middle_2', 'Middle_3', 'Middle_4', 'Ring_1', 'Ring_2', 'Ring_3', 'Ring_4', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Pinly_4')
ho3d_joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinly_4')
def mano_to_ho3d_joints3d(joints3d, root_joint_idx=0):
    global mano_joints_name, ho3d_joints_name
    # @note 转换 joint
    joints3d = transform_joint_to_other_db(joints3d, mano_joints_name, ho3d_joints_name)

    joints3d = np.asarray(joints3d, dtype=np.float32)

    return joints3d


def load_objects(obj_root):
    object_names = [obj_name for obj_name in os.listdir(obj_root) if ".tgz" not in obj_name]
    objects = {}
    for obj_name in object_names:
        obj_path = os.path.join(obj_root, obj_name, "textured_simple_ds.obj")
        mesh = trimesh.load(obj_path)
        objects[obj_name] = {"verts": mesh.vertices, "faces": mesh.faces}
    return objects


def center_vert_bbox(vertices, bbox_center=None, bbox_scale=None, scale=True):
    if bbox_center is None:
        bbox_center = (vertices.min(0) + vertices.max(0)) / 2
    vertices = vertices - bbox_center
    if scale:
        if bbox_scale is None:
            bbox_scale = np.linalg.norm(vertices, 2, 1).max()
        vertices = vertices / bbox_scale
    else:
        bbox_scale = 1
    return vertices, bbox_center, bbox_scale


class HO3DSeq(torch.utils.data.Dataset):
    def __init__(self, data_dir, model_dir="/home/red0orange/Data/HO3D_v3/HO3D_v3/YCB_models_supp"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.rgb_image_dir = os.path.join(data_dir, "rgb")
        self.depth_image_dir = os.path.join(data_dir, "depth")
        self.meta_dir = os.path.join(data_dir, "meta")
        
        self.image_names = sorted(os.listdir(self.rgb_image_dir))
        self.image_paths = [os.path.join(self.rgb_image_dir, image_name) for image_name in self.image_names]
        self.depth_image_paths = [os.path.join(self.depth_image_dir, image_name.replace("jpg", "png")) for image_name in self.image_names]
        self.meta_paths = [os.path.join(self.meta_dir, image_name.replace("jpg", "pkl")) for image_name in self.image_names]

        # 提取 K
        self.K = None
        for meta_path in self.meta_paths:
            meta = np.load(meta_path, allow_pickle=True)
            if meta["camMat"] is not None:
                self.K = meta["camMat"]
                break
        assert self.K is not None

        # gt 相关
        mano_assets_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang", use_pca=False, mano_root=os.path.join(mano_assets_path, "mano"), center_idx=None, flat_hand_mean=True,
        )
        self.obj_meshes = load_objects(self.model_dir)
        self.cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pass

    def read_ho3d_depth_image(self, depth_path):
        depth_scale = 0.00012498664727900177
        depth_img = cv2.imread(depth_path)
        depth = depth_img[:, :, 2] + depth_img[:, :, 1] * 256
        depth = depth * depth_scale
        return depth

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_name, image_path, depth_image_path, meta_path = self.image_names[index], self.image_paths[index], self.depth_image_paths[index], self.meta_paths[index]
        rgb = cv2.imread(image_path)
        depth = self.read_ho3d_depth_image(depth_image_path)
        meta = np.load(meta_path, allow_pickle=True)
        return rgb, depth, meta, image_name
    
    def get_gt(self, index):
        meta_path = self.meta_paths[index]
        meta = np.load(meta_path, allow_pickle=True)

        # 获得 objTrans, objRot
        objTrans, objRot = meta["objTrans"], meta["objRot"]
        objTrans = self.preprocess_obj_transf(objTrans, objRot, objName=meta["objName"])
        obj_rot = objTrans[:3, :3]  # (3, 3)
        obj_tsl = objTrans[:3, 3:]  # (3, 1)

        obj_faces = self.obj_meshes[meta["objName"]]["faces"]
        obj_verts_can, _, _ = self.get_obj_verts_can(meta["objName"])
        obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()

        # 获得手的
        hand_shape = meta["handBeta"].astype(np.float32)  # (10,)
        ori_hand_pose = meta["handPose"]  # (48,)
        ori_hand_tsl = meta["handTrans"]  # (3,)
        ori_hand_verts = self.get_hand_verts3d(hand_shape, ori_hand_pose)  # (778, 3)
        ori_hand_verts = ori_hand_verts + ori_hand_tsl  # (778, 3)
        ori_hand_verts = self.cam_extr[:3, :3].dot(ori_hand_verts.transpose()).transpose()  # (778, 3)
        ori_hand_verts = ori_hand_verts.astype(np.float32)

        hand_pose = self.preprocess_hand_pose(ori_hand_pose)  # (48,)
        process_hand_verts = self.get_hand_verts3d(hand_shape, hand_pose)  # (778, 3)
        hand_tsl = (ori_hand_verts - process_hand_verts)[0]  # (778, 3)

        data_dict = {
            "obj_verts_cam": obj_verts_pred,
            "obj_faces": obj_faces,
            "hand_shape": hand_shape,
            "hand_pose": hand_pose,
            "hand_tsl": hand_tsl,

            "ori_hand_pose": ori_hand_pose,
            "ori_hand_tsl": ori_hand_tsl,
            # "hand_verts_cam": process_hand_verts[0],
        }
        return data_dict
    
    def get_hand_verts3d(self, hand_shape, hand_pose):
        hand_pose = torch.from_numpy(hand_pose).unsqueeze(0)
        hand_shape = torch.from_numpy(hand_shape).unsqueeze(0)
        hand_verts, _ = self.mano_layer(hand_pose, hand_shape)
        hand_verts /= 1000.0
        hand_verts = np.array(hand_verts.squeeze(0))
        return hand_verts

    def preprocess_hand_pose(self, handpose):  # pose = root_rot + ...
        # only the first 3 dimension needs to be transformed by cam_extr
        root, remains = handpose[:3], handpose[3:]
        root = SO3.exp(root).as_matrix()
        root = self.cam_extr[:3, :3] @ root
        root = SO3.log(SO3.from_matrix(root, normalize=True))
        handpose_transformed = np.concatenate((root, remains), axis=0)
        return handpose_transformed.astype(np.float32)

    def get_obj_verts_can(self, objName):
        verts = self.obj_meshes[objName]["verts"]
        verts = self.cam_extr[:3, :3].dot(verts.transpose()).transpose()
        verts_can, bbox_center, bbox_scale = center_vert_bbox(verts, scale=False)  # !! CENTERED HERE
        return verts_can, bbox_center, bbox_scale

    def preprocess_obj_transf(self, objTrans, objRot, objName):
        """
        预处理 meta 中的 objTrans, objRot
        """
        rot = cv2.Rodrigues(objRot)[0]
        tsl = objTrans

        verts_can, v_0, _ = self.get_obj_verts_can(objName)  # (N, 3), (3, ), 1

        """ HACK
        v_{can} = E * v_{raw} - v_0
        v_{cam} = E * (R * v_{raw} + t)

        => v_{raw} = E^{-1} * (v_{can} + v_0)
        => v_{cam} = E * (R * (E^{-1} * (v_{can} + v_0)) + t)
        =>         = E*R*E^{-1} * v_{can} + E*R*E^{-1} * v_0 + E * t
        """

        ext_rot = self.cam_extr[:3, :3]
        ext_rot_inv = np.linalg.inv(ext_rot)

        rot_wrt_cam = ext_rot @ (rot @ ext_rot_inv)  # (3, 3)
        tsl_wrt_cam = (ext_rot @ (rot @ ext_rot_inv)).dot(v_0) + ext_rot.dot(tsl)  # (3,)
        tsl_wrt_cam = tsl_wrt_cam[:, np.newaxis]  # (3, 1)

        obj_transf = np.concatenate([rot_wrt_cam, tsl_wrt_cam], axis=1)  # (3, 4)
        obj_transf = np.concatenate([obj_transf, np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)
        return obj_transf

    def get_gt_pose(self, index):
        meta_path = self.meta_paths[index]
        meta = np.load(meta_path, allow_pickle=True)
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt


class WDHO3DSeq(HO3DSeq):
    # 工作目录下的 HO3D 数据集抽象类
    def __init__(self, data_dir, model_dir="/home/red0orange/Data/HO3D_v3/HO3D_v3/YCB_models_supp"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.rgb_image_dir = os.path.join(data_dir, "rgb")
        self.depth_image_dir = os.path.join(data_dir, "depth")
        self.mask_image_dir = os.path.join(data_dir, "hand_mask")
        self.obj_mask_image_dir = os.path.join(data_dir, "object_mask")
        self.meta_dir = os.path.join(data_dir, "meta")
        
        self.image_names = sorted(os.listdir(self.rgb_image_dir))
        self.image_paths = [os.path.join(self.rgb_image_dir, image_name) for image_name in self.image_names]
        self.depth_image_paths = [os.path.join(self.depth_image_dir, image_name.replace("jpg", "npy")) for image_name in self.image_names]
        self.mask_image_paths = [os.path.join(self.mask_image_dir, image_name.replace("jpg", "png")) for image_name in self.image_names]
        self.obj_mask_image_paths = [os.path.join(self.obj_mask_image_dir, image_name.replace("jpg", "png")) for image_name in self.image_names]
        self.meta_paths = [os.path.join(self.meta_dir, image_name.replace("jpg", "npy")) for image_name in self.image_names]

        self.mask_flag = True
        for mask_path in self.mask_image_paths:
            if not os.path.exists(mask_path):
                self.mask_flag = False
                break

        # 提取 K
        self.K = None
        for meta_path in self.meta_paths:
            meta = np.load(meta_path, allow_pickle=True).item()
            if meta["camMat"] is not None:
                self.K = meta["camMat"]
                break
        assert self.K is not None

        # gt 相关
        mano_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets/mano")
        self.mano_layer = ManoLayer(
            joint_rot_mode="axisang", use_pca=False, mano_root=mano_dir, center_idx=None, flat_hand_mean=True,
        )
        self.obj_meshes = load_objects(self.model_dir)
        self.cam_extr = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        pass

    def read_ho3d_depth_image(self, depth_path):
        return np.load(depth_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_name, image_path, depth_image_path, meta_path = self.image_names[index], self.image_paths[index], self.depth_image_paths[index], self.meta_paths[index]
        rgb = cv2.imread(image_path)
        depth = np.load(depth_image_path)
        meta = np.load(meta_path, allow_pickle=True).item()
        
        mask_path = self.mask_image_paths[index]
        obj_mask_path = self.obj_mask_image_paths[index]

        item = {
            "rgb": rgb,
            "depth": depth,
            "meta": meta,
            "image_name": image_name,
            "mask_path": mask_path,
            "obj_mask_path": obj_mask_path
        }

        if self.mask_flag:
            hand_mask = cv2.imread(mask_path, 0)
            bbox = masks_to_boxes(hand_mask[None, ...])[0]
            obj_mask = cv2.imread(obj_mask_path, 0)
            obj_bbox = masks_to_boxes(obj_mask[None, ...])[0]

            item["hand_mask"] = hand_mask
            item["obj_mask"] = obj_mask
            item["hand_bbox"] = bbox
            item["obj_bbox"] = obj_bbox
        
        return item

    def get_gt(self, index):
        meta_path = self.meta_paths[index]
        meta = np.load(meta_path, allow_pickle=True).item()

        # 获得 objTrans, objRot
        objTrans, objRot = meta["objTrans"], meta["objRot"]
        if objTrans is None:
            return None

        objTrans = self.preprocess_obj_transf(objTrans, objRot, objName=meta["objName"])
        obj_rot = objTrans[:3, :3]  # (3, 3)
        obj_tsl = objTrans[:3, 3:]  # (3, 1)

        obj_faces = self.obj_meshes[meta["objName"]]["faces"]
        obj_verts_can, _, _ = self.get_obj_verts_can(meta["objName"])
        obj_verts_pred = (obj_rot.dot(obj_verts_can.transpose()) + obj_tsl).transpose()

        # 获得手的
        hand_joints3d = meta["handJoints3D"].astype(np.float32)  # (21, 3)
        hand_joints3d = hand_joints3d * [1, -1, -1]
        hand_shape = meta["handBeta"].astype(np.float32)  # (10,)
        ori_hand_pose = meta["handPose"].astype(np.float32)  # (48,)
        ori_hand_tsl = meta["handTrans"].astype(np.float32)  # (3,)
        ori_hand_verts = self.get_hand_verts3d(hand_shape, ori_hand_pose)  # (778, 3)
        ori_hand_verts = ori_hand_verts + ori_hand_tsl  # (778, 3)
        ori_hand_verts = self.cam_extr[:3, :3].dot(ori_hand_verts.transpose()).transpose()  # (778, 3)
        ori_hand_verts = ori_hand_verts.astype(np.float32)

        hand_pose = self.preprocess_hand_pose(ori_hand_pose)  # (48,)
        process_hand_verts = self.get_hand_verts3d(hand_shape, hand_pose)  # (778, 3)
        hand_tsl = (ori_hand_verts - process_hand_verts)[0]  # (778, 3)

        torch_hand_pose = torch.from_numpy(hand_pose).unsqueeze(0)
        torch_hand_shape = torch.from_numpy(hand_shape).unsqueeze(0)
        hand_verts, _ = self.mano_layer(torch_hand_pose, torch_hand_shape)
        hand_verts /= 1000.0
        hand_faces = self.mano_layer.th_faces.numpy()
        hand_verts = np.array(hand_verts.squeeze(0))
        hand_verts = hand_verts + hand_tsl

        data_dict = {
            "obj_verts_cam": obj_verts_pred,
            "obj_faces": obj_faces,

            "hand_joints3d": hand_joints3d,
            "hand_verts": hand_verts,
            "hand_shape": hand_shape,
            "hand_pose": hand_pose,
            "hand_tsl": hand_tsl,

            "ori_hand_pose": ori_hand_pose,
            "ori_hand_tsl": ori_hand_tsl,
        }
        return data_dict
    
    def get_gt_pose(self, index):
        meta_path = self.meta_paths[index]
        meta = np.load(meta_path, allow_pickle=True).item()
        ob_in_cam_gt = np.eye(4)
        if meta["objTrans"] is None:
            return None
        else:
            ob_in_cam_gt[:3, 3] = meta["objTrans"]
            ob_in_cam_gt[:3, :3] = cv2.Rodrigues(meta["objRot"].reshape(3))[0]
            ob_in_cam_gt = glcam_in_cvcam @ ob_in_cam_gt
        return ob_in_cam_gt

    def get_gt_mesh(self, video_name):
        video2name = {
            "AP": "019_pitcher_base",
            "MPM": "010_potted_meat_can",
            "SB": "021_bleach_cleanser",
            "SM": "006_mustard_bottle",
            "ABF": "021_bleach_cleanser",
            "BB": "011_banana",
            "GPMF": "010_potted_meat_can"
        }
        for k in video2name:
            if video_name.startswith(k):
                ob_name = video2name[k]
                break
        mesh = trimesh.load(f"{self.model_dir}/{ob_name}/textured_simple_ds.obj")
        return mesh

    def per_eval(self, all_pred_joints3d, all_pred_verts):
        """
        对每一帧进行评估，帮助找出错误的帧

        Args:
            all_pred_joints3d (_type_): _description_
            all_pred_verts (_type_): _description_
        """
        size = all_pred_joints3d.shape[0]
        print("size: {}".format(size))
        # get gt
        all_gt_hand_verts = []
        all_gt_hand_joints3d = []
        for i in range(size):
            gt_data_dict = self.get_gt(i)

            if gt_data_dict is None:
                all_gt_hand_verts.append(None)
                all_gt_hand_joints3d.append(None)
                continue

            # @note 暂时不考虑手的姿态
            obj_verts = gt_data_dict["obj_verts_cam"]
            obj_faces = gt_data_dict["obj_faces"]

            # hand
            hand_joints3d = gt_data_dict["hand_joints3d"]
            hand_verts = gt_data_dict["hand_verts"]

            all_gt_hand_verts.append(hand_verts)
            all_gt_hand_joints3d.append(hand_joints3d)
        
        # eval
        eval_xyz, eval_xyz_procrustes_aligned, eval_xyz_sc_tr_aligned = EvalUtil(), EvalUtil(), EvalUtil()
        eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
        f_score, f_score_aligned = list(), list()
        f_threshs = [0.005, 0.015]
        empty_cnt = 0
        for idx in range(size):
            gt_verts = all_gt_hand_verts[idx]
            pred_verts = all_pred_verts[idx]
            gt_joints3d = all_gt_hand_joints3d[idx]
            pred_joints3d = all_pred_joints3d[idx]

            pred_joints3d = mano_to_ho3d_joints3d(pred_joints3d)

            if gt_verts is None or pred_verts is None:
                empty_cnt += 1
                print("{}: gt is None, continue".format(empty_cnt))
                continue

            xyz, verts = gt_joints3d, gt_verts
            xyz, verts = [np.array(x) for x in [xyz, verts]]

            xyz_pred, verts_pred = pred_joints3d, pred_verts
            xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

            # Not aligned errors
            eval_xyz.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred,
                idx=idx
            )

            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred,
                idx=idx
            )

            # scale and translation aligned predictions for xyz
            xyz_pred_sc_tr_aligned = align_sc_tr(xyz, xyz_pred)
            eval_xyz_sc_tr_aligned.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred_sc_tr_aligned,
                idx=idx
            )

            # align predictions
            xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
            verts_pred_aligned = align_w_scale(verts, verts_pred)

            # Aligned errors
            eval_xyz_procrustes_aligned.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred_aligned,
                idx=idx
            )

            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned,
                idx=idx
            )

            # F-scores
            l, la = list(), list()
            for t in f_threshs:
                # for each threshold calculate the f score and the f score of the aligned vertices
                f, _, _ = calculate_fscore(verts, verts_pred, t)
                l.append(f)
                f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
                la.append(f)
            f_score.append(l)
            f_score_aligned.append(la)

        xyz_error = np.array(eval_xyz.data).T        
        align_xyz_error = np.array(eval_xyz_procrustes_aligned.data).T
        vert_error = np.array(eval_mesh_err.data).T
        align_vert_error = np.array(eval_mesh_err_aligned.data).T

        xyz_error = np.mean(xyz_error, axis=1)
        align_xyz_error = np.mean(align_xyz_error, axis=1)
        vert_error = np.mean(vert_error, axis=1)
        align_vert_error = np.mean(align_vert_error, axis=1)

        ids = eval_xyz.ids

        return ids, xyz_error, align_xyz_error, vert_error, align_vert_error

    def eval(self, all_pred_joints3d, all_pred_verts):
        size = all_pred_joints3d.shape[0]
        print("size: {}".format(size))
        # get gt
        all_gt_hand_verts = []
        all_gt_hand_joints3d = []
        for i in range(size):
            gt_data_dict = self.get_gt(i)

            if gt_data_dict is None:
                all_gt_hand_verts.append(None)
                all_gt_hand_joints3d.append(None)
                continue

            # @note 暂时不考虑手的姿态
            obj_verts = gt_data_dict["obj_verts_cam"]
            obj_faces = gt_data_dict["obj_faces"]

            # hand
            hand_joints3d = gt_data_dict["hand_joints3d"]
            hand_verts = gt_data_dict["hand_verts"]

            all_gt_hand_verts.append(hand_verts)
            all_gt_hand_joints3d.append(hand_joints3d)
        
        # eval
        eval_xyz, eval_xyz_procrustes_aligned, eval_xyz_sc_tr_aligned = EvalUtil(), EvalUtil(), EvalUtil()
        eval_mesh_err, eval_mesh_err_aligned = EvalUtil(num_kp=778), EvalUtil(num_kp=778)
        f_score, f_score_aligned = list(), list()
        f_threshs = [0.005, 0.015]
        empty_cnt = 0
        for idx in range(size):
            gt_verts = all_gt_hand_verts[idx]
            pred_verts = all_pred_verts[idx]
            gt_joints3d = all_gt_hand_joints3d[idx]
            pred_joints3d = all_pred_joints3d[idx]

            pred_joints3d = mano_to_ho3d_joints3d(pred_joints3d)

            if gt_verts is None or pred_verts is None:
                empty_cnt += 1
                print("{}: gt is None, continue".format(empty_cnt))
                continue

            xyz, verts = gt_joints3d, gt_verts
            xyz, verts = [np.array(x) for x in [xyz, verts]]

            xyz_pred, verts_pred = pred_joints3d, pred_verts
            xyz_pred, verts_pred = [np.array(x) for x in [xyz_pred, verts_pred]]

            # Not aligned errors
            eval_xyz.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred,
                idx=idx
            )

            eval_mesh_err.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred,
                idx=idx
            )

            # scale and translation aligned predictions for xyz
            xyz_pred_sc_tr_aligned = align_sc_tr(xyz, xyz_pred)
            eval_xyz_sc_tr_aligned.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred_sc_tr_aligned,
                idx=idx
            )

            # align predictions
            xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
            verts_pred_aligned = align_w_scale(verts, verts_pred)

            # Aligned errors
            eval_xyz_procrustes_aligned.feed(
                xyz,
                np.ones_like(xyz[:, 0]),
                xyz_pred_aligned,
                idx=idx
            )

            eval_mesh_err_aligned.feed(
                verts,
                np.ones_like(verts[:, 0]),
                verts_pred_aligned,
                idx=idx
            )

            # F-scores
            l, la = list(), list()
            for t in f_threshs:
                # for each threshold calculate the f score and the f score of the aligned vertices
                f, _, _ = calculate_fscore(verts, verts_pred, t)
                l.append(f)
                f, _, _ = calculate_fscore(verts, verts_pred_aligned, t)
                la.append(f)
            f_score.append(l)
            f_score_aligned.append(la)

        # Calculate results
        xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

        xyz_procrustes_al_mean3d, _, xyz_procrustes_al_auc3d, pck_xyz_procrustes_al, thresh_xyz_procrustes_al = eval_xyz_procrustes_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP PROCRUSTES ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm' % (xyz_procrustes_al_auc3d, xyz_procrustes_al_mean3d * 100.0))

        xyz_sc_tr_al_mean3d, _, xyz_sc_tr_al_auc3d, pck_xyz_sc_tr_al, thresh_xyz_sc_tr_al = eval_xyz_sc_tr_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D KP SCALE-TRANSLATION ALIGNED results:')
        print('auc=%.3f, mean_kp3d_avg=%.2f cm\n' % (xyz_sc_tr_al_auc3d, xyz_sc_tr_al_mean3d * 100.0))

        mesh_mean3d, _, mesh_auc3d, pck_mesh, thresh_mesh = eval_mesh_err.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH results:')
        print('auc=%.3f, mean_vert3d_avg=%.2f cm' % (mesh_auc3d, mesh_mean3d * 100.0))

        mesh_al_mean3d, _, mesh_al_auc3d, pck_mesh_al, thresh_mesh_al = eval_mesh_err_aligned.get_measures(0.0, 0.05, 100)
        print('Evaluation 3D MESH ALIGNED results:')
        print('auc=%.3f, mean_vert3d_avg=%.2f cm\n' % (mesh_al_auc3d, mesh_al_mean3d * 100.0))

        print('F-scores')
        f_out = list()
        f_score, f_score_aligned = np.array(f_score).T, np.array(f_score_aligned).T
        for f, fa, t in zip(f_score, f_score_aligned, f_threshs):
            print('F@%.1fmm = %.3f' % (t*1000, f.mean()), '\tF_aligned@%.1fmm = %.3f' % (t*1000, fa.mean()))
            f_out.append('f_score_%d: %f' % (round(t*1000), f.mean()))
            f_out.append('f_al_score_%d: %f' % (round(t*1000), fa.mean()))
        pass


if __name__ == "__main__":
    import open3d as o3d

    ho3d_dataset = HO3DDataset("/home/red0orange/Data/HO3D_v3/HO3D_v3/train/ABF10")

    mano_layer = ManoLayer(
        joint_rot_mode="axisang", use_pca=False, mano_root="assets/mano", center_idx=None, flat_hand_mean=True,
    )
    gt_data_dict = ho3d_dataset.get_gt(118)

    obj_verts = gt_data_dict["obj_verts_cam"]
    obj_faces = gt_data_dict["obj_faces"]
    hand_pose = gt_data_dict["hand_pose"]
    hand_shape = gt_data_dict["hand_shape"]
    hand_tsl = gt_data_dict["hand_tsl"]
    
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj_mesh.compute_vertex_normals()

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_pose = torch.from_numpy(hand_pose).unsqueeze(0)
    hand_shape = torch.from_numpy(hand_shape).unsqueeze(0)
    hand_verts, _ = mano_layer(hand_pose, hand_shape)
    hand_verts /= 1000.0
    hand_faces = mano_layer.th_faces.numpy()
    hand_verts = np.array(hand_verts.squeeze(0))
    hand_verts = hand_verts + hand_tsl
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.compute_vertex_normals()

    o3d.visualization.draw_geometries([obj_mesh, hand_mesh])
    pass