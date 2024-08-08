# From HandOccNet

import os
import os.path as osp
import numpy as np
import torch
import torchvision
import cv2
import random
import json
import math
import copy
from pycocotools.coco import COCO

from my_utils.hand_recon.mano import MANO
from my_utils.new_crop_image import crop_square_image, K_trans
mano = MANO()


def load_img(path, order="BGR"):
    img = cv2.imread(path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(img, np.ndarray):
        raise IOError("Fail to read %s" % path)

    if order == "RGB":
        img = img[:, :, ::-1].copy()

    img = img.astype(np.float32)
    return img

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

def cam2pixel(cam_coord, f, c):
    x = cam_coord[:,0] / cam_coord[:,2] * f[0] + c[0]
    y = cam_coord[:,1] / cam_coord[:,2] * f[1] + c[1]
    z = cam_coord[:,2]
    return np.stack((x,y,z),1)

def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:,0] - c[0]) / f[0] * pixel_coord[:,2]
    y = (pixel_coord[:,1] - c[1]) / f[1] * pixel_coord[:,2]
    z = pixel_coord[:,2]
    return np.stack((x,y,z),1)

def world2cam(world_coord, R, t):
    cam_coord = np.dot(R, world_coord.transpose(1,0)).transpose(1,0) + t.reshape(1,3)
    return cam_coord

def cam2world(cam_coord, R, t):
    world_coord = np.dot(np.linalg.inv(R), (cam_coord - t.reshape(1,3)).transpose(1,0)).transpose(1,0)
    return world_coord

def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'


class HO3D(torch.utils.data.Dataset):
    def __init__(self, root_dir, data_split, input_img_size, bbox_scale_ratio=1.5, img_channel_order="BGR"):
        self.img_channel_order = img_channel_order
        self.input_img_size = input_img_size
        self.bbox_scale_ratio = bbox_scale_ratio
        self.transform = torchvision.transforms.ToTensor()
        self.data_split = data_split if data_split == 'train' else 'evaluation'
        self.root_dir = root_dir
        self.annot_path = osp.join(self.root_dir, 'annotations')
        self.root_joint_idx = 0

        self.datalist = self.my_load_data()

        self.joints_name = ('Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3', 'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4', 'Pinly_4')

    def my_load_data(self):
        file_name_list_path = osp.join(self.root_dir, "{}.txt".format(self.data_split))
        with open(file_name_list_path) as f:
            file_name_list = f.readlines()
        file_name_list = [f.strip() for f in file_name_list]
        assert len(file_name_list) == db_size(self.data_split, version='v3'), "file_list size is not correct"

        datalist = []
        for file_name in file_name_list:
            seq_name = file_name.split('/')[0]
            file_id = file_name.split('/')[1]

            img_path = osp.join(self.root_dir, self.data_split, seq_name, "rgb", file_id + ".jpg")
            meta_path = osp.join(self.root_dir, self.data_split, seq_name, "meta", file_id + ".pkl")

            meta = np.load(meta_path, allow_pickle=True)
            # img = cv2.imread(img_path)
            img_shape = (480, 640)

            if self.data_split == 'train':
                # joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32) # meter
                # cam_param = {k:np.array(v, dtype=np.float32) for k,v in ann['cam_param'].items()}
                # joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
                # bbox = get_bbox(joints_coord_img[:,:2], np.ones_like(joints_coord_img[:,0]), expansion_factor=1.5)
                # bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
                # if bbox is None:
                #     continue

                # mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
                # mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)

                # data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam, "joints_coord_img": joints_coord_img,
                #         "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape}
                pass
            else:
                root_joint_cam = np.array([meta['handJoints3D'][0], -meta['handJoints3D'][1], -meta['handJoints3D'][2]], dtype=np.float32)
                cam_param = {"focal": np.array([meta['camMat'][0, 0], meta['camMat'][1, 1]], dtype=np.float32), 
                             "princpt": np.array([meta['camMat'][0, 2], meta['camMat'][1, 2]], dtype=np.float32)}
                bbox = np.array(meta['handBoundingBox'], dtype=np.float32)
                
                data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
                        "bbox": bbox, "cam_param": cam_param}

            datalist.append(data)

        return datalist
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path, order=self.img_channel_order)

        img, img2bb_trans, bb2img_trans = crop_square_image(img, bbox, self.input_img_size, bbox_expansion_factor=self.bbox_scale_ratio)
        scale, rot = 1.0, 0.0
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            # ## 2D joint coordinate
            # joints_img = data['joints_coord_img']
            # joints_img_xy1 = np.concatenate((joints_img[:,:2], np.ones_like(joints_img[:,:1])),1)
            # joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1,0)).transpose(1,0)[:,:2]
            # # normalize to [0,1]
            # joints_img[:,0] /= cfg.input_img_shape[1]
            # joints_img[:,1] /= cfg.input_img_shape[0]

            # ## 3D joint camera coordinate
            # joints_coord_cam = data['joints_coord_cam']
            # root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            # joints_coord_cam -= joints_coord_cam[self.root_joint_idx,None,:] # root-relative
            # # 3D data rotation augmentation
            # rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0], 
            # [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            # [0, 0, 1]], dtype=np.float32)
            # joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1,0)).transpose(1,0)
            
            # ## mano parameter
            # mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            # # 3D data rotation augmentation
            # mano_pose = mano_pose.reshape(-1,3)
            # root_pose = mano_pose[self.root_joint_idx,:]
            # root_pose, _ = cv2.Rodrigues(root_pose)
            # root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat,root_pose))
            # mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            # mano_pose = mano_pose.reshape(-1)

            # inputs = {'img': img}
            # targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose, 'mano_shape': mano_shape}
            # meta_info = {'root_joint_cam': root_joint_cam}
            pass

        else:
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam, 'img_path': img_path}

        return inputs, targets, meta_info
                  
    def export_eval_pred(self, save_path, all_3d_joints, all_3d_verts):
        """
        结果要按照 datalist 的顺序排列
        """
        eval_result = [[],[]]

        for i, (joints_out, verts_out) in enumerate(zip(all_3d_joints, all_3d_verts)):
            annot = self.datalist[i]

            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
                
            # convert to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            # @note 转换 joint
            joints_out = transform_joint_to_other_db(joints_out, mano.joints_name, self.joints_name)

            joints_out = np.asarray(joints_out, dtype=np.float32)
            verts_out = np.asarray(verts_out, dtype=np.float32)

            eval_result[0].append(joints_out.tolist())
            eval_result[1].append(verts_out.tolist())

        # write file
        output_json_file = osp.join(save_path) 
        output_zip_file = osp.join(save_path.replace('.json', '.zip'))
        
        with open(output_json_file, 'w') as f:
            json.dump(eval_result, f)
        print('Dumped %d joints and %d verts predictions to %s' % (len(eval_result[0]), len(eval_result[1]), output_json_file))

        cmd = 'zip -j ' + output_zip_file + ' ' + output_json_file
        print(cmd)
        os.system(cmd)
        pass