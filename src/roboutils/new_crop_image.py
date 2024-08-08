# modify from https://github.com/namepllet/HandOccNet
import math
import random
import cv2
import numpy as np


def process_bbox(bbox, img_width, img_height, expansion_factor=1.25):
    # sanitize bboxes
    x, y, w, h = bbox
    x1 = np.max((0, x))
    y1 = np.max((0, y))
    x2 = np.min((img_width - 1, x1 + np.max((0, w - 1))))
    y2 = np.min((img_height - 1, y1 + np.max((0, h - 1))))
    if w * h > 0 and x2 >= x1 and y2 >= y1:
        bbox = np.array([x1, y1, x2 - x1, y2 - y1])
    else:
        return None

    # aspect ratio preserving bbox
    w = bbox[2]
    h = bbox[3]
    c_x = bbox[0] + w / 2.0
    c_y = bbox[1] + h / 2.0
    aspect_ratio = img_width / img_height
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    bbox[2] = w * expansion_factor
    bbox[3] = h * expansion_factor
    bbox[0] = c_x - bbox[2] / 2.0
    bbox[1] = c_y - bbox[3] / 2.0

    return bbox


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(
    c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False
):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def crop_square_image(cvimg, bbox, square_size, bbox_expansion_factor=1.25):
    out_shape = [square_size, square_size]
    scale = 1.0
    rot = 0.0

    # image
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    # bbox  ->  square bbox
    bbox = process_bbox(bbox, img_width, img_height, bbox_expansion_factor)

    bb_c_x = float(bbox[0] + 0.5 * bbox[2])
    bb_c_y = float(bbox[1] + 0.5 * bbox[3])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])

    trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale=scale, rot=rot
    )
    img_patch = cv2.warpAffine(
        img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, 
    borderValue=(255, 255, 255)
    )

    img_patch = img_patch.astype(np.float32)
    inv_trans = gen_trans_from_patch_cv(
        bb_c_x, bb_c_y, bb_width, bb_height, out_shape[1], out_shape[0], scale, rot, inv=True,
    )

    return img_patch.astype(np.uint8), trans, inv_trans


def K_trans(K, trans):
    assert K.shape == (3, 3)
    assert trans.shape == (2, 3)

    trans_homo = np.concatenate([trans, np.array([[0, 0, 1]])], axis=0)
    K_homo = np.concatenate([K, np.zeros((3, 1))], axis=-1)

    K_crop_homo = trans_homo @ K_homo # [3, 4]
    K_crop = K_crop_homo[:3, :3]

    return K_crop


def resize_image(cvimg, new_img_width, new_img_height):
    # image
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    src = np.array([
        0, 0,
        0, img_height,
        img_width, 0
    ]).reshape([3, 2]).astype(np.float32)
    dst = np.array([
        0, 0,
        0, new_img_height,
        new_img_width, 0
    ]).reshape([3, 2]).astype(np.float32)

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    img_patch = cv2.warpAffine(
        img, trans, (int(new_img_width), int(new_img_height)), flags=cv2.INTER_LINEAR 
    )
    
    return img_patch, trans, inv_trans


def crop_scale_square_roi(image, center, square_size, out_square_size):
    center_x, center_y = center
    scale = out_square_size / square_size

    image_height, image_width = image.shape[:2]
    roi_x1 = max(center_x - square_size // 2, 0)
    roi_y1 = max(center_y - square_size // 2, 0)
    roi_x2 = min(center_x + square_size // 2, image_width)
    roi_y2 = min(center_y + square_size // 2, image_height)
    roi_actual_width = roi_x2 - roi_x1
    roi_actual_height = roi_y2 - roi_y1
    roi_start_x = max(0, square_size // 2 - roi_actual_width // 2)
    roi_end_x = min(square_size // 2 + roi_actual_width // 2, square_size)
    roi_start_y = max(0, square_size // 2 - roi_actual_height // 2)
    roi_end_y = min(square_size // 2 + roi_actual_height // 2, square_size)

    # if (roi_end_y - roi_start_y) != (roi_y2 - roi_y1):
    #     diff = (roi_end_y - roi_start_y) - (roi_y2 - roi_y1)
    #     roi_y2 += diff
    # if (roi_end_x - roi_start_x) != (roi_x2 - roi_x1):
    #     diff = (roi_end_x - roi_start_x) - (roi_x2 - roi_x1)
    #     roi_x2 += diff
    # assert (roi_end_y - roi_start_y) == (roi_y2 - roi_y1)
    # assert (roi_end_x - roi_start_x) == (roi_x2 - roi_x1)
    # roi_start_x, roi_start_y, roi_end_x, roi_end_y = map(int, [roi_start_x, roi_start_y, roi_end_x, roi_end_y])
    # roi_x1, roi_y1, roi_x2, roi_y2 = map(int, [roi_x1, roi_y1, roi_x2, roi_y2])

    src = np.array([
        roi_x1, roi_y1,
        roi_x2, roi_y1,
        roi_x1, roi_y2
    ]).reshape([3, 2]).astype(np.float32)
    dst = np.array([
        roi_start_x * scale, roi_start_y * scale,
        roi_end_x * scale, roi_start_y * scale,
        roi_start_x * scale, roi_end_y * scale 
    ]).reshape([3, 2]).astype(np.float32)

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    inv_trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    new_image = cv2.warpAffine(
        image, trans, (int(out_square_size), int(out_square_size)), flags=cv2.INTER_LINEAR, borderValue=(255,255,255) 
    )

    trans = np.concatenate([trans, np.array([[0, 0, 1]])], axis=0)
    inv_trans = np.concatenate([inv_trans, np.array([[0, 0, 1]])], axis=0)
    return new_image, trans, inv_trans
