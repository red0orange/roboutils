import cv2
import numpy as np


def get_trans(K, x, y, width, height):
    box = [x, y, x+width, y+height]
    K_orig = K
    resize_shape = [height, width]

    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]]) # w, h
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    return trans_crop_homo


def adjust_K_for_crop(K, x, y, width, height):
    K_cropped, _ = get_K_crop_resize([x, y, x+width, y+height], K, [height, width])
    return K_cropped


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)
    
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    
    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_K_crop_resize(box, K_orig, resize_shape):
    """Update K (crop an image according to the box, and resize the cropped image to resize_shape) 
    @param box: [x0, y0, x1, y1]
    @param K_orig: [3, 3] or [3, 4]
    @resize_shape: [h, w]
    """
    center = np.array([(box[0] + box[2]) / 2., (box[1] + box[3]) / 2.])
    scale = np.array([box[2] - box[0], box[3] - box[1]]) # w, h
    
    resize_h, resize_w = resize_shape
    trans_crop = get_affine_transform(center, scale, 0, [resize_w, resize_h])
    trans_crop_homo = np.concatenate([trans_crop, np.array([[0, 0, 1]])], axis=0)

    if K_orig.shape == (3, 3):
        K_orig_homo = np.concatenate([K_orig, np.zeros((3, 1))], axis=-1)
    else:
        K_orig_homo = K_orig.copy()
    assert K_orig_homo.shape == (3, 4)

    K_crop_homo = trans_crop_homo @ K_orig_homo # [3, 4]
    K_crop = K_crop_homo[:3, :3]
    
    return K_crop, K_crop_homo

    
def adjust_to_square_bbox(ori_width, ori_height, bbox, square_size=None):
    bbox_width, bbox_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if square_size is None:
        square_size = max([bbox_width, bbox_height])
    max_square_size = min([ori_width, ori_height])
    square_size = min(max_square_size, square_size)
    half_square_size = square_size // 2
    center_x, center_y = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    
    x0, x1, y0, y1 = center_x - half_square_size, center_x + half_square_size, center_y - half_square_size, center_y + half_square_size
    if x0 < 0: 
        x1 += (-x0)
        x0 = 0
    elif x1 > (ori_width-1):
        x0 -= (x1 - ori_width + 1)
        x1 = ori_width-1
    if y0 < 0:
        y1 += (-y0)
        y0 = 0
    elif y1 > (ori_height-1):
        y0 -= (y1 - ori_height + 1)
        y1 = ori_height-1
    
    return map(int, [x0, y0, x1, y1])


def crop_image(img, K, bbox):
    ori_shape = img.shape[:2]
    x0, y0, x1, y1 = bbox
    assert (x0 >= 0) and (y0 >= 0) and (x1 <= ori_shape[1]) and (y1 <= ori_shape[0])

    crop_img = img[y0:y1, x0:y1]
    height, width = (y1 - y0), (x1 - x0)
    crop_K = adjust_K_for_crop(K, x0, y0, width, height)
    return crop_img, crop_K


def crop_roi_with_padding(image, center_x, center_y, roi_width, roi_height):
    # 获取图像的高度和宽度
    image_height, image_width = image.shape[:2]

    # 计算ROI的左上角和右下角坐标
    roi_x1 = max(center_x - roi_width // 2, 0)
    roi_y1 = max(center_y - roi_height // 2, 0)
    roi_x2 = min(center_x + roi_width // 2, image_width)
    roi_y2 = min(center_y + roi_height // 2, image_height)

    # 计算ROI的实际宽度和高度
    roi_actual_width = roi_x2 - roi_x1
    roi_actual_height = roi_y2 - roi_y1

    # 创建一个黑色背景的图像
    roi = np.full((roi_height, roi_width, image.shape[2]), dtype=np.uint8, fill_value=255)

    # 计算ROI在新图像中的位置
    roi_start_x = max(0, roi_width // 2 - roi_actual_width // 2)
    roi_end_x = min(roi_width // 2 + roi_actual_width // 2, roi_width)
    roi_start_y = max(0, roi_height // 2 - roi_actual_height // 2)
    roi_end_y = min(roi_height // 2 + roi_actual_height // 2, roi_height)

    if (roi_end_y - roi_start_y) != (roi_y2 - roi_y1):
        diff = (roi_end_y - roi_start_y) - (roi_y2 - roi_y1)
        roi_y2 += diff
    if (roi_end_x - roi_start_x) != (roi_x2 - roi_x1):
        diff = (roi_end_x - roi_start_x) - (roi_x2 - roi_x1)
        roi_x2 += diff

    assert (roi_end_y - roi_start_y) == (roi_y2 - roi_y1)
    assert (roi_end_x - roi_start_x) == (roi_x2 - roi_x1)

    roi_start_x, roi_start_y, roi_end_x, roi_end_y = map(int, [roi_start_x, roi_start_y, roi_end_x, roi_end_y])
    roi_x1, roi_y1, roi_x2, roi_y2 = map(int, [roi_x1, roi_y1, roi_x2, roi_y2])
    # 复制ROI到新图像中
    roi[roi_start_y:roi_end_y, roi_start_x:roi_end_x] = image[roi_y1:roi_y2, roi_x1:roi_x2]

    return roi, [roi_x1, roi_y1, roi_x2, roi_y2]
