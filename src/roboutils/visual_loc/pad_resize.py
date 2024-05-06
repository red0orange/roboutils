import cv2
import numpy as np


def pad_resize_image(image, width, height):
    # 获取原始图像的尺寸
    original_height, original_width = image.shape[:2]

    # 计算缩放比例
    scale = min(width / original_width, height / original_height)

    # 计算缩放后的新尺寸
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建新的画布，并将缩放后的图像放置在其中心
    if len(resized_image.shape) == 3:
        padded_image = np.zeros((height, width, resized_image.shape[2]), dtype=np.uint8)
    else:
        padded_image = np.zeros((height, width), dtype=np.uint8)
    start_x = (width - new_width) // 2
    start_y = (height - new_height) // 2
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    # 计算从目标坐标系到原始坐标系的变换矩阵
    scale_matrix = np.eye(3)
    scale_matrix[0, 0] = scale
    scale_matrix[1, 1] = scale
    translation_matrix = np.eye(3)
    translation_matrix[0, 2] = start_x
    translation_matrix[1, 2] = start_y
    transformation_matrix = translation_matrix.dot(scale_matrix)

    return padded_image, transformation_matrix