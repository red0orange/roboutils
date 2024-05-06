import os
import cv2
import numpy as np


def vis_image_matching(image1, image2, kp1, kp2, matches, scores=None, max_vis_num=100):
    kp1_scales = np.full(len(kp1), fill_value=1)
    kp2_scales = np.full(len(kp1), fill_value=1)
    if scores is not None:
        for match, score in zip(matches, scores):
            kp1_scales[match[0]] = score
            kp2_scales[match[1]] = score

    kp1 = [cv2.KeyPoint(x=float(kp1[i][0]), y=float(kp1[i][1]), size=1) for i in range(len(kp1))]
    kp2 = [cv2.KeyPoint(x=float(kp2[i][0]), y=float(kp2[i][1]), size=1) for i in range(len(kp2))]
    matches = [
        cv2.DMatch(_imgIdx=0, _queryIdx=i[0], _trainIdx=i[1], _distance=0)
        for i in matches
    ]
    
    vis_num = min(max_vis_num, len(matches))
    sample_indexes = np.random.choice(len(matches), vis_num, replace=False)
    # kp1 = [kp1[i] for i in sample_indexes]
    # kp2 = [kp2[i] for i in sample_indexes]
    matches = [matches[i] for i in sample_indexes]

    background_color = [255, 255, 255]  # Blue color in this case
    height = max(image1.shape[0], image2.shape[0])
    width = image1.shape[1] + image2.shape[1]
    outImg = np.zeros((height, width, 3), dtype=np.uint8)
    outImg[:, :] = background_color

    result_img = cv2.drawMatches(
        image1,
        kp1,
        image2,
        kp2,
        matches,
        # matchesThickness=1,
        matchColor = (0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        outImg=outImg
    )
    return outImg