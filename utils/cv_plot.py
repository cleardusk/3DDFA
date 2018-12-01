#!/usr/bin/env python3
# coding: utf-8


"""
Modified from: https://sourcegraph.com/github.com/YadiraF/PRNet@master/-/blob/utils/cv_plot.py
"""

import numpy as np
import cv2

from utils.inference import calc_hypotenuse

end_list = np.array([17, 22, 27, 42, 48, 31, 36, 68], dtype=np.int32) - 1


def plot_kpt(image, kpt):
    ''' Draw 68 key points
    Args:
        image: the input image
        kpt: (68, 3).
    '''
    image = image.copy()
    kpt = np.round(kpt).astype(np.int32)
    for i in range(kpt.shape[0]):
        st = kpt[i, :2]
        image = cv2.circle(image, (st[0], st[1]), 1, (0, 0, 255), 2)
        if i in end_list:
            continue
        ed = kpt[i + 1, :2]
        image = cv2.line(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 1)
    return image


def build_camera_box(rear_size=90):
    point_3d = []
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = int(4 / 3 * rear_size)
    front_depth = int(4 / 3 * rear_size)
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float).reshape(-1, 3)

    return point_3d


def plot_pose_box(image, Ps, pts68s, color=(40, 255, 0), line_width=2):
    ''' Draw a 3D box as annotation of pose. Ref:https://github.com/yinguobing/head-pose-estimation/blob/master/pose_estimator.py
    Args:
        image: the input image
        P: (3, 4). Affine Camera Matrix.
        kpt: (2, 68) or (3, 68)
    '''
    image = image.copy()
    if not isinstance(pts68s, list):
        pts68s = [pts68s]
    if not isinstance(Ps, list):
        Ps = [Ps]
    for i in range(len(pts68s)):
        pts68 = pts68s[i]
        llength = calc_hypotenuse(pts68)
        point_3d = build_camera_box(llength)
        P = Ps[i]

        # Map to 2d image points
        point_3d_homo = np.hstack((point_3d, np.ones([point_3d.shape[0], 1])))  # n x 4
        point_2d = point_3d_homo.dot(P.T)[:, :2]

        point_2d[:, 1] = - point_2d[:, 1]
        point_2d[:, :2] = point_2d[:, :2] - np.mean(point_2d[:4, :2], 0) + np.mean(pts68[:2, :27], 1)
        point_2d = np.int32(point_2d.reshape(-1, 2))

        # Draw all the lines
        cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[1]), tuple(
            point_2d[6]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[2]), tuple(
            point_2d[7]), color, line_width, cv2.LINE_AA)
        cv2.line(image, tuple(point_2d[3]), tuple(
            point_2d[8]), color, line_width, cv2.LINE_AA)

    return image


def main():
    pass


if __name__ == '__main__':
    main()
