#!/usr/bin/env python3
# coding: utf-8

"""
Notation (2019.09.15): two versions of spliting AFLW2000-3D:
 1) AFLW2000-3D.pose.npy: according to the fitted pose
 2) AFLW2000-3D-new.pose: according to AFLW labels 
There is no obvious difference between these two splits.
"""

import os.path as osp
import numpy as np
from math import sqrt
from utils.io import _load

d = 'test.configs'

# [1312, 383, 305], current version
yaws_list = _load(osp.join(d, 'AFLW2000-3D.pose.npy'))

# [1306, 462, 232], same as paper
# yaws_list = _load(osp.join(d, 'AFLW2000-3D-new.pose.npy'))

# origin
pts68_all_ori = _load(osp.join(d, 'AFLW2000-3D.pts68.npy'))

# reannonated
pts68_all_re = _load(osp.join(d, 'AFLW2000-3D-Reannotated.pts68.npy'))
roi_boxs = _load(osp.join(d, 'AFLW2000-3D_crop.roi_box.npy'))


def ana(nme_list):
    yaw_list_abs = np.abs(yaws_list)
    ind_yaw_1 = yaw_list_abs <= 30
    ind_yaw_2 = np.bitwise_and(yaw_list_abs > 30, yaw_list_abs <= 60)
    ind_yaw_3 = yaw_list_abs > 60

    nme_1 = nme_list[ind_yaw_1]
    nme_2 = nme_list[ind_yaw_2]
    nme_3 = nme_list[ind_yaw_3]

    mean_nme_1 = np.mean(nme_1) * 100
    mean_nme_2 = np.mean(nme_2) * 100
    mean_nme_3 = np.mean(nme_3) * 100
    # mean_nme_all = np.mean(nme_list) * 100

    std_nme_1 = np.std(nme_1) * 100
    std_nme_2 = np.std(nme_2) * 100
    std_nme_3 = np.std(nme_3) * 100
    # std_nme_all = np.std(nme_list) * 100

    mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
    mean = np.mean(mean_all)
    std = np.std(mean_all)

    s1 = '[ 0, 30]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
    s2 = '[30, 60]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
    s3 = '[60, 90]\tMean: \x1b[32m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
    # s4 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: {:.3f}'.format(mean_nme_all, std_nme_all)
    s5 = '[ 0, 90]\tMean: \x1b[31m{:.3f}\x1b[0m, Std: \x1b[31m{:.3f}\x1b[0m'.format(mean, std)

    s = '\n'.join([s1, s2, s3, s5])
    print(s)

    return mean_nme_1, mean_nme_2, mean_nme_3, mean, std


def convert_to_ori(lms, i):
    std_size = 120
    sx, sy, ex, ey = roi_boxs[i]
    scale_x = (ex - sx) / std_size
    scale_y = (ey - sy) / std_size
    lms[0, :] = lms[0, :] * scale_x + sx
    lms[1, :] = lms[1, :] * scale_y + sy
    return lms


def calc_nme(pts68_fit_all, option='ori'):
    if option == 'ori':
        pts68_all = pts68_all_ori
    elif option == 're':
        pts68_all = pts68_all_re
    std_size = 120

    nme_list = []

    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        sx, sy, ex, ey = roi_boxs[i]
        scale_x = (ex - sx) / std_size
        scale_y = (ey - sy) / std_size
        pts68_fit[0, :] = pts68_fit[0, :] * scale_x + sx
        pts68_fit[1, :] = pts68_fit[1, :] * scale_y + sy

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        #
        dis = pts68_fit - pts68_gt[:2, :]
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list


def main():
    pass


if __name__ == '__main__':
    main()
