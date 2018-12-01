#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
import numpy as np
from math import sqrt
from utils.io import _load

d = 'test.configs'
yaw_list = _load(osp.join(d, 'AFLW_GT_crop_yaws.npy'))
roi_boxs = _load(osp.join(d, 'AFLW_GT_crop_roi_box.npy'))
pts68_all = _load(osp.join(d, 'AFLW_GT_pts68.npy'))
pts21_all = _load(osp.join(d, 'AFLW_GT_pts21.npy'))


def ana(nme_list):
    yaw_list_abs = np.abs(yaw_list)
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


def calc_nme(pts68_fit_all):
    std_size = 120
    ind_68to21 = [[18], [20], [22], [23], [25], [27], [37], [37, 38, 39, 40, 41, 42], [40], [43],
                  [43, 44, 45, 46, 47, 48],
                  [46], [3], [32], [31], [36], [15], [49], [61, 62, 63, 64, 65, 66, 67, 68], [55], [9]]
    for i in range(len(ind_68to21)):
        for j in range(len(ind_68to21[i])):
            ind_68to21[i][j] -= 1

    nme_list = []

    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]
        pts21_gt = pts21_all[i]

        # reconstruct 68 pts
        sx, sy, ex, ey = roi_boxs[i]
        scale_x = (ex - sx) / std_size
        scale_y = (ey - sy) / std_size
        pts68_fit[0, :] = pts68_fit[0, :] * scale_x + sx
        pts68_fit[1, :] = pts68_fit[1, :] * scale_y + sy

        # pts68 -> pts21
        pts21_est = np.zeros_like(pts21_gt, dtype=np.float32)
        for i in range(21):
            ind = ind_68to21[i]
            tmp = np.mean(pts68_fit[:, ind], 1)
            pts21_est[:, i] = tmp

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))

        # nme
        pt_valid = (pts21_gt[0, :] != -1) & (pts21_gt[1, :] != -1)
        dis = pts21_est[:, pt_valid] - pts21_gt[:, pt_valid]
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
