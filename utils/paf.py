#!/usr/bin/env python3
# coding: utf-8

import numpy as np
from .ddfa import _parse_param
from .params import u_filter, w_filter, w_exp_filter, std_size, param_mean, param_std


def reconstruct_paf_anchor(param, whitening=True):
    if whitening:
        param = param * param_std + param_mean
    p, offset, alpha_shp, alpha_exp = _parse_param(param)
    anchor = p @ (u_filter + w_filter @ alpha_shp + w_exp_filter @ alpha_exp).reshape(3, -1, order='F') + offset
    anchor[1, :] = std_size + 1 - anchor[1, :]
    return anchor[:2, :]


def gen_offsets(kernel_size):
    offsets = np.zeros((2, kernel_size * kernel_size), dtype=np.int)
    ind = 0
    delta = (kernel_size - 1) // 2
    for i in range(kernel_size):
        y = i - delta
        for j in range(kernel_size):
            x = j - delta
            offsets[0, ind] = x
            offsets[1, ind] = y
            ind += 1
    return offsets


def gen_img_paf(img_crop, param, kernel_size=3):
    """Generate PAF image
    img_crop: 120x120
    kernel_size: kernel_size for convolution, should be even number like 3 or 5 or ...
    """
    anchor = reconstruct_paf_anchor(param)
    anchor = np.round(anchor).astype(np.int)
    delta = (kernel_size - 1) // 2
    anchor[anchor < delta] = delta
    anchor[anchor >= std_size - delta - 1] = std_size - delta - 1

    img_paf = np.zeros((64 * kernel_size, 64 * kernel_size, 3), dtype=np.uint8)
    offsets = gen_offsets(kernel_size)
    for i in range(kernel_size * kernel_size):
        ox, oy = offsets[:, i]
        index0 = anchor[0] + ox
        index1 = anchor[1] + oy
        p = img_crop[index1, index0].reshape(64, 64, 3).transpose(1, 0, 2)

        img_paf[oy + delta::kernel_size, ox + delta::kernel_size] = p

    return img_paf


def main():
    pass


if __name__ == '__main__':
    main()
