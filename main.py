#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
2. Refine code
"""

# import modules

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from math import sqrt
from ddfa_utils import ToTensorGjz, NormalizeGjz, str2bool
import matplotlib.pyplot as plt
from ddfa_utils import reconstruct_vertex
import scipy.io as sio
import time
import os
import argparse
import torch.backends.cudnn as cudnn


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_roi_box(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))


def dump_vertex(vertex, wfp):
    sio.savemat(wfp, {'vertex': vertex})
    print('Dump tp {}'.format(wfp))


def predict_68pts(param, roi_box):
    pts68 = reconstruct_vertex(param, dense=False)
    pts68 = pts68[:2, :]
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    pts68[0, :] = pts68[0, :] * scale_x + sx
    pts68[1, :] = pts68[1, :] * scale_y + sy

    return pts68


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flg=False):
    """Draw landmarks using matpliotlib"""
    plt.figure(figsize=(12, 8))
    plt.imshow(img[:, :, ::-1])

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        if style == 'simple':
            plt.plot(pts[i][0, :], pts[i][1, :], 'o', markersize=4, color='g')

        elif style == 'fancy':
            alpha = 0.8
            markersize = 4
            lw = 1.5
            color = 'w'
            markeredgecolor = 'black'

            nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

            # close eyes and mouths
            plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
            plot_close(41, 36)
            plot_close(47, 42)
            plot_close(59, 48)
            plot_close(67, 60)

            for ind in range(len(nums) - 1):
                l, r = nums[ind], nums[ind + 1]
                plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

                plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                         color=color,
                         markeredgecolor=markeredgecolor, alpha=alpha)

    plt.axis('off')
    plt.tight_layout()
    if wfp is not None:
        plt.savefig(wfp, dpi=200, bbox_inches='tight', pad_inches=0, transparent=True)
        print('Save visualization result to {}'.format(wfp))
    if show_flg:
        plt.show()


def main(args):
    # 1. load pretained model
    checkpoint_fp = 'models/phase1_wpdc_vdc_v2.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)

    model_dict = model.state_dict()
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model
    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_detector = dlib.get_frontal_face_detector()
    face_regressor = dlib.shape_predictor(dlib_landmark_model)

    # 3. forward pass: one step strategy
    tri = sio.loadmat('visualize/tri.mat')['tri']
    for img_fp in args.files:
        img_ori = cv2.imread(img_fp)
        if args.dlib_bbox:
            rects = face_detector(img_ori, 1)
        else:
            rects = []

        if len(rects) == 0:
            rects = dlib.rectangles()
            rect_fp = img_fp + '.bbox'
            lines = open(rect_fp).read().strip().split('\n')[1:]
            for l in lines:
                l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                rect = dlib.rectangle(l, r, t, b)
                rects.append(rect)

        pts_dlib = []
        pts_res = []
        ind = 0
        suffix = get_suffix(img_fp)
        for rect in rects:
            # landmark & crop
            pts = face_regressor(img_ori, rect).parts()
            pts = np.array([[pt.x, pt.y] for pt in pts]).T
            pts_dlib.append(pts)

            roi_box = calc_roi_box(pts)
            img = crop_img(img_ori, roi_box)

            # forward: one step
            img = cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
            transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

            # 68 pts
            pts68 = predict_68pts(param, roi_box)

            if args.box_init == 'two':
                roi_box_step2 = calc_roi_box(pts68)
                img_step2 = crop_img(img_ori, roi_box_step2)
                img_step2 = cv2.resize(img_step2, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                pts68 = predict_68pts(param, roi_box_step2)

            pts_res.append(pts68)

            # dense face vertices
            if args.dump_ply or args.dump_vertex:
                vertices = reconstruct_vertex(param, dense=True)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))

            ind += 1

        draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='false', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--box_init', default='one', type=str, help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='true', type=str2bool)
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool)

    args = parser.parse_args()
    main(args)
