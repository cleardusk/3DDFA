#!/usr/bin/env python3
# coding: utf-8
__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
0. Dump to obj with texture
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

# import modules

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, calc_roi_box, crop_img, predict_68pts, dump_to_ply, dump_vertex, draw_landmarks, \
    predict_dense
import argparse
import torch.backends.cudnn as cudnn


def main(args):
    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc_v2.pth.tar'
    arch = 'mobilenet_1'

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        model_dict[k.replace('module.', '')] = checkpoint[k]
    model.load_state_dict(model_dict)
    if args.mode == 'gpu':
        cudnn.benchmark = True
        model = model.cuda()
    model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    face_detector = dlib.get_frontal_face_detector()
    face_regressor = dlib.shape_predictor(dlib_landmark_model)

    # 3. forward
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

            # two-step for more accurate bbox to crop face
            if args.box_init == 'two':
                roi_box = calc_roi_box(pts68)
                img_step2 = crop_img(img_ori, roi_box)
                img_step2 = cv2.resize(img_step2, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
                input = transform(img_step2).unsqueeze(0)
                with torch.no_grad():
                    if args.mode == 'gpu':
                        input = input.cuda()
                    param = model(input)
                    param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
                pts68 = predict_68pts(param, roi_box)

            pts_res.append(pts68)

            # dense face vertices
            if args.dump_ply or args.dump_vertex:
                vertices = predict_dense(param, roi_box)
            if args.dump_ply:
                dump_to_ply(vertices, tri, '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_vertex:
                dump_vertex(vertices, '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))
            if args.dump_pts:
                wfp = '{}_{}.txt'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, pts68, fmt='%.3f')
                print('Save 68 3d landmarks to {}'.format(wfp))
            if args.dump_roi_box:
                wfp = '{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind)
                np.savetxt(wfp, roi_box, fmt='%.3f')
                print('Save roi box to {}'.format(wfp))

            ind += 1
        if args.dump_res:
            draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='True', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--box_init', default='one', type=str, help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='true', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')

    args = parser.parse_args()
    main(args)
