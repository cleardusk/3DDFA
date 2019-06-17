#!/usr/bin/env python3
# coding: utf-8
import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz
import scipy.io as sio
from utils.inference import (
    parse_roi_box_from_landmark,
    crop_img,
    predict_68pts,
    predict_dense,
)
from utils.cv_plot import plot_kpt
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn

STD_SIZE = 120


def main(args):
    # 0. open video
    # vc = cv2.VideoCapture(str(args.video) if len(args.video) == 1 else args.video)
    vc = cv2.VideoCapture(args.video if int(args.video) != 0 else 0)

    # 1. load pre-tained model
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    tri = sio.loadmat('visualize/tri.mat')['tri']
    transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])

    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)[
        'state_dict'
    ]
    model = getattr(mobilenet_v1, arch)(
        num_classes=62
    )  # 62 = 12(pose) + 40(shape) +10(expression)

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
    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    success, frame = vc.read()
    last_frame_pts = []

    while success:
        if len(last_frame_pts) == 0:
            rects = face_detector(frame, 1)
            for rect in rects:
                pts = face_regressor(frame, rect).parts()
                pts = np.array([[pt.x, pt.y] for pt in pts]).T
                last_frame_pts.append(pts)

        vertices_lst = []
        for lmk in last_frame_pts:
            roi_box = parse_roi_box_from_landmark(lmk)
            img = crop_img(frame, roi_box)
            img = cv2.resize(
                img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR
            )
            input = transform(img).unsqueeze(0)
            with torch.no_grad():
                if args.mode == 'gpu':
                    input = input.cuda()
                param = model(input)
                param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
            pts68 = predict_68pts(param, roi_box)
            vertex = predict_dense(param, roi_box)
            lmk[:] = pts68[:2]
            vertices_lst.append(vertex)

        pncc = cpncc(frame, vertices_lst, tri - 1) / 255.0
        frame = frame / 255.0 * (1.0 - pncc)
        cv2.imshow('3ddfa', frame)
        cv2.waitKey(1)
        success, frame = vc.read()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument(
        '-v',
        '--video',
        default='0',
        type=str,
        help='video file path or opencv cam index',
    )
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')

    args = parser.parse_args()
    main(args)
