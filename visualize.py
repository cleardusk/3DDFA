#!/usr/bin/env python3
# coding: utf-8

from benchmark import extract_param
from utils.ddfa import reconstruct_vertex
from utils.io import _dump, _load
import os.path as osp
from skimage import io
import matplotlib.pyplot as plt
from benchmark_aflw2000 import convert_to_ori
import scipy.io as sio


def aflw2000():
    arch = 'mobilenet_1'
    device_ids = [0]
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'

    params = extract_param(
        checkpoint_fp=checkpoint_fp,
        root='test.data/AFLW2000-3D_crop',
        filelists='test.data/AFLW2000-3D_crop.list',
        arch=arch,
        device_ids=device_ids,
        batch_size=128)
    _dump('res/params_aflw2000.npy', params)


def draw_landmarks():
    filelists = 'test.data/AFLW2000-3D_crop.list'
    root = 'AFLW-2000-3D/'
    fns = open(filelists).read().strip().split('\n')
    params = _load('res/params_aflw2000.npy')

    for i in range(2000):
        plt.close()
        img_fp = osp.join(root, fns[i])
        img = io.imread(img_fp)
        lms = reconstruct_vertex(params[i], dense=False)
        lms = convert_to_ori(lms, i)

        # print(lms.shape)
        fig = plt.figure(figsize=plt.figaspect(.5))
        # fig = plt.figure(figsize=(8, 4))
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(img)

        alpha = 0.8
        markersize = 4
        lw = 1.5
        color = 'w'
        markeredgecolor = 'black'

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]
        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            ax.plot(lms[0, l:r], lms[1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            ax.plot(lms[0, l:r], lms[1, l:r], marker='o', linestyle='None', markersize=markersize, color=color,
                    markeredgecolor=markeredgecolor, alpha=alpha)

        ax.axis('off')

        # 3D
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        lms[1] = img.shape[1] - lms[1]
        lms[2] = -lms[2]

        # print(lms)
        ax.scatter(lms[0], lms[2], lms[1], c="cyan", alpha=1.0, edgecolor='b')

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            ax.plot3D(lms[0, l:r], lms[2, l:r], lms[1, l:r], color='blue')

        ax.view_init(elev=5., azim=-95)
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        plt.tight_layout()
        # plt.show()

        wfp = f'res/AFLW-2000-3D/{osp.basename(img_fp)}'
        plt.savefig(wfp, dpi=200)


def gen_3d_vertex():
    filelists = 'test.data/AFLW2000-3D_crop.list'
    root = 'AFLW-2000-3D/'
    fns = open(filelists).read().strip().split('\n')
    params = _load('res/params_aflw2000.npy')

    sel = ['00427', '00439', '00475', '00477', '00497', '00514', '00562', '00623', '01045', '01095', '01104', '01506',
           '01621', '02214', '02244', '03906', '04157']
    sel = list(map(lambda x: f'image{x}.jpg', sel))
    for i in range(2000):
        fn = fns[i]
        if fn in sel:
            vertex = reconstruct_vertex(params[i], dense=True)
            wfp = osp.join('res/AFLW-2000-3D_vertex/', fn.replace('.jpg', '.mat'))
            print(wfp)
            sio.savemat(wfp, {'vertex': vertex})


def main():
    # step1: extract params
    # aflw2000()

    # step2: draw landmarks
    # draw_landmarks()

    # step3: visual 3d vertex
    gen_3d_vertex()


if __name__ == '__main__':
    main()
