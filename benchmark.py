# !/usr/bin/env python3
# coding: utf-8

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import mobilenet_v1
import time
import numpy as np

from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from ddfa_utils import ToTensorGjz, NormalizeGjz, DDFATestDataset
from params import *


def _reconstruct_vertex(param, whitening=True, dense=False):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp"""
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)

    if dense:
        vertex = p @ (u + w_shp @ alpha_shp + w_exp @ alpha_exp).reshape(3, -1, order='F') + offset
    else:
        """For 68 pts"""
        vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).reshape(3, -1, order='F') + offset
        # for landmarks
        vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex


def extract_param(checkpoint_fp, root='', filelists=None, arch='mobilenet_1', num_classes=62, device_ids=[0],
                  batch_size=128, num_workers=4):
    map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    checkpoint = torch.load(checkpoint_fp, map_location=map_location)['state_dict']
    torch.cuda.set_device(device_ids[0])
    model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    model.load_state_dict(checkpoint)

    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs.cuda()
            output = model(inputs)

            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()

                outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print(f'{time.time() - end: .3f}s')
    return outputs


def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = _reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params):
    outputs = []
    for i in range(params.shape[0]):
        lm = _reconstruct_vertex(params[i])
        outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline():
    device_ids = [0]
    checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    arch = 'mobilenet_1'

    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW_GT_crop',
            filelists='test.data/AFLW_GT_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=128)

        benchmark_alfw_params(params)

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root='test.data/AFLW2000-3D_crop',
            filelists='test.data/AFLW2000-3D_crop.list',
            arch=arch,
            device_ids=device_ids,
            batch_size=128)

        benchmark_aflw2000_params(params)

    aflw2000()
    aflw()


def main():
    benchmark_pipeline()


if __name__ == '__main__':
    main()
