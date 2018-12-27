#!/usr/bin/env python3
# coding: utf-8

import os.path as osp
import numpy as np
from .io import _load


def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)


d = make_abs_path('../train.configs')
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
meta = _load(osp.join(d, 'param_whitening.pkl'))
# param_mean and param_std are used for re-whitening
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp
w = np.concatenate((w_shp, w_exp), axis=1)
w_base = w[keypoints]
w_norm = np.linalg.norm(w, axis=0)
w_base_norm = np.linalg.norm(w_base, axis=0)

# for inference
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120

# for paf (pac)
paf = _load(osp.join(d, 'Model_PAF.pkl'))
u_filter = paf.get('mu_filter')
w_filter = paf.get('w_filter')
w_exp_filter = paf.get('w_exp_filter')

# pncc code (mean shape)
pncc_code = _load(osp.join(d, 'pncc_code.npy'))
