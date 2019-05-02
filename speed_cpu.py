#!/usr/bin/env python3
# coding: utf-8

import timeit
import numpy as np

SETUP_CODE = '''
import mobilenet_v1
import torch

model = mobilenet_v1.mobilenet_1()
model.eval()
data = torch.rand(1, 3, 120, 120)
'''

TEST_CODE = '''
with torch.no_grad():
    model(data)
'''


def main():
    repeat, number = 5, 100
    res = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=repeat,
                        number=number)
    res = np.array(res, dtype=np.float32)
    res /= number
    mean, var = np.mean(res), np.std(res)
    print('Inference speed: {:.2f}±{:.2f} ms'.format(mean * 1000, var * 1000))


if __name__ == '__main__':
    main()
