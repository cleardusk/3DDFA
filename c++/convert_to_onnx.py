#!/usr/bin/env python3
# coding: utf-8

import torch
import mobilenet_v1


def main():
    # checkpoint_fp = 'weights/phase1_wpdc_vdc.pth.tar'
    checkpoint_fp = 'weights/mb_1.p'
    arch = 'mobilenet_1'
    checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    for k in checkpoint.keys():
        kc = k.replace('module.', '')
        if kc in model_dict.keys():
            model_dict[kc] = checkpoint[k]
        if kc in ['fc_param.bias', 'fc_param.weight']:
            model_dict[kc.replace('_param', '')] = checkpoint[k]
    model.load_state_dict(model_dict)

    # conversion
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 120, 120)
    torch.onnx.export(model, dummy_input, checkpoint_fp.replace('.p', '.onnx'))
    # torch.onnx.export(model, dummy_input, checkpoint_fp.replace('.pth.tar', '.onnx'))


if __name__ == '__main__':
    main()
