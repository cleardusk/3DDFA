# Face Alignment in Full Pose Range: A 3D Total Solution

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Introduction
The pytorch implementation of paper [Face Alignment in Full Pose Range: A 3D Total Solution](https://arxiv.org/abs/1804.01005).

<p align="center">
  <img src="imgs/landmark_3d.jpg" alt="Landmark 3D" width="1000px">
</p>

<p align="center">
  <img src="imgs/vertex_3d.jpg" alt="Vertex 3D" width="750px">
</p>

## Citation
    @article{zhu2017face,
      title={Face Alignment in Full Pose Range: A 3D Total Solution},
      author={Zhu, Xiangyu and Lei, Zhen and Li, Stan Z and others},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2017},
      publisher={IEEE}
    }


## Requirements
 - PyTorch >= 0.4.0
 - Python3.6

I strongly recommend using Python3.6 instead of older version for its better design.

## Evaluation
First, you should download the cropped testset ALFW and ALFW-2000-3D in [test.data.zip](https://pan.baidu.com/s/1DTVGCG5k0jjjhOc8GcSLOw), then unzip it and put it in the root directory.
Next, run the benchmark code by providing trained model path.
I have already provided four pre-trained models in `models` directory. These models are trained using different loss in the first stage. The model size is about 13M due to the high efficiency of mobilenet-v1 structure.
```
python3 ./benchmark.py -c models/phase1_wpdc_vdc.pth.tar
```

The performances of pre-trained models are shown below. In the first stage, the effectiveness of different loss is in order: WPDC > VDC > PDC. While the strategy using VDC to finetune WPDC achieves the best result.

| Model | AFLW (21 pts) | AFLW 2000-3D (68 pts) |
|:-:|:-:|:-:|
| phase1_pdc.pth.tar  | 6.956±0.981 | 5.644±1.323 |
| phase1_vdc.pth.tar  | 6.717±0.924 | 5.030±1.044 |
| phase1_wpdc.pth.tar | 6.348±0.929 | 4.759±0.996 |
| phase1_wpdc_vdc.pth.tar | **5.401±0.754** | **4.252±0.976** |

