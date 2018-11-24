**Todo:** Update the neck-removing processing pipeline from original BFM model.

The original version with neck:
<p align="center">
  <img src="imgs/bfm.png" alt="neck" width="400px">
</p>

The refined version without neck:
<p align="center">
  <img src="imgs/bfm_refine.png" alt="no neck" width="400px">
</p>

These two images are rendered by MeshLab.

`bfm_show.m` shows how to render it with 68 keypoints in Matlab.

<p align="center">
  <img src="imgs/bfm_refine.jpg" alt="no neck">
</p>

Attention: the z-axis value of `bfm.ply` and `bfm_refine.ply` file are opposed in `model_refine.mat`, do not use these two `ply` file in training.
