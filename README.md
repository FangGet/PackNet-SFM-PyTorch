# PackNet-SFM PyTorch version
This codebase almost implements(unofficial) the system described in the paper:

> **PackNet-SfM: 3D Packing for Self-Supervised Monocular Depth Estimation**
>
> V. Guizilini, R. Ambrus, S. Pillai, and A. Gaidon 
>
> \[ [pdf](https://arxiv.org/pdf/1905.02693), [video](https://youtu.be/-N8QFtL3ees), [ICML'19 SSL wkshp](https://drive.google.com/open?id=0B4M2lUVyJzS4RmpJWTVRTHMtWHZPQ3V5VG85VDA0bENfSUJJ) \]

This codebase mainly based on [monodepth2](https://github.com/nianticlabs/monodepth2) and only for academic research.

See the [packnet-sfm(code published)](https://github.com/TRI-ML/packnet-sfm) and [spillai's homepage](http://people.csail.mit.edu/spillai/) for more detail.

## Preamble

This codebase was developed and tested with PyTorch 1.0.0, CUDA 10 and Ubuntu 16.04, and we only did depth evaluation.

## Differences with Paper
As we only have a RTX2060 8G, to make it possible train on it,  we slim the network while keeping main network structure consistent with paper, changes are as follows:
* For Packing and unpacking Block, we set D from 8 to 4, and add a 1x1 conv to reduce channel for Packing 3Dconv;
* For encoder, we change network stack from [64, 64, 128, 256, 512] to [64, 64, 64, 128, 128]; For decoder, from [512, 256, 128, 64, 64] to [128, 128, 64, 32, 16];
* For encoder, we change ResidualBlock number from [x2, x2, x3, x3] to [x2, x2, x2, x2];

All former changes make this implementation be suitable for training on RTX2060 8G with batch size set to 6, and we believe with more powerful GPU and original network complexity, we may get better prediction result.


## Depth Results

| Abs Rel | Sq Rel | RMSE  | RMSE(log) | Acc.1 | Acc.2 | Acc.3 |
| ------- | ------ | ----- | --------- | ----- | ----- | ----- |
| 0.119   | 0.890  | 4.855 | 0.198     | 0.862 | 0.954 | 0.980 |

For now, we can not get similar result as paper, but we are struggle to ease the gap. 

## Setup and Training

As this codebase mainly based on monodepth2, all environment setup, data preparation and training/ evaluation step are the same, please visit monodepth2 [Readme.md](https://github.com/nianticlabs/monodepth2/blob/master/README.md) for detail.



