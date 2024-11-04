# FLatten Transformer

This repo contains the official **PyTorch** code and pre-trained models for FLatten Transformer (ICCV 2023).

+ [FLatten Transformer: Vision Transformer with Focused Linear Attention](https://arxiv.org/abs/2308.00442) [[中文讲解]](https://www.bilibili.com/video/BV14j41127da/?share_source=copy_web&vd_source=b3825529abfc8e637393b7471e25233a)

## Updates

- May 28 2024: **Fix numerical instability problem.** Now FLatten Transformers can be trained with auto mixed precision (amp) or float16.

## Introduction

### Motivation

<p align="center">
    <img src="figures/attention.png" width= "600">
</p>

The quadratic computation complexity of self-attention $\mathcal{O}(N^2)$ has been a long-standing problem when applying Transformer models to vision tasks. Apart from reducing attention regions, linear attention is also considered as an effective solution to avoid excessive computation costs. By approximating Softmax with carefully designed mapping functions, linear attention can switch the computation order in the self-attention operation and achieve linear complexity $\mathcal{O}(N)$. Nevertheless, current linear attention approaches either suffer from severe performance drop or involve additional computation overhead from the mapping function. In this paper, we propose a novel **Focused Linear Attention** module to achieve both high efficiency and expressiveness.


### Method

<p align="center">
    <img src="figures/fp.png" width= "600">
</p>

<p align="center">
    <img src="figures/rank.png" width= "600">
</p>

 In this paper, we first perform a detailed analysis of the inferior performances of linear attention from two perspectives: focus ability and feature diversity. Then, we introduce a simple yet effective mapping function and an efficient rank restoration module and propose our **Focused Linear Attention (FLatten)** which adequately addresses these concerns and achieves high efficiency and expressive capability.

### Results

- Comparison of different models on ImageNet-1K.

<p align="center">
    <img src="figures/result1.png" width= "500">
</p>

- Accuracy-Runtime curve on ImageNet.

<p align="center">
    <img src="figures/result2.png" width= "900">
</p>

## Dependencies

- Python 3.9
- PyTorch == 1.11.0
- torchvision == 0.12.0
- numpy
- timm == 0.4.12
- einops
- yacs

## Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

## Pretrained Models

Based on different model architectures, we provide several pretrained models, as listed below.

| model  | Reso | acc@1 | config | pretrained weights |
| :---: | :---: | :---: | :---: | :---: |
| FLatten-PVT-T | $224^2$ | 77.8 (+2.7) | [config](cfgs/flatten_pvt_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/3ab1d773f19d45648690/?dl=1) |
| FLatten-PVTv2-B0 | $224^2$ | 71.1 (+0.6) | [config](cfgs/flatten_pvt_v2_b0.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/5d1f01532b104da28e7b/?dl=1) |
| FLatten-Swin-T | $224^2$ | 82.1 (+0.8) | [config](cfgs/flatten_swin_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/e1518e76703e4e57a7f2/?dl=1) |
| FLatten-Swin-S | $224^2$ | 83.5 (+0.5) | [config](cfgs/flatten_swin_s.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/94188e52af354bf4a88b/?dl=1) |
| FLatten-Swin-B | $224^2$ | 83.8 (+0.3) | [config](cfgs/flatten_swin_b.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/7a9e5186bad04e7fb3a9/?dl=1) |
| FLatten-Swin-B | $384^2$ | 85.0 (+0.5) | [config](cfgs/flatten_swin_b_384.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/0d0330cf2e5249f1abb6/?dl=1) |
| FLatten-CSwin-T | $224^2$ | 83.1 (+0.4) | [config](cfgs/flatten_cswin_t.yaml) | [TsinghuaCloud](https://cloud.tsinghua.edu.cn/f/48ba765ba8b0451d9d5a/?dl=1) |

Evaluate one model on ImageNet:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg <path-to-config-file> --data-path <imagenet-path> --output <output-path> --eval --resume <path-to-pretrained-weights>
```

Outputs of the four T/B0 pretrained models are:

```
[2023-07-21 07:50:09 flatten_pvt_tiny] (main.py 294): INFO  * Acc@1 77.758 Acc@5 93.910
[2023-07-21 07:50:09 flatten_pvt_tiny] (main.py 149): INFO Accuracy of the network on the 50000 test images: 77.8%

[2023-07-21 07:51:36 flatten_pvt_v2_b0] (main.py 294): INFO  * Acc@1 71.098 Acc@5 90.596
[2023-07-21 07:51:36 flatten_pvt_v2_b0] (main.py 149): INFO Accuracy of the network on the 50000 test images: 71.1%

[2023-07-21 07:46:13 flatten_swin_tiny_patch4_224] (main.py 294): INFO  * Acc@1 82.106 Acc@5 95.900
[2023-07-21 07:46:13 flatten_swin_tiny_patch4_224] (main.py 149): INFO Accuracy of the network on the 50000 test images: 82.1%

[2023-07-21 07:52:46 FLatten_CSWin_tiny](main.py 294): INFO  * Acc@1 83.130 Acc@5 96.376
[2023-07-21 07:52:46 FLatten_CSWin_tiny](main.py 149): INFO Accuracy of the network on the 50000 test images: 83.1%
```

## Train Models from Scratch

- **To train `FLatten-PVT-T/S/M/B` on ImageNet from scratch, run:**

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_t.yaml --data-path <imagenet-path> --output <output-path> --find-unused-params
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_s.yaml --data-path <imagenet-path> --output <output-path> --find-unused-params
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_m.yaml --data-path <imagenet-path> --output <output-path> --find-unused-params
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_b.yaml --data-path <imagenet-path> --output <output-path> --find-unused-params
```

- **To train `FLatten-PVT-v2-b0/1/2/3/4` on ImageNet from scratch, run:**

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_v2_b0.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_v2_b1.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_v2_b2.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_v2_b3.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_pvt_v2_b4.yaml --data-path <imagenet-path> --output <output-path>
```

- **To train `FLatten-Swin-T/S/B` on ImageNet from scratch, run:**

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_swin_t.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_swin_s.yaml --data-path <imagenet-path> --output <output-path>
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_swin_b.yaml --data-path <imagenet-path> --output <output-path>
```

- **To train `FLatten-CSwin-T/S/B` on ImageNet from scratch, run:**

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/flatten_cswin_t.yaml --data-path <imagenet-path> --output <output-path> --model-ema --model-ema-decay 0.99984
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/flatten_cswin_s.yaml --data-path <imagenet-path> --output <output-path> --model-ema --model-ema-decay 0.99984
```

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/flatten_cswin_b.yaml --data-path <imagenet-path> --output <output-path> --model-ema --model-ema-decay 0.99982
```

## Fine-tuning on higher resolution

Fine-tune a `FLatten-Swin-B` model pre-trained on 224x224 resolution to 384x384 resolution:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main.py --cfg ./cfgs/flatten_swin_b_384.yaml --data-path <imagenet-path> --output <output-path> --pretrained <path-to-224x224-pretrained-weights>
```

Fine-tune a `FLatten-CSwin-B` model pre-trained on 224x224 resolution to 384x384 resolution:

```shell
python -m torch.distributed.launch --nproc_per_node=8 main_ema.py --cfg ./cfgs/flatten_cswin_b_384.yaml --data-path <imagenet-path> --output <output-path> --pretrained <path-to-224x224-pretrained-weights> --model-ema --model-ema-decay 0.99982
```

## Visualization

We provide code for visualizing flatten attention. For example, to visualize flatten attention in FLatten-Swin-T, add the following to [this line](https://github.com/LeapLabTHU/FLatten-Transformer/blob/96b7dac65e9688d947a3afa01a0c70b92d9654c8/models/flatten_swin.py#L229). 

```python
from visualize import AttnVisualizer
visualizer = AttnVisualizer(qk=[q, k], kernel=self.dwc.weight, name='flatten_swin_t')
visualizer.visualize_all_attn(max_num=196, image='./visualize/img_ori_00809.png')
```

Then run:

```shell
python visualize.py
```

**Note:** Don't forget to modify the path of FLatten-Swin-T pretrained weight in `visualize.py`.

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer). The computational resources supporting this work are provided by [Hangzhou High-Flyer AI Fundamental Research Co.,Ltd](https://www.high-flyer.cn/)

## Citation

If you find this repo helpful, please consider citing us.

```latex
@InProceedings{han2023flatten,
  title={FLatten Transformer: Vision Transformer using Focused Linear Attention},
  author={Han, Dongchen and Pan, Xuran and Han, Yizeng and Song, Shiji and Huang, Gao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```

## Contact

If you have any questions, please feel free to contact the authors. 

Dongchen Han: [hdc23@mails.tsinghua.edu.cn](mailto:hdc23@mails.tsinghua.edu.cn)

Xuran Pan:  [pxr18@mails.tsinghua.edu.cn](mailto:pxr18@mails.tsinghua.edu.cn)
