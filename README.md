<div align="center">
  <h2><b> [SIGKDD'25 PatchSTG] Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective
 </b></h2>
</div>

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/efficient-large-scale-traffic-forecasting/traffic-prediction-on-largest)](https://paperswithcode.com/sota/traffic-prediction-on-largest?p=efficient-large-scale-traffic-forecasting)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=PatchSTG&color=red&logo=arxiv)](https://arxiv.org/abs/2412.09972)
![](https://img.shields.io/badge/PRs-Welcome-green)

</div>

<div align="center">

**[<a href="https://www.bilibili.com/video/BV1f5C3Y5Eb2/?spm_id_from=333.999.0.0&vd_source=ff21219783fd1e65d312b09da53b31c5">AITIME Talk</a>]**
**[<a href="https://mp.weixin.qq.com/s/M5w6I52tCIVsQXZ1ombVqA">时空探索之旅中文解读</a>]**
**[<a href="https://zhuanlan.zhihu.com/p/14255424377">知乎中文解读</a>]**

</div>

<div align="center">

<img src="./imgs/sketch.png" width="500">

</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{fang2024efficient,
  title={Efficient Large-Scale Traffic Forecasting with Transformers: A Spatial Data Management Perspective},
  author={Fang, Yuchen and Liang, Yuxuan and Hui, Bo and Shao, Zezhi and Deng, Liwei and Liu, Xu and Jiang, Xinke and Zheng, Kai},
  journal={arXiv preprint arXiv:2412.09972},
  year={2024}
}
```

## Introduction
PatchSTG is an attention-based dynamic spatial modeling method that uses irregular spatial patching for efficient large-scale traffic forecasting.
Notably, we show that spatiotemporal graphs can be patched on the spatial dimension, effectively reducing complexity in attention.

<p align="center">
<img src="./imgs/frame.png" height = "300" alt="" align=center />
</p>

- PatchSTG comprises four components: (1) embedding the input traffic into high-dimensional representations with spatio-temporal properties, (2) segmenting the large-scale input into balanced and non-overlapped patches on the spatial dimension with irregularly distributed points, (3) using depth and breadth attentions on the patched input to capture local and global spatial dependencies efficiently, and (4) projecting representations to the predicted future traffic.

<p align="center">
<img src="./imgs/patching.png"  width="600" alt="" align=center />
</p>

## Hyper-parameters
- tps: temporal patch size
- tpn: temporal patch number ($tps\times tpn==input\_len$)
- recur: KDTree depth ($recur<\lfloor log_2(nodes)\rfloor$)
- sps: KDTree leaf size
- spn: KDTree nodes ($spn=2^{recur}$)
- factors: $sps\times factors$ is the spatial patch size and $spn/factors$ is the spatial patch number

## Requirements
- torch==1.11.0
- timm==1.0.12
- scikit_learn==1.0.2
- tqdm==4.67.1
- pandas==1.4.1
- numpy==1.22.3

## Folder Structure

```tex
└── code-and-data
    ├── config                 # Including detail configurations
    ├── cpt                    # Storing pre-trained weight files (manually create the folder and download files)
    ├── data                   # Including traffic data (download), adj files (generated), and the meta data
    ├── lib
    │   |──  utils.py          # Codes of preprocessing datasets and calculating metrics
    ├── log                    # Storing log files
    ├── model
    │   |──  models.py         # The core source code of our PatchSTG
    ├── main.py                # This is the main file for training and testing
    └── README.md              # This document
```

## Datasets
You can access the well pre-processed datasets from [[Google Drive]](https://drive.google.com/drive/folders/1BDH1C66BCKBe7ge8G-rBaj1j3p0iR0TC?usp=sharing), then place the downloaded contents under the correspond dataset folder such as `./data/SD`.

## PreTrained Weights
You can access the pretrained weights from [[Google Drive]](https://drive.google.com/drive/folders/1hFyV2C10P3wl3OJkNNhhHb2LTKXcJ2mO?usp=sharing), then place the downloaded contents under the constructed cpt folder `./cpt`.

## Quick Demos
1. Download datasets and place them under `./data`
2. We provide pre-trained weights of results in the paper and the detail configurations under the folder `./config`. For example, you can test the SD dataset by:

```
python main.py --config ./config/SD.conf
```

3. If you want to train the model yourself, you can use the code at line 262 of the main file.


## Further Reading
1, [**When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks**](https://ieeexplore.ieee.org/abstract/document/10184591), in *ICDE* 2023.
[\[GitHub Repo\]](https://github.com/LMissher/STWave)

**Authors**: Yuchen Fang, Yanjun Qin, Haiyong Luo, Fang Zhao, Bingbing Xu, Liang Zeng, Chenxing Wang.

```bibtex
@inproceedings{fang2023spatio,
  title={When spatio-temporal meet wavelets: Disentangled traffic forecasting via efficient spectral graph attention networks},
  author={Fang, Yuchen and Qin, Yanjun and Luo, Haiyong and Zhao, Fang and Xu, Bingbing and Zeng, Liang and Wang, Chenxing},
  booktitle={2023 IEEE 39th International Conference on Data Engineering (ICDE)},
  pages={517--529},
  year={2023}
}
```
