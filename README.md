# INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection

### Official Pytorch Implementation of [INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection](https://www.mdpi.com/1424-8220/24/4/1168)
#### Authors: [Sangin Lee](https://sites.google.com/rcv.sejong.ac.kr/silee/%ED%99%88), [Taejoo Kim](https://sites.google.com/view/xown3197), [Jeongmin Shin](https://sites.google.com/view/jeongminshin), [Namil Kim](https://scholar.google.com/citations?user=IYyLBQYAAAAJ&hl=ko&oi=sra), and [Yukyung Choi](https://scholar.google.com/citations?user=vMrPtrAAAAAJ&hl=ko&oi=sra)

#### 📢Notice : Multispectral Pedestrian Detection Challenge Leaderboard is available.
 [![Leaderboard](https://img.shields.io/badge/Leaderboard-Multispectral%20Pedestrian%20Detection-blue)](https://eval.ai/web/challenges/challenge-page/1247/leaderboard/3137)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/insanet-intra-inter-spectral-attention/multispectral-object-detection-on-kaist)](https://paperswithcode.com/sota/multispectral-object-detection-on-kaist?p=insanet-intra-inter-spectral-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/insanet-intra-inter-spectral-attention/pedestrian-detection-on-llvip)](https://paperswithcode.com/sota/pedestrian-detection-on-llvip?p=insanet-intra-inter-spectral-attention)


<p align="center"><img src="fig/architecture.png" width="900"></p>
<p align="center"><img src="fig/insa.png" width="700"></p>

## Abstract
Pedestrian detection is a critical task for safety-critical systems, but detecting pedestrians is challenging in low-light and adverse weather conditions. Thermal images can be used to improve robustness by providing complementary information to RGB images. Previous studies have shown that multi-modal feature fusion using convolution operation can be effective, but such methods rely solely on local feature correlations, which can degrade the performance capabilities. To address this issue, we propose an attention-based novel fusion network, referred to as INSANet (INtra- INter Spectral Attention Network), that captures global intra- and inter-information. It consists of intra- and inter-spectral attention blocks that allow the model to learn mutual spectral relationships. Additionally, we identified an imbalance in the multispectral dataset caused by several factors and designed an augmentation strategy that mitigates concentrated distributions and enables the model to learn the diverse locations of pedestrians. Extensive experiments demonstrate the effectiveness of the proposed methods, which achieve state-of-the-art performance on the KAIST dataset and LLVIP dataset. Finally, we conduct a regional performance evaluation to demonstrate the effectiveness of our proposed network in various regions.

> **PDF**: [INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection](https://www.mdpi.com/1424-8220/24/4/1168/pdf)

---

## Usage

### Recommended Environment
- OS: Ubuntu 20.04
- CUDA-cuDNN: 11.3.0-8
- GPU: NVIDIA-A100
- Python-Torch: 3.7-1.11.0
  
See [environment.yaml](https://github.com/sejong-rcv/INSANet/blob/main/environment.yaml) for more details

### Installation
The environment file has all the dependencies that are needed for INSANet.

We offer guides on how to install dependencies via docker and conda.

First, clone the repository:
#### Git Clone
```
git clone https://github.com/sejong-rcv/INSANet.git
cd INSANet
```

#### 1. Docker
- **Prerequisite**
  - [nvidia-container-toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
  - Note that nvidia-cuda:11.3.0 is deprecated. See [issue](https://github.com/NVIDIA/nvidia-docker/issues/1745).
 ```
cd docker
make docker-make
```

- **Make Container**
 ```
nvidia-docker run -it --name insanet -v $PWD:/workspace -p 8888:8888 -e NVIDIA_VISIBLE_DEVICES=all --shm-size=32G insanet:maintainer /bin/bash
```

#### 2. Conda
- **Prerequisite**
  - Required dependencies are listed in environment.yaml.
```
conda env create -f environment.yml
conda activate insanet
```

If your environment support CUDA 11.3,
```
conda env create -f environment_cu113.yml
conda activate insanet
```

## Dataset
The datasets used to train and evaluate model are as follows:
- [KAIST Multispectral Pedestrian Detection Benchmark](https://github.com/SoonminHwang/rgbt-ped-detection)
- [LLVIP: A Visible-infrared Paired Dataset for Low-light Vision](https://github.com/bupt-ai-cz/LLVIP)

The dataloader in [datasets.py](https://github.com/sejong-rcv/INSANet/blob/main/src/datasets.py) assumes that the dataset is located in the data folder and structured as follows:

- **KAIST**
  - First, you should download the dataset. we provide the script to download the dataset (please see data/download_kaist).
  - Train: We use paired annotations provided in [AR-CNN](https://github.com/luzhang16/AR-CNN).
  - Evaluation:  We use sanitized (improved) annotations provided in [MSDS-RCNN](https://github.com/Li-Chengyang/MSDS-RCNN).
```
├── data
   └── kaist-rgbt
      ├── annotations_paired
         ├── set00
            ├── V000
               ├── lwir
                  ├── I00000.txt
                  ├── ...
               ├── visible
                  ├── I00000.txt
                  ├── ...
            ├── V001
               ├── lwir
                  ├── I00000.txt
                  ├── ...
               ├── visible
                  ├── I00000.txt
                  ├── ...
            └── ...
         ├── ... (set02-set10)
         └── set11
            ├── V000
               ├── lwir
                  ├── I00019.txt
                  ├── ...
               ├── visible
                  ├── I00019.txt
                  ├── ...
      ├── images
         ├─ The structure is identical to the "annotations_paired", but
         └─ files are altered from text(.txt) to image(.jpg).

├── src
   ├── kaist_annotations_test20.json
   ├── imageSets
      ├── train-all-02.txt
      └── test-all-20.txt
```

- **LLVIP**
  - First, you should download the dataset. Please see download_dataset.md in [LLVIP](https://github.com/bupt-ai-cz/LLVIP).
  - A pair of visible and infrared images share the same annotation with the same file name.
  - The annotations are in VOC format and we evaluate in annotations that have been modified to COCO format.
```
├── data
   └── LLVIP
      ├── Annotations
         ├── 010001.xml
         ├── 010002.xml
         ├── ...
         ├── 260535.xml
         └── 260536.xml
      ├── infrared
         ├── train
            ├── 010001.jpg
            ├── 010002.jpg
            ├── ...
            └── 250423.jpg
         └── test
            ├── 190001.jpg
            ├── 190002.jpg
            ├── ...
            └── 260536.jpg
      └── visible
         └── The structure is identical to the "infrared".
```
