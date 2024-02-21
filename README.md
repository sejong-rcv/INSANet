## INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection

### Official Pytorch Implementation of [INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection](https://www.mdpi.com/1424-8220/24/4/1168)

#### 📢Notice : Multispectral Pedestrian Detection Challenge Leaderboard is available.
 [![Leaderboard](https://img.shields.io/badge/Leaderboard-Multispectral%20Pedestrian%20Detection-blue)](https://eval.ai/web/challenges/challenge-page/1247/leaderboard/3137)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/insanet-intra-inter-spectral-attention/multispectral-object-detection-on-kaist)](https://paperswithcode.com/sota/multispectral-object-detection-on-kaist?p=insanet-intra-inter-spectral-attention)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/insanet-intra-inter-spectral-attention/pedestrian-detection-on-llvip)](https://paperswithcode.com/sota/pedestrian-detection-on-llvip?p=insanet-intra-inter-spectral-attention)


<p align="center"><img src="fig/architecture.png" width="900"></p>
<p align="center"><img src="fig/insa.png" width="700"></p>

> **PDF**: [INSANet: INtra-INter Spectral Attention Network for Effective Feature Fusion of Multispectral Pedestrian Detection](https://www.mdpi.com/1424-8220/24/4/1168/pdf)

---

## Abstract
Pedestrian detection is a critical task for safety-critical systems, but detecting pedestrians is challenging in low-light and adverse weather conditions. Thermal images can be used to improve robustness by providing complementary information to RGB images. Previous studies have shown that multi-modal feature fusion using convolution operation can be effective, but such methods rely solely on local feature correlations, which can degrade the performance capabilities. To address this issue, we propose an attention-based novel fusion network, referred to as INSANet (INtra- INter Spectral Attention Network), that captures global intra- and inter-information. It consists of intra- and inter-spectral attention blocks that allow the model to learn mutual spectral relationships. Additionally, we identified an imbalance in the multispectral dataset caused by several factors and designed an augmentation strategy that mitigates concentrated distributions and enables the model to learn the diverse locations of pedestrians. Extensive experiments demonstrate the effectiveness of the proposed methods, which achieve state-of-the-art performance on the KAIST dataset and LLVIP dataset. Finally, we conduct a regional performance evaluation to demonstrate the effectiveness of our proposed network in various regions.

