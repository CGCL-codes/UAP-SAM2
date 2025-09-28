# UAP-SAM2
This repository contains the source code for our paper "Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2".

![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![Pytorch 2.1.0](https://img.shields.io/badge/pytorch-2.4.0-red.svg?style=plastic)
## Abstract
Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can directly transfer to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: prompt-specific interference in the first frame and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For the cross-prompt ability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. Comparative results show that UAP-SAM2 significantly outperforms SOTA attacks by a large margin.

<img src="image/pipeline.jpg"/>

## Setup
- **Build environment**
```shell
Setup
Build environment
cd UAP-SAM2
# use anaconda to build environment 
conda create -n UAP-SAM2 python=3.8
conda activate UAP-SAM2
# install packages
pip install -r requirements.txt
```

- **The final project should be like this:**
    ```shell
    UAP-SAM2
    └- sam2
        └- sam2
        └- checkpoints
          └- sam2_hiera_tiny.pt
          └- download_ckpts.sh
    └- data
      └- YOUTUBE
    └- ...
    ```
- **Download Victim Pre-trained Encoders**
  - Our pre-trained encoders were obtained from the [SAM2](https://github.com/facebookresearch/sam2) repository.
  - Please move the downloaded pre-trained encoder into  /sam2/checkpoints.

## Quick Start

- **The dataset can be downloaded [here](https://youtube-vos.org/).**
- **run `download_ckpts.sh` to init repos and download basic SAM2 checkpoints**
```shell
cd sam2
bash checkpoints/download_ckpts.sh
cd ..
```

- **Train UAP**
```shell 
python uap_attack.py   # results saved in uap_file/YOUTUBE.pth
```
- **Test performance of UAP-SAM2**
```shell 
python uap_atk_test.py # results saved in /result/test
```



