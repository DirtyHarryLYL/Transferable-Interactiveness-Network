
# TIN: Transferable Interactiveness Network   

#### **News**: (2022.12.19) HAKE 2.0 is accepted by TPAMI!

(2022.11.19) We release the interactive object bounding boxes & classes in the interactions within AVA dataset (2.1 & 2.2)! [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA), [[Paper]](https://arxiv.org/abs/2211.07501). BTW, we also release a CLIP-based human body part states recognizer in [CLIP-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/CLIP-Activity2Vec)!

(2022.07.29) Our new work PartMap (ECCV'22) is released! [Paper](https://github.com/enlighten0707/Body-Part-Map-for-Interactiveness/blob/main), [Code](https://github.com/DirtyHarryLYL/HAKE-Action-Torch)

(2022.02.14) We release the human body part state labels based on AVA: [HAKE-AVA](https://github.com/DirtyHarryLYL/HAKE-AVA) and [HAKE 2.0 paper](https://arxiv.org/abs/2202.06851).

(2021.10.06) Our extended version of [SymNet](https://github.com/DirtyHarryLYL/SymNet) is accepted by TPAMI! Paper and code are coming soon.

(2021.2.7) Upgraded [HAKE-Activity2Vec](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/Activity2Vec) is released! Images/Videos --> human box + ID + skeleton + part states + action + representation. [[Description]](https://drive.google.com/file/d/1iZ57hKjus2lKbv1MAB-TLFrChSoWGD5e/view?usp=sharing)
<p align='center'>
    <img src="https://github.com/DirtyHarryLYL/HAKE-Action-Torch/blob/Activity2Vec/demo/a2v-demo.gif", height="400">
</p>

<!-- ## Full demo: [[YouTube]](https://t.co/hXiAYPXEuL?amp=1), [[bilibili]](https://www.bilibili.com/video/BV1s54y1Y76s) -->

(2021.1.15) Our extended version of [TIN](https://arxiv.org/abs/2101.10292) is accepted by **TPAMI**!

(2020.10.27) The code of [IDN](https://github.com/DirtyHarryLYL/HAKE-Action-Torch/tree/IDN-(Integrating-Decomposing-Network)) ([Paper](https://arxiv.org/abs/2010.16219)) in NeurIPS'20 is released!

(2020.6.16) Our larger version [HAKE-Large](https://github.com/DirtyHarryLYL/HAKE#hake-large-for-instance-level-hoi-detection) (>120K images, activity and part state labels) is released!

**We have opened a tiny repo: HOI learning list (https://github.com/DirtyHarryLYL/HOI-Learning-List). It includes most of the recent HOI-related papers, code, datasets and leaderboard on widely-used benchmarks. Hope it could help everybody interested in HOI.**

Code of *"Transferable Interactiveness Knowledge for Human-Object Interaction Detection"*.

Created by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Siyuan Zhou, [Xijie Huang](https://huangowen.github.io/), [Liang Xu](https://liangxuy.github.io/), Ze Ma, [Hao-Shu Fang](https://fang-haoshu.github.io/), Yan-Feng Wang, [Cewu Lu](http://mvig.sjtu.edu.cn).

Link: [[CVPR arXiv]](https://arxiv.org/abs/1811.08264), [[TPAMI arXiv]](https://arxiv.org/abs/2101.10292)

## Citation
If you find our work useful in your research, please consider citing:
```
@inproceedings{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Zhou, Siyuan and Huang, Xijie and Xu, Liang and Ma, Ze and Fang, Hao-Shu and Wang, Yanfeng and Lu, Cewu},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={3585--3594},
  year={2019}
}
@article{li2019transferable,
  title={Transferable Interactiveness Knowledge for Human-Object Interaction Detection},
  author={Li, Yong-Lu and Liu, Xinpeng and Wu, Xiaoqian and Huang, Xijie and Xu, Liang and Lu, Cewu},
  journal={TPAMI},
  year={2021}
}
```

## Introduction
Interactiveness Knowledge indicates whether human and object interact with each other or not. It can be learned across HOI datasets, regardless of HOI category settings. We exploit an Interactiveness Network to learn the general interactiveness knowledge from multiple HOI datasets and perform Non-Interaction Suppression before HOI classification in inference. On account of the generalization of interactiveness, our **TIN: Transferable Interactiveness Network** is a transferable knowledge learner and can be cooperated with any HOI detection models to achieve desirable results. *TIN* outperforms state-of-the-art HOI detection results by a great margin, verifying its efficacy and flexibility.

![Overview of Our Framework](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/blob/master/images/overview.jpg?raw=true)

## Results on HICO-DET and V-COCO (CVPR)

**Our Results on HICO-DET dataset**

|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RC<sub>D</sub>(paper)| 13.75 | 10.23 | 15.45 | 15.34| 10.98|17.02|
|RP<sub>D</sub>C<sub>D</sub>(paper)| 17.03 | 13.42| 18.11| 19.17| 15.51|20.26|
|RC<sub>T</sub>(paper)| 10.61  | 7.78 | 11.45 | 12.47 | 8.87|13.54|
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 16.91   | 13.32 | 17.99 | 19.05 | 15.22|20.19|
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 17.22   | 13.51 | 18.32 | 19.38 | 15.38|20.57|
|Interactiveness-optimized| **17.54**  | **13.80** | **18.65** | **19.75** | **15.70** |**20.96**|

**Our Results on V-COCO dataset**

|Method| Full(def) |
|:---:|:---:|
|RC<sub>D</sub>(paper)| 43.2|
|RP<sub>D</sub>C<sub>D</sub>(paper)| 47.8 |
|RC<sub>T</sub>(paper)| 38.5 |
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 48.3  |
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 48.7 |
|Interactiveness-optimized| **49.0** |

**Please note that we have reimplemented TIN (e.g. replacing the vanilla HOI classifier with iCAN and using cosine_decay lr), thus the result here is different and slight better than the one in [[Arxiv]](https://arxiv.org/abs/1811.08264).**

## Extended Version (TPAMI 2021)
![Part Interactiveness Attention](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/blob/master/images/tin-pami.PNG?raw=true)
Besides the instance-level interactiveness between humans and objects, we further propose the **part-level** interactiveness between body parts and objects (whether a body part is interacted with an object or not). A new large-scale HOI benchmark based on the data from [HAKE (CVPR2020)](http://hake-mvig.cn), i.e., **PaStaNet-HOI** is also constructed. It contains **110K+ images with 520 HOIs** (without the 80 "no_interaction" HOIs of HICO-DET to avoid the incomplete labeling) and is more difficult than HICO-DET. We hope it can help to benchmark the HOI detection method better. More details please refer to our paper [arXiv](https://arxiv.org/abs/2101.10292).

The related code is coming soon.

### New results of TPAMI version
RCD in new version: R (representation extractor), C (interaction classifier), D (interactiveness discriminator), slightly different from the CVPR 2019 version.

**HICO-DET**
|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|R+iCAN+D<sub>3|17.58 |13.75|18.33|19.13|15.06|19.94|
|RCD| 17.84 |13.08 |18.78 |20.58 |16.19 |21.45|
|RCD<sub>1|17.49|12.23|18.53|20.28|15.25|21.27|
|RCD<sub>2|18.43|13.93|19.32|21.10|16.56|22.00|
|RCD<sub>3|**20.93**|**18.95**|**21.32**|**23.02**|**20.96**|**23.42**|

**V-COCO**
|Method| Scenario-1 |
|:---:|:---:|
|R+iCAN+D<sub>3|45.8 (46.1)|
|RCD| 48.4|
|RCD<sub>1|48.5|
|RCD<sub>2|48.7|
|RCD<sub>3|**49.1**|

**PaStaNet-HOI**
|Method| mAP |
|:---:|:---:|
|iCAN|11.0|
|R+iCAN+D<sub>3|13.13|
|RCD| **15.38**|

## Getting Started

### Installation

1.Clone this repository.

```
git clone https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network.git
```

2.Download dataset and setup evaluation and API. (The detection results (person and object boudning boxes) are collected from: iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/).)

```
chmod +x ./script/Dataset_download.sh 
./script/Dataset_download.sh
```

3.Install Python dependencies.

```
pip install -r requirements.txt
```

If you have trouble installing requirements, try to update your pip or try to use conda/virtualenv.

4.Download our pre-trained weight (Optional)

```
python script/Download_data.py 1f_w7HQxTfXGxOPrkriu7jTyCTC-KPEH3 Weights/TIN_HICO.zip
python script/Download_data.py 1iU9dN9rLtekcHX2MT_zU_df3Yf0paL9s Weights/TIN_VCOCO.zip
```

### Training

1.Train on HICO-DET dataset

```
python tools/Train_TIN_HICO.py --num_iteration 2000000 --model TIN_HICO_test
```

2.Train on V-COCO dataset

```
python tools/Train_TIN_VCOCO.py --num_iteration 20000 --model TIN_VCOCO_test
```

3.Train Interactiveness (TPAMI version)

```
python tools/Train_TIN_binary.py --num_iteration 100000 --model TIN_binary
```

### Testing

1.Test on HICO-DET dataset

```
python tools/Test_TIN_HICO.py --num_iteration 1700000 --model TIN_HICO
```

2.Test on V-COCO dataset

```
python tools/Test_TIN_VCOCO.py --num_iteration 6000 --model TIN_VCOCO
```

3.Test Interactiveness (TPAMI version)

```
python tools/Test_TIN_binary.py --num_iteration 80000 --model TIN_binary
```

### Evaluation

1.Download test result [files](https://drive.google.com/drive/folders/1xAt6f28qoQs-T71xELRCY9o3uLoqljzo?usp=sharing) and put them into the appropriate folder

```
Transferable-Interactiveness-Network
 |─ -Results
 │   |- HICO-DET_50000_TIN_10w_with_part.pkl
 |   |- HICO-DET_1700000_TIN_HICO.pkl
 |
 :       :

```

2.Get Interactiveness mAP

```
python eval/eval_HICO_DET_binary.py
```

3.Get HOI mAP

```
python eval/eval_HICO_DET.py
```



### Notes on training and Q&A

Since the interactiveness branch is easier to converge, first pre-training the whole model with HOI classification loss only then finetuning with both HOI and interactiveness loss is preferred to get the best performance.

Q: How is the used loss weights generated? 

A: Please refer to this [issue](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/issues/36) for detailed explanation.

## [HAKE](http://hake-mvig.cn/home/)
You may also be interested in our new work **HAKE**[[website]](http://hake-mvig.cn/home/), HAKE is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the HOI recognition performance on HICO and some other widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

## Acknowledgement

Some of the codes are built upon iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/ ). Thanks them for their great work! The pose estimation results are obtained from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) . **Alpha Pose** is an accurate multi-person pose estimator, which is the **first real-time** open-source system that achieves **70+ mAP (72.3 mAP)** on COCO dataset and **80+ mAP (82.1 mAP)** on MPII dataset. You may also use your own pose estimation results to train the interactiveness predictor, thus you could directly donwload the train and test pkl files from iCAN [[website]](http://chengao.vision/iCAN/) and insert your pose results.

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

TIN(Transferable Interactiveness Network) is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
