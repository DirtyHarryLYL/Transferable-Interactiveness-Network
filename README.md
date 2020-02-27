
# TIN: Transferable Interactiveness Network   

**News: Our paper "PaStaNet: Toward Human Activity Knowledge Engine" is accepted in CVPR2020, which contains the image-level and instance-level human part state annotations ([[HAKE]](https://github.com/DirtyHarryLYL/HAKE) and the corresponding models. We will release them soon. Besides, we will also release the code and data of our two related papers "Detailed 2D-3D Joint Representation for Human-Object Interaction" and  "Symmetry in Attribute-Object Compositions" (also accepted in CVPR2020).**

Code for our **CVPR2019** paper *"Transferable Interactiveness Knowledge for Human-Object Interaction Detection"*.

Created by [Yong-Lu Li](https://dirtyharrylyl.github.io/), Siyuan Zhou, [Xijie Huang](https://huangowen.github.io/), [Liang Xu](https://liangxuy.github.io/), Ze Ma, [Hao-Shu Fang](https://fang-haoshu.github.io/), Yan-Feng Wang, [Cewu Lu](http://mvig.sjtu.edu.cn).

Link: [[Arxiv]](https://arxiv.org/abs/1811.08264)

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
```

## Introduction
Interactiveness Knowledge indicates whether human and object interact with each other or not. It can be learned across HOI datasets, regardless of HOI category settings. We exploit an Interactiveness Network to learn the general interactiveness knowledge from multiple HOI datasets and perform Non-Interaction Suppression before HOI classification in inference. On account of the generalization of interactiveness, our **TIN: Transferable Interactiveness Network** is a transferable knowledge learner and can be cooperated with any HOI detection models to achieve desirable results. *TIN* outperforms state-of-the-art HOI detection results by a great margin, verifying its efficacy and flexibility.

![Overview of Our Framework](https://github.com/DirtyHarryLYL/Transferable-Interactiveness-Network/blob/master/images/overview.jpg?raw=true)

## Results on HICO-DET and V-COCO

**Our Results on HICO-DET dataset**

|Method| Full(def) | Rare(def) | None-Rare(def)| Full(ko) | Rare(ko) | None-Rare(ko) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|RC<sub>D</sub>(paper)| 13.75 | 10.23 | 15.45 | 15.34| 10.98|17.02|
|RP<sub>D</sub>C<sub>D</sub>(paper)| **17.03** | **13.42**| **18.11**| **19.17**| **15.51**|**20.26**|
|RC<sub>T</sub>(paper)| 10.61  | 7.78 | 11.45 | 12.47 | 8.87|13.54|
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 16.91   | 13.32 | 17.99 | 19.05 | 15.22|20.19|
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 17.22   | 13.51 | 18.32 | 19.38 | 15.38|20.57|
|RP<sub>T2</sub>C<sub>D</sub>(optimized)| **17.54**  | **13.80** | **18.65** | **19.75** | **15.70** |**20.96**|

**Our Results on V-COCO dataset**

|Method| Full(def) |
|:---:|:---:|
|RC<sub>D</sub>(paper)| 43.2|
|RP<sub>D</sub>C<sub>D</sub>(paper)| **47.8** |
|RC<sub>T</sub>(paper)| 38.5 |
|RP<sub>T1</sub>C<sub>D</sub>(paper)| 48.3  |
|RP<sub>T2</sub>C<sub>D</sub>(paper)| 48.7 |
|RP<sub>T2</sub>C<sub>D</sub>(optimized)| **49.0** |

**Please note that we have reimplemented TIN (e.g. replacing the vanilla HOI classifier with iCAN and using cosine_decay lr), thus the result here is different and slight better than the one in [[Arxiv]](https://arxiv.org/abs/1811.08264).**

## [HAKE](http://hake-mvig.cn/home/)
You may also be interested in our new work **HAKE**[[website]](http://hake-mvig.cn/home/), HAKE is a new large-scale knowledge base and engine for human activity understanding. HAKE provides elaborate and abundant **body part state** labels for active human instances in a large scale of images and videos. With HAKE, we boost the HOI recognition performance on HICO and some other widely-used human activity benchmarks. Now we are still enlarging and enriching it, and looking forward to working with outstanding researchers around the world on its applications and further improvements. If you have any pieces of advice or interests, please feel free to contact [Yong-Lu Li](https://dirtyharrylyl.github.io/) (yonglu_li@sjtu.edu.cn).

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

### Testing

1.Test on HICO-DET dataset

```
python tools/Test_TIN_HICO.py --num_iteration 1700000 --model TIN_HICO
```

2.Test on V-COCO dataset

```
python tools/Test_TIN_VCOCO.py --num_iteration 6000 --model TIN_VCOCO
```

## Acknowledgement

Some of the codes are built upon iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/ ). Thanks them for their great work! The pose estimation results are obtained from [AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) . **Alpha Pose** is an accurate multi-person pose estimator, which is the **first real-time** open-source system that achieves **70+ mAP (72.3 mAP)** on COCO dataset and **80+ mAP (82.1 mAP)** on MPII dataset. You may also use your own pose estimation results to train the interactiveness predictor, thus you could directly donwload the train and test pkl files from iCAN [[website]](http://chengao.vision/iCAN/) and insert your pose results.

If you get any problems or if you find any bugs, don't hesitate to comment on GitHub or make a pull request! 

TIN(Transferable Interactiveness Network) is freely available for free non-commercial use, and may be redistributed under these conditions. For commercial queries, please drop an e-mail. We will send the detail agreement to you.
