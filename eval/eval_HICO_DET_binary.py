# --------------------------------------------------------
# Tensorflow iCAN
# Licensed under The MIT License [see LICENSE for details]
# Written by Chen Gao
# --------------------------------------------------------

"""
Change the HICO-DET detection results to the right format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import pickle
import shutil
import numpy as np
import os
import sys
import argparse
from ult.config import cfg
import os.path as osp

obj_range = [
    (161, 170), (11, 24), (66, 76), (147, 160), (1, 10),
    (55, 65), (187, 194), (568, 576), (32, 46), (563, 567),
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31),
    (77, 86), (112, 129), (130, 146), (175, 186), (97, 107),
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214),
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342),
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232),
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54),
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488),
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407),
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313),
    (265, 273), (87, 92), (93, 96), (171, 174), (240, 243),
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397),
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414),
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295),
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]

hoi_no_inter_all = [
    10, 24, 31, 46, 54, 65, 76, 86, 92, 96, 107, 111, 129, 146, 160, 170, 174, 186, 194, 198, 208, 214,
    224, 232, 235, 239, 243, 247, 252, 257, 264, 273, 283, 290, 295, 305, 313, 325, 330, 336, 342, 348,
    352, 356, 363, 368, 376, 383, 389, 393, 397, 407, 414, 418, 429, 434, 438, 445, 449, 453, 463, 474,
    483, 488, 502, 506, 516, 528, 533, 538, 546, 550, 558, 562, 567, 576, 584, 588, 595, 600
]


def iou(bb1, bb2, debug=False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[2] - bb2[0]
    y2 = bb2[3] - bb2[1]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[2]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[1])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)

def calc_binary(keys, bboxes, score):
    hit = []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open(cfg.ROOT_DIR + '/lib/ult/gt_binary_HICO_DET.pkl', 'rb'))
    npos = 0
    used = {}
    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0

    for j in range(len(idx)):
        pair_id = idx[j]
        bbox = bboxes[pair_id]
        key = keys[pair_id]

        if key in gt_bbox.keys():
            maxi = 0.0
            k = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox[0], gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)

    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(np.array(hit))

    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)

def Generate_HICO_detection(Binary, HICO_DIR):
    keys, scores, bboxes = [], [], []

    for key, value in Binary.items():
        for i in range(len(value)):
            element = value[i]
            keys.append(key)
            scores.append(element[3][0][0])
            hbox = element[0].reshape(1, -1)
            obox = element[1].reshape(1, -1)
            bboxes.append(np.concatenate([hbox, obox], axis=1))

    print("Preparation finished")

    ap, _ = calc_binary(keys, bboxes, scores)

    print("Binary mAP: %.4f" % (ap))
    f = open(os.path.join(HICO_DIR, "result_binary.txt"), 'w')
    f.write('mAP: %.4f \n' % (ap))


def parse_arg():
    parser = argparse.ArgumentParser(description="Generate detection file")
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        default="./-Results/HICO-DET_50000_TIN_10w_with_part.pkl",
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        default="./-Results/result_hico-det",
        type=str,
    )

    args = parser.parse_args()
    return args


def main():
    args = parse_arg()

    Binary = pickle.load(open(args.input_dir, "rb"))
    print("Pickle loaded")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    Generate_HICO_detection(Binary, args.output_dir)


if __name__ == '__main__':
    main()
