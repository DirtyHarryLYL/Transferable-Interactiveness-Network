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

hoi_inter_all = [i for i in range(600) if i + 1 not in hoi_no_inter_all]


def getSigmoid(b, c, d, x, a=6):
    e = 2.718281828459
    return a / (1 + e ** (b - c * x)) + d


def iou(bb1, bb2, debug=False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[1] - bb2[0]
    y2 = bb2[3] - bb2[2]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[1]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[2])
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


def calc_ap(scores, bboxes, keys, hoi_id, begin):
    score = scores[:, hoi_id - begin]
    hit = []
    idx = np.argsort(score)[::-1]

    gt_bbox = pickle.load(open(cfg.ROOT_DIR + '/lib/ult/gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'))

    npos = 0
    used = {}

    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
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
    hit = np.cumsum(hit)
    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    return ap, np.max(rec)


def Generate_HICO_detection(HICO, HICO_DIR, Binary):
    thresh_det = pickle.load(open(osp.join(HICO_DIR, 'params_det.pkl'), 'rb'))
    thresh_nis = pickle.load(open(osp.join(HICO_DIR, 'params_nis.pkl'), 'rb'))
    keys, scores, bboxes, hdet, odet, pos, neg = [], [], [], [], [], [], []

    for i in range(80):
        keys.append([])
        scores.append([])
        bboxes.append([])
        hdet.append([])
        odet.append([])
        pos.append([])
        neg.append([])

    for key, value in HICO.items():
        assert len(HICO[key]) == len(Binary[key])

        for i in range(len(value)):
            element = value[i]
            classid = element[2] - 1
            keys[classid].append(key)
            scores[classid].append(
                element[3].reshape(1, -1) * getSigmoid(9, 1, 3, 0, element[4]) * getSigmoid(9, 1, 3, 0, element[5]))
            hbox = element[0].reshape(1, -1)
            obox = element[1].reshape(1, -1)
            bboxes[classid].append(np.concatenate([hbox, obox], axis=1))
            hdet[classid].append(element[4])
            odet[classid].append(element[5])

            pos[classid].append(Binary[key][i][3][0][0])
            neg[classid].append(Binary[key][i][3][0][1])

    print("Preparation finished")

    map = np.zeros(600)

    for i in range(80):
        if len(keys[i]) == 0:
            continue
        scores[i] = np.concatenate(scores[i], axis=0)
        bboxes[i] = np.concatenate(bboxes[i], axis=0)
        keys[i] = np.array(keys[i])
        hdet[i] = np.array(hdet[i])
        odet[i] = np.array(odet[i])
        pos[i] = np.array(pos[i])
        neg[i] = np.array(neg[i])

        begin = obj_range[i][0] - 1
        end = obj_range[i][1]

        for hoi_id in range(begin, end):
            hthresh = (thresh_det[hoi_id] // 10.) / 10.
            othresh = (thresh_det[hoi_id] % 10.) / 10.
            athresh = thresh_nis[hoi_id]

            hmask = hdet[i] > hthresh
            omask = odet[i] > othresh
            amask = pos[i] > athresh

            fmask = hmask * omask * amask
            select = np.where(fmask > 0)[0]
            score = scores[i][select, :]
            bbox = bboxes[i][select, :]
            key = keys[i][select]
            map[hoi_id], _ = calc_ap(score, bbox, key, hoi_id, begin)

            print(hoi_id, map[hoi_id], hthresh, othresh, athresh)

    print("mAP: %.4f" % np.mean(map))
    f = open(osp.join(HICO_DIR, 'result.txt'), 'w')
    f.write('mAP: %.4f \n' % (float(np.mean(map))))


def parse_arg():
    parser = argparse.ArgumentParser(description="Generate detection file")
    parser.add_argument(
        "--input_dir",
        dest="input_dir",
        default="./-Results/HICO-DET_1700000_TIN_HICO.pkl",
        type=str,
    )
    parser.add_argument(
        "--binary_dir",
        dest="binary_dir",
        default = "./-Results/HICO-DET_50000_TIN_10w_with_part.pkl",
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

    HICO = pickle.load(open(args.input_dir, "rb"))
    print("Pickle loaded")
    Binary = pickle.load(open(args.binary_dir, 'rb'))

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    Generate_HICO_detection(HICO, args.output_dir, Binary)


if __name__ == '__main__':
    main()
