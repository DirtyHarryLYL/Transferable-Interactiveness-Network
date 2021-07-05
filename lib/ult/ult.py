# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Generating training instance
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import pickle
import random
from random import randint
import tensorflow as tf
import cv2

#from config import cfg # cause error because of "from __future__ import absolute_import"
from ult import config # use absolute import and config.cfg


list_no_inter = [10,24,31,46,54,65,76,86,92,96,107,111,
129,146,160,170,174,186,194,198,208,214,224,232,235,239,
243,247,252,257,264,273,283,290,295,305,313,325,330,336,
342,348,352,356,363,368,376,383,389,393,397,407,414,418,
429,434,438,445,449,453,463,474,483,488,502,506,516,528,
533,538,546,550,558,562,567,576,584,588,595,600]

image_folder_list = {'hico-train': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/train2015/',
                     'hico-test': '/Disk1/yonglu/iCAN/Data/hico_20160224_det/images/test2015/',
                     'hcvrd': '/Disk2/yonglu/behaviour_universe/hcvrd/image/',
                     'openimage': '/Disk2/yonglu/behaviour_universe/openimage/image/',
                     'vcoco': '/Disk1/yonglu/iCAN/Data/v-coco/coco/images/all/',
                     'pic': '/Disk2/yonglu/behaviour_universe/pic/All/',
                     'long_tail_1': '/Disk2/yonglu/behaviour_universe/long_tail_round_1/long_tail_final/img_all/all/',
                     'long_tail_2': '/Disk2/yonglu/behaviour_universe/long_tail_round_2/img_all/'}

OBJECT_MASK = [
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0]
]

hico_obj_range = [
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

hico_obj_mask = np.zeros((600, 80), dtype=np.int32)
for i in range(80):
    x, y = hico_obj_range[i]
    hico_obj_mask[x - 1:y, i] = 1


print("*************data path:*************")
print(config.cfg.DATA_DIR)
print("************************************")



def bbox_trans(human_box_ori, object_box_ori, ratio, size = 64):

    human_box  = human_box_ori.copy()
    object_box = object_box_ori.copy()
    
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]    

    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    
    if height > width:
        ratio = 'height'
    else:
        ratio = 'width'
        
    # shift the top-left corner to (0,0)
    
    human_box[0] -= InteractionPattern[0]
    human_box[2] -= InteractionPattern[0]
    human_box[1] -= InteractionPattern[1]
    human_box[3] -= InteractionPattern[1]    
    object_box[0] -= InteractionPattern[0]
    object_box[2] -= InteractionPattern[0]
    object_box[1] -= InteractionPattern[1]
    object_box[3] -= InteractionPattern[1] 

    if ratio == 'height': # height is larger than width
        
        human_box[0] = 0 + size * human_box[0] / height
        human_box[1] = 0 + size * human_box[1] / height
        human_box[2] = (size * width / height - 1) - size * (width  - 1 - human_box[2]) / height
        human_box[3] = (size - 1)                  - size * (height - 1 - human_box[3]) / height

        object_box[0] = 0 + size * object_box[0] / height
        object_box[1] = 0 + size * object_box[1] / height
        object_box[2] = (size * width / height - 1) - size * (width  - 1 - object_box[2]) / height
        object_box[3] = (size - 1)                  - size * (height - 1 - object_box[3]) / height
        
        # Need to shift horizontally  
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[3] == 63) & (InteractionPattern[2] <= 63)
        if human_box[3] > object_box[3]:
            human_box[3] = size - 1
        else:
            object_box[3] = size - 1



        shift = size / 2 - (InteractionPattern[2] + 1) / 2 
        human_box += [shift, 0 , shift, 0]
        object_box += [shift, 0 , shift, 0]
     
    else: # width is larger than height

        human_box[0] = 0 + size * human_box[0] / width
        human_box[1] = 0 + size * human_box[1] / width
        human_box[2] = (size - 1)                  - size * (width  - 1 - human_box[2]) / width
        human_box[3] = (size * height / width - 1) - size * (height - 1 - human_box[3]) / width
        

        object_box[0] = 0 + size * object_box[0] / width
        object_box[1] = 0 + size * object_box[1] / width
        object_box[2] = (size - 1)                  - size * (width  - 1 - object_box[2]) / width
        object_box[3] = (size * height / width - 1) - size * (height - 1 - object_box[3]) / width
        
        # Need to shift vertically 
        InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
        
        
        #assert (InteractionPattern[0] == 0) & (InteractionPattern[1] == 0) & (InteractionPattern[2] == 63) & (InteractionPattern[3] <= 63)
        

        if human_box[2] > object_box[2]:
            human_box[2] = size - 1
        else:
            object_box[2] = size - 1

        shift = size / 2 - (InteractionPattern[3] + 1) / 2 
        
        human_box = human_box + [0, shift, 0 , shift]
        object_box = object_box + [0, shift, 0 , shift]

 
    return np.round(human_box), np.round(object_box)

def Get_next_sp(human_box, object_box):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')
    
    Pattern = np.zeros((64,64,2))
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1


    return Pattern

def bb_IOU(boxA, boxB):

    ixmin = np.maximum(boxA[0], boxB[0])
    iymin = np.maximum(boxA[1], boxB[1])
    ixmax = np.minimum(boxA[2], boxB[2])
    iymax = np.minimum(boxA[3], boxB[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((boxB[2] - boxB[0] + 1.) * (boxB[3] - boxB[1] + 1.) +
           (boxA[2] - boxA[0] + 1.) *
           (boxA[3] - boxA[1] + 1.) - inters)

    overlaps = inters / uni
    return overlaps

# def Augmented_box(bbox, shape, image_id, augment = 15, break_flag = True):

#     thres_ = 0.7

#     box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
#     box = box.astype(np.float64)
        
#     count = 0
#     time_count = 0
#     while count < augment:
        
#         time_count += 1
#         height = bbox[3] - bbox[1]
#         width  = bbox[2] - bbox[0]

#         height_cen = (bbox[3] + bbox[1]) / 2
#         width_cen  = (bbox[2] + bbox[0]) / 2

#         ratio = 1 + randint(-10,10) * 0.01

#         height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
#         width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

#         H_0 = max(0, width_cen + width_shift - ratio * width / 2)
#         H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
#         H_1 = max(0, height_cen + height_shift - ratio * height / 2)
#         H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
#         if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
#             box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
#             box  = np.concatenate((box,     box_),     axis=0)
#             count += 1
#         if break_flag == True and time_count > 150:
#             return box
            
#     return box

def Augmented_box(bbox, shape, image_id, augment=15, break_flag=True, mode=True):
    # mode = True for instance human box and object box
    # False for human part box
    thres_ = 0.7

    if mode == False:
        bbox = np.array([bbox[1], bbox[2], bbox[3], bbox[4]])

    box = np.array([0, bbox[0], bbox[1], bbox[2], bbox[3]]).reshape(1, 5)
    box = box.astype(np.float64)

    count = 0
    time_count = 0
    while count < augment:

        time_count += 1
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10, 10) * 0.01

        height_shift = randint(-np.floor(height), np.floor(height)) * 0.1
        width_shift = randint(-np.floor(width), np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)

        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1, 5)
            box = np.concatenate((box, box_), axis=0)
            count += 1
        if break_flag == True and time_count > 150:
            return box

    return box


def Generate_action(action_list):
    action_ = np.zeros(29)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,29)
    return action_


def Generate_action_HICO(action_list):
    action_ = np.zeros(600)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,600)
    return action_



##############################################################################
##############################################################################
# tools added
##############################################################################
##############################################################################

def Generate_action_30(action_list):
    action_ = np.zeros(30)
    for GT_idx in action_list:
        action_[GT_idx] = 1
    action_ = action_.reshape(1,30)
    return action_

def draw_relation(human_pattern, joints, size = 64):

    joint_relation = [[1,3],[2,4],[0,1],[0,2],[0,17],[5,17],[6,17],[5,7],[6,8],[7,9],[8,10],[11,17],[12,17],[11,13],[12,14],[13,15],[14,16]]
    color = [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]
    skeleton = np.zeros((size, size, 1), dtype="float32")

    for i in range(len(joint_relation)):
        cv2.line(skeleton, tuple(joints[joint_relation[i][0]]), tuple(joints[joint_relation[i][1]]), (color[i]))
    
    # cv2.rectangle(skeleton, (int(human_pattern[0]), int(human_pattern[1])), (int(human_pattern[2]), int(human_pattern[3])), (255))
    # cv2.imshow("Joints", skeleton)
    # cv2.waitKey(0)
    # print(skeleton[:,:,0])

    return skeleton

def get_skeleton(human_box, human_pose, human_pattern, num_joints = 17, size = 64):
    width = human_box[2] - human_box[0] + 1
    height = human_box[3] - human_box[1] + 1
    pattern_width = human_pattern[2] - human_pattern[0] + 1
    pattern_height = human_pattern[3] - human_pattern[1] + 1
    joints = np.zeros((num_joints + 1, 2), dtype='int32')

    for i in range(num_joints):
        joint_x, joint_y, joint_score = human_pose[3 * i : 3 * (i + 1)]
        x_ratio = (joint_x - human_box[0]) / float(width)
        y_ratio = (joint_y - human_box[1]) / float(height)
        joints[i][0] = min(size - 1, int(round(x_ratio * pattern_width + human_pattern[0])))
        joints[i][1] = min(size - 1, int(round(y_ratio * pattern_height + human_pattern[1])))
    joints[num_joints] = (joints[5] + joints[6]) / 2

    return draw_relation(human_pattern, joints)


# def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    
#     InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
#     height = InteractionPattern[3] - InteractionPattern[1] + 1
#     width = InteractionPattern[2] - InteractionPattern[0] + 1
#     if height > width:
#         H, O = bbox_trans(human_box, object_box, 'height')
#     else:
#         H, O  = bbox_trans(human_box, object_box, 'width')

#     Pattern = np.zeros((64,64,2), dtype='float32')
#     Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
#     Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

#     if human_pose != None and len(human_pose) == 51:
#         skeleton = get_skeleton(human_box, human_pose, H, num_joints)
#     else:
#         skeleton = np.zeros((64,64,1), dtype='float32')
#         skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05

#     Pattern = np.concatenate((Pattern, skeleton), axis=2)

#     return Pattern


def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]),
                          max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64, 64, 2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 1
    Pattern[int(O[1]):int(O[3]) + 1, int(O[0]):int(O[2]) + 1, 1] = 1

    if human_pose is None:
        skeleton = np.zeros((64, 64, 1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 0.05
    elif len(human_pose) != 51:
        skeleton = np.zeros((64, 64, 1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1, int(H[0]):int(H[2]) + 1, 0] = 0.05
    else:
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)

    Pattern = np.concatenate((Pattern, skeleton), axis=2)

    return Pattern

##############################################################################
# for vcoco with pose pattern
##############################################################################


def Get_Next_Instance_HO_Neg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes_solo']= Human_augmented_solo
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, Human_augmented_solo, action_H = [], [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        Human_augmented_solo.extend(Human_augmented_temp)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)
    Human_augmented_solo.extend(Human_augmented_temp)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label


def Get_Next_Instance_HO_spNeg_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['Hsp_boxes']   = Human_augmented_sp
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_sp'] = action_sp
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_H']  = action_H
    blobs['Mask_sp']     = mask_sp
    blobs['Mask_HO']     = mask_HO
    blobs['Mask_H']      = mask_H
    blobs['sp']          = Pattern
    blobs['H_num']       = len(action_H)
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_spNeg_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_H__temp = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    action_sp = np.array(action_HO).copy()

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label


##############################################################################
# for hico with pose map
##############################################################################


def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_HICO_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO = [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_HO__temp = Generate_action_HICO(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    num_pos = len(Human_augmented)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 3) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label

def Image_Resize(GT):
    image_name = GT[0][0]  # image_name is a string
    image_dataset = GT[0][4]

    im_file = image_folder_list[image_dataset] + image_name  # new path added
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape

    ratio = 1.0
    if im_shape[0] > im_shape[1]:
        if im_shape[0] > 512:
            ratio = 512. / im_shape[0]
    else:
        if im_shape[1] > 512:
            ratio = 512. / im_shape[1]

    im_orig = cv2.resize(im_orig, (
        int(ratio * im_shape[1]), int(ratio * im_shape[0])))  # pay attention:width and height inversed for cv2.imread
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape, ratio


##############################################################################
# for hico with pose map + with local part
##############################################################################


def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2_with_part(trainval_GT, Trainval_Neg, image_id, Pos_augment,
                                                                  Neg_select, Data_length):
    GT = trainval_GT[image_id]  # image_id is int number
    im_orig, im_shape, ratio = Image_Resize(GT)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, Part0, Part1, Part2, Part3, Part4, Part5, Part6, Part7, Part8, Part9, binary_label_10v = Augmented_HO_Neg_HICO_pose_pattern_version2_with_part(
        GT, Trainval_Neg, im_shape, Pos_augment, Neg_select, ratio)

    blobs = {}
    blobs['image'] = im_orig
    blobs['H_boxes'] = Human_augmented
    blobs['O_boxes'] = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_O'] = np.matmul(action_HO, hico_obj_mask) > 0
    blobs['gt_class_O'] = blobs['gt_class_O'].astype(np.float32)
    blobs['sp'] = Pattern
    blobs['H_num'] = num_pos
    blobs['binary_label'] = binary_label
    blobs['Part0'] = Part0
    blobs['Part1'] = Part1
    blobs['Part2'] = Part2
    blobs['Part3'] = Part3
    blobs['Part4'] = Part4
    blobs['Part5'] = Part5
    blobs['Part6'] = Part6
    blobs['Part7'] = Part7
    blobs['Part8'] = Part8
    blobs['Part9'] = Part9
    blobs['binary_label_10v'] = binary_label_10v

    return blobs


def Correct_box(Part_box_all, shape):
    for Part in Part_box_all:
        if Part[1] < 0.:
            Part[1] = 0.
        if Part[2] < 0.:
            Part[2] = 0.
        if Part[3] >= shape[1] or Part[3] < Part[1]:
            Part[3] = shape[1] - 1.
        if Part[4] >= shape[0] or Part[4] < Part[2]:
            Part[4] = shape[0] - 1.
    return Part_box_all


def Augmented_HO_Neg_HICO_pose_pattern_version2_with_part(GT, Trainval_Neg, shape, Pos_augment, Neg_select, ratio):
    # added: resize the box by parameter-ratio
    # add human local part
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1)
    image_id = GT[0][0]  # here it is a string

    Human_augmented, Object_augmented, action_HO = [], [], []
    Pattern = np.empty((0, 64, 64, 3), dtype=np.float32)
    Part0, Part1, Part2, Part3, Part4, Part5, Part6, Part7, Part8, Part9 = [], [], [], [], [], [], [], [], [], []
    hp10_list_pos = []

    for i in range(GT_count - 1):
        Human = np.round(np.array(GT[i][2]).astype(np.float32) * ratio)
        Object = np.round(np.array(GT[i][3]).astype(np.float32) * ratio)
        Part_box = np.round(np.array(GT[i][8]).astype(np.float32) * ratio)

        Human_augmented_temp = Augmented_box(Human, shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        if Part_box is None or Part_box.shape != (10, 5):
            # use human instance instead
            Part_0 = Augmented_box(Human, shape, image_id, aug_all)
            Part_1 = Augmented_box(Human, shape, image_id, aug_all)
            Part_2 = Augmented_box(Human, shape, image_id, aug_all)
            Part_3 = Augmented_box(Human, shape, image_id, aug_all)
            Part_4 = Augmented_box(Human, shape, image_id, aug_all)
            Part_5 = Augmented_box(Human, shape, image_id, aug_all)
            Part_6 = Augmented_box(Human, shape, image_id, aug_all)
            Part_7 = Augmented_box(Human, shape, image_id, aug_all)
            Part_8 = Augmented_box(Human, shape, image_id, aug_all)
            Part_9 = Augmented_box(Human, shape, image_id, aug_all)
        else:
            Part_box = Correct_box(Part_box, shape)
            Part_0 = Augmented_box(Part_box[0], shape, image_id, aug_all, True, False)
            Part_1 = Augmented_box(Part_box[1], shape, image_id, aug_all, True, False)
            Part_2 = Augmented_box(Part_box[2], shape, image_id, aug_all, True, False)
            Part_3 = Augmented_box(Part_box[3], shape, image_id, aug_all, True, False)
            Part_4 = Augmented_box(Part_box[4], shape, image_id, aug_all, True, False)
            Part_5 = Augmented_box(Part_box[5], shape, image_id, aug_all, True, False)
            Part_6 = Augmented_box(Part_box[6], shape, image_id, aug_all, True, False)
            Part_7 = Augmented_box(Part_box[7], shape, image_id, aug_all, True, False)
            Part_8 = Augmented_box(Part_box[8], shape, image_id, aug_all, True, False)
            Part_9 = Augmented_box(Part_box[9], shape, image_id, aug_all, True, False)

        length_min = min(len(Human_augmented_temp), len(Object_augmented_temp), len(Part_0), len(Part_1), len(Part_2),
                         len(Part_3), len(Part_4), len(Part_5), len(Part_6), len(Part_7), len(Part_8), len(Part_9))

        Human_augmented_temp = Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
        Part_0 = Part_0[:length_min]
        Part_1 = Part_1[:length_min]
        Part_2 = Part_2[:length_min]
        Part_3 = Part_3[:length_min]
        Part_4 = Part_4[:length_min]
        Part_5 = Part_5[:length_min]
        Part_6 = Part_6[:length_min]
        Part_7 = Part_7[:length_min]
        Part_8 = Part_8[:length_min]
        Part_9 = Part_9[:length_min]

        action_HO__temp = Generate_action_HICO(GT[i][1])
        action_HO_temp = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        hp10_list__temp = np.zeros(10)
        hp10_list__temp[list(GT[i][9]['hp10_list'])] = 1.
        hp10_list_temp = hp10_list__temp
        for j in range(length_min - 1):
            hp10_list_temp = np.concatenate((hp10_list_temp, hp10_list__temp), axis=0)

        for j in range(length_min):
            if GT[i][7] is None:
                Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:],
                                                 None).reshape(1, 64, 64, 3)
            else:
                Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:],
                                                 np.array(GT[i][7]) * ratio).reshape(1, 64, 64, 3)
            Pattern = np.concatenate((Pattern, Pattern_), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        hp10_list_pos.extend(hp10_list_temp)
        Part0.extend(Part_0)
        Part1.extend(Part_1)
        Part2.extend(Part_2)
        Part3.extend(Part_3)
        Part4.extend(Part_4)
        Part5.extend(Part_5)
        Part6.extend(Part_6)
        Part7.extend(Part_7)
        Part8.extend(Part_8)
        Part9.extend(Part_9)

    Human = np.round(np.array(GT[GT_count - 1][2]).astype(np.float32) * ratio)
    Object = np.round(np.array(GT[GT_count - 1][3]).astype(np.float32) * ratio)
    Part_box = np.round(np.array(GT[GT_count - 1][8]).astype(np.float32) * ratio)

    Human_augmented_temp = Augmented_box(Human, shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    if Part_box is None or Part_box.shape != (10, 5):
        Part_0 = Augmented_box(Human, shape, image_id, aug_last)
        Part_1 = Augmented_box(Human, shape, image_id, aug_last)
        Part_2 = Augmented_box(Human, shape, image_id, aug_last)
        Part_3 = Augmented_box(Human, shape, image_id, aug_last)
        Part_4 = Augmented_box(Human, shape, image_id, aug_last)
        Part_5 = Augmented_box(Human, shape, image_id, aug_last)
        Part_6 = Augmented_box(Human, shape, image_id, aug_last)
        Part_7 = Augmented_box(Human, shape, image_id, aug_last)
        Part_8 = Augmented_box(Human, shape, image_id, aug_last)
        Part_9 = Augmented_box(Human, shape, image_id, aug_last)
    else:
        Part_box = Correct_box(Part_box, shape)
        # print(Part_box)
        Part_0 = Augmented_box(Part_box[0], shape, image_id, aug_last, True, False)
        Part_1 = Augmented_box(Part_box[1], shape, image_id, aug_last, True, False)
        Part_2 = Augmented_box(Part_box[2], shape, image_id, aug_last, True, False)
        Part_3 = Augmented_box(Part_box[3], shape, image_id, aug_last, True, False)
        Part_4 = Augmented_box(Part_box[4], shape, image_id, aug_last, True, False)
        Part_5 = Augmented_box(Part_box[5], shape, image_id, aug_last, True, False)
        Part_6 = Augmented_box(Part_box[6], shape, image_id, aug_last, True, False)
        Part_7 = Augmented_box(Part_box[7], shape, image_id, aug_last, True, False)
        Part_8 = Augmented_box(Part_box[8], shape, image_id, aug_last, True, False)
        Part_9 = Augmented_box(Part_box[9], shape, image_id, aug_last, True, False)

    length_min = min(len(Human_augmented_temp), len(Object_augmented_temp), len(Part_0), len(Part_1), len(Part_2),
                     len(Part_3), len(Part_4), len(Part_5), len(Part_6), len(Part_7), len(Part_8), len(Part_9))

    Human_augmented_temp = Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]
    Part_0 = Part_0[:length_min]
    Part_1 = Part_1[:length_min]
    Part_2 = Part_2[:length_min]
    Part_3 = Part_3[:length_min]
    Part_4 = Part_4[:length_min]
    Part_5 = Part_5[:length_min]
    Part_6 = Part_6[:length_min]
    Part_7 = Part_7[:length_min]
    Part_8 = Part_8[:length_min]
    Part_9 = Part_9[:length_min]

    action_HO__temp = Generate_action_HICO(GT[GT_count - 1][1])
    action_HO_temp = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    hp10_list__temp = np.zeros(10)
    hp10_list__temp[list(GT[GT_count - 1][9]['hp10_list'])] = 1.
    hp10_list_temp = hp10_list__temp
    for j in range(length_min - 1):
        hp10_list_temp = np.concatenate((hp10_list_temp, hp10_list__temp), axis=0)

    for j in range(length_min):
        if GT[GT_count - 1][7] is None:
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], None).reshape(1,
                                                                                                                      64,
                                                                                                                      64,
                                                                                                                      3)
        else:
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:],
                                             np.array(GT[GT_count - 1][7]) * ratio).reshape(1, 64, 64, 3)
        Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    hp10_list_pos.extend(hp10_list_temp)
    Part0.extend(Part_0)
    Part1.extend(Part_1)
    Part2.extend(Part_2)
    Part3.extend(Part_3)
    Part4.extend(Part_4)
    Part5.extend(Part_5)
    Part6.extend(Part_6)
    Part7.extend(Part_7)
    Part8.extend(Part_8)
    Part9.extend(Part_9)

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:
        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented = np.concatenate((Human_augmented, (
                        ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, (
                        ratio * np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]])).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Part_box = np.array(Neg[7])
                if Part_box is None or Part_box.shape != (10, 5):
                    Part0 = np.concatenate(
                        (Part0, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part1 = np.concatenate(
                        (Part1, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part2 = np.concatenate(
                        (Part2, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part3 = np.concatenate(
                        (Part3, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part4 = np.concatenate(
                        (Part4, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part5 = np.concatenate(
                        (Part5, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part6 = np.concatenate(
                        (Part6, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part7 = np.concatenate(
                        (Part7, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part8 = np.concatenate(
                        (Part8, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part9 = np.concatenate(
                        (Part9, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                else:
                    Part_box = Correct_box(Part_box, shape)
                    Part0 = np.concatenate((Part0, (ratio * np.array(
                        [0, Part_box[0][1], Part_box[0][2], Part_box[0][3], Part_box[0][4]])).reshape(1, 5)), axis=0)
                    Part1 = np.concatenate((Part1, (ratio * np.array(
                        [0, Part_box[1][1], Part_box[1][2], Part_box[1][3], Part_box[1][4]])).reshape(1, 5)), axis=0)
                    Part2 = np.concatenate((Part2, (ratio * np.array(
                        [0, Part_box[2][1], Part_box[2][2], Part_box[2][3], Part_box[2][4]])).reshape(1, 5)), axis=0)
                    Part3 = np.concatenate((Part3, (ratio * np.array(
                        [0, Part_box[3][1], Part_box[3][2], Part_box[3][3], Part_box[3][4]])).reshape(1, 5)), axis=0)
                    Part4 = np.concatenate((Part4, (ratio * np.array(
                        [0, Part_box[4][1], Part_box[4][2], Part_box[4][3], Part_box[4][4]])).reshape(1, 5)), axis=0)
                    Part5 = np.concatenate((Part5, (ratio * np.array(
                        [0, Part_box[5][1], Part_box[5][2], Part_box[5][3], Part_box[5][4]])).reshape(1, 5)), axis=0)
                    Part6 = np.concatenate((Part6, (ratio * np.array(
                        [0, Part_box[6][1], Part_box[6][2], Part_box[6][3], Part_box[6][4]])).reshape(1, 5)), axis=0)
                    Part7 = np.concatenate((Part7, (ratio * np.array(
                        [0, Part_box[7][1], Part_box[7][2], Part_box[7][3], Part_box[7][4]])).reshape(1, 5)), axis=0)
                    Part8 = np.concatenate((Part8, (ratio * np.array(
                        [0, Part_box[8][1], Part_box[8][2], Part_box[8][3], Part_box[8][4]])).reshape(1, 5)), axis=0)
                    Part9 = np.concatenate((Part9, (ratio * np.array(
                        [0, Part_box[9][1], Part_box[9][2], Part_box[9][3], Part_box[9][4]])).reshape(1, 5)), axis=0)

                if Neg[6] is None:
                    Pattern_ = Get_next_sp_with_pose(ratio * np.array(Neg[2], dtype='float64'),
                                                     ratio * np.array(Neg[3], dtype='float64'), None).reshape(1, 64, 64,
                                                                                                              3)
                else:
                    Pattern_ = Get_next_sp_with_pose(ratio * np.array(Neg[2], dtype='float64'),
                                                     ratio * np.array(Neg[3], dtype='float64'),
                                                     ratio * np.array(Neg[6])).reshape(1, 64, 64, 3)
                Pattern = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented = np.concatenate((Human_augmented, (
                        ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, (
                        ratio * np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]])).reshape(1, 5)), axis=0)
                action_HO = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                Part_box = np.array(Neg[7])
                if Part_box is None or Part_box.shape != (10, 5):
                    Part0 = np.concatenate(
                        (Part0, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part1 = np.concatenate(
                        (Part1, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part2 = np.concatenate(
                        (Part2, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part3 = np.concatenate(
                        (Part3, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part4 = np.concatenate(
                        (Part4, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part5 = np.concatenate(
                        (Part5, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part6 = np.concatenate(
                        (Part6, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part7 = np.concatenate(
                        (Part7, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part8 = np.concatenate(
                        (Part8, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                    Part9 = np.concatenate(
                        (Part9, (ratio * np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]])).reshape(1, 5)),
                        axis=0)
                else:
                    Part_box = Correct_box(Part_box, shape)
                    Part0 = np.concatenate((Part0, (ratio * np.array(
                        [0, Part_box[0][1], Part_box[0][2], Part_box[0][3], Part_box[0][4]])).reshape(1, 5)), axis=0)
                    Part1 = np.concatenate((Part1, (ratio * np.array(
                        [0, Part_box[1][1], Part_box[1][2], Part_box[1][3], Part_box[1][4]])).reshape(1, 5)), axis=0)
                    Part2 = np.concatenate((Part2, (ratio * np.array(
                        [0, Part_box[2][1], Part_box[2][2], Part_box[2][3], Part_box[2][4]])).reshape(1, 5)), axis=0)
                    Part3 = np.concatenate((Part3, (ratio * np.array(
                        [0, Part_box[3][1], Part_box[3][2], Part_box[3][3], Part_box[3][4]])).reshape(1, 5)), axis=0)
                    Part4 = np.concatenate((Part4, (ratio * np.array(
                        [0, Part_box[4][1], Part_box[4][2], Part_box[4][3], Part_box[4][4]])).reshape(1, 5)), axis=0)
                    Part5 = np.concatenate((Part5, (ratio * np.array(
                        [0, Part_box[5][1], Part_box[5][2], Part_box[5][3], Part_box[5][4]])).reshape(1, 5)), axis=0)
                    Part6 = np.concatenate((Part6, (ratio * np.array(
                        [0, Part_box[6][1], Part_box[6][2], Part_box[6][3], Part_box[6][4]])).reshape(1, 5)), axis=0)
                    Part7 = np.concatenate((Part7, (ratio * np.array(
                        [0, Part_box[7][1], Part_box[7][2], Part_box[7][3], Part_box[7][4]])).reshape(1, 5)), axis=0)
                    Part8 = np.concatenate((Part8, (ratio * np.array(
                        [0, Part_box[8][1], Part_box[8][2], Part_box[8][3], Part_box[8][4]])).reshape(1, 5)), axis=0)
                    Part9 = np.concatenate((Part9, (ratio * np.array(
                        [0, Part_box[9][1], Part_box[9][2], Part_box[9][3], Part_box[9][4]])).reshape(1, 5)), axis=0)

                if Neg[6] is None:
                    Pattern_ = Get_next_sp_with_pose(ratio * np.array(Neg[2], dtype='float64'),
                                                     ratio * np.array(Neg[3], dtype='float64'), None).reshape(1, 64, 64,
                                                                                                              3)
                else:
                    Pattern_ = Get_next_sp_with_pose(ratio * np.array(Neg[2], dtype='float64'),
                                                     ratio * np.array(Neg[3], dtype='float64'),
                                                     ratio * np.array(Neg[6])).reshape(1, 64, 64, 3)
                Pattern = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern = Pattern.reshape(num_pos_neg, 64, 64, 3)
    Human_augmented = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5)
    Object_augmented = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5)
    action_HO = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600)
    Part0 = np.array(Part0, dtype='float32').reshape(num_pos_neg, 5)
    Part1 = np.array(Part1, dtype='float32').reshape(num_pos_neg, 5)
    Part2 = np.array(Part2, dtype='float32').reshape(num_pos_neg, 5)
    Part3 = np.array(Part3, dtype='float32').reshape(num_pos_neg, 5)
    Part4 = np.array(Part4, dtype='float32').reshape(num_pos_neg, 5)
    Part5 = np.array(Part5, dtype='float32').reshape(num_pos_neg, 5)
    Part6 = np.array(Part6, dtype='float32').reshape(num_pos_neg, 5)
    Part7 = np.array(Part7, dtype='float32').reshape(num_pos_neg, 5)
    Part8 = np.array(Part8, dtype='float32').reshape(num_pos_neg, 5)
    Part9 = np.array(Part9, dtype='float32').reshape(num_pos_neg, 5)

    hp10_list_pos = np.array(hp10_list_pos, dtype='int32').reshape(num_pos, 10)

    # 10v -> 6v
    # 10v: 0Right_foot, 1Right_leg, 2Left_leg, 3Left_foot, 4Hip, 5Head, 6Right_hand, 7Right_arm, 8Left_arm, 9Left_hand
    # 6v: 0foot, 1leg, 2hip, 3hand, 4arm, 5head
    # 0 -> (0,3), 1->(1,2), 2->(4), 3->(6,9), 4->(7,8), 5->5

    binary_label_10v = np.zeros((num_pos_neg, 10, 2), dtype='int32')
    for i in range(num_pos):
        for j in range(10):
            binary_label_10v[i, j, 0] = hp10_list_pos[i, j]
            binary_label_10v[i, j, 1] = 1 - binary_label_10v[i, j, 0]
    for i in range(num_pos + 1, num_pos_neg):
        for j in range(10):
            binary_label_10v[i, j, 1] = 1

    binary_label = np.zeros((num_pos_neg, 2), dtype='int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1  # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1  # neg is at 1

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, Part0, Part1, Part2, Part3, Part4, Part5, Part6, Part7, Part8, Part9, binary_label_10v
