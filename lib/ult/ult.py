# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
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

def Augmented_box(bbox, shape, image_id, augment = 15, break_flag = True):

    thres_ = 0.7

    box = np.array([0, bbox[0],  bbox[1],  bbox[2],  bbox[3]]).reshape(1,5)
    box = box.astype(np.float64)
        
    count = 0
    time_count = 0
    while count < augment:
        
        time_count += 1
        height = bbox[3] - bbox[1]
        width  = bbox[2] - bbox[0]

        height_cen = (bbox[3] + bbox[1]) / 2
        width_cen  = (bbox[2] + bbox[0]) / 2

        ratio = 1 + randint(-10,10) * 0.01

        height_shift = randint(-np.floor(height),np.floor(height)) * 0.1
        width_shift  = randint(-np.floor(width),np.floor(width)) * 0.1

        H_0 = max(0, width_cen + width_shift - ratio * width / 2)
        H_2 = min(shape[1] - 1, width_cen + width_shift + ratio * width / 2)
        H_1 = max(0, height_cen + height_shift - ratio * height / 2)
        H_3 = min(shape[0] - 1, height_cen + height_shift + ratio * height / 2)
        
        
        if bb_IOU(bbox, np.array([H_0, H_1, H_2, H_3])) > thres_:
            box_ = np.array([0, H_0, H_1, H_2, H_3]).reshape(1,5)
            box  = np.concatenate((box,     box_),     axis=0)
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

def Get_next_sp_with_pose(human_box, object_box, human_pose, num_joints=17):
    
    InteractionPattern = [min(human_box[0], object_box[0]), min(human_box[1], object_box[1]), max(human_box[2], object_box[2]), max(human_box[3], object_box[3])]
    height = InteractionPattern[3] - InteractionPattern[1] + 1
    width = InteractionPattern[2] - InteractionPattern[0] + 1
    if height > width:
        H, O = bbox_trans(human_box, object_box, 'height')
    else:
        H, O  = bbox_trans(human_box, object_box, 'width')

    Pattern = np.zeros((64,64,2), dtype='float32')
    Pattern[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 1
    Pattern[int(O[1]):int(O[3]) + 1,int(O[0]):int(O[2]) + 1,1] = 1

    if human_pose != None and len(human_pose) == 51:
        skeleton = get_skeleton(human_box, human_pose, H, num_joints)
    else:
        skeleton = np.zeros((64,64,1), dtype='float32')
        skeleton[int(H[1]):int(H[3]) + 1,int(H[0]):int(H[2]) + 1,0] = 0.05

    Pattern = np.concatenate((Pattern, skeleton), axis=2)

    return Pattern

##############################################################################
##############################################################################














##############################################################################
##############################################################################
##############################################################################
##############################################################################
# for vcoco
##############################################################################
##############################################################################
##############################################################################
##############################################################################


# for vcoco
def Get_Next_Instance_HO_Neg(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_Neg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()



    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)
    

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label



# for vcoco
def Get_Next_Instance_HO_spNeg(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_spNeg(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action(GT[1])
    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label




##############################################################################################################

# for vcoco label with no_interaction (action_id = 30)



def Get_Next_Instance_HO_Neg_30(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_30(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_Neg_30(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()



    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)
    

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label




def Get_Next_Instance_HO_spNeg_30(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_30(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_30(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action_30(GT[1])
    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label









##############################################################################
##############################################################################
# for vcoco with pose vector
##############################################################################
##############################################################################




##############################################################################

def Get_Next_Instance_HO_Neg_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose = Augmented_HO_Neg_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_Neg_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()


    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)
    

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)
    pose              = pose.reshape(num_pos_neg, 51)

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose


def Get_Next_Instance_HO_spNeg_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose = Augmented_HO_spNeg_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_spNeg_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action(GT[1])
    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose




##############################################################################################################

'''
for vcoco label with no_interaction (action_id = 30)
'''

def Get_Next_Instance_HO_Neg_30_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose = Augmented_HO_Neg_30_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_Neg_30_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose



def Get_Next_Instance_HO_spNeg_30_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose = Augmented_HO_spNeg_30_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_spNeg_30_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action_30(GT[1])
    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose











##############################################################################
##############################################################################
# for vcoco with pose pattern
##############################################################################
##############################################################################




##############################################################################

def Get_Next_Instance_HO_Neg_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_Neg_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()


    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

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


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)


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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label


def Get_Next_Instance_HO_spNeg_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action(GT[1])
    action_HO_ = Generate_action(GT[1])
    action_H_  = Generate_action(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

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

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label


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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label




##############################################################################################################

'''
for vcoco label with no_interaction (action_id = 30)
'''

def Get_Next_Instance_HO_Neg_30_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_30_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_Neg_30_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    

    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)
    Human_augmented_solo = Human_augmented.copy()

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

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


    action_HO = action_HO_
    action_H  = action_H_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label



def Get_Next_Instance_HO_spNeg_30_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_30_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_30_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_sp_ = Generate_action_30(GT[1])
    action_HO_ = Generate_action_30(GT[1])
    action_H_  = Generate_action_30(GT[4])
    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
        
    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

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

    action_sp = action_sp_
    action_HO = action_HO_
    action_H  = action_H_
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        action_sp = np.concatenate((action_sp, action_sp_), axis=0)
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
        action_H  = np.concatenate((action_H,  action_H_),  axis=0)
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label















####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
# for hico
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################




def Get_Next_Instance_HO_Neg_HICO(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

    return blobs

def Augmented_HO_Neg_HICO(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label





########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_520(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(len(Pattern)))
    f.write('\n')
    f.close()

    return blobs

def Augmented_HO_Neg_HICO_520(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label








##############################################################################
##############################################################################
##############################################################################
# for hico with pose vector
##############################################################################
##############################################################################
##############################################################################



def Get_Next_Instance_HO_Neg_HICO_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose = Augmented_HO_Neg_HICO_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['H_pose']      = pose

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

    return blobs

def Augmented_HO_Neg_HICO_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose





########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520_pose_vector(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose = Augmented_HO_Neg_HICO_520_pose_vector(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['H_pose']      = pose

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(len(Pattern)))
    f.write('\n')
    f.close()

    return blobs

def Augmented_HO_Neg_HICO_520_pose_vector(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    if GT[5] == None:
        pose = np.zeros((num_pos, 51), dtype = 'float32')
    else:
        pose = np.array(GT[5], dtype = 'float32').reshape(1,51)
        pose_ = pose
        for i in range(num_pos - 1):
            pose = np.concatenate((pose, pose_), axis=0)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose





##############################################################################
##############################################################################
##############################################################################
# for hico with pose pattern
##############################################################################
##############################################################################
##############################################################################



def Get_Next_Instance_HO_Neg_HICO_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

    return blobs

def Augmented_HO_Neg_HICO_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
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
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label



########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520_pose_pattern(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_520_pose_pattern(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(len(Pattern)))
    f.write('\n')
    f.close()

    return blobs

def Augmented_HO_Neg_HICO_520_pose_pattern(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0]
    Human    = GT[2]
    Object   = GT[3]
    
    action_HO_ = Generate_action_HICO(GT[1])
    action_HO  = action_HO_

    Human_augmented  = Augmented_box(Human,  shape, image_id, Pos_augment)
    Object_augmented = Augmented_box(Object, shape, image_id, Pos_augment)

    Human_augmented  =  Human_augmented[:min(len(Human_augmented),len(Object_augmented))]
    Object_augmented = Object_augmented[:min(len(Human_augmented),len(Object_augmented))]

    num_pos = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)
    for i in range(num_pos):
        Pattern_ = Get_next_sp_with_pose(Human_augmented[i][1:], Object_augmented[i][1:], GT[5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    for i in range(num_pos - 1):
        action_HO = np.concatenate((action_HO, action_HO_), axis=0)
    
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
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label



































#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

# For the use of more positive pairs (version2)

#######################################################################################################
########################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################










#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
# for vcoco
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################



def Get_Next_Instance_HO_Neg_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_Neg_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        action_H__temp  = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

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

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    action_H__temp  = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    Human_augmented_solo = Human_augmented.copy()
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label






























def Get_Next_Instance_HO_Neg_version2_for_hico_520(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, binary_label, H_num = Augmented_HO_Neg_version2_for_hico_520(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes_solo']= Human_augmented_solo
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

# for vcoco
def Augmented_HO_Neg_version2_for_hico_520(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    Human_augmented_solo = Human_augmented.copy()
    
    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(600).reshape(1,600)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    action_HO         = np.array(action_HO, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    
    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, binary_label, num_pos





























# for vcoco
def Get_Next_Instance_HO_spNeg_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_spNeg_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_sp, action_HO, action_H = [], [], [], [], []

    for i in range(GT_count - 1):

        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_sp__temp  = Generate_action(GT[i][1])
        action_sp_temp  = action_sp__temp
        for j in range(length_min - 1):
            action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)

        action_HO_temp = action_sp_temp.copy()

        action_H__temp  = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_sp.extend(action_sp_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_sp__temp  = Generate_action(GT[GT_count - 1][1])
    action_sp_temp  = action_sp__temp
    for j in range(length_min - 1):
        action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)

    action_HO_temp = action_sp_temp.copy()

    action_H__temp  = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_sp.extend(action_sp_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)

    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label




##############################################################################################################
##############################################################################################################
# for vcoco label with no_interaction (action_id = 30)
##############################################################################################################
##############################################################################################################


def Get_Next_Instance_HO_Neg_30_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_30_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_Neg_30_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action_30(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        action_H__temp  = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

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

    action_HO__temp = Generate_action_30(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    action_H__temp  = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################

    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    Human_augmented_solo = Human_augmented.copy()

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)

    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label



def Get_Next_Instance_HO_spNeg_30_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_30_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_30_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_sp, action_HO, action_H = [], [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_sp__temp  = Generate_action_30(GT[i][1])
        action_sp_temp  = action_sp__temp
        for j in range(length_min - 1):
            action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)
    
        action_HO_temp = action_sp_temp.copy()

        action_H__temp  = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_sp.extend(action_sp_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_sp__temp  = Generate_action_30(GT[GT_count - 1][1])
    action_sp_temp  = action_sp__temp
    for j in range(length_min - 1):
        action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)

    action_HO_temp = action_sp_temp.copy()

    action_H__temp  = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_sp.extend(action_sp_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)

    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label




















##############################################################################
##############################################################################
# for vcoco with pose vector
##############################################################################
##############################################################################




##############################################################################

def Get_Next_Instance_HO_Neg_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose = Augmented_HO_Neg_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_Neg_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO, pose, Human_augmented_solo, action_H = [], [], [], [], [], []

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

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32')  
    action_H          = np.array(action_H, dtype = 'int32') 
    pose              = np.array(pose, dtype = 'float32')


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)
    pose              = pose.reshape(num_pos_neg, 51)

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose


def Get_Next_Instance_HO_spNeg_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose = Augmented_HO_spNeg_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_spNeg_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, pose, Human_augmented_solo, action_H = [], [], [], [], [], []

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

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 
    pose              = np.array(pose, dtype = 'float32')


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 29)
    action_HO         = action_HO.reshape(num_pos, 29) 
    action_H          = action_H.reshape( num_pos, 29)
    mask_sp           = mask_sp.reshape(num_pos_neg, 29)
    mask_HO           = mask_HO.reshape(  num_pos, 29)
    mask_H            = mask_H.reshape(   num_pos, 29)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose




##############################################################################################################

'''
for vcoco label with no_interaction (action_id = 30)
'''

def Get_Next_Instance_HO_Neg_30_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose = Augmented_HO_Neg_30_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_Neg_30_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, pose, Human_augmented_solo, action_H = [], [], [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        Human_augmented_solo.extend(Human_augmented_temp)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action_30(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    action_H__temp = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action_30(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 
    pose              = np.array(pose, dtype = 'float32')


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label, pose



def Get_Next_Instance_HO_spNeg_30_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose = Augmented_HO_spNeg_30_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['H_pose']      = pose

    return blobs

def Augmented_HO_spNeg_30_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, pose, Human_augmented_solo, action_H = [], [], [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)
        Human_augmented_solo.extend(Human_augmented_temp)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_H__temp = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action_30(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    action_H__temp = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action_30(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 
    pose              = np.array(pose, dtype = 'float32')


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)
    pose              = pose.reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, pose











##############################################################################
##############################################################################
# for vcoco with pose pattern
##############################################################################
##############################################################################




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
    for i in range(num_pos + 1, num_pos_neg):
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label










def Get_Next_Instance_HO_spNeg_pose_pattern_mask_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, mask_object = Augmented_HO_spNeg_pose_pattern_mask_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['Mask_Object'] = mask_object

    return blobs

def Augmented_HO_spNeg_pose_pattern_mask_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H, mask_object = [], [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]
        Object_class = GT[i][6]
        this_mask = OBJECT_MASK[Object_class]

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
        
        for j in range(length_min):
            mask_object.append(this_mask)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]
    Object_class = GT[GT_count - 1][6]
    this_mask = OBJECT_MASK[Object_class]

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

    for j in range(length_min):
        mask_object.append(this_mask)
    
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
                Object_class = Neg[5]
                this_mask = OBJECT_MASK[Object_class]
                mask_object.append(this_mask)

        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
                Object_class = Neg[5]
                this_mask = OBJECT_MASK[Object_class]
                mask_object.append(this_mask)

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


    mask_object       = np.array(mask_object, dtype = 'int32').reshape(num_pos_neg, 29) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, mask_object











def reverse(mask, alpha, beta): # alpha = 10, beta = 1
    mask_new = []
    for i in range(29):
        if mask[i] == 0:
            mask_new.append(alpha)
        else:
            mask_new.append(beta)
    return mask_new

def Get_Next_Instance_HO_spNeg_pose_pattern_mask_reversed_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, mask_object = Augmented_HO_spNeg_pose_pattern_mask_reversed_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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
    blobs['Mask_Object'] = mask_object

    return blobs

def Augmented_HO_spNeg_pose_pattern_mask_reversed_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 

    Human_augmented, Object_augmented, action_HO, action_H, mask_object = [], [], [], [], []
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float32)

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]
        Object_class = GT[i][6]
        this_mask = reverse(OBJECT_MASK[Object_class], 10, 1)

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
        
        for j in range(length_min):
            mask_object.append(this_mask)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]
    Object_class = GT[GT_count - 1][6]
    this_mask = reverse(OBJECT_MASK[Object_class], 10, 1)

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

    for j in range(length_min):
        mask_object.append(this_mask)
    
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
                Object_class = Neg[5]
                this_mask = reverse(OBJECT_MASK[Object_class], 10, 1)
                mask_object.append(this_mask)

        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
                Object_class = Neg[5]
                this_mask = reverse(OBJECT_MASK[Object_class], 10, 1)
                mask_object.append(this_mask)

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


    mask_object       = np.array(mask_object, dtype = 'int32').reshape(num_pos_neg, 29) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label, mask_object














def Get_Next_Instance_HO_spNeg_pose_pattern_negfull_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_pose_pattern_negfull_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_pose_pattern_negfull_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
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
            Neg_length = len(Trainval_Neg[image_id])
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][i % Neg_length]
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label













##############################################################################################################

'''
for vcoco label with no_interaction (action_id = 30)
'''

def Get_Next_Instance_HO_Neg_30_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_30_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_Neg_30_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
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

        action_H__temp = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action_30(GT[i][1])
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

    action_H__temp = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action_30(GT[GT_count - 1][1])
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


    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Human_augmented_solo = np.array(Human_augmented_solo, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented_solo = Human_augmented_solo.reshape( -1, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_HO           = mask_HO.reshape(  num_pos_neg, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label



def Get_Next_Instance_HO_spNeg_30_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_30_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

def Augmented_HO_spNeg_30_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
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

        action_H__temp = Generate_action_30(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)
    
        action_HO__temp = Generate_action_30(GT[i][1])
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

    action_H__temp = Generate_action_30(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    action_HO__temp = Generate_action_30(GT[GT_count - 1][1])
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

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1,1]).reshape(1,30)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,30)
    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]).reshape(1,30)), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented_sp= Human_augmented.reshape( num_pos_neg, 5) 
    Human_augmented   = Human_augmented[:num_pos].reshape( num_pos, 5) 
    Object_augmented  = Object_augmented[:num_pos].reshape(num_pos, 5)
    action_sp         = action_sp.reshape(num_pos_neg, 30)
    action_HO         = action_HO.reshape(num_pos, 30) 
    action_H          = action_H.reshape( num_pos, 30)
    mask_sp           = mask_sp.reshape(num_pos_neg, 30)
    mask_HO           = mask_HO.reshape(  num_pos, 30)
    mask_H            = mask_H.reshape(   num_pos, 30)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label


















#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
# For hico
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################


def Get_Next_Instance_HO_Neg_HICO_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    return blobs


def Augmented_HO_Neg_HICO_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    #######################################################################################

    num_pos = len(Human_augmented)
    
    if image_id in Trainval_Neg:
        if len(Trainval_Neg[image_id]) < Neg_select:
            for Neg in Trainval_Neg[image_id]:

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype = 'float32')
    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = np.array(Human_augmented, dtype = 'float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype = 'float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype = 'int32').reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label






########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_520_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_HICO_520_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype = 'float32')
    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = np.array(Human_augmented, dtype = 'float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype = 'float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype = 'int32').reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    
    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label





##############################################################################
##############################################################################
##############################################################################
# for hico with pose vector
##############################################################################
##############################################################################
##############################################################################



def Get_Next_Instance_HO_Neg_HICO_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose = Augmented_HO_Neg_HICO_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['H_pose']      = pose

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

    return blobs

def Augmented_HO_Neg_HICO_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO, pose = [], [], [], []

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

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600) 
    pose              = np.array(pose, dtype='float32').reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose





########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520_pose_vector_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose = Augmented_HO_Neg_HICO_520_pose_vector_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['H_pose']      = pose

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(len(Pattern)))
    f.write('\n')
    f.close()

    return blobs

def Augmented_HO_Neg_HICO_520_pose_vector_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO, pose = [], [], [], []

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

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
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
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i]]

                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600) 
    pose              = np.array(pose, dtype='float32').reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose







##############################################################################
##############################################################################
##############################################################################
# for hico with pose pattern
##############################################################################
##############################################################################
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

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(num_pos_neg))
    # f.write('\n')
    # f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label









def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2_for_classification(trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select):

    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, gt_600 = Augmented_HO_Neg_HICO_pose_pattern_version2_for_classification(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['gt_600']      = gt_600

    return blobs

def Augmented_HO_Neg_HICO_pose_pattern_version2_for_classification(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select):
    GT_info = trainval_GT[image_id]
    gt_list = GT_info['gt_list']
    pair_info = GT_info['pair_info']
    pair_num = len(pair_info)

    if pair_num >= Pos_augment:
        List = random.sample(range(pair_num), Pos_augment)
        GT = []
        for i in range(Pos_augment):
            GT.append(pair_info[List[i]])
    else:
        GT = pair_info
        for i in range(Pos_augment - pair_num):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    Object_augmented = np.empty((0, 5), dtype=np.float64)
    action_HO = np.empty((0, 600), dtype=np.float64)
    Pattern   = np.empty((0, 64, 64, 3), dtype=np.float64)

    for i in range(Pos_augment):
        Human    = GT[i][2]
        Object   = GT[i][3]
        Human_augmented = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        Object_augmented = np.concatenate((Object_augmented, np.array([0, Object[0],  Object[1],  Object[2],  Object[3]]).reshape(1,5).astype(np.float64)), axis=0)
        action_HO = np.concatenate((action_HO, Generate_action_HICO(GT[i][1])), axis=0)
        Pattern  = np.concatenate((Pattern, Get_next_sp_with_pose(np.array(Human, dtype='float64'), np.array(Object, dtype='float64'), GT[i][5]).reshape(1, 64, 64, 3)), axis=0)

    num_pos = Pos_augment
    
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
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype='int64')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1
    
    gt_600 = np.zeros(600, dtype='int64')
    gt_600[gt_list] = 1
    gt_600 = gt_600.reshape((1,600))

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, gt_600
















########################################################################################################
########################################################################################################
# for hico 520-80
########################################################################################################
########################################################################################################


def Get_Next_Instance_HO_Neg_HICO_520_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label = Augmented_HO_Neg_HICO_520_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(len(Pattern)))
    f.write('\n')
    f.close()

    return blobs

def Augmented_HO_Neg_HICO_520_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
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

    # for every positive input pair ,the binary label is 10 if hoi is in 520, if its 80 go to 01 label
    for i in range(num_pos):

        flag_inter = 0

        for num_hoi in range(600):
            if (int(action_HO[i][num_hoi]) == 1) and (num_hoi + 1 not in list_no_inter):
                flag_inter = 1
                break
        # caution: there are situations that inter and no inter in a same pair, viewed as no inter
        if flag_inter == 1:
            binary_label[i][0] = 1
        else:
            binary_label[i][1] = 1
     
    # for every negative input pair, the binary label is 01
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1

    f = open('test.txt','w')
    f.write(str(num_pos))
    f.write('\n')
    f.write(str(num_pos_neg))
    f.write('\n')
    f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label














#############################################################################################################
# hico vcoco combine training
#############################################################################################################
def Get_Next_Instance_HO_Neg_HICO_520_80_VCOCO_Early_combine_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO':
        im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    elif dataset == 'VCOCO':
        im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    else:
        return
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, binary_label, H_num = Augmented_HO_Neg_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    key = dataset + '_' + str(image_id)

    #############################################################################
    GT_count = len(GT) - 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')

    if key in Trainval_Neg:

        if len(Trainval_Neg[key]) < Neg_select:
            for Neg in Trainval_Neg[key]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[key])), len(Trainval_Neg[key]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[key][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    if dataset == 'HICO':
        for i in range(num_pos):
            flag_inter = 0

            for num_hoi in range(600):
                if int(action_HO[i][num_hoi]) == 1 and num_hoi + 1 not in list_no_inter:
                    flag_inter = 1
                    break
            # caution: there are situations that inter and no inter in a same pair, viewed as no inter
            if flag_inter == 1:
                binary_label[i][0] = 1
            else:
                binary_label[i][1] = 1
    else:
        for i in range(num_pos):
            binary_label[i][0] = 1 # pos is at 0

    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, binary_label, num_pos








#############################################################################################################
# hcvrd hico vcoco combine training
#############################################################################################################
def Get_Next_Instance_HO_Neg_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO':
        im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    elif dataset == 'VCOCO':
        im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    elif dataset == 'HCVRD':
        im_file = '/disk1/zhousy/analyze_hcvrd/classification/image_used_all/' + image_id
    else:
        return
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, binary_label, H_num = Augmented_HO_Neg_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset != 'HCVRD':
        key = dataset + '_' + str(image_id)
    else:
        key = dataset + '_' + image_id

    #############################################################################
    GT_count = len(GT) - 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')

    if key in Trainval_Neg:

        if len(Trainval_Neg[key]) < Neg_select:
            for Neg in Trainval_Neg[key]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[key])), len(Trainval_Neg[key]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[key][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    if dataset == 'HICO':
        for i in range(num_pos):
            flag_inter = 0

            for num_hoi in range(600):
                if int(action_HO[i][num_hoi]) == 1 and num_hoi + 1 not in list_no_inter:
                    flag_inter = 1
                    break
            # caution: there are situations that inter and no inter in a same pair, viewed as no inter
            if flag_inter == 1:
                binary_label[i][0] = 1
            else:
                binary_label[i][1] = 1
    else:
        for i in range(num_pos):
            binary_label[i][0] = 1 # pos is at 0

    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, binary_label, num_pos











#############################################################################################################
# openimage hcvrd hico vcoco combine training
#############################################################################################################
def Get_Next_Instance_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO':
        im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    elif dataset == 'VCOCO':
        im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    elif dataset == 'HCVRD':
        im_file = '/disk1/zhousy/analyze_hcvrd/classification/image_used_all/' + image_id
    elif dataset == 'OPENIMAGE':
        im_file = '/disk1/zhousy/analyze_openimage/classification/image_used_all/' + image_id
    else:
        return
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, binary_label, H_num = Augmented_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO' or dataset == 'VCOCO':
        key = dataset + '_' + str(image_id)
    else:
        key = dataset + '_' + image_id

    #############################################################################
    GT_count = len(GT) - 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')

    if key in Trainval_Neg:

        if len(Trainval_Neg[key]) < Neg_select:
            for Neg in Trainval_Neg[key]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[key])), len(Trainval_Neg[key]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[key][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    if dataset == 'HICO':
        for i in range(num_pos):
            flag_inter = 0

            for num_hoi in range(600):
                if int(action_HO[i][num_hoi]) == 1 and num_hoi + 1 not in list_no_inter:
                    flag_inter = 1
                    break
            # caution: there are situations that inter and no inter in a same pair, viewed as no inter
            if flag_inter == 1:
                binary_label[i][0] = 1
            else:
                binary_label[i][1] = 1
    else:
        for i in range(num_pos):
            binary_label[i][0] = 1 # pos is at 0

    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, binary_label, num_pos








#############################################################################################################
# openimage hcvrd hico vcoco combine training
#############################################################################################################
def Get_Next_Instance_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_pose_pattern_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO':
        im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    elif dataset == 'VCOCO':
        im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    elif dataset == 'HCVRD':
        im_file = '/disk1/zhousy/analyze_hcvrd/classification/image_used_all/' + image_id
    elif dataset == 'OPENIMAGE':
        im_file = '/disk1/zhousy/analyze_openimage/classification/image_used_all/' + image_id
    else:
        return
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, binary_label, H_num = Augmented_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_pose_pattern_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_OPENIMAGE_HCVRD_HICO_520_80_VCOCO_Early_combine_pose_pattern_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'HICO' or dataset == 'VCOCO':
        key = dataset + '_' + str(image_id)
    else:
        key = dataset + '_' + image_id

    #############################################################################
    GT_count = len(GT) - 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
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
    
        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_HO.extend(action_HO_temp)

        for j in range(length_min):
            Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[i][5]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    for j in range(length_min):
        Pattern_ = Get_next_sp_with_pose(Human_augmented_temp[j][1:], Object_augmented_temp[j][1:], GT[GT_count - 1][5]).reshape(1, 64, 64, 3)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')

    if key in Trainval_Neg:

        if len(Trainval_Neg[key]) < Neg_select:
            for Neg in Trainval_Neg[key]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[key])), len(Trainval_Neg[key]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[key][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
                Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    num_pos_neg = len(Human_augmented)
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    if dataset == 'HICO':
        for i in range(num_pos):
            flag_inter = 0

            for num_hoi in range(600):
                if int(action_HO[i][num_hoi]) == 1 and num_hoi + 1 not in list_no_inter:
                    flag_inter = 1
                    break
            # caution: there are situations that inter and no inter in a same pair, viewed as no inter
            if flag_inter == 1:
                binary_label[i][0] = 1
            else:
                binary_label[i][1] = 1
    else:
        for i in range(num_pos):
            binary_label[i][0] = 1 # pos is at 0

    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, binary_label, num_pos











#############################################################################################################
# VTRANSE D training
#############################################################################################################
def Get_Next_Instance_HO_Neg_VTRANSE_version2(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    if dataset == 'VTRANSE':
        im_file = '/Disk1/yonglu/vtranse/dataset/VG/images/VG_100K/' + image_id
    else:
        return

    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, binary_label, H_num = Augmented_HO_Neg_VTRANSE_version2(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)

    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['sp']          = Pattern
    blobs['H_num']       = H_num
    blobs['binary_label'] = binary_label

    return blobs

def Augmented_HO_Neg_VTRANSE_version2(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    dataset = GT[len(GT) - 1]
    image_id = GT[0][0]
    key = dataset + '_' + image_id

    #############################################################################
    GT_count = len(GT) - 1
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO = [], [], []

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

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')

    if key in Trainval_Neg:

        if len(Trainval_Neg[key]) < Neg_select:
            for Neg in Trainval_Neg[key]:
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            List = random.sample(range(len(Trainval_Neg[key])), len(Trainval_Neg[key]))
            for i in range(Neg_select):
                Neg = Trainval_Neg[key][List[i]]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
    Human_augmented   = Human_augmented.reshape( num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 

    # binary label for vcoco, different from hico, vcoco does not have the no-interaction label, so give it binary label accord to the locations of pos and neg directly
    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')

    for i in range(num_pos):
        binary_label[i][0] = 1 # pos is at 0

    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, binary_label, num_pos


















#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################

# For the use of repeat gt and more neg pairs (version3)

#######################################################################################################
########################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################










#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################
# for vcoco
#######################################################################################################
#######################################################################################################
#######################################################################################################
#######################################################################################################



def Get_Next_Instance_HO_Neg_version3(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label = Augmented_HO_Neg_version3(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_Neg_version3(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_HO, action_H = [], [], [], []

    for i in range(GT_count - 1):
        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]
    
        action_HO__temp = Generate_action(GT[i][1])
        action_HO_temp  = action_HO__temp
        for j in range(length_min - 1):
            action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

        action_H__temp  = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

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

    action_HO__temp = Generate_action(GT[GT_count - 1][1])
    action_HO_temp  = action_HO__temp
    for j in range(length_min - 1):
        action_HO_temp = np.concatenate((action_HO_temp, action_HO__temp), axis=0)

    action_H__temp  = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################
    
    num_pos = len(Human_augmented)
    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    Human_augmented_solo = Human_augmented.copy()
    
    if image_id in Trainval_Neg:

        neg_length = len(Trainval_Neg[image_id]) - 1
        if neg_length < Neg_select:
            for i in range(neg_length):
                Neg = Trainval_Neg[image_id][i]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            repeat_index = Trainval_Neg[image_id][neg_length]
            List = list(range(repeat_index * Neg_select, (repeat_index + 1) * Neg_select))
            Trainval_Neg[image_id][neg_length] += 1
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i] % neg_length]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29) # some actions do not have objects
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)
        
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_HO = np.concatenate((action_HO, np.zeros(29).reshape(1,29)), axis=0)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
    
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 


    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Human_augmented_solo, Object_augmented, action_HO, action_H, mask_HO, mask_H, binary_label



# for vcoco
def Get_Next_Instance_HO_spNeg_version3(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file  = config.cfg.DATA_DIR + '/' + 'v-coco/coco/images/train2014/COCO_train2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)


    Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label = Augmented_HO_spNeg_version3(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
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

# for vcoco
def Augmented_HO_spNeg_version3(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    image_id = GT[0][0]

    #############################################################################
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    
    Human_augmented, Object_augmented, action_sp, action_HO, action_H = [], [], [], [], []

    for i in range(GT_count - 1):

        Human    = GT[i][2]
        Object   = GT[i][3]

        Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_all)
        Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_all)

        length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

        Human_augmented_temp  =  Human_augmented_temp[:length_min]
        Object_augmented_temp = Object_augmented_temp[:length_min]

        action_sp__temp  = Generate_action(GT[i][1])
        action_sp_temp  = action_sp__temp
        for j in range(length_min - 1):
            action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)

        action_HO_temp = action_sp_temp.copy()

        action_H__temp  = Generate_action(GT[i][4])
        action_H_temp  = action_H__temp
        for j in range(length_min - 1):
            action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

        Human_augmented.extend(Human_augmented_temp)
        Object_augmented.extend(Object_augmented_temp)
        action_sp.extend(action_sp_temp)
        action_HO.extend(action_HO_temp)
        action_H.extend(action_H_temp)

    Human    = GT[GT_count - 1][2]
    Object   = GT[GT_count - 1][3]

    Human_augmented_temp  = Augmented_box(Human,  shape, image_id, aug_last)
    Object_augmented_temp = Augmented_box(Object, shape, image_id, aug_last)

    length_min = min(len(Human_augmented_temp),len(Object_augmented_temp))

    Human_augmented_temp  =  Human_augmented_temp[:length_min]
    Object_augmented_temp = Object_augmented_temp[:length_min]

    action_sp__temp  = Generate_action(GT[GT_count - 1][1])
    action_sp_temp  = action_sp__temp
    for j in range(length_min - 1):
        action_sp_temp = np.concatenate((action_sp_temp, action_sp__temp), axis=0)

    action_HO_temp = action_sp_temp.copy()

    action_H__temp  = Generate_action(GT[GT_count - 1][4])
    action_H_temp  = action_H__temp
    for j in range(length_min - 1):
        action_H_temp = np.concatenate((action_H_temp, action_H__temp), axis=0)

    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_sp.extend(action_sp_temp)
    action_HO.extend(action_HO_temp)
    action_H.extend(action_H_temp)

    #######################################################################################

    num_pos = len(Human_augmented)

    if image_id in Trainval_Neg:

        neg_length = len(Trainval_Neg[image_id]) - 1
        if neg_length < Neg_select:
            for i in range(neg_length):
                Neg = Trainval_Neg[image_id][i]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
        else:
            repeat_index = Trainval_Neg[image_id][neg_length]
            List = list(range(repeat_index * Neg_select, (repeat_index + 1) * Neg_select))
            Trainval_Neg[image_id][neg_length] += 1
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i] % neg_length]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)

    num_pos_neg = len(Human_augmented)

    mask_sp_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_HO_   = np.asarray([1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0,1]).reshape(1,29)
    mask_H_    = np.asarray([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]).reshape(1,29)

    mask_sp   = mask_sp_
    mask_HO   = mask_HO_
    mask_H    = mask_H_
    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos - 1):
        mask_HO   = np.concatenate((mask_HO,   mask_HO_),   axis=0)
        mask_H    = np.concatenate((mask_H,    mask_H_),    axis=0)

    for i in range(num_pos_neg - 1):
        mask_sp   = np.concatenate((mask_sp,   mask_sp_),   axis=0)

    for i in range(num_pos_neg - num_pos):
        action_sp = np.concatenate((action_sp, np.zeros(29).reshape(1,29)), axis=0) # neg has all 0 in 29 label

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)

    Human_augmented   = np.array(Human_augmented, dtype = 'float32')
    Object_augmented  = np.array(Object_augmented, dtype = 'float32')
    action_HO         = np.array(action_HO, dtype = 'int32') 
    action_sp         = np.array(action_sp, dtype = 'int32') 
    action_H          = np.array(action_H, dtype = 'int32') 

    Pattern           = Pattern.reshape( num_pos_neg, 64, 64, 2) 
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
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented_sp, Human_augmented, Object_augmented, action_sp, action_HO, action_H, mask_sp, mask_HO, mask_H, binary_label


















##############################################################################
##############################################################################
##############################################################################
# for hico with pose vector
##############################################################################
##############################################################################
##############################################################################



def Get_Next_Instance_HO_Neg_HICO_pose_vector_version3(trainval_GT, Trainval_Neg, iter, Pos_augment, Neg_select, Data_length):

    GT       = trainval_GT[iter%Data_length]
    image_id = GT[0][0]
    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose = Augmented_HO_Neg_HICO_pose_vector_version3(GT, Trainval_Neg, im_shape, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['gt_class_HO'] = action_HO
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['H_pose']      = pose

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(len(Pattern)))
    # f.write('\n')
    # f.close()

    return blobs

def Augmented_HO_Neg_HICO_pose_vector_version3(GT, Trainval_Neg, shape, Pos_augment, Neg_select):
    GT_count = len(GT)
    aug_all = int(Pos_augment / GT_count)
    aug_last = Pos_augment - aug_all * (GT_count - 1) 
    image_id = GT[0][0]

    Human_augmented, Object_augmented, action_HO, pose = [], [], [], []

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

        if GT[i][5] == None:
            pose_temp = np.zeros((length_min, 51), dtype = 'float32')
        else:
            pose_temp = np.array(GT[i][5], dtype = 'float32').reshape(1,51)
            pose__temp = pose_temp
            for j in range(length_min - 1):
                pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

        pose.extend(pose_temp)
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

    if GT[GT_count - 1][5] == None:
        pose_temp = np.zeros((length_min, 51), dtype = 'float32')
    else:
        pose_temp = np.array(GT[GT_count - 1][5], dtype = 'float32').reshape(1,51)
        pose__temp = pose_temp
        for j in range(length_min - 1):
            pose_temp = np.concatenate((pose_temp, pose__temp), axis=0)

    pose.extend(pose_temp)
    Human_augmented.extend(Human_augmented_temp)
    Object_augmented.extend(Object_augmented_temp)
    action_HO.extend(action_HO_temp)

    num_pos = len(Human_augmented)
    
    if image_id in Trainval_Neg:

        neg_length = len(Trainval_Neg[image_id]) - 1
        if neg_length < Neg_select:

            for i in range(neg_length):
                Neg = Trainval_Neg[image_id][i]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)
        else:
            repeat_index = Trainval_Neg[image_id][neg_length]
            List = list(range(repeat_index * Neg_select, (repeat_index + 1) * Neg_select))
            Trainval_Neg[image_id][neg_length] += 1
            for i in range(Neg_select):
                Neg = Trainval_Neg[image_id][List[i] % neg_length]
                Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
                Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
                action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
                if Neg[7] == None:
                    pose = np.concatenate((pose, np.zeros((1,51), dtype='float32')), axis=0)
                else:
                    pose = np.concatenate((pose, np.array(Neg[7]).reshape(1,51)), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern   = np.empty((0, 64, 64, 2), dtype=np.float32)

    for i in range(num_pos_neg):
        Pattern_ = Get_next_sp(Human_augmented[i][1:], Object_augmented[i][1:]).reshape(1, 64, 64, 2)
        Pattern  = np.concatenate((Pattern, Pattern_), axis=0)


    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 2) 
    Human_augmented   = np.array(Human_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    Object_augmented  = np.array(Object_augmented, dtype='float32').reshape(num_pos_neg, 5) 
    action_HO         = np.array(action_HO, dtype='int32').reshape(num_pos_neg, 600) 
    pose              = np.array(pose, dtype='float32').reshape(num_pos_neg, 51)

    binary_label = np.zeros((num_pos_neg, 2), dtype = 'int32')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    # different choice: 1. 600--1 0, no label no-inter--0 1; 2. 520--1 0, 80 and no label no-inter--0 1
    # now which do we choose??? the first?

    # f = open('test.txt','w')
    # f.write(str(num_pos))
    # f.write('\n')
    # f.write(str(num_pos_neg))
    # f.write('\n')
    # f.close()

    return Pattern, Human_augmented, Object_augmented, action_HO, num_pos, binary_label, pose




##############################################################################
##############################################################################
##############################################################################
# for hico det PVP TODO:
##############################################################################
##############################################################################
##############################################################################

def Generate_part_bbox(joint_bbox, Human_bbox=None):
    part_bbox = np.zeros([1, 10, 5], dtype=np.float64)
    if isinstance(joint_bbox, int):
        if Human_bbox is None:
            raise ValueError
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, Human_bbox[0], Human_bbox[1], Human_bbox[2], Human_bbox[3]], dtype=np.float64)
    
    else:
        for i in range(10):
            part_bbox[0, i, :] = np.array([0, joint_bbox[i]['x1'],  joint_bbox[i]['y1'],  joint_bbox[i]['x2'],  joint_bbox[i]['y2']], dtype=np.float64)
    
    return part_bbox

def Generate_action_PVP(idx, num_pvp):
    action_PVP = np.zeros([1, num_pvp], dtype=np.float64)
    if isinstance(idx, int):
        action_PVP[:, idx] = 1
    else:
        action_PVP[:, list(idx)] = 1
    return action_PVP

def Get_Next_Instance_HO_Neg_HICO_pose_pattern_version2_for_PVP(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select):

    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Pattern, Human_augmented, Object_augmented, Part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_10v, gt_verb = Augmented_HO_Neg_HICO_pose_pattern_version2_for_PVP(image_id, Trainval_GT, Trainval_Neg, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['P_boxes']     = Part_bbox
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_P0'] = action_PVP0
    blobs['gt_class_P1'] = action_PVP1
    blobs['gt_class_P2'] = action_PVP2
    blobs['gt_class_P3'] = action_PVP3
    blobs['gt_class_P4'] = action_PVP4
    blobs['gt_class_P5'] = action_PVP5
    blobs['sp']          = Pattern
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['gt_10v']      = gt_10v
    blobs['gt_verb']     = gt_verb

    return blobs

def Augmented_HO_Neg_HICO_pose_pattern_version2_for_PVP(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select):
    pair_info = trainval_GT[image_id]
    pair_num = len(pair_info)

    # if not sufficient, repeat data
    if pair_num >= Pos_augment:
        List = random.sample(range(pair_num), Pos_augment)
        GT = []
        for i in range(Pos_augment):
            GT.append(pair_info[List[i]])
    else:
        GT = pair_info
        for i in range(Pos_augment - pair_num):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    Object_augmented = np.empty((0, 5), dtype=np.float64)
    action_HO   = np.empty((0, 600), dtype=np.float64)
    Pattern     = np.empty((0, 64, 64, 3), dtype=np.float64)
    part_bbox   = np.empty((0, 10, 5), dtype=np.float64)
    action_PVP0 = np.empty((0, 12), dtype=np.float64)
    action_PVP1 = np.empty((0, 10), dtype=np.float64)
    action_PVP2 = np.empty((0, 5), dtype=np.float64)
    action_PVP3 = np.empty((0, 31), dtype=np.float64)
    action_PVP4 = np.empty((0, 5), dtype=np.float64)
    action_PVP5 = np.empty((0, 13), dtype=np.float64)
    action_verb = np.empty((0, 117), dtype=np.float64)
    gt_hp10     = np.empty((0, 10), dtype=np.float64)
    # TODO:
    for i in range(Pos_augment):
        Human    = GT[i][2] # [x1, y1, x2, y2]
        Object   = GT[i][3] # [x1, y1, x2, y2]
        Human_augmented  = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        Object_augmented = np.concatenate((Object_augmented, np.array([0, Object[0],  Object[1],  Object[2],  Object[3]]).reshape(1,5).astype(np.float64)), axis=0)
        action_HO        = np.concatenate((action_HO, Generate_action_HICO(GT[i][1])), axis=0)
        Pattern          = np.concatenate((Pattern, Get_next_sp_with_pose(np.array(Human, dtype='float64'), np.array(Object, dtype='float64'), GT[i][4]['joints_16']).reshape(1, 64, 64, 3)), axis=0)
        part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(GT[i][4]['part_bbox'])), axis=0)
        action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(GT[i][4]['pvp76_ankle2'], 12)), axis=0) # ankle
        action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(GT[i][4]['pvp76_knee2'], 10)), axis=0) # knee
        action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(GT[i][4]['pvp76_hip'], 5)), axis=0) # hand
        action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(GT[i][4]['pvp76_hand2'], 31)), axis=0) # shoulder
        action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(GT[i][4]['pvp76_shoulder2'], 5)), axis=0) # hip
        action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(GT[i][4]['pvp76_head'], 13)), axis=0) # head
        action_verb      = np.concatenate((action_verb, Generate_action_PVP(list(GT[i][4]['verb117_list']), 117)), axis=0)
        gt_hp10          = np.concatenate((gt_hp10, Generate_action_PVP(GT[i][4]['hp10_list'], 10)), axis=0)
    num_pos = Pos_augment
    
    if image_id in Trainval_Neg.keys():
        if len(Trainval_Neg[image_id]) < Neg_select:
            List = range(len(Trainval_Neg[image_id]))
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))

        for i in range(len(List)):
            Neg = Trainval_Neg[image_id][List[i]]
            Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
            Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
            action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
            Pattern_ = Get_next_sp_with_pose(np.array(Neg[2], dtype='float64'), np.array(Neg[3], dtype='float64'), Neg[7]).reshape(1, 64, 64, 3)
            Pattern  = np.concatenate((Pattern, Pattern_), axis=0)
            # print('id, List[i]', image_id, List[i])
            # print('Neg[8]', Neg[8])
            # print('Neg[9]', Neg[9])
            # print('Neg[-2]', Neg[-2])
            # print('Neg[-1]', Neg[-1])
            try:
                part_bbox = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
            except:
                print('id, List[i]', image_id, List[i])
                print('Neg[8]', Neg[8])
                print('Neg[9]', Neg[9])
                print('Neg[10]', Neg[10])
                print('Neg[11]', Neg[11])
                raise ValueError
            action_verb      = np.concatenate((action_verb, Generate_action_PVP(Neg[-2], 117)), axis=0)
            action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(Neg[-1]['pvp76_ankle2'], 12)), axis=0) # ankle
            action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(Neg[-1]['pvp76_knee2'], 10)), axis=0) # knee
            action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(Neg[-1]['pvp76_hip'], 5)), axis=0) # hand
            action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(Neg[-1]['pvp76_hand2'], 31)), axis=0) # shoulder
            action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(Neg[-1]['pvp76_shoulder2'], 5)), axis=0) # hip
            action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(Neg[-1]['pvp76_head'], 13)), axis=0) # head
            gt_hp10          = np.concatenate((gt_hp10, np.zeros([1, 10])), axis=0)

    num_pos_neg = len(Human_augmented)

    Pattern           = Pattern.reshape(num_pos_neg, 64, 64, 3) 
    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype='int64')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Pattern, Human_augmented, Object_augmented, part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_hp10, action_verb


########################################################################################################


def Get_Next_Instance_HO_HICO_DET_for_only_PVP(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select, pvp=76):

    im_file = config.cfg.DATA_DIR + '/' + 'hico_20160224_det/images/train2015/HICO_train2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    if pvp == 76:
        Human_augmented, Object_augmented, Part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_10v, gt_verb = Augmented_HO_Neg_HICO_DET_for_only_PVP76(image_id, Trainval_GT, Trainval_Neg, Pos_augment, Neg_select)
    else:
        Human_augmented, Object_augmented, Part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_10v, gt_verb = Augmented_HO_Neg_HICO_DET_for_only_PVP55(image_id, Trainval_GT, Trainval_Neg, Pos_augment, Neg_select)

    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['O_boxes']     = Object_augmented
    blobs['P_boxes']     = Part_bbox
    blobs['gt_class_HO'] = action_HO
    blobs['gt_class_P0'] = action_PVP0
    blobs['gt_class_P1'] = action_PVP1
    blobs['gt_class_P2'] = action_PVP2
    blobs['gt_class_P3'] = action_PVP3
    blobs['gt_class_P4'] = action_PVP4
    blobs['gt_class_P5'] = action_PVP5
    blobs['H_num']       = num_pos
    blobs['binary_label'] = binary_label
    blobs['gt_10v']      = gt_10v
    blobs['gt_verb']     = gt_verb

    return blobs

def Augmented_HO_Neg_HICO_DET_for_only_PVP76(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select):
    pair_info = trainval_GT[image_id]
    pair_num = len(pair_info)

    # if not sufficient, repeat data
    if pair_num >= Pos_augment:
        List = random.sample(range(pair_num), Pos_augment)
        GT = []
        for i in range(Pos_augment):
            GT.append(pair_info[List[i]])
    else:
        GT = pair_info
        for i in range(Pos_augment - pair_num):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    Object_augmented = np.empty((0, 5), dtype=np.float64)
    action_HO   = np.empty((0, 600), dtype=np.float64)
    part_bbox   = np.empty((0, 10, 5), dtype=np.float64)
    action_PVP0 = np.empty((0, 12), dtype=np.float64)
    action_PVP1 = np.empty((0, 10), dtype=np.float64)
    action_PVP2 = np.empty((0, 5), dtype=np.float64)
    action_PVP3 = np.empty((0, 31), dtype=np.float64)
    action_PVP4 = np.empty((0, 5), dtype=np.float64)
    action_PVP5 = np.empty((0, 13), dtype=np.float64)
    action_verb = np.empty((0, 117), dtype=np.float64)
    gt_hp10     = np.empty((0, 10), dtype=np.float64)
    # TODO:
    for i in range(Pos_augment):
        Human    = GT[i][2] # [x1, y1, x2, y2]
        Object   = GT[i][3] # [x1, y1, x2, y2]
        Human_augmented  = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        Object_augmented = np.concatenate((Object_augmented, np.array([0, Object[0],  Object[1],  Object[2],  Object[3]]).reshape(1,5).astype(np.float64)), axis=0)
        action_HO        = np.concatenate((action_HO, Generate_action_HICO(GT[i][1])), axis=0)
        part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(GT[i][4]['part_bbox'])), axis=0)
        action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(GT[i][4]['pvp76_ankle2'], 12)), axis=0) # ankle
        action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(GT[i][4]['pvp76_knee2'], 10)), axis=0) # knee
        action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(GT[i][4]['pvp76_hip'], 5)), axis=0) # hand
        action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(GT[i][4]['pvp76_hand2'], 31)), axis=0) # shoulder
        action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(GT[i][4]['pvp76_shoulder2'], 5)), axis=0) # hip
        action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(GT[i][4]['pvp76_head'], 13)), axis=0) # head
        action_verb      = np.concatenate((action_verb, Generate_action_PVP(list(GT[i][4]['verb117_list']), 117)), axis=0)
        gt_hp10          = np.concatenate((gt_hp10, Generate_action_PVP(GT[i][4]['hp10_list'], 10)), axis=0)
    num_pos = Pos_augment
    
    if image_id in Trainval_Neg.keys():
        if len(Trainval_Neg[image_id]) < Neg_select:
            List = range(len(Trainval_Neg[image_id]))
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))

        for i in range(len(List)):
            Neg = Trainval_Neg[image_id][List[i]]
            Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
            Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
            action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
            try:
                part_bbox = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
            except:
                print('id, List[i]', image_id, List[i])
                print('Neg[8]', Neg[8])
                print('Neg[9]', Neg[9])
                print('Neg[10]', Neg[10])
                print('Neg[11]', Neg[11])
                raise ValueError
            action_verb      = np.concatenate((action_verb, Generate_action_PVP(Neg[-2], 117)), axis=0)
            action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(Neg[-1]['pvp76_ankle2'], 12)), axis=0) # ankle
            action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(Neg[-1]['pvp76_knee2'], 10)), axis=0) # knee
            action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(Neg[-1]['pvp76_hip'], 5)), axis=0) # hand
            action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(Neg[-1]['pvp76_hand2'], 31)), axis=0) # shoulder
            action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(Neg[-1]['pvp76_shoulder2'], 5)), axis=0) # hip
            action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(Neg[-1]['pvp76_head'], 13)), axis=0) # head
            gt_hp10          = np.concatenate((gt_hp10, np.zeros([1, 10])), axis=0)

    num_pos_neg = len(Human_augmented)

    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype='int64')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Human_augmented, Object_augmented, part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_hp10, action_verb

def Augmented_HO_Neg_HICO_DET_for_only_PVP55(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select):
    pair_info = trainval_GT[image_id]
    pair_num = len(pair_info)

    # if not sufficient, repeat data
    if pair_num >= Pos_augment:
        List = random.sample(range(pair_num), Pos_augment)
        GT = []
        for i in range(Pos_augment):
            GT.append(pair_info[List[i]])
    else:
        GT = pair_info
        for i in range(Pos_augment - pair_num):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    Object_augmented = np.empty((0, 5), dtype=np.float64)
    action_HO   = np.empty((0, 600), dtype=np.float64)
    part_bbox   = np.empty((0, 10, 5), dtype=np.float64)
    action_PVP0 = np.empty((0, 6), dtype=np.float64)
    action_PVP1 = np.empty((0, 6), dtype=np.float64)
    action_PVP2 = np.empty((0, 3), dtype=np.float64)
    action_PVP3 = np.empty((0, 23), dtype=np.float64)
    action_PVP4 = np.empty((0, 5), dtype=np.float64)
    action_PVP5 = np.empty((0, 12), dtype=np.float64)
    action_verb = np.empty((0, 117), dtype=np.float64)
    gt_hp10     = np.empty((0, 10), dtype=np.float64)
    # TODO:
    for i in range(Pos_augment):
        Human    = GT[i][2] # [x1, y1, x2, y2]
        Object   = GT[i][3] # [x1, y1, x2, y2]
        Human_augmented  = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        Object_augmented = np.concatenate((Object_augmented, np.array([0, Object[0],  Object[1],  Object[2],  Object[3]]).reshape(1,5).astype(np.float64)), axis=0)
        action_HO        = np.concatenate((action_HO, Generate_action_HICO(GT[i][1])), axis=0)
        part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(GT[i][4]['part_bbox'])), axis=0)
        action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(GT[i][4]['pvp55_ankle2'], 6)), axis=0) # ankle
        action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(GT[i][4]['pvp55_knee2'], 6)), axis=0) # knee
        action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(GT[i][4]['pvp55_hip'], 3)), axis=0) # hand
        action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(GT[i][4]['pvp55_hand2'], 23)), axis=0) # shoulder
        action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(GT[i][4]['pvp55_shoulder2'], 5)), axis=0) # hip
        action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(GT[i][4]['pvp55_head'], 12)), axis=0) # head
        action_verb      = np.concatenate((action_verb, Generate_action_PVP(list(GT[i][4]['verb117_list']), 117)), axis=0)
        gt_hp10          = np.concatenate((gt_hp10, Generate_action_PVP(GT[i][4]['hp10_list'], 10)), axis=0)
    num_pos = Pos_augment
    
    if image_id in Trainval_Neg.keys():
        if len(Trainval_Neg[image_id]) < Neg_select:
            List = range(len(Trainval_Neg[image_id]))
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))

        for i in range(len(List)):
            Neg = Trainval_Neg[image_id][List[i]]
            Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
            Object_augmented = np.concatenate((Object_augmented, np.array([0, Neg[3][0], Neg[3][1], Neg[3][2], Neg[3][3]]).reshape(1,5)), axis=0)
            action_HO        = np.concatenate((action_HO, Generate_action_HICO([Neg[1]])), axis=0)
            try:
                part_bbox = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
            except:
                print('id, List[i]', image_id, List[i])
                print('Neg[8]', Neg[8])
                print('Neg[9]', Neg[9])
                print('Neg[10]', Neg[10])
                print('Neg[11]', Neg[11])
                raise ValueError
            action_verb      = np.concatenate((action_verb, Generate_action_PVP(Neg[-2], 117)), axis=0)
            action_PVP0      = np.concatenate((action_PVP0, Generate_action_PVP(Neg[-1]['pvp55_ankle2'], 6)), axis=0) # ankle
            action_PVP1      = np.concatenate((action_PVP1, Generate_action_PVP(Neg[-1]['pvp55_knee2'], 6)), axis=0) # knee
            action_PVP2      = np.concatenate((action_PVP2, Generate_action_PVP(Neg[-1]['pvp55_hip'], 3)), axis=0) # hand
            action_PVP3      = np.concatenate((action_PVP3, Generate_action_PVP(Neg[-1]['pvp55_hand2'], 23)), axis=0) # shoulder
            action_PVP4      = np.concatenate((action_PVP4, Generate_action_PVP(Neg[-1]['pvp55_shoulder2'], 5)), axis=0) # hip
            action_PVP5      = np.concatenate((action_PVP5, Generate_action_PVP(Neg[-1]['pvp55_head'], 12)), axis=0) # head
            gt_hp10          = np.concatenate((gt_hp10, np.zeros([1, 10])), axis=0)

    num_pos_neg = len(Human_augmented)

    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 
    Object_augmented  = Object_augmented.reshape(num_pos_neg, 5) 
    action_HO         = action_HO.reshape(num_pos_neg, 600) 

    binary_label = np.zeros((num_pos_neg, 2), dtype='int64')
    for i in range(num_pos):
        # if hoi_id belong to 80 then is 0 1, else 1 0
        binary_label[i][0] = 1 # pos is at 0 #############problem: here in pos, it contains the no interaction 80, so we may give this 80 '0 1' and 520 '1 0', and test
    for i in range(num_pos + 1, num_pos_neg):
        binary_label[i][1] = 1 # neg is at 1

    return Human_augmented, Object_augmented, part_bbox, action_HO, action_PVP0, action_PVP1, action_PVP2, action_PVP3, action_PVP4, action_PVP5, num_pos, binary_label, gt_hp10, action_verb

################################### for AVA #######################################

def Get_Next_Instance_verb_AVA_for_only_PVP(Trainval_GT, Trainval_Neg, image_id, Pos_augment, Neg_select):

    im_file = '/Disk1/AVA/input/train/' + str(Trainval_GT[image_id][0][0]) + '_0.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= config.cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)

    Human_augmented, Part_bbox, num_pos, gt_verb = Augmented_HO_AVA_for_only_verb(image_id, Trainval_GT, Trainval_Neg, Pos_augment, Neg_select)
    
    blobs = {}
    blobs['image']       = im_orig
    blobs['H_boxes']     = Human_augmented
    blobs['P_boxes']     = Part_bbox
    blobs['H_num']       = num_pos
    blobs['gt_verb']     = gt_verb

    return blobs

def Augmented_HO_AVA_for_only_verb(image_id, trainval_GT, Trainval_Neg, Pos_augment, Neg_select):
    pair_info = trainval_GT[image_id]
    pair_num = len(pair_info)

    # if not sufficient, repeat data
    if pair_num >= Pos_augment:
        List = random.sample(range(pair_num), Pos_augment)
        GT = []
        for i in range(Pos_augment):
            GT.append(pair_info[List[i]])
    else:
        GT = pair_info
        for i in range(Pos_augment - pair_num):
            index = random.randint(0, pair_num - 1)
            GT.append(pair_info[index])

    Human_augmented = np.empty((0, 5), dtype=np.float64)
    part_bbox   = np.empty((0, 10, 5), dtype=np.float64)
    action_verb = np.empty((0, 80), dtype=np.float64)
    # TODO:
    for i in range(Pos_augment):
        Human    = GT[i][1] # [x1, y1, x2, y2]
        Human_augmented  = np.concatenate((Human_augmented, np.array([0, Human[0],  Human[1],  Human[2],  Human[3]]).reshape(1,5).astype(np.float64)), axis=0)
        part_bbox        = np.concatenate((part_bbox, Generate_part_bbox(GT[i][5])), axis=0)
        for k in range(len(GT[i][2])):
            GT[i][2][k] -= 1
        action_verb      = np.concatenate((action_verb, Generate_action_PVP(GT[i][2], 80)), axis=0)
    num_pos = Pos_augment
    
    if image_id in Trainval_Neg.keys():
        if len(Trainval_Neg[image_id]) < Neg_select:
            List = range(len(Trainval_Neg[image_id]))
        else:
            List = random.sample(range(len(Trainval_Neg[image_id])), len(Trainval_Neg[image_id]))

        for i in range(len(List)):
            Neg = Trainval_Neg[image_id][List[i]]
            Human_augmented  = np.concatenate((Human_augmented,  np.array([0, Neg[2][0], Neg[2][1], Neg[2][2], Neg[2][3]]).reshape(1,5)), axis=0)
            try:
                part_bbox = np.concatenate((part_bbox, Generate_part_bbox(Neg[8], Neg[2])), axis=0)
            except:
                print('id, List[i]', image_id, List[i])
                print('Neg[8]', Neg[8])
                print('Neg[9]', Neg[9])
                print('Neg[10]', Neg[10])
                print('Neg[11]', Neg[11])
                raise ValueError
            action_verb      = np.concatenate((action_verb, Generate_action_PVP(Neg[-2], 117)), axis=0)

    num_pos_neg = len(Human_augmented)

    Human_augmented   = Human_augmented.reshape(num_pos_neg, 5) 

    return Human_augmented, part_bbox, num_pos, action_verb

