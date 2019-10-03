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
