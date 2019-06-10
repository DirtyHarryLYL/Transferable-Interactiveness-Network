# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np

# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()

cfg = __C

#
# Training options
#
__C.TRAIN = edict()
###########################################################################################
# Training loss setting: 1--H+O+SP+D, 2--H+O+SP, 3--D, 4--H+O, 5--SP 6--without h o sp
__C.TRAIN_MODULE = 1

# Training setting: 1--Update_all_parameter, 2--Only_Update_D, 3--Update_H+O+SP, 4--update except classifiers of S(fc), 5--without h o p
__C.TRAIN_MODULE_UPDATE = 1

# Initializing weight: 1--from fast RCNN  2--from previous best  3--from our model with d
__C.TRAIN_INIT_WEIGHT = 3

#refer to models/train_solver

# Continue Training or Training from iter 0  1--continue training  2--from iter 0  
__C.TRAIN_MODULE_CONTINUE = 2

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.0001 #0.00001

# Drop out for binary_discriminator
__C.TRAIN_DROP_OUT_BINARY = 0.8

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 100000 #20000 

# 1--weight = 0.1  2--no loss weight of D, consider put the wts of binary loss here directly
# __C.TRAIN_WEIGHT_LOSS = 2
#############################################################################################
# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.96

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = 20000

# The number of snapshots kept, older ones are deleted to save space
__C.TRAIN.SNAPSHOT_KEPT = None

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10

# The step interval for saving tensorflow summaries
__C.TRAIN.SUMMARY_INTERVAL = 200


# ResNet options
#
__C.RESNET = edict()

# Option to set if max-pooling is appended after crop_and_resize. 
# if true, the region will be resized to a square of 2xPOOLING_SIZE, 
# then 2x2 max-pooling is applied; otherwise the region will be directly
# resized to a square of POOLING_SIZE
__C.RESNET.MAX_POOL = False

# Number of fixed blocks during training, by default the first of all 4 blocks is fixed
# Range: 0 (none) to 3 (all)
__C.RESNET.FIXED_BLOCKS = 1


#
# MISC
#

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'Data'))


# Place outputs under an experiments directory
__C.EXP_DIR = 'default'

# Use GPU implementation of non-maximum suppression
__C.USE_GPU_NMS = True

# Default pooling mode, only 'crop' is available
__C.POOLING_MODE = 'crop'

# Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

# Anchor scales for RPN
__C.ANCHOR_SCALES = [8,16,32]

# Anchor ratios for RPN
__C.ANCHOR_RATIOS = [0.5,1,2]

# Number of filters for the RPN layer
__C.RPN_CHANNELS = 512