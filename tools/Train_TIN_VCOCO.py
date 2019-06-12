# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import ipdb
import os

from ult.config_vcoco import cfg
from models.train_Solver_VCOCO_pose_pattern_inD_more_positive import train_net
from networks.TIN_VCOCO import ResNet50

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0, 1

def parse_args():
    parser = argparse.ArgumentParser(description='Train an TIN on VCOCO')
    parser.add_argument('--num_iteration', dest='max_iters',
            help='Number of iterations to perform',
            default=20000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_VCOCO_default', type=str)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
            help='Number of Res5 blocks',
            default=5, type=int)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
            help='Number of augmented detection for each one. (By jittering the object detections)',
            default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
            help='Number of Negative example selected for each image',
            default=60, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    Trainval_GT       = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_GT_VCOCO_with_pose.pkl', "rb" ) )
    Trainval_N        = pickle.load( open( cfg.DATA_DIR + '/' + 'Trainval_Neg_VCOCO_with_pose.pkl', "rb" ) ) 

    np.random.seed(cfg.RNG_SEED)

    Early_flag = 0

    if cfg.TRAIN_MODULE_CONTINUE == 1:
            weight = cfg.ROOT_DIR + '/Weights/iCAN_ResNet50_VCOCO_D_more_positive_15-30/HOI_iter_10000.ckpt' 
    else:
            if cfg.TRAIN_INIT_WEIGHT == 1:
                    weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn/res50_faster_rcnn_iter_1190000.ckpt'
            elif cfg.TRAIN_INIT_WEIGHT == 2:
                    # add other pre-trained weight here
                    weight = cfg.ROOT_DIR + '/Weights/VCOCO_pretrained/HOI_iter_100000.ckpt'

    # output directory where the logs are saved
    tb_dir     = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model + '/'

    net = ResNet50()
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, Early_flag, args.Restore_flag, weight, max_iters=args.max_iters)
