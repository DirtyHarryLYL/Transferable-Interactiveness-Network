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

from ult.config import cfg
from models.train_Solver_HICO_with_part_pose_pattern_inD_more_positive_coslr_random import train_net
from networks.TIN_HICO_with_part import ResNet50

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # use GPU 0,1

def parse_args():
    parser = argparse.ArgumentParser(description='Train TIN on HICO')
    parser.add_argument('--num_iteration', dest='max_iters',
                        help='Number of iterations to perform',
                        default=100000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='TIN_10w_with_part', type=str)
    parser.add_argument('--Pos_augment', dest='Pos_augment',
                        help='Number of augmented detection for each one. (By jittering the object detections)',
                        default=15, type=int)
    parser.add_argument('--Neg_select', dest='Neg_select',
                        help='Number of Negative example selected for each image',
                        default=60, type=int)
    parser.add_argument('--Restore_flag', dest='Restore_flag',
                        help='How many ResNet blocks are there?',
                        default=5, type=int)
    parser.add_argument('--debug', dest='debug',
                        help='Debug',
                        action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    # Trainval_GT = pickle.load(open('Data/Train_GT_10w_new.pkl', "rb"))
    # Trainval_N = pickle.load(open('Data/Train_Neg_10w_new.pkl', "rb"))
    Trainval_GT = pickle.load(open('Data/Train_GT_10w_new_demo.pkl', "rb"))
    Trainval_N = pickle.load(open('Data/Train_Neg_10w_new_demo.pkl', "rb"))

    np.random.seed(cfg.RNG_SEED)
    # change this to trained model of TIN for finetune, 1800000, '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    # change the module of init weight in config.py TRAIN_INIT_WEIGHT
    if cfg.TRAIN_MODULE_CONTINUE == 1:
        weight = cfg.ROOT_DIR + '/Weights/TIN_HICO_bin/HOI_iter_1800000.ckpt'  # from ckpt which you wish to continue
    else:
        if cfg.TRAIN_INIT_WEIGHT == 1:
            weight = cfg.ROOT_DIR + '/Weights/res50_faster_rcnn/res50_faster_rcnn_iter_1190000.ckpt'  # from faster R-CNN
        if cfg.TRAIN_INIT_WEIGHT == 2:
            weight = cfg.ROOT_DIR + '/Weights/best_ResNet50_HICO/HOI_iter_1800000.ckpt'  # from previous best model, best performance
        if cfg.TRAIN_INIT_WEIGHT == 3:
            weight = cfg.ROOT_DIR + '/Weights/sd_TIN/HOI_iter_1800000.ckpt'  # from our model with d

    # output directory where the logs are saved
    tb_dir = cfg.ROOT_DIR + '/logs/' + args.model + '/'

    # output directory where the models are saved
    output_dir = cfg.ROOT_DIR + '/Weights/' + args.model

    net = ResNet50()
    train_net(net, Trainval_GT, Trainval_N, output_dir, tb_dir, args.Pos_augment, args.Neg_select, args.Restore_flag,
              weight, max_iters=args.max_iters)
