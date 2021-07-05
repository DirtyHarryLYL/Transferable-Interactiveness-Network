# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import numpy as np
import tensorflow as tf
import argparse
import pickle
import json
import ipdb
import os

from ult.config import cfg
from networks.TIN_HICO_with_part import ResNet50
from models.test_Solver_HICO_DET_binary import test_net

os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # use GPU 0

def parse_args():
    parser = argparse.ArgumentParser(description='Test TIN on HICO dataset')
    parser.add_argument('--num_iteration', dest='iteration',
                        help='Specify which weight to load',
                        default=200000, type=int)
    parser.add_argument('--model', dest='model',
                        help='Select model',
                        default='TIN_10w_with_part', type=str)
    parser.add_argument('--object_thres', dest='object_thres',
                        help='Object threshold',
                        default=0.3, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
                        help='Human threshold',
                        default=0.8, type=float)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    Test_RCNN = pickle.load(open('Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose_with_part.pkl', "rb"))

    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    print('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(
        args.iteration) + ', path = ' + weight)

    output_file = '-Results/' + str(args.iteration) + '_' + args.model + '.pkl'
    print('output file = ', output_file)

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)

    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')

    test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres)
    sess.close()
