# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import tensorflow as tf
import numpy as np
import argparse
import pickle
import json
import ipdb
import os

from networks.TIN_HICO import ResNet50
from ult.config import cfg
from models.test_HICO_pose_pattern_all_wise_pair import test_net

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # use GPU 0 

def parse_args():
    parser = argparse.ArgumentParser(description='Test TIN on HICO dataset')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=1700000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_HICO', type=str)
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

    # test detections result
    Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl', "rb" ) ) 

    # pretrain model
    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
    output_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '_' + str(args.human_thres) + '_' + str(args.object_thres) +  'new_pair_selection_H2O2.pkl'

    # init session

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)

    net = ResNet50()
    net.create_architecture(False)
    
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, Test_RCNN, output_file, args.object_thres, args.human_thres)
    sess.close()

  
    # thres_X and thres_Y indicate the NIS threshold to suppress the pair which might be no-interaction
    thres_x = 0.1
    thres_y = 0.9

    os.chdir(cfg.ROOT_DIR + '/HICO-DET_Benchmark/')
    os.system("python Generate_HICO_detection_nis.py " + output_file + ' ' + cfg.ROOT_DIR + "/-Results/"+ args.model + "NIS_thres_x" + str(thres_x) + "_y" + str(thres_y) + "/ " + str(thres_y) + " " + str(thres_x))
    os.system('matlab -nodesktop -nosplash -r "Generate_detection '+ cfg.ROOT_DIR + '/-Results/'+ args.model+ 'NIS_thres_x'+ str(thres_x)+ '_y'+ str(thres_y)+ '/;quit;"')

