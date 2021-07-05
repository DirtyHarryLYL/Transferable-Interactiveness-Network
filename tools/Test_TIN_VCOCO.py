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
import json
import ipdb
import os

# for error of 'ImportError: No module named pycocotools.coco'
from ult.config_vcoco import cfg
import sys
sys.path.append(cfg.ROOT_DIR + '/Data/v-coco/coco/PythonAPI')

from pycocotools.coco import COCO # for test
from ult.vsrl_eval_output_txt import VCOCOeval
from models.test_VCOCO_D_pose_pattern_naked import test_net

from networks.TIN_VCOCO import ResNet50

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' # use GPU 0, 1


def parse_args():
    parser = argparse.ArgumentParser(description='Test an TIN on VCOCO')
    parser.add_argument('--num_iteration', dest='iteration',
            help='Specify which weight to load',
            default=6000, type=int)
    parser.add_argument('--model', dest='model',
            help='Select model',
            default='TIN_VCOCO', type=str)
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag or not',
            default=3, type=int)
    parser.add_argument('--object_thres', dest='object_thres',
            help='Object threshold',
            default=0.4, type=float)
    parser.add_argument('--human_thres', dest='human_thres',
            help='Human threshold',
            default=0.6, type=float)

    args = parser.parse_args()
    return args

# prior flag: see ROOT/lib/ult/apply_prior.py for more detail
#     apply_prior   prior_mask
# 0        -             -          
# 1        Y             - 
# 2        -             Y
# 3        Y             Y


if __name__ == '__main__':

    args = parse_args()
  
    Test_RCNN      = pickle.load( open( cfg.DATA_DIR + '/' + 'Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl', "rb" ) )
    prior_mask     = pickle.load( open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) )
    Action_dic     = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y:x for x,y in Action_dic.iteritems()}

    vcocoeval      = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json', cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json', cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids')     

    weight = cfg.ROOT_DIR + '/Weights/' + args.model + '/HOI_iter_' + str(args.iteration) + '.ckpt'
    print ('Human thres = ' + str(args.human_thres) + ', Object thres = ' + str(args.object_thres) + ', iter = ' + str(args.iteration) + ', path = ' + weight ) 
  
    output_file = cfg.ROOT_DIR + '/-Results/' + str(args.iteration) + '_' + args.model + '_' + str(args.human_thres) + '_' + str(args.object_thres) + '_naked.pkl'

    # init session
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    sess = tf.Session(config=tfconfig)
        
    net = ResNet50()
    net.create_architecture(False)
    
    saver = tf.train.Saver()
    saver.restore(sess, weight)

    print('Pre-trained weights loaded.')
    
    test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_file, args.object_thres, args.human_thres, args.prior_flag)
    sess.close()
    
    os.system("python tools/Vcoco_lis_nis.py --prior_flag 3 --mode best --input_S " + output_file)
