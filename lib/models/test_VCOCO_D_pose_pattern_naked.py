# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config_vcoco import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose
from ult.apply_prior import apply_prior

import cv2
import pickle
import numpy as np
import os
import sys
import glob
import time
import ipdb

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'v-coco/coco/images/val2014/COCO_val2014_' + (str(image_id)).zfill(12) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection):

    im_orig, im_shape = get_blob(image_id)
    
    blobs = {}
    blobs['H_num']       = 1
   
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            
            # Predict actrion using human appearance only
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)

            prediction_H = net.test_image_H(sess, im_orig, blobs)

            # save image information
            dic = {}
            dic['image_id']   = image_id
            dic['person_box'] = Human_out[2]
            dic['H_det'] = np.max(Human_out[5])
            dic['H_Score'] = prediction_H

            # binary score for all objects in image
            Binary_Score = []

            # all object bboxes in image
            Object_bbox = []

            # classes for objects
            Object_class = []

            # object detection score
            Object_det = []

            # prediction_HO
            HO_Score = []

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object

                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)

                    prediction_HO, prediction_binary  = net.test_image_HO(sess, im_orig, blobs)

                    # remain iCAN format
                    prediction_HO = [prediction_HO]

                    Binary_Score.append(prediction_binary[0])
                    Object_bbox.append(Object[2])
                    Object_class.append(Object[4])
                    HO_Score.append(prediction_HO)
                    Object_det.append(np.max(Object[5]))

            # There is only a single human detected in this image. I just ignore it. Might be better to add Nan as object box.
            if len(Object_bbox) == 0:
                continue

            # binary score for all objects in image
            dic['binary_score'] = Binary_Score

            # all object bboxes in image
            dic['object_box'] = Object_bbox

            # classes for objects 
            dic['object_class'] = Object_class

            # object detection score
            dic['O_det'] = Object_det

            # prediction_HO
            dic['HO_Score'] = HO_Score

            detection.append(dic)


def test_net(sess, net, Test_RCNN, prior_mask, Action_dic_inv, output_dir, object_thres, human_thres, prior_flag):


    np.random.seed(cfg.RNG_SEED)
    detection = []
    count = 0

    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}


    for line in open(cfg.DATA_DIR + '/' + '/v-coco/data/splits/vcoco_test.ids', 'r'):

        _t['im_detect'].tic()
 
        image_id   = int(line.rstrip())
        
        im_detect(sess, net, image_id, Test_RCNN, prior_mask, Action_dic_inv, object_thres, human_thres, prior_flag, detection)

        _t['im_detect'].toc()

        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 4946, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )




