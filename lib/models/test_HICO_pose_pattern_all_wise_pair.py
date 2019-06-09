# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ult.config import cfg
from ult.timer import Timer
from ult.ult import Get_next_sp_with_pose

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

human_num_thres = 4
object_num_thres = 4

def get_blob(image_id):
    im_file  = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(8) + '.jpg'
    im       = cv2.imread(im_file)
    im_orig  = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig  = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape

def im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):

    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id) 
    blobs = {}
    blobs['H_num']       = 1
   
    for Human_out in Test_RCNN[image_id]:
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'): # This is a valid human
            
            blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)

            for Object in Test_RCNN[image_id]:
                if (np.max(Object[5]) > object_thres) and not (np.all(Object[2] == Human_out[2])): # This is a valid object
                    # 1.the object detection result should > thres  2.the bbox detected is not an object
 
                    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                    blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)
                    

                    prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)

                    #print ("prediction_binary",prediction_binary[0])
                    #print ("prediction_HO",prediction_HO[0])

                    temp = []
                    temp.append(Human_out[2])           # Human box
                    temp.append(Object[2])              # Object box
                    temp.append(Object[4])              # Object class
                    temp.append(prediction_HO[0])       # Score (600)
                    temp.append(Human_out[5])           # Human score
                    temp.append(Object[5])              # Object score
                    temp.append(prediction_binary[0])   # binary score
                    # print("the HO score is",prediction_HO[0][0])
                    # print("the binary score is",prediction_binary[0][0])
                    This_image.append(temp)

    if len(This_image) == 0:

        print('Dealing with zero-sample test Image '+str(image_id))

        list_human_included = []
        list_object_included = []
        Human_out_list = []
        Object_list = []

        test_pair_all = Test_RCNN[image_id]
        length = len(test_pair_all)

        flag_continue_searching = 1

        while (len(list_human_included) < human_num_thres) or (len(list_object_included) < object_num_thres):
            h_max = [-1, -1.0]
            o_max = [-1, -1.0]
            flag_continue_searching = 0
            for i in range(length):
                if test_pair_all[i][1] == 'Human':
                    if (np.max(test_pair_all[i][5]) > h_max[1]) and not (i in list_human_included) and len(list_human_included) < human_num_thres:
                        h_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1
                else:
                    if np.max(test_pair_all[i][5]) > o_max[1] and not (i in list_object_included) and len(list_object_included) < object_num_thres:
                        o_max = [i, np.max(test_pair_all[i][5])]
                        flag_continue_searching = 1

            if flag_continue_searching == 0:
                break

            list_human_included.append(h_max[0])
            list_object_included.append(o_max[0])

            Human_out_list.append(test_pair_all[h_max[0]])
            Object_list.append(test_pair_all[o_max[0]])


        for Human_out in Human_out_list:
            for Object in Object_list:
                blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
                blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
                blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)
                    
                prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)

                #This_image = []

                temp = []
                temp.append(Human_out[2])           # Human box
                temp.append(Object[2])              # Object box
                temp.append(Object[4])              # Object class
                temp.append(prediction_HO[0])       # Score (600)
                temp.append(Human_out[5])           # Human score
                temp.append(Object[5])              # Object score
                temp.append(prediction_binary[0])   # binary score
                This_image.append(temp)
            
    detection[image_id] = This_image


'''
def im_detect_remaining(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):

    # for those pairs which do not have any pairs with given threshold
    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id)

    num_inter = 0
    num_no_inter = 0
    
    blobs = {}
    blobs['H_num']       = 1

    if image_id not in all_remaining:
        return 0, 0
    
    h_max = [-1, -1.0]
    o_max = [-1, -1.0]

    test_pair_all = Test_RCNN[image_id]

    length = len(test_pair_all)
    for i in range(length):
        if test_pair_all[i][1] == 'Human':
            if np.max(test_pair_all[i][5]) > h_max[1]:
                h_max = [i, np.max(test_pair_all[i][5])]
        else:
            if np.max(test_pair_all[i][5]) > o_max[1]:
                o_max = [i, np.max(test_pair_all[i][5])]

    Human_out = test_pair_all[h_max[0]]
    Object = test_pair_all[o_max[0]]
            
    blobs['H_boxes'] = np.array([0, Human_out[2][0],  Human_out[2][1],  Human_out[2][2],  Human_out[2][3]]).reshape(1,5)
    blobs['O_boxes'] = np.array([0, Object[2][0],  Object[2][1],  Object[2][2],  Object[2][3]]).reshape(1,5)
    blobs['sp']      = Get_next_sp_with_pose(Human_out[2], Object[2], Human_out[6]).reshape(1, 64, 64, 3)
                    
    prediction_HO, prediction_binary = net.test_image_HO(sess, im_orig, blobs)

    This_image = []

    temp = []
    temp.append(Human_out[2])           # Human box
    temp.append(Object[2])              # Object box
    temp.append(Object[4])              # Object class
    temp.append(prediction_HO[0])       # Score (600)
    temp.append(Human_out[5])           # Human score
    temp.append(Object[5])              # Object score
    temp.append(prediction_binary[0])   # binary score

    This_image.append(temp)
            
    detection[image_id] = This_image

    return num_inter, num_no_inter
'''

def test_net(sess, net, Test_RCNN, output_dir, object_thres, human_thres):


    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):

        _t['im_detect'].tic()
 
        image_id   = int(line[-9:-4])

        #if image_id in all_remaining:
        #    im_detect_remaining(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)
        #    print('dealing with remaining image')
        #else:
        #    im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)
        im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    pickle.dump( detection, open( output_dir, "wb" ) )




