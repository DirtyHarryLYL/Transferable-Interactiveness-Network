# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
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


def get_blob(image_id):
    im_file = cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/HICO_test2015_' + (str(image_id)).zfill(
        8) + '.jpg'
    im = cv2.imread(im_file)
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS
    im_shape = im_orig.shape
    im_orig = im_orig.reshape(1, im_shape[0], im_shape[1], 3)
    return im_orig, im_shape


def im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection):
    # save image information
    This_image = []

    im_orig, im_shape = get_blob(image_id)

    for h_id, Human_out in enumerate(Test_RCNN[image_id]):
        if (np.max(Human_out[5]) > human_thres) and (Human_out[1] == 'Human'):  # This is a valid human
            blobs = {}
            blobs['H_num'] = 0
            blobs['H_boxes'] = []
            blobs['O_boxes'] = []
            blobs['sp'] = []
            blobs['binary_label_10v'] = []
            for i in range(10):
                blobs['Part%d' % i] = []
            keys = []
            odets = []

            for o_id, Object in enumerate(Test_RCNN[image_id]):

                if (np.max(Object[5]) > object_thres) and not (
                        np.all(Object[2] == Human_out[2])):  # This is a valid object
                    # 1.the object detection result should > thres  2.the bbox detected is not an object
                    blobs['H_boxes'].append(
                        np.array([0, Human_out[2][0], Human_out[2][1], Human_out[2][2], Human_out[2][3]]))
                    blobs['O_boxes'].append(
                        np.array([0, Object[2][0], Object[2][1], Object[2][2], Object[2][3]]))

                    if Human_out[6] is None:
                        blobs['sp'].append(
                            Get_next_sp_with_pose(np.array(Human_out[2]), np.array(Object[2]),
                                                  None).reshape(64, 64, 3))
                    else:
                        blobs['sp'].append(
                            Get_next_sp_with_pose(np.array(Human_out[2]), np.array(Object[2]),
                                                  np.array(Human_out[6])).reshape(64, 64, 3))

                    for i in range(10):
                        blobs['Part%d' % i].append(
                            np.array([0, Human_out[7][i][0], Human_out[7][i][1], Human_out[7][i][2],
                                      Human_out[7][i][3]]))

                    blobs['H_num'] += 1
                    blobs['binary_label_10v'].append(np.zeros((10, 2)))
                    keys.append(o_id)
                    odets.append(Object[5])

            keys = np.array(keys)

            if blobs['H_num'] == 0:
                continue

            prediction_binary = net.test_image_binary(sess, im_orig, blobs)

            for i in range(len(keys)):
                temp = []
                Object = Test_RCNN[image_id][keys[i]]
                temp.append(Human_out[2])  # Human box
                temp.append(Object[2])  # Object box
                temp.append(Object[4])  # Object class
                temp.append(np.array(prediction_binary)[:, i])  # Score (600)
                temp.append(Human_out[5])  # Human score
                temp.append(Object[5])  # Object score
                This_image.append(temp)

    detection[image_id] = This_image


def test_net(sess, net, Test_RCNN, output_dir, object_thres, human_thres):
    np.random.seed(cfg.RNG_SEED)
    detection = {}
    count = 0
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for line in glob.iglob(cfg.DATA_DIR + '/' + 'hico_20160224_det/images/test2015/*.jpg'):
        _t['im_detect'].tic()

        image_id = int(line[-9:-4])

        im_detect(sess, net, image_id, Test_RCNN, object_thres, human_thres, detection)

        _t['im_detect'].toc()
        print('im_detect: {:d}/{:d} {:.3f}s'.format(count + 1, 9658, _t['im_detect'].average_time))
        count += 1

    pickle.dump(detection, open(output_dir, "wb"))
