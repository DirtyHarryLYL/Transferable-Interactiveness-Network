# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import arg_scope
from tensorflow.contrib.slim.python.slim.nets import resnet_utils
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.python.framework import ops

from ult.config import cfg
from ult.visualization import draw_bounding_boxes_HOI

import numpy as np
import ipdb


def resnet_arg_scope(is_training=True,
                     weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': False,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': False,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }
    with arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            weights_initializer=slim.variance_scaling_initializer(),
            biases_regularizer=tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
            biases_initializer=tf.constant_initializer(0.0),
            trainable=is_training,
            activation_fn=tf.nn.relu,
            normalizer_fn=slim.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc


class ResNet50():  # 64--128--256--512--512
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image = tf.placeholder(tf.float32, shape=[1, None, None, 3], name='image')
        self.spatial = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='sp')
        self.H_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='H_boxes')
        self.O_boxes = tf.placeholder(tf.float32, shape=[None, 5], name='O_boxes')
        self.Part0 = tf.placeholder(tf.float32, shape=[None, 5], name='Part0_boxes')
        self.Part1 = tf.placeholder(tf.float32, shape=[None, 5], name='Part1_boxes')
        self.Part2 = tf.placeholder(tf.float32, shape=[None, 5], name='Part2_boxes')
        self.Part3 = tf.placeholder(tf.float32, shape=[None, 5], name='Part3_boxes')
        self.Part4 = tf.placeholder(tf.float32, shape=[None, 5], name='Part4_boxes')
        self.Part5 = tf.placeholder(tf.float32, shape=[None, 5], name='Part5_boxes')
        self.Part6 = tf.placeholder(tf.float32, shape=[None, 5], name='Part6_boxes')
        self.Part7 = tf.placeholder(tf.float32, shape=[None, 5], name='Part7_boxes')
        self.Part8 = tf.placeholder(tf.float32, shape=[None, 5], name='Part8_boxes')
        self.Part9 = tf.placeholder(tf.float32, shape=[None, 5], name='Part9_boxes')
        self.gt_binary_label = tf.placeholder(tf.float32, shape=[None, 2], name='gt_binary_label')
        self.gt_binary_label_10v = tf.placeholder(tf.float32, shape=[None, 10, 2], name='gt_binary_label')

        self.num_vec = 10
        self.H_num = tf.placeholder(tf.int32)
        self.binary_weight = np.array([1.6094379124341003, 0.22314355131420976], dtype='float32').reshape(1, 2)
        self.num_classes = 600  # HOI
        self.num_binary = 2  # existence (0 or 1) of HOI
        self.num_fc = 1024
        self.scope = 'resnet_v1_50'
        self.stride = [16, ]
        self.lr = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks = [resnet_utils.Block('block1', resnet_v1.bottleneck, [(256, 64, 1)] * 2 + [(256, 64, 2)]),
                           resnet_utils.Block('block2', resnet_v1.bottleneck, [(512, 128, 1)] * 3 + [(512, 128, 2)]),
                           resnet_utils.Block('block3', resnet_v1.bottleneck, [(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                           resnet_utils.Block('block4', resnet_v1.bottleneck, [(2048, 512, 1)] * 3),
                           resnet_utils.Block('block5', resnet_v1.bottleneck, [(2048, 512, 1)] * 3)]
        else:  # we use tf 1.2.0 here, Resnet-50
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64, num_units=3, stride=2),
                           # a resnet_v1 bottleneck block
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1),  # feature former
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1')  # conv2d + subsample
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

    # Number of fixed blocks during training, by default ***the first of all 4 blocks*** is fixed (Resnet-50 block)
    # Range: 0 (none) to 3 (all)
    # __C.RESNET.FIXED_BLOCKS = 1

    # feature extractor
    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS],  # now 1, block 1
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2],  # now 1, block 2~3
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head

    # spatial configuration, conv-pool-conv-pool-flatten
    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            # (num_pos_neg,64,64,2)->(num_pos_neg,60,60,64)
            conv1_sp = slim.conv2d(self.spatial[:, :, :, 0:2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            # (num_pos_neg,60,60,64)->(num_pos_neg,30,30,64)
            pool1_sp = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            # (num_pos_neg,30,30,64)->(num_pos_neg,26,26,32)
            conv2_sp = slim.conv2d(pool1_sp, 32, [5, 5], padding='VALID', scope='conv2_sp')
            # (num_pos_neg,26,26,32)->(num_pos_neg,13,13,32)
            pool2_sp = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            # (num_pos_neg,13,13,32)->(num_pos_neg,5408)
            pool2_flat_sp = slim.flatten(pool2_sp)

        return pool2_flat_sp

    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_H, _ = resnet_v1.resnet_v1(pool5_H,  # H input, one block
                                           self.blocks[-2:-1],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])

            fc7_O, _ = resnet_v1.resnet_v1(pool5_O,  # O input, one block
                                           self.blocks[-1:],
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])

        return fc7_H, fc7_O

    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH = tf.concat([fc7_H, fc7_SH], 1)
            fc8_SH = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH')  # fc size = 1024
            fc8_SH = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH = slim.dropout(fc9_SH, keep_prob=0.5, is_training=is_training, scope='dropout9_SH')

            Concat_SO = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO = slim.dropout(fc9_SO, keep_prob=0.5, is_training=is_training, scope='dropout9_SO')

            Concat_SHsp = tf.concat([fc7_H, sp], 1)
            Concat_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp = slim.dropout(fc7_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

        return fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO

    def crop_pool_layer(self, bottom, rois, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            batch_ids = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size],
                                                 name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids),
                                                 [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.variable_scope(name) as scope:
            fc1 = slim.fully_connected(fc7_H, 512, scope='fc1_b')
            fc1 = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1 = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att

    def attention_norm_H(self, att, name):
        with tf.variable_scope(name) as scope:
            att = tf.transpose(att, [0, 3, 1, 2])
            att_shape = tf.shape(att)
            att = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att = tf.nn.softmax(att)
            att = tf.reshape(att, att_shape)
            att = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:
            fc1 = slim.fully_connected(fc7_O, 512, scope='fc1_b')
            fc1 = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1 = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att

    def attention_norm_O(self, att, name):
        with tf.variable_scope(name) as scope:
            att = tf.transpose(att, [0, 3, 1, 2])
            att_shape = tf.shape(att)
            att = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att = tf.nn.softmax(att)  ###
            att = tf.reshape(att, att_shape)
            att = tf.transpose(att, [0, 2, 3, 1])
        return att

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:
            if reuse:
                scope.reuse_variables()

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name)  # 1x1, 1024, fc

        return head_bottleneck

    def ho_att(self, head, fc7_H, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:
            # whole image feature
            head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')

            # whole image feature
            head_g = slim.conv2d(head, 512, [1, 1], scope='head_g')

            Att_H = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')
            Att_H = self.attention_norm_H(Att_H, 'Norm_Att_H')  # softmax
            att_head_H = tf.multiply(head_g, Att_H)

            Att_O = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
            Att_O = self.attention_norm_O(Att_O, 'Norm_Att_O')  # softmax
            att_head_O = tf.multiply(head_g, Att_O)

            pool5_SH = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
            pool5_SO = self.bottleneck(att_head_O, is_training, 'bottleneck', True)
            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            return fc7_SH, fc7_SO

    def ROI_for_parts(self, head, name):
        with tf.variable_scope(name) as scope:
            pool5_P0 = self.crop_pool_layer(head, self.Part0, 'crop_P0')  # RAnk
            pool5_P1 = self.crop_pool_layer(head, self.Part1, 'crop_P1')  # RKnee
            pool5_P2 = self.crop_pool_layer(head, self.Part2, 'crop_P2')  # LKnee
            pool5_P3 = self.crop_pool_layer(head, self.Part3, 'crop_P3')  # LAnk
            pool5_P4 = self.crop_pool_layer(head, self.Part4, 'crop_P4')  # Hip
            pool5_P5 = self.crop_pool_layer(head, self.Part5, 'crop_P5')  # Head
            pool5_P6 = self.crop_pool_layer(head, self.Part6, 'crop_P6')  # RHand
            pool5_P7 = self.crop_pool_layer(head, self.Part7, 'crop_P7')  # RSho
            pool5_P8 = self.crop_pool_layer(head, self.Part8, 'crop_P8')  # LSho
            pool5_P9 = self.crop_pool_layer(head, self.Part9, 'crop_P9')  # LHand

        return pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6, pool5_P7, pool5_P8, pool5_P9

    def vec_classification(self, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6, pool5_P7,
                           pool5_P8, pool5_P9, is_training, initializer,
                           name):
        with tf.variable_scope(name) as scope:
            pool5_all = tf.concat(
                [pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6, pool5_P7, pool5_P8, pool5_P9],
                axis=3)  # (?, 7, 7, 45*6)
            pool5_all_flat = slim.flatten(pool5_all)  # (?, 7*7*45*6)

            fc8_vec_roi = slim.fully_connected(pool5_all_flat, 1024, weights_initializer=initializer,
                                               trainable=is_training)
            fc8_vec_roi = slim.dropout(fc8_vec_roi, keep_prob=0.5, is_training=is_training)
            fc9_vec_roi = slim.fully_connected(fc8_vec_roi, 1024, weights_initializer=initializer,
                                               trainable=is_training)
            fc9_vec_roi = slim.dropout(fc9_vec_roi, keep_prob=0.5, is_training=is_training)
            cls_score_vec = slim.fully_connected(fc9_vec_roi, self.num_vec,
                                                 weights_initializer=initializer,
                                                 trainable=is_training,
                                                 activation_fn=None)
            cls_prob_vec = tf.nn.sigmoid(cls_score_vec)

            tf.reshape(cls_prob_vec, [1, self.num_vec])

        self.predictions["cls_score_vec"] = cls_score_vec
        self.predictions["cls_prob_vec"] = cls_prob_vec
        return cls_prob_vec

    def vec_attention(self, cls_prob_vec, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6,
                      pool5_P7, pool5_P8, pool5_P9, is_training,
                      name):
        with tf.variable_scope(name) as scope:
            # att * roi feature, not the fc feature
            pool5_P0 = tf.reduce_mean(pool5_P0, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P1 = tf.reduce_mean(pool5_P1, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P2 = tf.reduce_mean(pool5_P2, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P3 = tf.reduce_mean(pool5_P3, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P4 = tf.reduce_mean(pool5_P4, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P5 = tf.reduce_mean(pool5_P5, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P6 = tf.reduce_mean(pool5_P6, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P7 = tf.reduce_mean(pool5_P7, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P8 = tf.reduce_mean(pool5_P8, axis=[1, 2]) #(?,7,7,64)->(?,64)
            pool5_P9 = tf.reduce_mean(pool5_P9, axis=[1, 2]) #(?,7,7,64)->(?,64)

            pool5_P0 = tf.multiply(pool5_P0, cls_prob_vec[:, 0:1])
            pool5_P1 = tf.multiply(pool5_P1, cls_prob_vec[:, 1:2])
            pool5_P2 = tf.multiply(pool5_P2, cls_prob_vec[:, 2:3])
            pool5_P3 = tf.multiply(pool5_P3, cls_prob_vec[:, 3:4])
            pool5_P4 = tf.multiply(pool5_P4, cls_prob_vec[:, 4:5])
            pool5_P5 = tf.multiply(pool5_P5, cls_prob_vec[:, 5:6])
            pool5_P6 = tf.multiply(pool5_P6, cls_prob_vec[:, 6:7])
            pool5_P7 = tf.multiply(pool5_P7, cls_prob_vec[:, 7:8])
            pool5_P8 = tf.multiply(pool5_P8, cls_prob_vec[:, 8:9])
            pool5_P9 = tf.multiply(pool5_P9, cls_prob_vec[:, 9:10]) # (?,64)
            
            pool5_P_att = tf.concat([pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6,
                                     pool5_P7, pool5_P8, pool5_P9], axis=1) #(?,640)

        return pool5_P_att

    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, fc7_P, sp, is_training, name):
        with tf.variable_scope(name) as scope:

            # print('---<', name, '>---')
            # print("fc7_P.shape", fc7_P.shape) #(?,64) or (?,31360)

            conv1_pose_map = slim.conv2d(self.spatial[:, :, :, 2:], 32, [5, 5], padding='VALID', scope='conv1_pose_map')
            pool1_pose_map = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map = slim.conv2d(pool1_pose_map, 16, [5, 5], padding='VALID', scope='conv2_pose_map')
            pool2_pose_map = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)

            # fc7_H + fc7_SH + sp---fc1024---fc8_binary_1
            fc_binary_1 = tf.concat([fc7_H, fc7_SH, fc7_P], 1)  # (?,2048+1024+64=3136)
            fc_binary_1 = tf.concat([fc_binary_1, sp, pool2_flat_pose_map], 1)  # (?,11248)
            fc8_binary_1 = slim.fully_connected(fc_binary_1, 1024, scope='fc8_binary_1')  # 1024? we should try
            fc8_binary_1 = slim.dropout(fc8_binary_1, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                        scope='dropout8_binary_1')  # [pos + neg,1024]
            

            # fc7_O + fc7_SO---fc1024---fc8_binary_2
            fc_binary_2 = tf.concat([fc7_O, fc7_SO], 1)  # (?, 3072)
            fc8_binary_2 = slim.fully_connected(fc_binary_2, 1024, scope='fc8_binary_2')
            fc8_binary_2 = slim.dropout(fc8_binary_2, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                        scope='dropout8_binary_2')  # (?,1024)

            # fc8_binary_1 + fc8_binary_2---fc1024---fc9_binary
            fc8_binary = tf.concat([fc8_binary_1, fc8_binary_2], 1) # (?,2048)
            fc9_binary = slim.fully_connected(fc8_binary, 1024, scope='fc9_binary')
            fc9_binary = slim.dropout(fc9_binary, keep_prob=cfg.TRAIN_DROP_OUT_BINARY, is_training=is_training,
                                      scope='dropout9_binary')
            
            # print('---</', name, '>---')

        return fc9_binary

    def binary_classification(self, fc9_binary, is_training, initializer, name, mode=True):
        with tf.variable_scope(name) as scope:
            cls_score_binary = slim.fully_connected(fc9_binary, self.num_binary,
                                                    weights_initializer=initializer,
                                                    trainable=is_training,
                                                    activation_fn=None, scope='cls_score_binary')
            tf.reshape(cls_score_binary, [1, self.num_binary])
            cls_prob_binary = tf.nn.sigmoid(cls_score_binary, name='cls_prob_binary')
            if mode:  # restore instance binary score in predictions , for part , not change
                self.predictions["cls_score_binary"] = cls_score_binary
                self.predictions["cls_prob_binary"] = cls_prob_binary

        return cls_score_binary

    def binary_cls_part(self, fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P, sp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            fc7_P = tf.reduce_mean(pool5_P, axis=[1, 2]) # (?, 7,7,64)->(?, 64)

            fc9_P = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, fc7_P, sp, is_training, 'fc_0')
            cls_score_binary_P = self.binary_classification(fc9_P, is_training, initializer, 'cls_P', False)
            tf.reshape(cls_score_binary_P, [1, self.num_binary])

        return cls_score_binary_P

    def binary_cls_all(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4,
                       pool5_P5, pool5_P6, pool5_P7, pool5_P8, pool5_P9, is_training, initializer, name):

        with tf.variable_scope(name) as scope:
            cls_score_binary0 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P0, sp, is_training,
                                                     initializer, 'score_0')  # (None,2)
            cls_score_binary1 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P1, sp, is_training,
                                                     initializer, 'score_1')
            cls_score_binary2 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P2, sp, is_training,
                                                     initializer, 'score_2')
            cls_score_binary3 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P3, sp, is_training,
                                                     initializer, 'score_3')
            cls_score_binary4 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P4, sp, is_training,
                                                     initializer, 'score_4')
            cls_score_binary5 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P5, sp, is_training,
                                                     initializer, 'score_5')
            cls_score_binary6 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P6, sp, is_training,
                                                     initializer, 'score_6')
            cls_score_binary7 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P7, sp, is_training,
                                                     initializer, 'score_7')
            cls_score_binary8 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P8, sp, is_training,
                                                     initializer, 'score_8')
            cls_score_binary9 = self.binary_cls_part(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P9, sp, is_training,
                                                     initializer, 'score_9')

            cls_score_binary_all = tf.concat(
                [tf.expand_dims(cls_score_binary0, 1), tf.expand_dims(cls_score_binary1, 1),
                 tf.expand_dims(cls_score_binary2, 1), tf.expand_dims(cls_score_binary3, 1),
                 tf.expand_dims(cls_score_binary4, 1), tf.expand_dims(cls_score_binary5, 1),
                 tf.expand_dims(cls_score_binary6, 1), tf.expand_dims(cls_score_binary7, 1),
                 tf.expand_dims(cls_score_binary8, 1), tf.expand_dims(cls_score_binary9, 1)
                 ], axis=1)  # (None,10,2)

            cls_score_binary_all = tf.reduce_max(cls_score_binary_all, axis=1, name='cls_score_binary_all')  # (None,2)
            self.predictions["cls_score_binary_all"] = cls_score_binary_all

            cls_prob_binary_all = tf.nn.sigmoid(cls_score_binary_all, name='cls_prob_binary_all')
            self.predictions["cls_prob_binary_all"] = cls_prob_binary_all

        return cls_score_binary_all

    def compress(self, head, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            head = slim.conv2d(head, 64, [1, 1], padding='VALID', scope='dim_red_head')
            return head

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # ResNet Backbone
        head = self.image_to_head(is_training)  # (1, ?, ?, 1024)
        sp = self.sp_to_head()  # (?,5408)
        
        pool5_H = self.crop_pool_layer(head, self.H_boxes, 'Crop_H')  # (?, 7, 7, 1024)
        pool5_O = self.crop_pool_layer(head, self.O_boxes, 'Crop_O')  # (?, 7, 7, 1024)
        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, sp, is_training, 'res5')  # (?, 2048)
        fc7_SH, fc7_SO = self.ho_att(head, fc7_H, fc7_O, is_training, 'ho_att') # (?, 1024)

        head = self.compress(head, is_training, initializer, 'Compress') # (1, ?, ?, 64)

        pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6, pool5_P7, pool5_P8, pool5_P9 = self.ROI_for_parts(
            head, 'ROI_for_parts') # (?,7,7,64)

        # get 10 part binary score, ultilize reduce-max
        cls_score_binary_all = self.binary_cls_all(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, pool5_P0, pool5_P1, pool5_P2,
                                                   pool5_P3, pool5_P4, pool5_P5, pool5_P6, pool5_P7, pool5_P8, pool5_P9,
                                                   is_training, initializer,
                                                   'binary_cls_all')

        # get 10 part att score, for part-level attention as human feature
        cls_prob_vec = self.vec_classification(pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5, pool5_P6,
                                               pool5_P7, pool5_P8, pool5_P9, is_training,
                                               initializer=initializer, name='vec_classification')  # 10v
        pool5_P_att = self.vec_attention(cls_prob_vec, pool5_P0, pool5_P1, pool5_P2, pool5_P3, pool5_P4, pool5_P5,
                                         pool5_P6, pool5_P7, pool5_P8, pool5_P9, is_training, name='vec_attention_roi')

        # concat attention to cls binary
        fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, pool5_P_att, sp, is_training, 'fc_binary')
        cls_score_binary = self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')

        self.score_summaries.update(self.predictions)
        # self.visualize["attention_map_H"] = (Att_H - tf.reduce_min(Att_H[0,:,:,:])) / tf.reduce_max((Att_H[0,:,:,:] - tf.reduce_min(Att_H[0,:,:,:])))
        # self.visualize["attention_map_O"] = (Att_O - tf.reduce_min(Att_O[0,:,:,:])) / tf.reduce_max((Att_O[0,:,:,:] - tf.reduce_min(Att_O[0,:,:,:])))

        return

    def create_architecture(self, is_training):

        self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            # val_summaries.append(self.add_gt_image_summary_H())
            # val_summaries.append(self.add_gt_image_summary_HO())
            # tf.summary.image('ATTENTION_MAP_H',  self.visualize["attention_map_H"], max_outputs=1)
            # tf.summary.image('ATTENTION_MAP_O',  self.visualize["attention_map_O"], max_outputs=1)
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            # for key, var in self.score_summaries.items():
            #    self.add_score_summary(key, var)
            # for var in self.train_summaries:
            #    self.add_train_summary(var)

        val_summaries.append(tf.summary.scalar('lr', self.lr))
        self.summary_op = tf.summary.merge_all()
        self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output

    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_binary = self.predictions["cls_score_binary"]
            cls_score_binary_all = self.predictions["cls_score_binary_all"]  # here use cls_score, not cls_prob
            cls_score_binary_with_weight = tf.multiply(cls_score_binary, self.binary_weight)
            cls_score_binary_all_with_weight = tf.multiply(cls_score_binary_all, self.binary_weight)

            label_binary = self.gt_binary_label

            binary_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary_with_weight))
            binary_all_cross_entropy = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label_binary, logits=cls_score_binary_all_with_weight))
            mse = tf.losses.mean_squared_error(cls_score_binary_with_weight, cls_score_binary_all_with_weight)
            loss = binary_cross_entropy + binary_all_cross_entropy + mse

            self.losses['binary_cross_entropy'] = binary_cross_entropy
            self.losses['binary_all_cross_entropy'] = binary_all_cross_entropy
            self.losses['mse'] = mse
            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss

    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num'],
                     self.gt_binary_label: blobs['binary_label'], self.Part0: blobs['Part0'],
                     self.Part1: blobs['Part1'], self.Part2: blobs['Part2'], self.Part3: blobs['Part3'],
                     self.Part4: blobs['Part4'], self.Part5: blobs['Part5'], self.Part6: blobs['Part6'],
                     self.Part7: blobs['Part7'], self.Part8: blobs['Part8'], self.Part9: blobs['Part9'],
                     self.gt_binary_label_10v: blobs['binary_label_10v']}

        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                           feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num'],
                     self.gt_binary_label: blobs['binary_label'], self.Part0: blobs['Part0'],
                     self.Part1: blobs['Part1'], self.Part2: blobs['Part2'], self.Part3: blobs['Part3'],
                     self.Part4: blobs['Part4'], self.Part5: blobs['Part5'], self.Part6: blobs['Part6'],
                     self.Part7: blobs['Part7'], self.Part8: blobs['Part8'], self.Part9: blobs['Part9'],
                     self.gt_binary_label_10v: blobs['binary_label_10v']
                     }

        loss, summary, _ = sess.run([self.losses['total_loss'],
                                     self.summary_op,
                                     train_op],
                                    feed_dict=feed_dict)
        return loss, summary

    # return late fusion prediction, cls_prob
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'],
                     self.H_num: blobs['H_num'], self.Part0: blobs['Part0'],
                     self.Part1: blobs['Part1'], self.Part2: blobs['Part2'], self.Part3: blobs['Part3'],
                     self.Part4: blobs['Part4'], self.Part5: blobs['Part5'], self.Part6: blobs['Part6'],
                     self.Part7: blobs['Part7'], self.Part8: blobs['Part8'], self.Part9: blobs['Part9'],
                     self.gt_binary_label_10v: blobs['binary_label_10v']
                     }
        cls_prob_HO = sess.run([self.predictions["cls_prob_HO"]], feed_dict=feed_dict)
        return cls_prob_HO

    def test_image_binary(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'],
                     self.spatial: blobs['sp'],
                     self.H_num: blobs['H_num'], self.Part0: blobs['Part0'],
                     self.Part1: blobs['Part1'], self.Part2: blobs['Part2'], self.Part3: blobs['Part3'],
                     self.Part4: blobs['Part4'], self.Part5: blobs['Part5'], self.Part6: blobs['Part6'],
                     self.Part7: blobs['Part7'], self.Part8: blobs['Part8'], self.Part9: blobs['Part9'],
                     self.gt_binary_label_10v: blobs['binary_label_10v']
                     }
        cls_prob_binary = sess.run([self.predictions["cls_prob_binary"]], feed_dict=feed_dict)

        return cls_prob_binary
