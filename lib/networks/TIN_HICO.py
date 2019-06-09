# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
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
        weights_regularizer = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY),
        weights_initializer = slim.variance_scaling_initializer(),
        biases_regularizer  = tf.contrib.layers.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), 
        biases_initializer  = tf.constant_initializer(0.0),
        trainable           = is_training,
        activation_fn       = tf.nn.relu,
        normalizer_fn       = slim.batch_norm,
        normalizer_params   = batch_norm_params):
        with arg_scope([slim.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

"""Contains definitions for the original form of Residual Networks.
The 'v1' residual networks (ResNets) implemented in this module were proposed
by:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Other variants were introduced in:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027
The networks defined in this module utilize the bottleneck building block of
[1] with projection shortcuts only for increasing depths. They employ batch
normalization *after* every weight layer. This is the architecture used by
MSRA in the Imagenet and MSCOCO 2016 competition models ResNet-101 and
ResNet-152. See [2; Fig. 1a] for a comparison between the current 'v1'
architecture and the alternative 'v2' architecture of [2] which uses batch
normalization *before* every weight layer in the so-called full pre-activation
units.
Typical use:
   from tensorflow.contrib.slim.python.slim.nets import resnet_v1
ResNet-101 for image classification into 1000 classes:
   # inputs has shape [batch, 224, 224, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs, 1000, is_training=False)
ResNet-101 for semantic segmentation into 21 classes:
   # inputs has shape [batch, 513, 513, 3]
   with slim.arg_scope(resnet_v1.resnet_arg_scope()):
      net, end_points = resnet_v1.resnet_v1_101(inputs,
                                                21,
                                                is_training=False,
                                                global_pool=False,
                                                output_stride=16)
"""
"""
Helper function for creating a resnet_v1 bottleneck block.
  Args:
    scope: The scope of the block.
    base_depth: The depth of the bottleneck layer for each unit.
    num_units: The number of units in the block.
    stride: The stride of the block, implemented as a stride in the last unit.
      All other units have stride=1.
  Returns:
    A resnet_v1 bottleneck block.
"""
"""Generator for v1 ResNet models.
  This function generates a family of ResNet v1 models. See the resnet_v1_*()
  methods for specific model instantiations, obtained by selecting different
  block instantiations that produce ResNets of various depths.
  Training for image classification on Imagenet is usually done with [224, 224]
  inputs, resulting in [7, 7] feature maps at the output of the last ResNet
  block for the ResNets defined in [1] that have nominal stride equal to 32.
  However, for dense prediction tasks we advise that one uses inputs with
  spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
  this case the feature maps at the ResNet output will have spatial shape
  [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
  and corners exactly aligned with the input image corners, which greatly
  facilitates alignment of the features to the image. Using as input [225, 225]
  images results in [8, 8] feature maps at the output of the last ResNet block.
  For dense prediction tasks, the ResNet needs to run in fully-convolutional
  (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
  have nominal stride equal to 32 and a good choice in FCN mode is to use
  output_stride=16 in order to increase the density of the computed features at
  small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

  Args:
    inputs: A tensor of size [batch, height_in, width_in, channels].
    blocks: A list of length equal to the number of ResNet blocks. Each element
      is a resnet_utils.Block object describing the units in the block.
    num_classes: Number of predicted classes for classification tasks. If None
      we return the features before the logit layer.
    is_training: whether batch_norm layers are in training mode.
    global_pool: If True, we perform global average pooling before computing the
      logits. Set to True for image classification, False for dense prediction.
    output_stride: If None, then the output will be computed at the nominal
      network stride. If output_stride is not None, it specifies the requested
      ratio of input to output spatial resolution.
    include_root_block: If True, include the initial convolution followed by
      max-pooling, if False excludes it.
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.
    scope: Optional variable_scope.
  Returns:
    net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
      If global_pool is False, then height_out and width_out are reduced by a
      factor of output_stride compared to the respective height_in and width_in,
      else both height_out and width_out equal one. If num_classes is None, then
      net is the output of the last ResNet block, potentially after global
      average pooling. If num_classes is not None, net contains the pre-softmax
      activations.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: If the target output_stride is not valid.
"""
class ResNet50(): # 64--128--256--512--512
    def __init__(self):
        self.visualize = {}
        self.intermediate = {}
        self.predictions = {}
        self.score_summaries = {}
        self.event_summaries = {}
        self.train_summaries = []
        self.losses = {}

        self.image       = tf.placeholder(tf.float32, shape=[1, None, None, 3], name = 'image')
        self.spatial     = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name = 'sp')
        self.H_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'H_boxes')
        self.O_boxes     = tf.placeholder(tf.float32, shape=[None, 5], name = 'O_boxes')
        self.gt_class_HO = tf.placeholder(tf.float32, shape=[None, 600], name = 'gt_class_HO')
        self.gt_binary_label = tf.placeholder(tf.float32, shape=[None, 2], name = 'gt_binary_label')
        self.H_num       = tf.placeholder(tf.int32)
        self.HO_weight   = np.array([
                9.192927, 9.778443, 10.338059, 9.164914, 9.075144, 10.045923, 8.714437, 8.59822, 12.977117, 6.2745423, 
                11.227917, 6.765012, 9.436157, 9.56762, 11.0675745, 11.530198, 9.609821, 9.897503, 6.664475, 6.811699, 
                6.644726, 9.170454, 13.670264, 3.903943, 10.556748, 8.814335, 9.519224, 12.753973, 11.590822, 8.278912, 
                5.5245695, 9.7286825, 8.997436, 10.699849, 9.601237, 11.965516, 9.192927, 10.220277, 6.056692, 7.734048, 
                8.42324, 6.586457, 6.969533, 10.579222, 13.670264, 4.4531965, 9.326459, 9.288238, 8.071842, 10.431585, 
                12.417501, 11.530198, 11.227917, 4.0678477, 8.854023, 12.571651, 8.225684, 10.996116, 11.0675745, 10.100731, 
                7.0376034, 7.463688, 12.571651, 14.363411, 5.4902234, 11.0675745, 14.363411, 8.45805, 10.269067, 9.820116, 
                14.363411, 11.272368, 11.105314, 7.981595, 9.198626, 3.3284247, 14.363411, 12.977117, 9.300817, 10.032678, 
                12.571651, 10.114916, 10.471591, 13.264799, 14.363411, 8.01953, 10.412168, 9.644913, 9.981384, 7.2197933, 
                14.363411, 3.1178555, 11.031207, 8.934066, 7.546675, 6.386472, 12.060826, 8.862153, 9.799063, 12.753973, 
                12.753973, 10.412168, 10.8976755, 10.471591, 12.571651, 9.519224, 6.207762, 12.753973, 6.60636, 6.2896967, 
                4.5198326, 9.7887, 13.670264, 11.878505, 11.965516, 8.576513, 11.105314, 9.192927, 11.47304, 11.367679, 
                9.275815, 11.367679, 9.944571, 11.590822, 10.451388, 9.511381, 11.144535, 13.264799, 5.888291, 11.227917, 
                10.779892, 7.643191, 11.105314, 9.414651, 11.965516, 14.363411, 12.28397, 9.909063, 8.94731, 7.0330057, 
                8.129001, 7.2817025, 9.874775, 9.758241, 11.105314, 5.0690055, 7.4768796, 10.129305, 9.54313, 13.264799, 
                9.699972, 11.878505, 8.260853, 7.1437693, 6.9321113, 6.990665, 8.8104515, 11.655361, 13.264799, 4.515912, 
                9.897503, 11.418972, 8.113436, 8.795067, 10.236277, 12.753973, 14.363411, 9.352776, 12.417501, 0.6271591, 
                12.060826, 12.060826, 12.166186, 5.2946343, 11.318889, 9.8308115, 8.016022, 9.198626, 10.8976755, 13.670264, 
                11.105314, 14.363411, 9.653881, 9.503599, 12.753973, 5.80546, 9.653881, 9.592727, 12.977117, 13.670264, 
                7.995224, 8.639826, 12.28397, 6.586876, 10.929424, 13.264799, 8.94731, 6.1026597, 12.417501, 11.47304, 
                10.451388, 8.95624, 10.996116, 11.144535, 11.031207, 13.670264, 13.670264, 6.397866, 7.513285, 9.981384, 
                11.367679, 11.590822, 7.4348736, 4.415428, 12.166186, 8.573451, 12.977117, 9.609821, 8.601359, 9.055143, 
                11.965516, 11.105314, 13.264799, 5.8201604, 10.451388, 9.944571, 7.7855496, 14.363411, 8.5463, 13.670264, 
                7.9288645, 5.7561946, 9.075144, 9.0701065, 5.6871653, 11.318889, 10.252538, 9.758241, 9.407584, 13.670264, 
                8.570397, 9.326459, 7.488179, 11.798462, 9.897503, 6.7530537, 4.7828183, 9.519224, 7.6492405, 8.031909, 
                7.8180614, 4.451856, 10.045923, 10.83705, 13.264799, 13.670264, 4.5245686, 14.363411, 10.556748, 10.556748, 
                14.363411, 13.670264, 14.363411, 8.037262, 8.59197, 9.738439, 8.652985, 10.045923, 9.400566, 10.9622135, 
                11.965516, 10.032678, 5.9017305, 9.738439, 12.977117, 11.105314, 10.725825, 9.080208, 11.272368, 14.363411, 
                14.363411, 13.264799, 6.9279733, 9.153925, 8.075553, 9.126969, 14.363411, 8.903826, 9.488214, 5.4571533, 
                10.129305, 10.579222, 12.571651, 11.965516, 6.237189, 9.428937, 9.618479, 8.620408, 11.590822, 11.655361, 
                9.968962, 10.8080635, 10.431585, 14.363411, 3.796231, 12.060826, 10.302968, 9.551227, 8.75394, 10.579222, 
                9.944571, 14.363411, 6.272396, 10.625742, 9.690582, 13.670264, 11.798462, 13.670264, 11.724354, 9.993963, 
                8.230013, 9.100721, 10.374427, 7.865129, 6.514087, 14.363411, 11.031207, 11.655361, 12.166186, 7.419324, 
                9.421769, 9.653881, 10.996116, 12.571651, 13.670264, 5.912144, 9.7887, 8.585759, 8.272101, 11.530198, 8.886948, 
                5.9870906, 9.269661, 11.878505, 11.227917, 13.670264, 8.339964, 7.6763024, 10.471591, 10.451388, 13.670264, 
                11.185357, 10.032678, 9.313555, 12.571651, 3.993144, 9.379805, 9.609821, 14.363411, 9.709451, 8.965248, 
                10.451388, 7.0609145, 10.579222, 13.264799, 10.49221, 8.978916, 7.124196, 10.602211, 8.9743395, 7.77862, 
                8.073695, 9.644913, 9.339531, 8.272101, 4.794418, 9.016304, 8.012526, 10.674532, 14.363411, 7.995224, 
                12.753973, 5.5157638, 8.934066, 10.779892, 7.930471, 11.724354, 8.85808, 5.9025764, 14.363411, 12.753973, 
                12.417501, 8.59197, 10.513264, 10.338059, 14.363411, 7.7079706, 14.363411, 13.264799, 13.264799, 10.752493, 
                14.363411, 14.363411, 13.264799, 12.417501, 13.670264, 6.5661197, 12.977117, 11.798462, 9.968962, 12.753973, 
                11.47304, 11.227917, 7.6763024, 10.779892, 11.185357, 14.363411, 7.369478, 14.363411, 9.944571, 10.779892, 
                10.471591, 9.54313, 9.148476, 10.285873, 10.412168, 12.753973, 14.363411, 6.0308623, 13.670264, 10.725825, 
                12.977117, 11.272368, 7.663911, 9.137665, 10.236277, 13.264799, 6.715625, 10.9622135, 14.363411, 13.264799, 
                9.575919, 9.080208, 11.878505, 7.1863923, 9.366199, 8.854023, 9.874775, 8.2857685, 13.670264, 11.878505, 
                12.166186, 7.616999, 9.44343, 8.288065, 8.8104515, 8.347254, 7.4738197, 10.302968, 6.936267, 11.272368, 
                7.058223, 5.0138307, 12.753973, 10.173757, 9.863602, 11.318889, 9.54313, 10.996116, 12.753973, 7.8339925, 
                7.569945, 7.4427395, 5.560738, 12.753973, 10.725825, 10.252538, 9.307165, 8.491293, 7.9161053, 7.8849015, 
                7.782772, 6.3088884, 8.866243, 9.8308115, 14.363411, 10.8976755, 5.908519, 10.269067, 9.176025, 9.852551, 
                9.488214, 8.90809, 8.537411, 9.653881, 8.662968, 11.965516, 10.143904, 14.363411, 14.363411, 9.407584, 
                5.281472, 11.272368, 12.060826, 14.363411, 7.4135547, 8.920994, 9.618479, 8.891141, 14.363411, 12.060826, 
                11.965516, 10.9622135, 10.9622135, 14.363411, 5.658909, 8.934066, 12.571651, 8.614018, 11.655361, 13.264799, 
                10.996116, 13.670264, 8.965248, 9.326459, 11.144535, 14.363411, 6.0517673, 10.513264, 8.7430105, 10.338059, 
                13.264799, 6.878481, 9.065094, 8.87035, 14.363411, 9.92076, 6.5872955, 10.32036, 14.363411, 9.944571, 
                11.798462, 10.9622135, 11.031207, 7.652888, 4.334878, 13.670264, 13.670264, 14.363411, 10.725825, 12.417501, 
                14.363411, 13.264799, 11.655361, 10.338059, 13.264799, 12.753973, 8.206432, 8.916674, 8.59509, 14.363411, 
                7.376845, 11.798462, 11.530198, 11.318889, 11.185357, 5.0664344, 11.185357, 9.372978, 10.471591, 9.6629305, 
                11.367679, 8.73579, 9.080208, 11.724354, 5.04781, 7.3777695, 7.065643, 12.571651, 11.724354, 12.166186, 
                12.166186, 7.215852, 4.374113, 11.655361, 11.530198, 14.363411, 6.4993753, 11.031207, 8.344818, 10.513264, 
                10.032678, 14.363411, 14.363411, 4.5873594, 12.28397, 13.670264, 12.977117, 10.032678, 9.609821
            ], dtype = 'float32').reshape(1,600)
        self.binary_weight = np.array([1.6094379124341003, 0.22314355131420976], dtype = 'float32').reshape(1,2)
        self.num_classes = 600 # HOI
        self.num_binary  = 2 # existence (0 or 1) of HOI
        self.num_fc      = 1024
        self.scope       = 'resnet_v1_50'
        self.stride      = [16, ]
        self.lr          = tf.placeholder(tf.float32)
        if tf.__version__ == '1.1.0':
            self.blocks     = [resnet_utils.Block('block1', resnet_v1.bottleneck,[(256,   64, 1)] * 2 + [(256,   64, 2)]),
                               resnet_utils.Block('block2', resnet_v1.bottleneck,[(512,  128, 1)] * 3 + [(512,  128, 2)]),
                               resnet_utils.Block('block3', resnet_v1.bottleneck,[(1024, 256, 1)] * 5 + [(1024, 256, 1)]),
                               resnet_utils.Block('block4', resnet_v1.bottleneck,[(2048, 512, 1)] * 3),
                               resnet_utils.Block('block5', resnet_v1.bottleneck,[(2048, 512, 1)] * 3)]
        else: # we use tf 1.2.0 here, Resnet-50
            from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import resnet_v1_block
            self.blocks = [resnet_v1_block('block1', base_depth=64,  num_units=3, stride=2), # a resnet_v1 bottleneck block
                           resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
                           resnet_v1_block('block3', base_depth=256, num_units=6, stride=1), # feature former
                           resnet_v1_block('block4', base_depth=512, num_units=3, stride=1),
                           resnet_v1_block('block5', base_depth=512, num_units=3, stride=1)]
  # Args:
  #   scope: The scope of the block.
  #   base_depth: The depth of the bottleneck layer for each unit.
  #   num_units: The number of units in the block.
  #   stride: The stride of the block, implemented as a stride in the last unit.
  #     All other units have stride=1.

    def build_base(self):
        with tf.variable_scope(self.scope, self.scope):
            net = resnet_utils.conv2d_same(self.image, 64, 7, stride=2, scope='conv1') # conv2d + subsample
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID', scope='pool1')

        return net

# Number of fixed blocks during training, by default ***the first of all 4 blocks*** is fixed (Resnet-50 block)
# Range: 0 (none) to 3 (all)
# __C.RESNET.FIXED_BLOCKS = 1

    # feature extractor
    def image_to_head(self, is_training):
        with slim.arg_scope(resnet_arg_scope(is_training=False)):
            net    = self.build_base()
            net, _ = resnet_v1.resnet_v1(net,
                                         self.blocks[0:cfg.RESNET.FIXED_BLOCKS], # now 1, block 1
                                         global_pool=False,
                                         include_root_block=False,
                                         scope=self.scope)
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):
            head, _ = resnet_v1.resnet_v1(net,
                                          self.blocks[cfg.RESNET.FIXED_BLOCKS:-2], # now 1, block 2~3
                                          global_pool=False,
                                          include_root_block=False,
                                          scope=self.scope)
        return head

    # spatial configuration, conv-pool-conv-pool-flatten
    def sp_to_head(self):
        with tf.variable_scope(self.scope, self.scope):
            # (num_pos_neg,64,64,2)->(num_pos_neg,60,60,64)
            conv1_sp      = slim.conv2d(self.spatial[:,:,:,0:2], 64, [5, 5], padding='VALID', scope='conv1_sp')
            # (num_pos_neg,60,60,64)->(num_pos_neg,30,30,64)
            pool1_sp      = slim.max_pool2d(conv1_sp, [2, 2], scope='pool1_sp')
            # (num_pos_neg,30,30,64)->(num_pos_neg,26,26,32)
            conv2_sp      = slim.conv2d(pool1_sp,     32, [5, 5], padding='VALID', scope='conv2_sp')
            # (num_pos_neg,26,26,32)->(num_pos_neg,13,13,32)
            pool2_sp      = slim.max_pool2d(conv2_sp, [2, 2], scope='pool2_sp')
            # (num_pos_neg,13,13,32)->(num_pos_neg,5408)
            pool2_flat_sp = slim.flatten(pool2_sp)

        return pool2_flat_sp

    # the fifth residual block in paper, in block before GAP
    def res5(self, pool5_H, pool5_O, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_H, _ = resnet_v1.resnet_v1(pool5_H, # H input, one block
                                           self.blocks[-2:-1], #fourth block
                                           global_pool=False,
                                           include_root_block=False,
                                           reuse=False,
                                           scope=self.scope)

            fc7_H = tf.reduce_mean(fc7_H, axis=[1, 2])


            fc7_O, _ = resnet_v1.resnet_v1(pool5_O, # O input, one block
                                       self.blocks[-1:], #fifth block
                                       global_pool=False,
                                       include_root_block=False,
                                       reuse=False,
                                       scope=self.scope)

            fc7_O = tf.reduce_mean(fc7_O, axis=[1, 2])
        
        return fc7_H, fc7_O

    # GAP + fc
    def head_to_tail(self, fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, name):
        with slim.arg_scope(resnet_arg_scope(is_training=is_training)):

            fc7_SH = tf.reduce_mean(pool5_SH, axis=[1, 2])
            fc7_SO = tf.reduce_mean(pool5_SO, axis=[1, 2])

            Concat_SH     = tf.concat([fc7_H, fc7_SH], 1)
            fc8_SH        = slim.fully_connected(Concat_SH, self.num_fc, scope='fc8_SH') #fc size = 1024
            fc8_SH        = slim.dropout(fc8_SH, keep_prob=0.5, is_training=is_training, scope='dropout8_SH')
            fc9_SH        = slim.fully_connected(fc8_SH, self.num_fc, scope='fc9_SH')
            fc9_SH        = slim.dropout(fc9_SH, keep_prob=0.5, is_training=is_training, scope='dropout9_SH')  

            Concat_SO     = tf.concat([fc7_O, fc7_SO], 1)
            fc8_SO        = slim.fully_connected(Concat_SO, self.num_fc, scope='fc8_SO')
            fc8_SO        = slim.dropout(fc8_SO, keep_prob=0.5, is_training=is_training, scope='dropout8_SO')
            fc9_SO        = slim.fully_connected(fc8_SO, self.num_fc, scope='fc9_SO')
            fc9_SO        = slim.dropout(fc9_SO, keep_prob=0.5, is_training=is_training, scope='dropout9_SO')  

            Concat_SHsp   = tf.concat([fc7_H, sp], 1)
            Concat_SHsp   = slim.fully_connected(Concat_SHsp, self.num_fc, scope='Concat_SHsp')
            Concat_SHsp   = slim.dropout(Concat_SHsp, keep_prob=0.5, is_training=is_training, scope='dropout6_SHsp')
            fc7_SHsp      = slim.fully_connected(Concat_SHsp, self.num_fc, scope='fc7_SHsp')
            fc7_SHsp      = slim.dropout(fc7_SHsp,  keep_prob=0.5, is_training=is_training, scope='dropout7_SHsp')

            # Concat_HOS   = tf.concat([fc7_H, \
            #                           fc7_O, \
            #                           fc7_SH,\
            #                           fc7_SO, sp], 1)


        # add 2 returns here
        return fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO #Concat_SH, Concat_SO

    # binary discriminator for 0/1 classification of interaction, fc7_H, fc7_SH, fc7_O, fc7_SO, sp
    def binary_discriminator(self, fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, name):
        with tf.variable_scope(name) as scope:
            # fc7_H  [37,2048], first dimension is pos + neg
            # fc7_O  [16,2048], first dimension is pos
            # fc7_SH [37,1024], first dimension is pos + neg
            # fc7_SO [16,1024], first dimension is pos
            # sp     first dimension is pos + neg
            conv1_pose_map      = slim.conv2d(self.spatial[:,:,:,2:], 32, [5, 5], padding='VALID', scope='conv1_pose_map')
            pool1_pose_map      = slim.max_pool2d(conv1_pose_map, [2, 2], scope='pool1_pose_map')
            conv2_pose_map      = slim.conv2d(pool1_pose_map,     16, [5, 5], padding='VALID', scope='conv2_pose_map')
            pool2_pose_map      = slim.max_pool2d(conv2_pose_map, [2, 2], scope='pool2_pose_map')
            pool2_flat_pose_map = slim.flatten(pool2_pose_map)

            # fc7_H + fc7_SH + sp---fc1024---fc8_binary_1
            fc_binary_1    = tf.concat([fc7_H, fc7_SH], 1) # [pos + neg, 3072]
            fc_binary_1    = tf.concat([fc_binary_1, sp, pool2_flat_pose_map], 1) # [pos + neg, 8480]
            fc8_binary_1     = slim.fully_connected(fc_binary_1, 1024, scope = 'fc8_binary_1') # 1024? we should try
            fc8_binary_1     = slim.dropout(fc8_binary_1, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout8_binary_1') # [pos + neg,1024]

            # fc7_O + fc7_SO---fc1024---fc8_binary_2
            fc_binary_2    = tf.concat([fc7_O, fc7_SO], 1) # [pos, 3072]
            fc8_binary_2     = slim.fully_connected(fc_binary_2, 1024, scope = 'fc8_binary_2')
            fc8_binary_2     = slim.dropout(fc8_binary_2, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout8_binary_2') # [pos,1024]
            fc8_binary_2   = tf.concat([fc8_binary_2, fc8_binary_1[self.H_num:,:]], 0) # copt the neg part of fc8_binary_1 and concat

            # fc8_binary_1 + fc8_binary_2---fc1024---fc9_binary
            fc8_binary     = tf.concat([fc8_binary_1, fc8_binary_2], 1)
            fc9_binary     = slim.fully_connected(fc8_binary, 1024, scope = 'fc9_binary')
            fc9_binary     = slim.dropout(fc9_binary, keep_prob = cfg.TRAIN_DROP_OUT_BINARY, is_training = is_training, scope = 'dropout9_binary')
        return fc9_binary

    # biary classification
    def binary_classification(self, fc9_binary, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
            cls_score_binary = slim.fully_connected(fc9_binary, self.num_binary, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_binary')
            cls_prob_binary  = tf.nn.sigmoid(cls_score_binary, name='cls_prob_binary') 
            tf.reshape(cls_prob_binary, [1, self.num_binary])

            self.predictions["cls_score_binary"]  = cls_score_binary
            self.predictions["cls_prob_binary"]  = cls_prob_binary
        return cls_prob_binary


    def crop_pool_layer(self, bottom, rois, name):
        with tf.variable_scope(name) as scope:

            batch_ids    = tf.squeeze(tf.slice(rois, [0, 0], [-1, 1], name="batch_id"), [1])
            bottom_shape = tf.shape(bottom)
            height       = (tf.to_float(bottom_shape[1]) - 1.) * np.float32(self.stride[0])
            width        = (tf.to_float(bottom_shape[2]) - 1.) * np.float32(self.stride[0])
            x1 = tf.slice(rois, [0, 1], [-1, 1], name="x1") / width
            y1 = tf.slice(rois, [0, 2], [-1, 1], name="y1") / height
            x2 = tf.slice(rois, [0, 3], [-1, 1], name="x2") / width
            y2 = tf.slice(rois, [0, 4], [-1, 1], name="y2") / height

            bboxes        = tf.stop_gradient(tf.concat([y1, x1, y2, x2], axis=1))
            if cfg.RESNET.MAX_POOL:
                pre_pool_size = cfg.POOLING_SIZE * 2
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [pre_pool_size, pre_pool_size], name="crops")
                crops = slim.max_pool2d(crops, [2, 2], padding='SAME')
            else:
                crops = tf.image.crop_and_resize(bottom, bboxes, tf.to_int32(batch_ids), [cfg.POOLING_SIZE, cfg.POOLING_SIZE], name="crops")
        return crops

    def attention_pool_layer_H(self, bottom, fc7_H, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_H, 512, scope='fc1_b')
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att


    def attention_norm_H(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att)
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def attention_pool_layer_O(self, bottom, fc7_O, is_training, name):
        with tf.variable_scope(name) as scope:

            fc1         = slim.fully_connected(fc7_O, 512, scope='fc1_b')
            fc1         = slim.dropout(fc1, keep_prob=0.8, is_training=is_training, scope='dropout1_b')
            fc1         = tf.reshape(fc1, [tf.shape(fc1)[0], 1, 1, tf.shape(fc1)[1]])
            att         = tf.reduce_mean(tf.multiply(bottom, fc1), 3, keep_dims=True)
        return att

    def attention_norm_O(self, att, name):
        with tf.variable_scope(name) as scope:

            att         = tf.transpose(att, [0, 3, 1, 2])
            att_shape   = tf.shape(att)
            att         = tf.reshape(att, [att_shape[0], att_shape[1], -1])
            att         = tf.nn.softmax(att) ###
            att         = tf.reshape(att, att_shape)
            att         = tf.transpose(att, [0, 2, 3, 1])
        return att

    def region_classification(self, fc7_H, fc7_O, fc7_SHsp, is_training, initializer, name):
        with tf.variable_scope(name) as scope:
 
            cls_score_H = slim.fully_connected(fc7_H, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_H')
            cls_prob_H  = tf.nn.sigmoid(cls_score_H, name='cls_prob_H') 
            tf.reshape(cls_prob_H, [1, self.num_classes]) 

            cls_score_O  = slim.fully_connected(fc7_O, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_O')
            cls_prob_O  = tf.nn.sigmoid(cls_score_O, name='cls_prob_O') 
            tf.reshape(cls_prob_O, [1, self.num_classes]) 

            cls_score_sp = slim.fully_connected(fc7_SHsp, self.num_classes, 
                                               weights_initializer=initializer,
                                               trainable=is_training,
                                               activation_fn=None, scope='cls_score_sp')
            cls_prob_sp  = tf.nn.sigmoid(cls_score_sp, name='cls_prob_sp') 
            tf.reshape(cls_prob_sp, [1, self.num_classes]) # actually did not use prob to conpute loss...

            #sss = tf.concat([cls_score_H, fc7_H],0)
            #ipdb.set_trace()
            self.predictions["cls_score_H"]  = cls_score_H
            self.predictions["cls_prob_H"]   = cls_prob_H
            self.predictions["cls_score_O"]  = cls_score_O
            self.predictions["cls_prob_O"]   = cls_prob_O
            self.predictions["cls_score_sp"] = cls_score_sp
            self.predictions["cls_prob_sp"]  = cls_prob_sp

            self.predictions["cls_prob_HO"]  = cls_prob_sp * (cls_prob_O + cls_prob_H) # late fusion of predictions

        return cls_prob_H, cls_prob_O, cls_prob_sp

    def bottleneck(self, bottom, is_training, name, reuse=False):
        with tf.variable_scope(name) as scope:

            if reuse:
                scope.reuse_variables() # won't generate new variable

            head_bottleneck = slim.conv2d(bottom, 1024, [1, 1], scope=name) # 1x1, 1024, fc

        return head_bottleneck

    def build_network(self, is_training):
        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)

        # ResNet Backbone
        head       = self.image_to_head(is_training) # (1, ?, ?, 1024)
        sp         = self.sp_to_head()  # (num_pos_neg,5408)
        pool5_H    = self.crop_pool_layer(head, self.H_boxes, 'Crop_H') # (?, 7, 7, 1024)
        pool5_O    = self.crop_pool_layer(head, self.O_boxes[:self.H_num,:], 'Crop_O') # (?, 7, 7, 1024)

        fc7_H, fc7_O = self.res5(pool5_H, pool5_O, sp, is_training, 'res5')

        # Phi, whole image feature
        head_phi = slim.conv2d(head, 512, [1, 1], scope='head_phi')

        # g, whole image feature
        head_g   = slim.conv2d(head, 512, [1, 1], scope='head_g')

        Att_H      = self.attention_pool_layer_H(head_phi, fc7_H, is_training, 'Att_H')
        Att_H      = self.attention_norm_H(Att_H, 'Norm_Att_H') # softmax
        att_head_H = tf.multiply(head_g, Att_H)

        Att_O      = self.attention_pool_layer_O(head_phi, fc7_O, is_training, 'Att_O')
        Att_O      = self.attention_norm_O(Att_O, 'Norm_Att_O') # softmax
        att_head_O = tf.multiply(head_g, Att_O)

        pool5_SH     = self.bottleneck(att_head_H, is_training, 'bottleneck', False)
        pool5_SO     = self.bottleneck(att_head_O, is_training, 'bottleneck', True)

        fc9_SH, fc9_SO, fc7_SHsp, fc7_SH, fc7_SO = self.head_to_tail(fc7_H, fc7_O, pool5_SH, pool5_SO, sp, is_training, 'fc_HO')
        #ipdb.set_trace()
        fc9_binary = self.binary_discriminator(fc7_H, fc7_O, fc7_SH, fc7_SO, sp, is_training, 'fc_binary')
        # with tf.Session() as sess:
        #   print("************************")
        #   print("fc7_SH:", fc7_SH.shape(), ", fc7_SO", fc7_SO.shape(), ", fc7_H", fc7_H.shape(), ", fc7_O", fc7_O.shape())
        #   print("************************")
        #   print("Concat_SH:", Concat_SH.shape(), ", Concat_SO", Concat_SO.shape(), ", sp", sp.shape())
        #   print("************************")
        # ************************
        # fc7_SH: (?, 1024) , fc7_SO (?, 1024) , fc7_H (?, 2048) , fc7_O (?, 2048)
        # ************************
        # Concat_SH: (?, 3072) , Concat_SO (?, 3072) , sp (?, 5408)
        # ************************
        cls_prob_H, cls_prob_O, cls_prob_sp = self.region_classification(fc9_SH, fc9_SO, fc7_SHsp, is_training, initializer, 'classification')

        # add a Discriminator here to make binary classification
        # 3072, 3072, 5408, concat--fc--fc--binary loss
        cls_prob_binary = self.binary_classification(fc9_binary, is_training, initializer, 'binary_classification')
        # ipdb.set_trace()
        self.score_summaries.update(self.predictions)
        self.visualize["attention_map_H"] = (Att_H - tf.reduce_min(Att_H[0,:,:,:])) / tf.reduce_max((Att_H[0,:,:,:] - tf.reduce_min(Att_H[0,:,:,:])))
        self.visualize["attention_map_O"] = (Att_O - tf.reduce_min(Att_O[0,:,:,:])) / tf.reduce_max((Att_O[0,:,:,:] - tf.reduce_min(Att_O[0,:,:,:])))

        return cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary


    def create_architecture(self, is_training):

        # cls_prob_H, cls_prob_O, cls_prob_sp = self.build_network(is_training)
        cls_prob_H, cls_prob_O, cls_prob_sp, cls_prob_binary = self.build_network(is_training)

        for var in tf.trainable_variables():
            self.train_summaries.append(var)

        self.add_loss()
        layers_to_output = {}
        layers_to_output.update(self.losses)

        val_summaries = []
        with tf.device("/cpu:0"):
            val_summaries.append(self.add_gt_image_summary_H())
            val_summaries.append(self.add_gt_image_summary_HO())
            tf.summary.image('ATTENTION_MAP_H',  self.visualize["attention_map_H"], max_outputs=1)
            tf.summary.image('ATTENTION_MAP_O',  self.visualize["attention_map_O"], max_outputs=1)
            for key, var in self.event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            #for key, var in self.score_summaries.items():
            #    self.add_score_summary(key, var)
            #for var in self.train_summaries:
            #    self.add_train_summary(var)
        
        val_summaries.append(tf.summary.scalar('lr', self.lr))
        self.summary_op     = tf.summary.merge_all()
        self.summary_op_val = tf.summary.merge(val_summaries)

        return layers_to_output


    def add_loss(self):

        with tf.variable_scope('LOSS') as scope:
            cls_score_binary = self.predictions["cls_score_binary"] # here use cls_score, not cls_prob
            cls_score_H  = self.predictions["cls_score_H"]
            cls_score_O  = self.predictions["cls_score_O"]
            cls_score_sp = self.predictions["cls_score_sp"]
            cls_score_H_with_weight = tf.multiply(cls_score_H, self.HO_weight)
            cls_score_O_with_weight = tf.multiply(cls_score_O, self.HO_weight)
            cls_score_sp_with_weight = tf.multiply(cls_score_sp, self.HO_weight)
            cls_score_binary_with_weight = tf.multiply(cls_score_binary, self.binary_weight)

            label_HO     = self.gt_class_HO
            label_binary = self.gt_binary_label

            # Discriminator needs all samples, pos + neg
            binary_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_binary, logits = cls_score_binary_with_weight))
            # H and O stream only use positive samples' grads, but H's inputs are pos + neg(because sp stream needs feature from H stream)
            H_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_H_with_weight[:self.H_num,:]))
            # O's inputs are pos
            O_cross_entropy  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO[:self.H_num,:], logits = cls_score_O_with_weight[:self.H_num,:])) # fake :self.H_num
            # sp stream uses all samples
            sp_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = label_HO, logits = cls_score_sp_with_weight))
            # ipdb.set_trace()
            self.losses['binary_cross_entropy'] = binary_cross_entropy
            self.losses['H_cross_entropy']  = H_cross_entropy
            self.losses['O_cross_entropy']  = O_cross_entropy
            self.losses['sp_cross_entropy'] = sp_cross_entropy

            # we may need interative training of S and D, so we add a switch here to control total loss
            # 1--H+O+SP+D, 2--H+O+SP, 3--D, 4--H+O, 5--SP
            if cfg.TRAIN_MODULE == 1:
              loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy + binary_cross_entropy
            elif cfg.TRAIN_MODULE == 2:
              loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy
            elif cfg.TRAIN_MODULE == 3:
              loss = binary_cross_entropy
            elif cfg.TRAIN_MODULE == 4:
              loss = H_cross_entropy + O_cross_entropy
            else:
              loss = sp_cross_entropy

            #loss = H_cross_entropy + O_cross_entropy + sp_cross_entropy + binary_cross_entropy
            self.losses['total_loss'] = loss
            self.event_summaries.update(self.losses)

        return loss


    def add_gt_image_summary_H(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.H_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_H")
        return tf.summary.image('GROUND_TRUTH_H', image)

    def add_gt_image_summary_HO(self):

        image = tf.py_func(draw_bounding_boxes_HOI, 
                      [tf.reverse(self.image+cfg.PIXEL_MEANS, axis=[-1]), self.O_boxes, self.gt_class_HO],
                      tf.float32, name="gt_boxes_HO")
        return tf.summary.image('GROUND_TRUTH_HO)', image)

    def add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores', tensor)

    def add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'],
                     self.spatial:blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num'], 
                     self.gt_binary_label: blobs['binary_label']}
        
        loss, _ = sess.run([self.losses['total_loss'],
                            train_op],
                            feed_dict=feed_dict)
        return loss

    def train_step_with_summary(self, sess, blobs, lr, train_op):
        feed_dict = {self.image: blobs['image'], self.H_boxes: blobs['H_boxes'],
                     self.O_boxes: blobs['O_boxes'], self.gt_class_HO: blobs['gt_class_HO'],
                     self.spatial:blobs['sp'], self.lr: lr,
                     self.H_num: blobs['H_num'],
                     self.gt_binary_label: blobs['binary_label']}

        loss, summary, _ = sess.run([self.losses['total_loss'],
                                     self.summary_op,
                                     train_op],
                                     feed_dict=feed_dict)
        return loss, summary

    # return late fusion prediction, cls_prob
    def test_image_HO(self, sess, image, blobs):
        feed_dict = {self.image: image, self.H_boxes: blobs['H_boxes'], self.O_boxes: blobs['O_boxes'], self.spatial: blobs['sp'], self.H_num: blobs['H_num']}
        cls_prob_HO, cls_prob_binary = sess.run([self.predictions["cls_prob_HO"], self.predictions["cls_prob_binary"]], feed_dict=feed_dict) #self.predictions["cls_prob_binary"]
        #cls_prob_binary = sess.run([self.predictions["cls_prob_binary"]], feed_dict=feed_dict)

        # binary_gt_value = self.gt_binary_label
        # print("binary_prob_output",cls_prob_binary)
        # print("binary_gt",binary_gt_value)

        return cls_prob_HO, cls_prob_binary
