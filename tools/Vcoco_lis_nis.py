# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# Written by Yonglu Li, Xijie Huang
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import numpy as np
import argparse
import pickle
import json
import os

from ult.config_vcoco import cfg
import sys
sys.path.append(cfg.ROOT_DIR + '/Data/v-coco/coco/PythonAPI')
from ult.vsrl_eval_output_txt import VCOCOeval

# prior nis threshold of different object
object_mask = [[1e-05, 0.85], [0.0015, 0.8], [0.0002, 0.85], [0.001, 0.8], [0.0007, 0.85], [0.0007, 0.85], [0.002, 0.8], [1e-04, 0.85], [5e-05, 0.85], [1e-05, 0.85], [0.0007, 0.85], [0.0007, 0.85], [0.0007, 0.85], [0.0002, 0.85], [0.0007, 0.85], [1e-05, 0.85], [1e-05, 0.85], [0.01, 0.95], [0.0008, 0.8], [1e-05, 0.85], [5e-05, 0.95], [0.0007, 0.85], [0.0007, 0.85], [0.0007, 0.85], [0.00015, 1.0], [5e-05, 0.85], [1e-04, 0.9], [0.004, 0.75], [0.0004, 0.9], [0.01, 0.9], [0.01, 0.9], [1e-05, 0.95], [0.005, 0.95], [0.0007, 0.85], [1e-05, 0.85], [1e-05, 0.85], [5e-05, 0.95], [1e-05, 0.95], [0.001, 0.95], [0.05, 0.8], [0.005, 0.8], [0.1, 0.75], [0.005, 0.8], [0.001, 0.8], [0.007, 0.75], [0.0008, 0.8], [0.0015, 0.85], [0.004, 0.75], [0.0007, 0.9], [0.0004, 0.85], [0.00015, 0.8], [0.005, 0.8], [1e-04, 0.8], [0.0035, 0.8], [0.002, 0.9], [5e-05, 0.85], [0.005, 0.8], [0.0015, 0.85], [1e-05, 0.85], [0.0025, 0.9], [0.05, 0.75], [0.0007, 0.85], [0.0007, 0.85], [1e-05, 0.9], [0.0007, 0.85], [0.05, 0.9], [1e-05, 0.85], [0.0015, 1.0], [1e-05, 0.85], [1e-05, 0.85], [0.0007, 0.85], [0.0007, 0.85], [0.0007, 0.85], [0.003, 0.75], [0.0007, 0.85], [1e-05, 0.85], [0.0035, 0.9], [0.05, 0.75], [1e-05, 0.85], [1e-05, 0.85]]

def parse_args():
    parser = argparse.ArgumentParser(description='Test TIN on VCOCO')
    parser.add_argument('--prior_flag', dest='prior_flag',
            help='whether use prior_flag',
            default=3, type=int)
    parser.add_argument('--mode', dest='Mode',
            help='best/default/mode1/mode2',
            default='best', type=str)
    parser.add_argument('--input_S', dest='S_pkl',
            help='input the path of S pkl of vcoco inference result',
            default=cfg.ROOT_DIR + '/-Results/6000_TIN_VCOCO_0.6_0.4_naked.pkl', type=str)
    args = parser.parse_args()
    return args

#     apply_prior   prior_mask
# 0        -             -          
# 1        Y             - 
# 2        -             Y
# 3        Y             Y


def getSigmoid(sigmoid_coeff, x):
    a, b, c, d = sigmoid_coeff
    e = 2.718281828459
    return a / (1 + e**(b - c * x)) + d


def apply_prior(Object_class, prediction):
    
    if Object_class != 32: # not a snowboard, then the action is impossible to be snowboard
        prediction[0][0][21] = 0

    if Object_class != 74: # not a book, then the action is impossible to be read
        prediction[0][0][24] = 0

    if Object_class != 33: # not a sports ball, then the action is impossible to be kick
        prediction[0][0][7] = 0   

    if (Object_class != 41) and (Object_class != 40) and (Object_class != 42) and (Object_class != 46): # not 'wine glass', 'bottle', 'cup', 'bowl', then the action is impossible to be drink
        prediction[0][0][13] = 0       

    if Object_class != 37: # not a skateboard, then the action is impossible to be skateboard
        prediction[0][0][26] = 0    

    if Object_class != 38: # not a surfboard, then the action is impossible to be surfboard
        prediction[0][0][0] = 0  
                            
    if Object_class != 31: # not a ski, then the action is impossible to be ski
        prediction[0][0][1] = 0      
                             
    if Object_class != 64: # not a laptop, then the action is impossible to be work on computer
        prediction[0][0][8] = 0
                        
    if (Object_class != 77) and (Object_class != 43) and (Object_class != 44): # not 'scissors', 'fork', 'knife', then the action is impossible to be cur instr
        prediction[0][0][2] = 0
                        
    if (Object_class != 33) and (Object_class != 30): # not 'sports ball', 'frisbee', then the action is impossible to be throw and catch
        prediction[0][0][15] = 0
        prediction[0][0][28] = 0
                              
    if Object_class != 68: # not a cellphone, then the action is impossible to be talk_on_phone
        prediction[0][0][6] = 0   
                            
    if (Object_class != 14) and (Object_class != 61) and (Object_class != 62) and (Object_class != 60) and (Object_class != 58)  and (Object_class != 57): # not 'bench', 'dining table', 'toilet', 'bed', 'couch', 'chair', then the action is impossible to be lay
        prediction[0][0][12] = 0
                            
    if (Object_class != 32) and (Object_class != 31) and (Object_class != 37) and (Object_class != 38): # not 'snowboard', 'skis', 'skateboard', 'surfboard', then the action is impossible to be jump
        prediction[0][0][11] = 0   
   
    if (Object_class != 47) and (Object_class != 48) and (Object_class != 49) and (Object_class != 50) and (Object_class != 51) and (Object_class != 52) and (Object_class != 53) and (Object_class != 54) and (Object_class != 55) and (Object_class != 56): # not ''banana', 'apple', 'sandwich', 'orange', 'carrot', 'broccoli', 'hot dog', 'pizza', 'cake', 'donut', then the action is impossible to be eat_obj
        prediction[0][0][9] = 0 

    if (Object_class != 43) and (Object_class != 44) and (Object_class != 45): # not 'fork', 'knife', 'spoon', then the action is impossible to be eat_instr
        prediction[0][0][16] = 0 
            
    if (Object_class != 39) and (Object_class != 35): # not 'tennis racket', 'baseball bat', then the action is impossible to be hit_instr
        prediction[0][0][19] = 0 

    if (Object_class != 33): # not 'sports ball, then the action is impossible to be hit_obj
        prediction[0][0][20] = 0 
                            
                            
    if (Object_class != 2) and (Object_class != 4) and (Object_class != 6) and (Object_class != 8) and (Object_class != 9) and (Object_class != 7) and (Object_class != 5) and (Object_class != 3) and (Object_class != 18) and (Object_class != 21): # not 'bicycle', 'motorcycle', 'bus', 'truck', 'boat', 'train', 'airplane', 'car', 'horse', 'elephant', then the action is impossible to be ride
        prediction[0][0][5] = 0 
                            
    if (Object_class != 2) and (Object_class != 4) and (Object_class != 18) and (Object_class != 21) and (Object_class != 14) and (Object_class != 57) and (Object_class != 58) and (Object_class != 60) and (Object_class != 62) and (Object_class != 61) and (Object_class != 29) and (Object_class != 27) and (Object_class != 25): # not 'bicycle', 'motorcycle', 'horse', 'elephant', 'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table', 'suitcase', 'handbag', 'backpack', then the action is impossible to be sit
        prediction[0][0][10] = 0 
        
    if (Object_class == 1):
        prediction[0][0][4] = 0 
    
    return prediction


def generate_pkl(mode, test_D, test_result, prior_mask, Action_dic_inv, sigmoid_coeff, prior_flag = 3):
    generate_result = []

    for fuck in range(len(test_result)):
        dic = test_result[fuck]
        dic_d = test_D[fuck]

        image_id = dic['image_id']
        bbox_H = dic['person_box']
        det_H = getSigmoid(sigmoid_coeff, dic['H_det'])
        # det_H = dic['H_det']
        score_H = dic['H_Score']
        score_binary = np.array(dic['binary_score'])
        bbox_O =  np.array(dic['object_box'])
        class_O =  np.array(dic['object_class'])
        det_O =  np.array(getSigmoid(sigmoid_coeff, np.array(dic['O_det'], dtype = 'float32')))
        # det_O = dic['O_det']
        score_HO =  np.array(dic['HO_Score'])
        length = len(score_binary)

        if length == 0:
            continue

        if mode == '3streamD':
            score_binary_d =  np.array(dic_d['binary_score_O'])
        else:
            score_binary_d =  np.array(dic_d['binary_score'])
        length_d = len(score_binary_d)

        if length != length_d:
            print('fuckkk')
            continue

        dic_new = {}
        dic_new['image_id'] = image_id
        dic_new['person_box'] = bbox_H
        dic_new['binary_score'] = score_binary
        dic_new['object_box'] = bbox_O
        dic_new['object_class'] = class_O
            
    ########################################################################################################################################
            
        # Predict actrion using human and object appearance 
        Score_obj     = np.empty((0, 4 + 29), dtype=np.float32) 

        prediction_H = score_H

        for i in range(length):

            prediction_HO = score_HO[i]
            class_id = class_O[i]
            O_bbox = bbox_O[i]
            O_det = det_O[i]

            if prior_flag == 1:
                prediction_HO  = apply_prior(class_id, prediction_HO)
            if prior_flag == 2:
                prediction_HO  = prediction_HO * prior_mask[:,class_id].reshape(1,29)
            if prior_flag == 3:
                prediction_HO  = apply_prior(class_id, prediction_HO)
                prediction_HO  = prediction_HO * prior_mask[:,class_id].reshape(1,29)

            This_Score_obj = np.concatenate((O_bbox.reshape(1,4), prediction_HO[0] * O_det), axis=1)
            Score_obj      = np.concatenate((Score_obj, This_Score_obj), axis=0)

        # Find out the object box associated with highest action score
        max_idx = np.argmax(Score_obj,0)[4:]

        # agent mAP
        for i in range(29):
            #'''
            # walk, smile, run, stand
            if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                agent_name      = Action_dic_inv[i] + '_agent'
                dic_new[agent_name] = det_H * prediction_H[0][0][i]
                continue
            
            # cut
            if i == 2:
                agent_name = 'cut_agent'
                dic_new[agent_name] = det_H * max(Score_obj[max_idx[2]][4 + 2], Score_obj[max_idx[4]][4 + 4])
                continue 
            if i == 4:
                continue   

            # eat
            if i == 9:
                agent_name = 'eat_agent'
                dic_new[agent_name] = det_H * max(Score_obj[max_idx[9]][4 + 9], Score_obj[max_idx[16]][4 + 16])
                continue  
            if i == 16:
                continue

            # hit
            if i == 19:
                agent_name = 'hit_agent'
                dic_new[agent_name] = det_H * max(Score_obj[max_idx[19]][4 + 19], Score_obj[max_idx[20]][4 + 20])
                continue  
            if i == 20:
                continue  

            # These 2 classes need to save manually because there is '_' in action name
            if i == 6:
                agent_name = 'talk_on_phone_agent'  
                dic_new[agent_name] = det_H * Score_obj[max_idx[i]][4 + i]
                continue

            if i == 8:
                agent_name = 'work_on_computer_agent'  
                dic_new[agent_name] = det_H * Score_obj[max_idx[i]][4 + i]
                continue 

            # all the rest
            agent_name =  Action_dic_inv[i].split("_")[0] + '_agent'
            dic_new[agent_name] = det_H * Score_obj[max_idx[i]][4 + i]
            #'''
   
        # role mAP
        for i in range(29):
            this_index = class_O[max_idx[i]]
            inter_bound, no_inter_bound = object_mask[this_index - 1]

            # walk, smile, run, stand. Won't contribute to role mAP
            if (i == 3) or (i == 17) or (i == 22) or (i == 27):
                dic_new[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), det_H * prediction_H[0][0][i]) 
                continue

            # Impossible to perform this action
            if det_H * Score_obj[max_idx[i]][4 + i] == 0:
                dic_new[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), det_H * Score_obj[max_idx[i]][4 + i])

            # perform nis on target object
            elif (score_binary_d[max_idx[i]][0] < inter_bound and score_binary_d[max_idx[i]][1] > no_inter_bound) or class_O[max_idx[i]] == 1:
                dic_new[Action_dic_inv[i]] = np.append(np.full(4, np.nan).reshape(1,4), 0.0)

            # Action with > 0 score
            else:
                dic_new[Action_dic_inv[i]] = np.append(Score_obj[max_idx[i]][:4], det_H * Score_obj[max_idx[i]][4 + i])

        generate_result.append(dic_new)

    ########################################################################################################################################

    return generate_result


if __name__ == '__main__':

    args = parse_args()
  
    prior_mask     = pickle.load( open( cfg.DATA_DIR + '/' + 'prior_mask.pkl', "rb" ) )
    Action_dic     = json.load(   open( cfg.DATA_DIR + '/' + 'action_index.json'))
    Action_dic_inv = {y:x for x,y in Action_dic.iteritems()}

    vcocoeval      = VCOCOeval(cfg.DATA_DIR + '/' + 'v-coco/data/vcoco/vcoco_test.json', cfg.DATA_DIR + '/' + 'v-coco/data/instances_vcoco_all_2014.json', cfg.DATA_DIR + '/' + 'v-coco/data/splits/vcoco_test.ids') 
    mode = args.Mode
    # now_best_vcoco_S
    input_S = args.S_pkl
    output_dir = cfg.ROOT_DIR+'/-Results/'
    
    output_file =  'CVPR_best_VCOCO'
    
    if mode == 'best':
        # vcoco default D
        input_D = cfg.ROOT_DIR + '/-Results/60000_TIN_VCOCO_D.pkl'
    elif mode == 'mode1':
        input_D = cfg.ROOT_DIR + 'mode1.pkl' # modify the file here for different binary D
    elif mode == 'mode2':
        input_D = cfg.ROOT_DIR + 'mode2.pkl'
    # elif ...
    # add different mode of nis here  
    else:
        print('wrong input')
        exit(0)

    with open(input_S, 'rb') as f:
        test_result = pickle.load(f)

    with open(input_D, 'rb') as f:
        test_D = pickle.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # (6,6,7,0) here represents the parameter of LIS function (sigmoid function)
    generate_result = generate_pkl(mode, test_D, test_result, prior_mask, Action_dic_inv, (6, 6, 7, 0), args.prior_flag)
    this_output = output_dir + output_file + '_nis_best_lis.pkl'
    with open(this_output, 'wb') as f:
        pickle.dump(generate_result, f)

    vcocoeval._do_eval(this_output, ovr_thresh = 0.5)
    os.remove(this_output) 
