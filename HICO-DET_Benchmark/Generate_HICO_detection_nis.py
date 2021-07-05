# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

"""
Change the HICO-DET detection results to the right format.
input arg: python Generate_HICO_detection_nis.py (1:pkl_path) (2:hico_dir) (3:rule_inter) (4:threshold_x) (5:threshold_y) 
"""

import pickle
import shutil
import numpy as np
import scipy.io as sio
import os
import sys
import matplotlib
import matplotlib.pyplot as plth
import random
import HICO_Benchmark_Binary as rank 

# all the no-interaction HOI index in HICO dataset
hoi_no_inter_all = [10,24,31,46,54,65,76,86,92,96,107,111,129,146,160,170,174,186,194,198,208,214,224,232,235,239,243,247,252,257,264,273,283,290,295,305,313,325,330,336,342,348,352,356,363,368,376,383,389,393,397,407,414,418,429,434,438,445,449,453,463,474,483,488,502,506,516,528,533,538,546,550,558,562,567,576,584,588,595,600]
# all HOI index range corresponding to different object id in HICO dataset
hoi_range = [(161, 170), (11, 24), (66, 76), (147, 160), (1, 10), (55, 65), (187, 194), (568, 576), (32, 46), (563, 567), (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), (77, 86), (112, 129), (130, 146), (175, 186), (97, 107), (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), (265, 273), (87, 92), (93, 96), (171, 174), (240, 243), (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)]
# all image index in test set without any pair
all_remaining = set([20, 25, 54, 60, 66, 71, 74, 94, 154, 155, 184, 200, 229, 235, 242, 249, 273, 280, 289, 292, 315, 323, 328, 376, 400, 421, 432, 436, 461, 551, 554, 578, 613, 626, 639, 641, 642, 704, 705, 768, 773, 776, 796, 809, 827, 845, 850, 855, 862, 886, 901, 947, 957, 963, 965, 1003, 1011, 1014, 1028, 1042, 1044, 1057, 1090, 1092, 1097, 1099, 1119, 1171, 1180, 1231, 1241, 1250, 1346, 1359, 1360, 1391, 1420, 1450, 1467, 1495, 1498, 1545, 1560, 1603, 1605, 1624, 1644, 1659, 1673, 1674, 1677, 1709, 1756, 1808, 1845, 1847, 1849, 1859, 1872, 1881, 1907, 1910, 1912, 1914, 1953, 1968, 1979, 2039, 2069, 2106, 2108, 2116, 2126, 2142, 2145, 2146, 2154, 2175, 2184, 2218, 2232, 2269, 2306, 2308, 2316, 2323, 2329, 2390, 2397, 2406, 2425, 2463, 2475, 2483, 2494, 2520, 2576, 2582, 2591, 2615, 2624, 2642, 2646, 2677, 2703, 2707, 2712, 2717, 2763, 2780, 2781, 2818, 2830, 2833, 2850, 2864, 2873, 2913, 2961, 2983, 3021, 3040, 3042, 3049, 3057, 3066, 3082, 3083, 3111, 3112, 3122, 3157, 3200, 3204, 3229, 3293, 3309, 3328, 3341, 3373, 3393, 3423, 3439, 3449, 3471, 3516, 3525, 3537, 3555, 3616, 3636, 3653, 3668, 3681, 3709, 3718, 3719, 3733, 3737, 3744, 3756, 3762, 3772, 3780, 3784, 3816, 3817, 3824, 3855, 3865, 3885, 3891, 3910, 3916, 3918, 3919, 3933, 3949, 3980, 4009, 4049, 4066, 4089, 4112, 4143, 4154, 4200, 4222, 4243, 4254, 4257, 4259, 4266, 4269, 4273, 4308, 4315, 4320, 4331, 4343, 4352, 4356, 4369, 4384, 4399, 4411, 4424, 4428, 4445, 4447, 4466, 4477, 4482, 4492, 4529, 4534, 4550, 4566, 4596, 4605, 4606, 4620, 4648, 4710, 4718, 4734, 4771, 4773, 4774, 4801, 4807, 4811, 4842, 4845, 4849, 4874, 4886, 4887, 4907, 4926, 4932, 4948, 4960, 4969, 5000, 5039, 5042, 5105, 5113, 5159, 5161, 5174, 5183, 5197, 5214, 5215, 5216, 5221, 5264, 5273, 5292, 5293, 5353, 5438, 5447, 5452, 5465, 5468, 5492, 5498, 5520, 5543, 5551, 5575, 5581, 5605, 5617, 5623, 5671, 5728, 5759, 5766, 5777, 5799, 5840, 5853, 5875, 5883, 5886, 5898, 5919, 5922, 5941, 5948, 5960, 5962, 5964, 6034, 6041, 6058, 6080, 6103, 6117, 6134, 6137, 6138, 6163, 6196, 6206, 6210, 6223, 6228, 6232, 6247, 6272, 6273, 6281, 6376, 6409, 6430, 6438, 6473, 6496, 6595, 6608, 6635, 6678, 6687, 6692, 6695, 6704, 6712, 6724, 6757, 6796, 6799, 6815, 6851, 6903, 6908, 6914, 6948, 6957, 7065, 7071, 7073, 7089, 7099, 7102, 7114, 7147, 7169, 7185, 7219, 7226, 7232, 7271, 7285, 7315, 7323, 7341, 7378, 7420, 7433, 7437, 7467, 7489, 7501, 7513, 7514, 7523, 7534, 7572, 7580, 7614, 7619, 7625, 7658, 7667, 7706, 7719, 7727, 7752, 7813, 7826, 7829, 7868, 7872, 7887, 7897, 7902, 7911, 7936, 7942, 7945, 8032, 8034, 8042, 8044, 8092, 8101, 8156, 8167, 8175, 8176, 8205, 8234, 8237, 8244, 8301, 8316, 8326, 8350, 8362, 8385, 8441, 8463, 8479, 8534, 8565, 8610, 8623, 8651, 8671, 8678, 8689, 8707, 8735, 8761, 8763, 8770, 8779, 8800, 8822, 8835, 8923, 8942, 8962, 8970, 8984, 9010, 9037, 9041, 9122, 9136, 9140, 9147, 9164, 9165, 9166, 9170, 9173, 9174, 9175, 9185, 9186, 9200, 9210, 9211, 9217, 9218, 9246, 9248, 9249, 9250, 9254, 9307, 9332, 9337, 9348, 9364, 9371, 9376, 9379, 9389, 9404, 9405, 9408, 9415, 9416, 9417, 9418, 9419, 9421, 9424, 9433, 9434, 9493, 9501, 9505, 9519, 9520, 9521, 9522, 9526, 9529, 9531, 9637, 9654, 9655, 9664, 9686, 9688, 9701, 9706, 9709, 9712, 9716, 9717, 9718, 9731, 9746, 9747, 9748, 9753, 9765])

pair_total_num = 999999


binary_score_nointer, binary_score_inter, a_pair, b_pair, c_pair = rank.cal_rank_600()

pair_is_del = np.zeros(pair_total_num, dtype = 'float32')
pair_in_the_result = np.zeros(9999, dtype = 'float32')


def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d 

def save_HICO(HICO, HICO_dir, thres_no_inter, thres_inter, classid, begin, finish):

    all_boxes = []
    possible_hoi_range = hoi_range[classid - 1]
    num_delete_pair_a = 0
    num_delete_pair_b = 0
    num_delete_pair_c = 0

    for i in range(finish - begin + 1): # for every verb, iteration all the pkl file
        total = []
        score = []
        pair_id = 0
      
        for key, value in HICO.iteritems():
            for element in value:
                if element[2] == classid:
                    temp = []
                    temp.append(element[0].tolist())  # Human box
                    temp.append(element[1].tolist())  # Object box
                    temp.append(int(key))             # image id
                    temp.append(int(i))               # action id (0-599)

                    human_score = element[4]
                    object_score = element[5]

                    d_score = binary_score_inter[pair_id]
                    d_score_noi = binary_score_nointer[pair_id]

                    # you could change the parameter of NIS (sigmoid function) here
                    # use (10, 1.4, 0) as the default 
                    score_old = element[3][begin - 1 + i] * getSigmoid(10,1.4,0,element[4]) * getSigmoid(10,1.4,0,element[5])

                    hoi_num = begin - 1 + i

                    score_new = score_old

                    if classid == 63:
                        thres_no_inter = 0.95 
                        thres_inter = 0.15
                    elif classid == 43:
                        thres_no_inter = 0.85
                        thres_inter = 0.1
                    elif classid == 57:
                        thres_no_inter = 0.85
                        thres_inter = 0.2
                    elif classid == 48:
                        thres_no_inter = 0.85
                        thres_inter = 0.2
                    elif classid == 41:
                        thres_no_inter = 0.85
                        thres_inter = 0.15 
                    elif classid == 2:
                        thres_inter = 0.2
                        thres_no_inter = 0.85
                    elif classid == 4:
                        thres_inter = 0.15
                        thres_no_inter = 0.85
                    elif classid == 31:
                        thres_inter = 0.1
                        thres_no_inter = 0.85
                    elif classid == 19:
                        thres_inter = 0.2
                        thres_no_inter = 0.85
                    elif classid == 1:
                        thres_inter = 0.05
                        thres_no_inter = 0.85
                    elif classid == 11:
                        thres_inter = 0.15
                        thres_no_inter = 0.85



                    # if Binary D score D[0] > no interaction threshold and D[1] <
                    if (d_score_noi > thres_no_inter) and (d_score < thres_inter) and not(int(key) in all_remaining):

                        if not((hoi_num + 1) in hoi_no_inter_all): # skiping all the 520 score
                            
                            if (a_pair[pair_id] == 1) and (pair_is_del[pair_id] == 0):
                                num_delete_pair_a += 1
                                pair_is_del[pair_id] = 1

                            elif (b_pair[pair_id] == 1) and (pair_is_del[pair_id] == 0):
                                num_delete_pair_b += 1
                                pair_is_del[pair_id] = 1

                            elif (c_pair[pair_id] == 1) and (pair_is_del[pair_id] == 0):
                                num_delete_pair_c += 1
                                pair_is_del[pair_id] = 1

                            pair_id += 1
                            continue  

                    temp.append(score_new)
                    total.append(temp)
                    score.append(score_new)
                    
                if not(int(key) in all_remaining):
                    pair_id += 1
                
        idx = np.argsort(score, axis=0)[::-1]
        for i_idx in range(min(len(idx),19999)):
            all_boxes.append(total[idx[i_idx]])

    
    # save the detection result in .mat file
    savefile = os.path.join(HICO_dir, 'detections_' + str(classid).zfill(2) + '.mat')
    if os.path.exists(savefile):
        os.remove(savefile)
    sio.savemat(savefile, {'all_boxes':all_boxes})

    print('class',classid,'finished')
    num_delete_inter = num_delete_pair_a + num_delete_pair_b

    return num_delete_inter, num_delete_pair_c
   

def Generate_HICO_detection(output_file, HICO_dir, thres_no_inter,thres_inter):

    if not os.path.exists(HICO_dir):
        os.makedirs(HICO_dir)

    HICO = pickle.load( open( output_file, "rb" ) )

    # del_i and del_ni 

    del_i = 0
    del_ni = 0

    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 1 ,161, 170)
    del_i += num_del_i
    del_ni += num_del_no_i
     # 1 person
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 2 ,11,  24 )
    del_i += num_del_i
    del_ni += num_del_no_i
     # 2 bicycle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 3 ,66,  76 )
    del_i += num_del_i
    del_ni += num_del_no_i
     # 3 car
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 4 ,147, 160)
    del_i += num_del_i
    del_ni += num_del_no_i
     # 4 motorcycle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 5 ,1,   10 )
    del_i += num_del_i
    del_ni += num_del_no_i
     # 5 airplane
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 6 ,55,  65 )
    del_i += num_del_i
    del_ni += num_del_no_i
     # 6 bus
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 7 ,187, 194)
    del_i += num_del_i
    del_ni += num_del_no_i
     # 7 train
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 8 ,568, 576)
    del_i += num_del_i
    del_ni += num_del_no_i
     # 8 truck
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 9 ,32,  46 )
    del_i += num_del_i
    del_ni += num_del_no_i
     # 9 boat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 10,563, 567)
    del_i += num_del_i
    del_ni += num_del_no_i
     # 10 traffic light
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 11,326,330) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 11 fire_hydrant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 12,503,506) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 12 stop_sign
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 13,415,418) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 13 parking_meter
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 14,244,247) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 14 bench
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 15,25,  31) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 15 bird
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 16,77,  86) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 16 cat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 17,112,129) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 17 dog
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 18,130,146) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 18 horse
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 19,175,186) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 19 sheep
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 20,97,107)  
    del_i += num_del_i
    del_ni += num_del_no_i
    # 20 cow
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 21,314,325) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 21 elephant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 22,236,239) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 22 bear
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 23,596,600) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 23 zebra
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 24,343,348) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 24 giraffe
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 25,209,214) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 25 backpack
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 26,577,584) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 26 umbrella
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 27,353,356) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 27 handbag
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 28,539,546) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 28 tie
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 29,507,516) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 29 suitcase
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 30,337,342) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 30 Frisbee
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 31,464,474) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 31 skis
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 32,475,483) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 32 snowboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 33,489,502) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 33 sports_ball
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 34,369,376) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 34 kite
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 35,225,232) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 35 baseball_bat
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 36,233,235) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 36 baseball_glove
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 37,454,463) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 37 skateboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 38,517,528) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 38 surfboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 39,534,538) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 39 tennis_racket
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 40,47,54)   
    del_i += num_del_i
    del_ni += num_del_no_i
    # 40 bottle
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 41,589,595) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 41 wine_glass
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 42,296,305) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 42 cup
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 43,331,336) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 43 fork
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 44,377,383) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 44 knife
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 45,484,488) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 45 spoon
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 46,253,257) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 46 bowl
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 47,215,224) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 47 banana
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 48,199,208) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 48 apple
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 49,439,445) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 49 sandwich
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 50,398,407) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 50 orange
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 51,258,264) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 51 broccoli
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 52,274,283) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 52 carrot
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 53,357,363) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 53 hot_dog
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 54,419,429) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 54 pizza
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 55,306,313) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 55 donut
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 56,265,273) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 56 cake
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 57,87,92)   
    del_i += num_del_i
    del_ni += num_del_no_i
    # 57 chair
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 58,93,96)   
    del_i += num_del_i
    del_ni += num_del_no_i
    # 58 couch
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 59,171,174) 
    del_i += num_del_i
    del_ni += num_del_no_i
    # 59 potted_plant
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 60,240,243) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #60 bed
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 61,108,111) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #61 dining_table
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 62,551,558) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #62 toilet
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 63,195,198) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #63 TV
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 64,384,389) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #64 laptop
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 65,394,397) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #65 mouse
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 66,435,438) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #66 remote
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 67,364,368) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #67 keyboard
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 68,284,290) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #68 cell_phone
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 69,390,393) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #69 microwave
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 70,408,414) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #70 oven
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 71,547,550) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #71 toaster
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 72,450,453) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #72 sink
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 73,430,434) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #73 refrigerator
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 74,248,252) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #74 book
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 75,291,295) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #75 clock
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 76,585,588) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #76 vase
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 77,446,449) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #77 scissors
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 78,529,533) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #78 teddy_bear
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 79,349,352) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #79 hair_drier
    num_del_i, num_del_no_i = save_HICO(HICO, HICO_dir, thres_no_inter,thres_inter, 80,559,562) 
    del_i += num_del_i
    del_ni += num_del_no_i
    #80 toothbrush 

    print('num_del_inter',del_i,'num_del_no_inter',del_ni)


def main():
    output_file = sys.argv[1]
    HICO_dir = sys.argv[2]
    thres_no_inter = float(sys.argv[3]) 
    thres_inter = float(sys.argv[4])

    print("the output file is",output_file)
    print("the threshold of no interaction score is",thres_no_inter)
    print("the threshold of interaction score is",thres_inter)

    Generate_HICO_detection(output_file, HICO_dir, thres_no_inter,thres_inter)


if __name__ == '__main__':
    main()
    