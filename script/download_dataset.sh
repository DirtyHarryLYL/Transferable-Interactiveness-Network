#!/bin/bash

# --------------------------------------------------------
# Tensorflow TIN
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# ---------------V-COCO Dataset------------------
echo "Downloading V-COCO Dataset"
git clone --recursive https://github.com/s-gupta/v-coco.git Data/v-coco/
cd Data/v-coco/coco

URL_2014_Train_images=http://images.cocodataset.org/zips/train2014.zip
URL_2014_Val_images=http://images.cocodataset.org/zips/val2014.zip
URL_2014_Test_images=http://images.cocodataset.org/zips/test2014.zip
URL_2014_Trainval_annotation=http://images.cocodataset.org/annotations/annotations_trainval2014.zip

wget -N $URL_2014_Train_images
wget -N $URL_2014_Val_images
wget -N $URL_2014_Test_images
wget -N $URL_2014_Trainval_annotation

mkdir images

unzip train2014.zip -d images/
unzip val2014.zip -d images/
unzip test2014.zip -d images/
unzip annotations_trainval2014.zip

rm train2014.zip
rm val2014.zip
rm test2014.zip
rm annotations_trainval2014.zip

cd ../
python script_pick_annotations.py coco/annotations

# Build V-COCO
echo "Building"
cd coco/PythonAPI/ && make install
cd ../../ && make
cd ../../

# ---------------HICO-DET Dataset------------------
echo "Downloading HICO-DET Dataset"

# URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz
# new url hico-det from their website: https://drive.google.com/open?id=1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk

# wget -N $URL_HICO_DET -P Data/
python script/Download_data.py 1QZcJmGVlF9f4h-XLWe9Gkmnmj2z1gSnk Data/hico_20160224_det.tar.gz
tar -xvzf Data/hico_20160224_det.tar.gz -C Data/
rm Data/hico_20160224_det.tar.gz

echo "Downloading HICO-DET Evaluation Code"
cd Data/
git clone https://github.com/ywchao/ho-rcnn.git
cd ../
cp script/Generate_detection.m Data/ho-rcnn/
cp script/save_mat.m Data/ho-rcnn/
cp script/load_mat.m Data/ho-rcnn/

mkdir Data/ho-rcnn/data/hico_20160224_det/
cp HICO-DET_Benchmark/data/hico_20160224_det/anno_bbox.mat Data/ho-rcnn/data/hico_20160224_det/
cp HICO-DET_Benchmark/data/hico_20160224_det/anno.mat Data/ho-rcnn/data/hico_20160224_det/

mkdir -Results/

echo "Downloading training data..."
# python script/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
# python script/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
# python script/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
# python script/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
# python script/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
# python script/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
# python script/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
# python script/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
# python script/Download_data.py 12-ZFl2_AwRkVpRe5sqOJRJrGzBeXtWLm Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl
# python script/Download_data.py 1y3cnbX12jwNAoSiXDLcdzn-nF_jvsaum Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl
# python script/Download_data.py 1IGxFW2Fe8uFRSoDtg5k_7dOxR7lK6SIA Data/Trainval_GT_HICO_with_pose.pkl
# python script/Download_data.py 1o59JGvhzI7CzdIVSxbcq0L6UnMLX0tDO Data/Trainval_GT_VCOCO_with_pose.pkl
# python script/Download_data.py 1qioujFz-jWBXPb-gCRqh6ttpf79H06Re Data/Trainval_Neg_HICO_with_pose.pkl
# python script/Download_data.py 1zEJqQcq2KF5QC8D2TIFAo0zog0PH2LF- Data/Trainval_Neg_VCOCO_with_pose.pkl
# python script/Download_data.py 1sjV6e916NIPcYYqbGwhKM6Vhl7SY6WqD -Results/80000_TIN_D_noS.pkl
# python script/Download_data.py 1sJipmoZ-5u0ymm8diqYd5Yqk2A-QQBXN -Results/60000_TIN_VCOCO_D.pkl

# old links are unavailable, here are the new replaced ones (if the version of files is wrong as much time has passed, please notice us):
# https://drive.google.com/file/d/12f2v5Jko5ESWO77KyhujTq4nD_6ilHCt/view?usp=share_link
python script/Download_data.py 12f2v5Jko5ESWO77KyhujTq4nD_6ilHCt Data/action_index.json

# https://drive.google.com/file/d/1IRooYkyaAAQNqO3Q6UzBoIW0QwbY8mB7/view?usp=share_link
python script/Download_data.py 1IRooYkyaAAQNqO3Q6UzBoIW0QwbY8mB7 Data/prior_mask.pkl

https://drive.google.com/file/d/1adsZcRYQiTJ6GR6f_0SAjrkZLVSDoxcP/view?usp=share_link
python script/Download_data.py 1adsZcRYQiTJ6GR6f_0SAjrkZLVSDoxcP Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl

https://drive.google.com/file/d/18-1UhKBw-xaPuXD50chzTtswRIuNdNWp/view?usp=share_link
python script/Download_data.py 18-1UhKBw-xaPuXD50chzTtswRIuNdNWp Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl

https://drive.google.com/file/d/11HpFAWxLMVox_OXWoBfYH6Q0chnFTRSo/view?usp=share_link
python script/Download_data.py 11HpFAWxLMVox_OXWoBfYH6Q0chnFTRSo Data/Trainval_GT_HICO.pkl

https://drive.google.com/file/d/1RcLUL0biwcGjdx76U_2oElNpfIaSSxUd/view?usp=share_link
python script/Download_data.py 1RcLUL0biwcGjdx76U_2oElNpfIaSSxUd Data/Trainval_GT_VCOCO.pkl

https://drive.google.com/file/d/1FDquAd_Z6AOwMfiuPAKOFfb1_FPbK8TC/view?usp=share_link
python script/Download_data.py 1FDquAd_Z6AOwMfiuPAKOFfb1_FPbK8TC Data/Trainval_Neg_HICO.pkl

https://drive.google.com/file/d/1wweekvLEBl6E9zBDyPVczRLoHIOq6aGR/view?usp=share_link
python script/Download_data.py 1wweekvLEBl6E9zBDyPVczRLoHIOq6aGR Data/Trainval_Neg_VCOCO.pkl

https://drive.google.com/file/d/11FJ55o_GNwYFAJLgQO25btZ5RPqi8thh/view?usp=share_link
python script/Download_data.py 11FJ55o_GNwYFAJLgQO25btZ5RPqi8thh Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl

https://drive.google.com/file/d/1rf73-jIQFyI9WbmLzLzmQoap-EU3cXxe/view?usp=share_link
python script/Download_data.py 1rf73-jIQFyI9WbmLzLzmQoap-EU3cXxe Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl

https://drive.google.com/file/d/1VZVUHdeCZ0d-LbpXNwbiNE-aJjOGD5An/view?usp=share_link
python script/Download_data.py 1VZVUHdeCZ0d-LbpXNwbiNE-aJjOGD5An Data/Trainval_GT_HICO_with_pose.pkl

https://drive.google.com/file/d/1cPZpU-L56EFHdsaE9U413_nsmpQOBW1d/view?usp=share_link
python script/Download_data.py 1cPZpU-L56EFHdsaE9U413_nsmpQOBW1d Data/Trainval_GT_VCOCO_with_pose.pkl

https://drive.google.com/file/d/15t5sNImWhvoA8aBied8hAcTSxE3fssgz/view?usp=share_link
python script/Download_data.py 15t5sNImWhvoA8aBied8hAcTSxE3fssgz Data/Trainval_Neg_HICO_with_pose.pkl

https://drive.google.com/file/d/1-MsytpBSSXkWBZ0eFp6w_G2IrPLXWy3o/view?usp=share_link
python script/Download_data.py 1-MsytpBSSXkWBZ0eFp6w_G2IrPLXWy3o Data/Trainval_Neg_VCOCO_with_pose.pkl

https://drive.google.com/file/d/1kQu3480HghtEdov5WaKdKDD8QhOlBzAm/view?usp=share_link
python script/Download_data.py 1kQu3480HghtEdov5WaKdKDD8QhOlBzAm -Results/80000_TIN_D_noS.pkl

https://drive.google.com/file/d/1uQG0n2cCi-JF_KLDujjn4Dg6E6vXb4p_/view?usp=share_link
python script/Download_data.py 1uQG0n2cCi-JF_KLDujjn4Dg6E6vXb4p_ -Results/60000_TIN_VCOCO_D.pkl
