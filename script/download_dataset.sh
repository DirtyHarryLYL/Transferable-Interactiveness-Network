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
rm annotations_trainval2014

cd ../
python script_pick_annotations.py coco/annotations

# Build V-COCO
echo "Building"
cd coco/PythonAPI/ && make install
cd ../../ && make
cd ../../

# ---------------HICO-DET Dataset------------------
echo "Downloading HICO-DET Dataset"

URL_HICO_DET=http://napoli18.eecs.umich.edu/public_html/data/hico_20160224_det.tar.gz

wget -N $URL_HICO_DET -P Data/
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
python script/Download_data.py 1z5iZWJsMl4hafo0u7f1mVj7A2S5HQBOD Data/action_index.json
python script/Download_data.py 1QeCGE_0fuQsFa5IxIOUKoRFakOA87JqY Data/prior_mask.pkl
python script/Download_data.py 1JRMaE35EbJYxkXNSADEgTtvAFDdU20Ru Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET.pkl
python script/Download_data.py 1Y9yRTntfThrKMJbqyMzVasua25GUucf4 Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl
python script/Download_data.py 1le4aziSn_96cN3dIPCYyNsBXJVDD8-CZ Data/Trainval_GT_HICO.pkl
python script/Download_data.py 1YrsQUcBEF31cvqgCZYmX5j-ns2tgYXw7 Data/Trainval_GT_VCOCO.pkl
python script/Download_data.py 1PPPya4M2poWB_QCoAheStEYn3rPMKIgR Data/Trainval_Neg_HICO.pkl
python script/Download_data.py 1oGZfyhvArB2WHppgGVBXeYjPvgRk95N9 Data/Trainval_Neg_VCOCO.pkl
python script/Download_data.py 12-ZFl2_AwRkVpRe5sqOJRJrGzBeXtWLm Data/Test_Faster_RCNN_R-50-PFN_2x_HICO_DET_with_pose.pkl
python script/Download_data.py 1y3cnbX12jwNAoSiXDLcdzn-nF_jvsaum Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO_with_pose.pkl
python script/Download_data.py 1IGxFW2Fe8uFRSoDtg5k_7dOxR7lK6SIA Data/Trainval_GT_HICO_with_pose.pkl
python script/Download_data.py 1o59JGvhzI7CzdIVSxbcq0L6UnMLX0tDO Data/Trainval_GT_VCOCO_with_pose.pkl
python script/Download_data.py 1qioujFz-jWBXPb-gCRqh6ttpf79H06Re Data/Trainval_Neg_HICO_with_pose.pkl
python script/Download_data.py 1zEJqQcq2KF5QC8D2TIFAo0zog0PH2LF- Data/Trainval_Neg_VCOCO_with_pose.pkl
python script/Download_data.py 1sjV6e916NIPcYYqbGwhKM6Vhl7SY6WqD -Results/80000_TIN_D_noS.pkl
