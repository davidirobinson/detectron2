#!/bin/sh

# set -e

conda activate vitis-ai-pytorch

# # python -m pip install -e detectron2
# # cd detectron2/demo

LD_LIBRARY_PATH=/opt/vitis_ai/conda/envs/vitis-ai-tensorflow2/lib/ python quantize.py  \
    --config-file ../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
    --input target/images/test_01.jpg target/images/test_02.jpg target/images/test_03.jpg target/images/test_04.jpg target/images/test_05.jpg \
    --output ../../test/mask_rcnn_R_50_FPN_1x/ \
    --opts MODEL.WEIGHTS \
    ../../pytorch_conversion/model_final_a54504.pkl

# Compile for target boards
source compile.sh zcu104 GeneralizedRCNN

# Make target folders
cp compiled_model/GeneralizedRCNN_zcu104.xmodel target/

# Deploy
scp -r target/ root@192.168.1.22:~/

# # Run locally
# cd target
# python3 app_mt.py
