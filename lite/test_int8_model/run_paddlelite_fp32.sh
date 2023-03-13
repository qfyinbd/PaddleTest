#!/bin/bash
# export XPU_VISIBLE_DEVICES=0
PYTHON="python3"

if [ -f "paddlelite_result_fp32.json" ];
then
    rm "paddlelite_result_fp32.json"
fi

### Detection

# PPYOLOE
echo "[Benchmark] Run PPYOLOE paddlelite fp32" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_crn_l_300e_coco --reader_config=configs/ppyoloe_reader.yml --precision=fp32 --model_name=ppyoloe_crn_l_300e_coco

# PPYOLOE+
echo "[Benchmark] Run PPYOLOE+ paddlelite fp32" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_plus_crn_s_80e_coco_no_nms --reader_config=configs/ppyoloe_plus_reader.yml --precision=fp32 --model_name=ppyoloe_plus_crn_s_80e_coco_no_nms --exclude_nms 

# PicoDet
echo "[Benchmark] Run PicoDet with nms fp32"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu --reader_config=configs/picodet_reader.yml --precision=fp32 --model_name=picodet_s_416_coco_npu

# PicoDet without nms
echo "[Benchmark] Run PicoDet without nms fp32"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu_no_postprocess --reader_config=configs/picodet_reader.yml --precision=fp32 --model_name=picodet_s_416_coco_npu_no_postprocess --exclude_nms

# YOLOv5s
echo "[Benchmark] Run YOLOv7 paddlelite fp32"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov5s_infer --precision=fp32 --model_name=yolov5s_infer

# YOLOv6s
echo "[Benchmark] Run YOLOv7 paddlelite fp32"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov6s_infer --precision=fp32 --model_name=yolov6s_infer

# YOLOv7
echo "[Benchmark] Run YOLOv7 paddlelite fp32"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov7_infer --precision=fp32 --model_name=yolov7_infer


### Classification

# ResNet_vd
echo "[Benchmark] Run ResNet_vd paddlelite fp32"
$PYTHON test_image_classification_infer.py --model_dir=models/ResNet50_vd_infer --precision=fp32 --model_name=ResNet_vd_infer

# # MobileNetV3_large
# echo "[Benchmark] Run MobileNetV3_large paddlelite fp32"
# $PYTHON test_image_classification_infer.py --model_dir=models/MobileNetV3_large_x1_0_infer --precision=fp32 --model_name=MobileNetV3_large_x1_0_infer 

# PPLCNetV2
echo "[Benchmark] Run PPLCNetV2 paddlelite fp32"
$PYTHON test_image_classification_infer.py --model_dir=models/PPLCNetV2_base_infer --precision=fp32 --model_name=PPLCNetV2_base_infer

# PPHGNet_tiny
echo "[Benchmark] Run PPHGNet_tiny paddlelite fp32"
$PYTHON test_image_classification_infer.py --model_dir=models/PPHGNet_tiny_infer --precision=fp32 --model_name=PPHGNet_tiny_infer

# EfficientNetB0 paddlelite fp32
echo "[Benchmark] Run EfficientNetB0 paddlelite fp32"
$PYTHON test_image_classification_infer.py --model_dir=models/EfficientNetB0_infer --precision=fp32 --model_name=EfficientNetB0_infer


### Segmentation

# PP-HumanSeg-Lite
echo "[Benchmark] Run PP-HumanSeg-Lite paddlelite fp32"
$PYTHON test_segmentation_infer.py --model_dir=models/ppseg_lite_portrait_398x224_with_softmax --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --precision=fp32 --model_name=ppseg_lite_portrait_398x224_with_softmax

# PP-Liteseg
echo "[Benchmark] Run PP-Liteseg paddlelite fp32"
$PYTHON test_segmentation_infer.py --model_dir=models/RES-paddle2-PPLIteSegSTDC1 --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=fp32 --model_name=RES-paddle2-PPLIteSegSTDC1

# HRNet
echo "[Benchmark] Run HRNet paddlelite fp32"
$PYTHON test_segmentation_infer.py --model_dir=models/RES-paddle2-HRNetW18-Seg --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=fp32 --model_name=RES-paddle2-HRNetW18-Seg

# UNet
echo "[Benchmark] Run UNet paddlelite fp32"
$PYTHON test_segmentation_infer.py --model_dir=models/RES-paddle2-UNet --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=fp32 --model_name=RES-paddle2-UNet

# Deeplabv3-ResNet50
echo "[Benchmark] Run Deeplabv3-ResNet50 paddlelite fp32"
$PYTHON test_segmentation_infer.py --model_dir=models/RES-paddle2-Deeplabv3-ResNet50 --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=fp32 --model_name=RES-paddle2-Deeplabv3-ResNet50


### todo(yinqifang) for NLP

# # ERNIE 3.0-Medium
# echo "[Benchmark] Run ERNIE 3.0-Medium paddlelite fp32"
# $PYTHON test_nlp_infer.py --model_dir=models/AFQMC --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --use_trt --precision=fp32 --model_name=ERNIE_3.0-Medium

# # PP-MiniLM
# echo "[Benchmark] Run PP-MiniLM paddlelite fp32"
# $PYTHON test_nlp_infer.py --model_dir=models/afqmc --task_name='afqmc' --use_trt --precision=fp32 --model_name=PP-MiniLM

# # BERT Base
# echo "[Benchmark] Run BERT Base paddlelite fp32"
# $PYTHON test_bert_infer.py --model_dir=models/x2paddle_cola --precision=fp32 --batch_size=1 --model_name=BERT_Base


