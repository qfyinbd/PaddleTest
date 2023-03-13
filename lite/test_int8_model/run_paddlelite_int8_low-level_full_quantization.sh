#!/bin/bash

export XPU_FULL_QUANTIZATION=false
export XPU_INT8_AUTOTUNE=false

PYTHON="python3"

if [ -f "paddlelite_result_low-level_full_quantization.json" ];
then
    rm "paddlelite_result_low-level_full_quantization.json"
fi

### Detection

# PPYOLOE ScoreThreshold==0.01
echo "[Benchmark] Run PPYOLOE paddlelite low-level_full_quantization-ScoreThreshold==0.01" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.01 --model_name=ppyoloe_crn_l_300e_coco_quant

# PPYOLOE ScoreThreshold==0.05
export ResetMulticlassNMSScoreThreshold=0.05
echo "[Benchmark] Run PPYOLOE paddlelite low-level_full_quantization-ScoreThreshold==0.05" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.05 --model_name=ppyoloe_crn_l_300e_coco_quant

# PPYOLOE ScoreThreshold==0.1
export ResetMulticlassNMSScoreThreshold=0.1
echo "[Benchmark] Run PPYOLOE paddlelite low-level_full_quantization-ScoreThreshold==0.1" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_crn_l_300e_coco_quant --reader_config=configs/ppyoloe_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.1 --model_name=ppyoloe_crn_l_300e_coco_quant
unset ResetMulticlassNMSScoreThreshold

# PPYOLOE+ 
echo "[Benchmark] Run PPYOLOE+ paddlelite low-level_full_quantization" 
$PYTHON test_ppyoloe_infer.py --model_dir=models/ppyoloe_plus_crn_s_80e_coco_no_nms_quant --reader_config=configs/ppyoloe_plus_reader.yml --precision=low-level_full_quantization --model_name=ppyoloe_plus_crn_s_80e_coco_no_nms_quant --exclude_nms 


# PicoDet ScoreThreshold==0.025
echo "[Benchmark] Run PicoDet with nms low-level_full_quantization-ScoreThreshold==0.025"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.025 --model_name=picodet_s_416_coco_npu_quant

# PicoDet ScoreThreshold==0.05
export ResetMulticlassNMSScoreThreshold=0.05
echo "[Benchmark] Run PicoDet with nms low-level_full_quantization-ScoreThreshold==0.05"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.05 --model_name=picodet_s_416_coco_npu_quant

# PicoDet ScoreThreshold==0.1
export ResetMulticlassNMSScoreThreshold=0.1
echo "[Benchmark] Run PicoDet with nms low-level_full_quantization-ScoreThreshold==0.1"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu_quant --reader_config=configs/picodet_reader.yml --precision=low-level_full_quantization-ScoreThreshold==0.1 --model_name=picodet_s_416_coco_npu_quant

# PicoDet without nms
echo "[Benchmark] Run PicoDet without nms low-level_full_quantization"
$PYTHON test_ppyoloe_infer.py --model_dir=models/picodet_s_416_coco_npu_no_postprocess_quant --reader_config=configs/picodet_reader.yml --precision=low-level_full_quantization --model_name=picodet_s_416_coco_npu_no_postprocess_quant --exclude_nms

# YOLOv5s
echo "[Benchmark] Run YOLOv5s paddlelite low-level_full_quantization"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov5s_quant --precision=low-level_full_quantization --model_name=yolov5s_quant 

# YOLOv6s
echo "[Benchmark] Run YOLOv6s paddlelite low-level_full_quantization"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov6s_quant --precision=low-level_full_quantization --model_name=yolov6s_quant

# YOLOv7
echo "[Benchmark] Run YOLOv7 paddlelite low-level_full_quantization"
$PYTHON test_yolo_series_infer.py --model_dir=models/yolov7_quant --precision=low-level_full_quantization --model_name=YOLOv7_quant 


### Classification

# ResNet_vd
echo "[Benchmark] Run ResNet_vd paddlelite low-level_full_quantization"
$PYTHON test_image_classification_infer.py --model_dir=models/ResNet50_vd_QAT --precision=low-level_full_quantization --model_name=ResNet_vd_QAT

# # MobileNetV3_large
# echo "[Benchmark] Run MobileNetV3_large paddlelite low-level_full_quantization"
# $PYTHON test_image_classification_infer.py --model_dir=models/MobileNetV3_large_x1_0_QAT --precision=low-level_full_quantization --model_name=MobileNetV3_large_x1_0_QAT 

# PPLCNetV2
echo "[Benchmark] Run PPLCNetV2 paddlelite low-level_full_quantization"
$PYTHON test_image_classification_infer.py --model_dir=models/PPLCNetV2_base_QAT --precision=low-level_full_quantization --model_name=PPLCNetV2_base_QAT

# PPHGNet_tiny
echo "[Benchmark] Run PPHGNet_tiny paddlelite low-level_full_quantization"
$PYTHON test_image_classification_infer.py --model_dir=models/PPHGNet_tiny_QAT --precision=low-level_full_quantization --model_name=PPHGNet_tiny_QAT

# EfficientNetB0 paddlelite low-level_full_quantization
echo "[Benchmark] Run EfficientNetB0 paddlelite low-level_full_quantization"
$PYTHON test_image_classification_infer.py --model_dir=models/EfficientNetB0_QAT --precision=low-level_full_quantization --model_name=EfficientNetB0_QAT


### Segmentation

# PP-HumanSeg-Lite
echo "[Benchmark] Run PP-HumanSeg-Lite paddlelite low-level_full_quantization"
$PYTHON test_segmentation_infer.py --model_dir=models/pp_humanseg_qat --dataset='human' --dataset_config=configs/humanseg_dataset.yaml --precision=low-level_full_quantization --model_name=PP-HumanSeg-Lite
# PP-Liteseg
echo "[Benchmark] Run PP-Liteseg paddlelite low-level_full_quantization"
$PYTHON test_segmentation_infer.py --model_dir=models/pp_liteseg_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=low-level_full_quantization --model_name=PP-Liteseg
# HRNet
echo "[Benchmark] Run HRNet paddlelite low-level_full_quantization"
$PYTHON test_segmentation_infer.py --model_dir=models/hrnet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=low-level_full_quantization --model_name=HRNet
# UNet
echo "[Benchmark] Run UNet paddlelite low-level_full_quantization"
$PYTHON test_segmentation_infer.py --model_dir=models/unet_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=low-level_full_quantization --model_name=UNet
# Deeplabv3-ResNet50
echo "[Benchmark] Run Deeplabv3-ResNet50 paddlelite low-level_full_quantization"
$PYTHON test_segmentation_infer.py --model_dir=models/deeplabv3_qat --dataset='cityscape' --dataset_config=configs/cityscapes_1024x512_scale1.0.yml --precision=low-level_full_quantization --model_name=Deeplabv3-ResNet50


# todo(yinqifang) for NLP
# # ERNIE 3.0-Medium
# echo "[Benchmark] Run ERNIE 3.0-Medium paddlelite low-level_full_quantization"
# $PYTHON test_nlp_infer.py --model_dir=models/save_ernie3_afqmc_new_cablib --model_filename=infer.pdmodel --params_filename=infer.pdiparams --task_name='afqmc' --use_trt --precision=low-level_full_quantization --model_name=ERNIE_3.0-Medium
# # PP-MiniLM MKLDNN
# echo "[Benchmark] Run PP-MiniLM paddlelite low-level_full_quantization"
# $PYTHON test_nlp_infer.py --model_dir=models/save_ppminilm_afqmc_new_calib --task_name='afqmc' --use_trt --precision=low-level_full_quantization --model_name=PP-MiniLM
# # BERT Base MKLDNN
# echo "[Benchmark] Run BERT Base paddlelite low-level_full_quantization"
# $PYTHON test_bert_infer.py --model_dir=models/x2paddle_cola_new_calib --precision=low-level_full_quantization --batch_size=1 --model_name=BERT_Base


