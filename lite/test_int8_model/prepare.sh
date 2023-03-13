# 矮油  竟然可以直接这样跑，pip装环境的脚本都都有了，可以的
pip install -r requirements.txt

# ======== download Val Dataset
# 啊 给了下载数据集的脚本，我咋还傻了吧唧的自己下了那么多的数据集
mkdir dataset
# download coco val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/coco_val2017.tar
# tar 可以设置目标目录，用-C
tar -xf coco_val2017.tar -C ./dataset
# 但是这里不用cd到目标目录吗，在这个目录下不太行吧
rm -rf coco_val2017.tar
# download small coco val dataset
# small coco dataset，有可能只是测试集？
wget https://paddle-qa.bj.bcebos.com/PaddleDetection/coco.zip
# unzip的目标目录选项是-d
unzip -q coco.zip -d ./dataset
rm -rf coco.zip
# download imagenet val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ILSVRC2012_val.tar
tar -xf ILSVRC2012_val.tar -C ./dataset
rm -rf ILSVRC2012_val.tar
# download cityscapes(城市风光) val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/cityscapes_val.tar
tar -xf cityscapes_val.tar -C ./dataset
rm -rf cityscapes_val.tar

# download portrait(人像）14k val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/portrait14k_val.tar
tar -xf portrait14k_val.tar -C ./dataset
rm -rf portrait14k_val.tar

mkdir models

# ====== download INT8 quant inference model ======
# 哟，已经把模型给量化好了，那我能直接用吗，用lite直接进行推理？会不会它的量化方法有些配置选项我用不了呢
# PPYOLOE with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco_quant.tar
tar -xf ppyoloe_crn_l_300e_coco_quant.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco_quant.tar
# PPYOLOE+ without nms  我跑的那个确实是有nms的
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar
tar -xf ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar -C ./models
rm -rf ppyoloe_plus_crn_s_80e_coco_no_nms_quant.tar
# PicoDet with nms  
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_quant.tar
tar -xf picodet_s_416_coco_npu_quant.tar -C ./models
rm -rf picodet_s_416_coco_npu_quant.tar
# PicoDet without postprocess  没有后处理，这里不能直接说成没有nms吗
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_no_postprocess_quant.tar
tar -xf picodet_s_416_coco_npu_no_postprocess_quant.tar -C ./models
rm -rf picodet_s_416_coco_npu_no_postprocess_quant.tar
# YOLOv5s 这里的YOLOv5s是不是带了nms呢
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_quant.tar
tar -xf yolov5s_quant.tar -C ./models
rm -rf yolov5s_quant.tar
# YOLOv6s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant.tar
tar -xf yolov6s_quant.tar -C ./models
rm -rf yolov6s_quant.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_quant.tar
tar -xf yolov7_quant.tar -C ./models
rm -rf yolov7_quant.tar
# Resnet50_vd  vd是啥意思呢，是主干吗
wget https://paddle-slim-models.bj.bcebos.com/act/ResNet50_vd_QAT.tar
tar -xf ResNet50_vd_QAT.tar -C ./models
rm -rf ResNet50_vd_QAT.tar
# MobileNetV3_large  large的MobileNetV3，厉害了
wget https://paddle-slim-models.bj.bcebos.com/act/MobileNetV3_large_x1_0_QAT.tar
tar -xf MobileNetV3_large_x1_0_QAT.tar -C ./models
rm -rf MobileNetV3_large_x1_0_QAT.tar
# PPLCNetV2  PPLCNetV2
wget https://paddle-slim-models.bj.bcebos.com/act/PPLCNetV2_base_QAT.tar
tar -xf PPLCNetV2_base_QAT.tar -C ./models
rm -rf PPLCNetV2_base_QAT.tar
# PPHGNet_tiny  # PPHGNet 这个应该是关键点检测吧
wget https://paddle-slim-models.bj.bcebos.com/act/PPHGNet_tiny_QAT.tar
tar -xf PPHGNet_tiny_QAT.tar -C ./models
rm -rf PPHGNet_tiny_QAT.tar
# EfficientNetB0  # 不知道是干啥的
wget https://paddle-slim-models.bj.bcebos.com/act/EfficientNetB0_QAT.tar
tar -xf EfficientNetB0_QAT.tar -C ./models
rm -rf EfficientNetB0_QAT.tar
# PP-HumanSeg-Lite  # PP人体分割
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/pp_humanseg_qat.tar
tar -xf pp_humanseg_qat.tar -C ./models
rm -rf pp_humanseg_qat.tar
# PP-Liteseg  # 物体分割？
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/pp_liteseg_qat.tar
tar -xf pp_liteseg_qat.tar -C ./models
rm -rf pp_liteseg_qat.tar
# HRNet  # 关键点检测
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/hrnet_qat.tar
tar -xf hrnet_qat.tar -C ./models
rm -rf hrnet_qat.tar
# UNet  # 3D分割啥的吧
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/unet_qat.tar
tar -xf unet_qat.tar -C ./models
rm -rf unet_qat.tar
# Deeplabv3-ResNet50
wget https://bj.bcebos.com/v1/paddle-slim-models/act/PaddleSeg/qat/deeplabv3_qat.tar
tar -xf deeplabv3_qat.tar -C ./models
rm -rf deeplabv3_qat.tar
# ERNIE 3.0-Medium  # 这个就是文心大模型吧
wget https://bj.bcebos.com/v1/paddle-slim-models/act/save_ernie3_afqmc_new_cablib.tar
tar -xf save_ernie3_afqmc_new_cablib.tar -C ./models
rm -rf save_ernie3_afqmc_new_cablib.tar
# PP-MiniLM  # MiniLM
wget https://bj.bcebos.com/v1/paddle-slim-models/act/save_ppminilm_afqmc_new_calib.tar
tar -xf save_ppminilm_afqmc_new_calib.tar -C ./models
rm -rf save_ppminilm_afqmc_new_calib.tar
# BERT Base  # BERT
wget https://bj.bcebos.com/v1/paddle-slim-models/act/x2paddle_cola_new_calib.tar
tar -xf x2paddle_cola_new_calib.tar -C ./models
rm -rf x2paddle_cola_new_calib.tar

# ====== download FP32 inference model ======
# PPYOLOE with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_crn_l_300e_coco.tar
tar -xf ppyoloe_crn_l_300e_coco.tar -C ./models
rm -rf ppyoloe_crn_l_300e_coco.tar
# PPYOLOE+ without nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/ppyoloe_plus_crn_s_80e_coco_no_nms.tar
tar -xf ppyoloe_plus_crn_s_80e_coco_no_nms.tar -C ./models
rm -rf ppyoloe_plus_crn_s_80e_coco_no_nms.tar
# PicoDet with nms
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu.tar
tar -xf picodet_s_416_coco_npu.tar -C ./models
rm -rf picodet_s_416_coco_npu.tar
# PicoDet without postprocess
wget https://bj.bcebos.com/v1/paddle-slim-models/act/picodet_s_416_coco_npu_no_postprocess.tar
tar -xf picodet_s_416_coco_npu_no_postprocess.tar -C ./models
rm -rf picodet_s_416_coco_npu_no_postprocess.tar
# YOLOv5s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov5s_infer.tar
tar -xf yolov5s_infer.tar -C ./models
rm -rf yolov5s_infer.tar
# YOLOv6s
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_infer.tar
tar -xf yolov6s_infer.tar -C ./models
rm -rf yolov6s_infer.tar
# YOLOv7
wget https://bj.bcebos.com/v1/paddle-slim-models/act/yolov7_infer.tar
tar -xf yolov7_infer.tar -C ./models
rm -rf yolov7_infer.tar
# Resnet50_vd
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/ResNet50_vd_infer.tar
tar -xf ResNet50_vd_infer.tar -C ./models
rm -rf ResNet50_vd_infer.tar
# MobileNetV3_large
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/MobileNetV3_large_x1_0_infer.tar
tar -xf MobileNetV3_large_x1_0_infer.tar -C ./models
rm -rf MobileNetV3_large_x1_0_infer.tar
# PPLCNetV2
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPLCNetV2_base_infer.tar
tar -xf PPLCNetV2_base_infer.tar -C ./models
rm -rf PPLCNetV2_base_infer.tar
# PPHGNet_tiny
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/PPHGNet_tiny_infer.tar
tar -xf PPHGNet_tiny_infer.tar -C ./models
rm -rf PPHGNet_tiny_infer.tar
# EfficientNetB0
wget https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/inference/EfficientNetB0_infer.tar
tar -xf EfficientNetB0_infer.tar -C ./models
rm -rf EfficientNetB0_infer.tar
# PP-HumanSeg-Lite
wget https://paddleseg.bj.bcebos.com/dygraph/ppseg/ppseg_lite_portrait_398x224_with_softmax.tar.gz
tar -xf ppseg_lite_portrait_398x224_with_softmax.tar.gz -C ./models
rm -rf ppseg_lite_portrait_398x224_with_softmax.tar.gz
# PP-Liteseg
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-PPLIteSegSTDC1.zip
unzip -q RES-paddle2-PPLIteSegSTDC1.zip -d ./models
rm -rf RES-paddle2-PPLIteSegSTDC1.zip
# HRNet
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-HRNetW18-Seg.zip
unzip -q RES-paddle2-HRNetW18-Seg.zip -d ./models
rm -rf RES-paddle2-HRNetW18-Seg.zip
# UNet
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-UNet.zip
unzip -q RES-paddle2-UNet.zip -d ./models
rm -rf RES-paddle2-UNet.zip
# Deeplabv3-ResNet50
wget https://paddleseg.bj.bcebos.com/tipc/easyedge/RES-paddle2-Deeplabv3-ResNet50.zip
unzip -q RES-paddle2-Deeplabv3-ResNet50.zip -d ./models
rm -rf RES-paddle2-Deeplabv3-ResNet50.zip
# ERNIE 3.0-Medium
wget https://bj.bcebos.com/v1/paddle-slim-models/act/NLP/ernie3.0-medium/fp32_models/AFQMC.tar
tar -xf AFQMC.tar -C ./models
rm -rf AFQMC.tar
# PP-MiniLM
wget https://bj.bcebos.com/v1/paddle-slim-models/act/afqmc.tar
tar -xf afqmc.tar -C ./models
rm -rf afqmc.tar
# BERT Base
wget https://paddle-slim-models.bj.bcebos.com/act/x2paddle_cola.tar
tar xf x2paddle_cola.tar -C ./models
rm -rf x2paddle_cola.tar


# ====== ocr model and datset======
# download val dataset
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr/test_set_det.tar   # 检测测试集
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr/test_list_det.txt  # 用来检测的测试列表
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr/test.jpg  # 用来测试的图片
tar -xf test_set_det.tar

# download inference model
# ocr rec
# fp32
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xf ch_PP-OCRv3_det_infer.tar -C ./models
rm -rf ch_PP-OCRv3_det_infer.tar

# int8
wget https://bj.bcebos.com/v1/paddle-slim-models/act/OCR/PPOCRV3_det_QAT.tar
tar -xf PPOCRV3_det_QAT.tar -C ./models
rm -rf PPOCRV3_det_QAT.tar

git clone -b release/2.6 https://github.com/PaddlePaddle/PaddleOCR.git

# ocr rec
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr/test_set_rec.tar  # 图像数据
wget https://bj.bcebos.com/v1/paddle-slim-models/data/ocr/test_list_rec.txt  # 标签
tar -xf test_set_rec.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar  # 识别
tar -xvf ch_PP-OCRv3_rec_infer.tar # fp32
wget https://bj.bcebos.com/v1/paddle-slim-models/act/OCR/PPOCRV3_rec_QAT.tar
tar -xvf PPOCRV3_rec_QAT.tar # int8

