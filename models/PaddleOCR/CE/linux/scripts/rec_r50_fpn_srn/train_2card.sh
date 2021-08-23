export FLAGS_cudnn_deterministic=True

cd /workspace/PaddleOCR/ce/Paddle_Cloud_CE/src/task/PaddleOCR
# data
rm -rf train_data
ln -s /home/data/cfs/models_ce/PaddleOCR/train_data train_data
# log
if [ ! -d "log" ]; then
  mkdir log
fi
if [ ! -d "../log" ]; then
  mkdir ../log
fi
# depencency
python -m pip install -r requirements.txt
# small data
sed -i 's!data_lmdb_release/training!data_lmdb_release/validation!g' configs/rec/rec_r50_fpn_srn.yml

python -m paddle.distributed.launch tools/train.py -c configs/rec/rec_r50_fpn_srn.yml -o Global.epoch_num=10 > log/rec_r50_fpn_srn_2card.log 2>&1
cat log/rec_r50_fpn_srn_2card.log | grep "10/10" > ../log/rec_r50_fpn_srn_2card.log