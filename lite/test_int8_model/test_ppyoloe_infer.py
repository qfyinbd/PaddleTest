"""
# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import os
import argparse
import time
import sys
import json
import numpy as np

from paddlelite.lite import *
import paddle
from backend import PaddleLiteEngine
from ppdet.core.workspace import load_config, create
from ppdet.metrics import COCOMetric
from utils.ppyoloe_post_process import PPYOLOEPostProcess
from utils.picodet_post_process import PicoDetPostProcess


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        help="inference model filepath")
    parser.add_argument("--reader_config",
                        type=str,
                        default=None,
                        help="path of datset and reader config.")

    parser.add_argument("--img_shape",
                        type=int,
                        default=640,
                        help="input_size")  # input size

    parser.add_argument("--model_name",
                        type=str,
                        default="",
                        help="model_name for benchmark")  # model name

    parser.add_argument("--exclude_nms",
                        action="store_true",
                        default=False,
                        help="Whether exclude nms or not.")

    parser.add_argument("--calibration_file",
                        type=str,
                        default=None,
                        help="quant onnx model calibration cache file.")

    parser.add_argument("--small_data",
                        action="store_true",
                        default=False,
                        help="Whether use small data to eval.")

    parser.add_argument(
        "--precision",
        type=str,
        default='fp32',
        # choices=("fp32", "fp16", "int8"),
        help="Precision of evaluaton, just for showing")

    return parser


def eval(predictor, val_loader, metric):
    """
    eval main func
    """
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    sample_nums = len(val_loader)
    print(sample_nums)
    warmup = 20
    repeats = 20 if FLAGS.small_data else 1

    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}
        if FLAGS.exclude_nms:
            predictor.prepare_data([data_all["image"]])
        else:
            predictor.prepare_data(
                [data_all["image"], data_all["scale_factor"]])

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        for j in range(repeats):
            outs = predictor.run()
        end_time = time.time()

        timed = (end_time - start_time) / repeats
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed

        if FLAGS.exclude_nms and "ppyoloe" in FLAGS.model_name:
            postprocess = PPYOLOEPostProcess(score_threshold=0.1,
                                             nms_threshold=0.6)
            res = postprocess(outs[0], data_all["scale_factor"])
        elif FLAGS.exclude_nms and "picodet" in FLAGS.model_name:
            np_score_list, np_boxes_list = [], []
            batch_size = data_all["image"].shape[0]
            for i, out in enumerate(outs):
                np_out = np.array(out)
                if np_out.shape[-1] == 32:
                    box_reg_shape = np_out.shape[-1]
                    np_boxes_list.append(
                        np_out.reshape(batch_size, -1, box_reg_shape))
                else:
                    num_classes = np_out.shape[-1]
                    np_score_list.append(
                        np_out.reshape(batch_size, -1, num_classes))
            postprocess = PicoDetPostProcess(
                data_all["image"].shape[2:],
                data_all["im_shape"],
                data_all["scale_factor"],
                score_threshold=0.01,
                nms_threshold=0.6,
            )
            res = postprocess(np_score_list, np_boxes_list)
        else:
            res = {"bbox": outs[0], "bbox_num": outs[1]}
        metric.update(data_all, res)
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()
    metric.accumulate()
    metric.log()
    map_res = metric.get_results()
    metric.reset()

    time_avg = predict_time / sample_nums
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2), round(time_max * 1000, 2),
        round(time_avg * 1000, 2)))
    print("[Benchmark] COCO mAP: {}".format(round(map_res["bbox"][0], 3)))
    final_res = {
        "model_name": FLAGS.model_name,
        "precision": FLAGS.precision,
        "batch_size": FLAGS.batch_size,
        "Accuracy": {
            "value": map_res["bbox"][0],
            "unit": "mAP",
        },
        "Performance": {
            "value": round(time_avg * 1000, 2),
            "unit": "ms",
        }
    }

    dump_file = "paddlelite_result_" + FLAGS.precision + ".json"
    if not os.path.exists(dump_file):
        json.dump({}, open(dump_file, "w"))

    with open(dump_file, 'r') as f:
        data = json.load(f)

    data[FLAGS.model_name] = final_res

    with open(dump_file, mode="w") as f:
        json.dump(data, f, indent=4)

    print("[Benchmark][final result]{}".format(final_res))
    sys.stdout.flush()


def main():
    """
    main func
    """
    reader_cfg = load_config(FLAGS.reader_config)  # reader config
    FLAGS.batch_size = reader_cfg["EvalReader"]["batch_size"]

    predictor = PaddleLiteEngine(model_dir=FLAGS.model_dir,
                                 platform_place=[
                                     Place(TargetType.XPU, PrecisionType.FP16),
                                     Place(TargetType.XPU, PrecisionType.FP32),
                                     Place(TargetType.X86, PrecisionType.FP32)
                                 ])

    if FLAGS.small_data:
        dataset = reader_cfg["TestDataset"]
    else:
        dataset = reader_cfg["EvalDataset"]
    global val_loader
    val_loader = create("EvalReader")(dataset,
                                      reader_cfg["worker_num"],
                                      return_list=True)
    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
    anno_file = dataset.get_anno()
    metric = COCOMetric(anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        IouType="bbox")
    eval(predictor, val_loader, metric)


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
