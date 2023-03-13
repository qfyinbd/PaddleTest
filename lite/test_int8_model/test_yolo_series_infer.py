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
import time
import sys
import argparse
import json
import numpy as np

from paddlelite.lite import *
import paddle
from backend import PaddleLiteEngine
from utils.dataset import COCOValDataset
from utils.yolo_series_post_process import YOLOPostProcess, coco_metric


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        type=str,
                        help="inference model filepath")
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="dataset/coco_val2017",
                        help="COCO dataset dir.")
    parser.add_argument("--val_image_dir",
                        type=str,
                        default="val2017",
                        help="COCO dataset val image dir.")
    parser.add_argument("--val_anno_path",
                        type=str,
                        default="annotations/instances_val2017.json",
                        help="COCO dataset anno path.")

    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="mode of running(fp32/fp16/int8, just for printing messages)")

    parser.add_argument(
        "--platform_place",
        default=[
            Place(TargetType.XPU, PrecisionType.FP16),
            Place(TargetType.XPU, PrecisionType.FP32),
            Place(TargetType.X86, PrecisionType.FP32)
        ],
        help=("The platform and precision of inference."),
    )

    parser.add_argument("--batch_size",
                        type=int,
                        default=1,
                        help="Batch size of model input.")
    parser.add_argument("--model_name",
                        type=str,
                        default="",
                        help="model name for benchmark")
    parser.add_argument("--small_data",
                        action="store_true",
                        default=False,
                        help="Whether use small data to eval.")
    return parser


def reader_wrapper(reader, input_field="image"):
    """
    reader wrapper func
    """

    def gen():
        for data in reader:
            yield np.array(data[input_field]).astype(np.float32)

    return gen


def eval(predictor, val_loader, anno_file):
    """
    eval main func
    """
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    cpu_mems, gpu_mems = 0, 0
    sample_nums = len(val_loader)
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20
    repeats = 20 if FLAGS.small_data else 1

    for batch_id, data in enumerate(val_loader):
        data_all = {k: np.array(v) for k, v in data.items()}

        predictor.prepare_data([data_all["image"]])

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

        postprocess = YOLOPostProcess(score_threshold=0.001,
                                      nms_threshold=0.65,
                                      multi_label=True)
        res = postprocess(np.array(outs), data_all["scale_factor"])
        bboxes_list.append(res["bbox"])
        bbox_nums_list.append(res["bbox_num"])
        image_id_list.append(np.array(data_all["im_id"]))
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()

    time_avg = predict_time / sample_nums
    print("[Benchmark]Inference time(ms): min={}, max={}, avg={}".format(
        round(time_min * 1000, 2), round(time_max * 1000, 2),
        round(time_avg * 1000, 2)))

    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list,
                          image_id_list)
    print("[Benchmark] COCO mAP: {}".format(round(map_res[0], 3)))
    final_res = {
        "model_name": FLAGS.model_name,
        "precision": FLAGS.precision,
        "batch_size": FLAGS.batch_size,
        "Accuracy": {
            "value": map_res[0],
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
    dataset = COCOValDataset(dataset_dir=FLAGS.dataset_dir,
                             image_dir=FLAGS.val_image_dir,
                             anno_path=FLAGS.val_anno_path)
    anno_file = dataset.ann_file
    val_loader = paddle.io.DataLoader(dataset,
                                      batch_size=FLAGS.batch_size,
                                      drop_last=True)

    predictor = PaddleLiteEngine(model_dir=FLAGS.model_dir,
                                 platform_place=FLAGS.platform_place)

    eval(predictor, val_loader, anno_file)


if __name__ == "__main__":
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    if FLAGS.small_data:
        # set small dataset
        FLAGS.dataset_dir = "dataset/coco"

    # DataLoader need run on cpu
    paddle.set_device("cpu")

    main()
