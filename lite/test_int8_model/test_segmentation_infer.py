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

import paddle
from paddlelite.lite import *
from paddleseg.cvlibs import Config as PaddleSegDataConfig
from paddleseg.core.infer import reverse_transform
from paddleseg.utils import metrics

from backend import PaddleLiteEngine


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir",
                        type=str,
                        help="inference model file diretory")

    parser.add_argument(
        "--dataset",
        type=str,
        default="human",
        choices=["human", "cityscape"],
        help="The type of given image which can be 'human' or 'cityscape'.",
    )

    parser.add_argument("--dataset_config",
                        type=str,
                        default=None,
                        help="path of dataset config.")
    parser.add_argument("--benchmark",
                        type=bool,
                        default=False,
                        help="Whether to run benchmark or not.")
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help=(
            "The precision of inference. It can be 'fp32', 'fp16' or 'int8'."),
    )
    parser.add_argument("--model_name",
                        type=str,
                        default="",
                        help="model_name for benchmark")
    parser.add_argument("--small_data",
                        action="store_true",
                        default=False,
                        help="Whether use small data to eval.")
    return parser


def eval(predictor, loader, eval_dataset):
    """
    eval mIoU func
    """
    intersect_area_all = 0
    pred_area_all = 0
    label_area_all = 0

    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20

    print("Start evaluating (total_samples: {}, total_iters: {}).".format(
        FLAGS.total_samples, FLAGS.sample_nums))

    for batch_id, data in enumerate(loader):
        image = np.array(data[0])
        label = np.array(data[1]).astype("int64")
        ori_shape = np.array(label).shape[-2:]

        predictor.prepare_data([image])

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        outs = predictor.run()
        end_time = time.time()

        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed

        logit = reverse_transform(paddle.to_tensor(outs[0]),
                                  ori_shape,
                                  eval_dataset.transforms.transforms,
                                  mode="bilinear")
        pred = paddle.to_tensor(logit)
        if len(
                pred.shape
        ) == 4:  # for humanseg model whose prediction is distribution but not class id
            pred = paddle.argmax(pred, axis=1, keepdim=True, dtype="int32")

        intersect_area, pred_area, label_area = metrics.calculate_area(
            pred,
            paddle.to_tensor(label),
            eval_dataset.num_classes,
            ignore_index=eval_dataset.ignore_index)
        intersect_area_all = intersect_area_all + intersect_area
        pred_area_all = pred_area_all + pred_area
        label_area_all = label_area_all + label_area
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()

        if FLAGS.small_data and batch_id > FLAGS.sample_nums:
            break

    _, miou = metrics.mean_iou(intersect_area_all, pred_area_all,
                               label_area_all)
    _, acc = metrics.accuracy(intersect_area_all, pred_area_all)
    kappa = metrics.kappa(intersect_area_all, pred_area_all, label_area_all)
    _, mdice = metrics.dice(intersect_area_all, pred_area_all, label_area_all)

    time_avg = predict_time / FLAGS.sample_nums
    print(
        "[Benchmark]Batch size: {}, Inference time(ms): min={}, max={}, avg={}"
        .format(FLAGS.batch_size, round(time_min * 1000, 2),
                round(time_max * 1000, 2), round(time_avg * 1000, 2)))
    infor = "[Benchmark] #Images: {} mIoU: {:.4f} Acc: {:.4f} Kappa: {:.4f} Dice: {:.4f}".format(
        FLAGS.total_samples, miou, acc, kappa, mdice)
    print(infor)
    final_res = {
        "model_name": FLAGS.model_name,
        "precision": FLAGS.precision,
        "batch_size": FLAGS.batch_size,
        "Accuracy": {
            "value": miou,
            "unit": "mIoU",
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
    data_cfg = PaddleSegDataConfig(FLAGS.dataset_config)
    eval_dataset = data_cfg.val_dataset

    batch_sampler = paddle.io.BatchSampler(eval_dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           drop_last=False)
    eval_loader = paddle.io.DataLoader(eval_dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=0,
                                       return_list=True)
    FLAGS.total_samples = len(eval_dataset) if not FLAGS.small_data else 100
    FLAGS.sample_nums = len(eval_loader) if not FLAGS.small_data else 100
    FLAGS.batch_size = int(FLAGS.total_samples / FLAGS.sample_nums)

    predictor = PaddleLiteEngine(model_dir=FLAGS.model_dir,
                                 platform_place=[
                                     Place(TargetType.XPU, PrecisionType.FP16),
                                     Place(TargetType.XPU, PrecisionType.FP32),
                                     Place(TargetType.X86, PrecisionType.FP32)
                                 ])

    eval(predictor, eval_loader, eval_dataset)


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()

    # DataLoader need to run on cpu
    paddle.set_device("cpu")

    main()
