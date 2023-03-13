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

from backend import PaddleLiteEngine
import paddle
from paddlelite.lite import *
from paddle.io import DataLoader
from utils.imagenet_reader import ImageNetDataset


def argsparser():
    """
    argsparser func
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", type=str, default="./models/MobileNetV3_large_x1_0_infer", help="model directory")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--resize_size", type=int, default=256)
    parser.add_argument("--data_path", type=str, default="./dataset/ILSVRC2012_val/")
    parser.add_argument(
        "--small_data",
        action="store_true",  
        default=False,
        help="whether val on full data, if not we will only val on 1000 samples",
    )
    parser.add_argument(
        "--precision",
        type=str,    
        default="fp32",   
        # choices=["fp32", "int8"],
        help=("The precision of inference. It can be 'fp32', 'fp16', 'int8' or 'bf16'."),
    )
    parser.add_argument("--model_name", type=str, default="", help="model_name for benchmark")
    return parser   


def eval_reader(data_dir, batch_size, crop_size, resize_size):
    """
    eval reader func
    """
    val_reader = ImageNetDataset(mode="val", data_dir=data_dir, crop_size=crop_size, resize_size=resize_size)
    val_loader = DataLoader(val_reader, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
    return val_loader


def reader_wrapper(reader, input_field="inputs"):
    """
    reader wrapper func
    """

    def gen():
        for batch_id, (image, label) in enumerate(reader):
            yield np.array(image).astype(np.float32)

    return gen


def eval(predictor, FLAGS):
    """
    eval func
    """
    if os.path.exists(FLAGS.data_path):
        val_loader = eval_reader(
            FLAGS.data_path, batch_size=FLAGS.batch_size, crop_size=FLAGS.img_size, resize_size=FLAGS.resize_size
        )
    else:
        image = np.ones((1, 3, FLAGS.img_size, FLAGS.img_size)).astype(np.float32)
        label = None
        val_loader = [[image, label]]
    results = []
    predict_time = 0.0
    time_min = float("inf")
    time_max = float("-inf")
    warmup = 20
    sample_nums = len(val_loader)
    if FLAGS.small_data:
        sample_nums = 1000

    for batch_id, (image, label) in enumerate(val_loader):
        image = np.array(image)
        image = np.expand_dims(image, 0)
        predictor.prepare_data(image)

        for i in range(warmup):
            predictor.run()
            warmup = 0

        start_time = time.time()
        all_output = predictor.run()
        # classfication model usually having only one output
        batch_output = all_output[0]
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed
        sort_array = batch_output.argsort(axis=1)
        top_1_pred = sort_array[:, -1:][:, ::-1]
        if label is None:
            results.append(top_1_pred)
            break
        label = np.array(label)
        top_1 = np.mean(label == top_1_pred)
        top_5_pred = sort_array[:, -5:][:, ::-1]
        acc_num = 0
        for i, _ in enumerate(label):
            if label[i][0] in top_5_pred[i]:
                acc_num += 1
        top_5 = float(acc_num) / len(label)
        results.append([top_1, top_5])
        if batch_id >= sample_nums:
            break
        if batch_id % 100 == 0:
            print("Eval iter:", batch_id)
            sys.stdout.flush()

    result = np.mean(np.array(results), axis=0)
    fp_message = FLAGS.precision
    print_msg = "Paddle-Lite-XPU"
    time_avg = predict_time / sample_nums
    print(
        "[Benchmark]{}\t{}\tbatch size: {}.Inference time(ms): min={}, max={}, avg={}".format(
            print_msg,
            fp_message,
            FLAGS.batch_size,
            round(time_min * 1000, 2),
            round(time_max * 1000, 2),
            round(time_avg * 1000, 2),
        )
    )
    print("[Benchmark] Evaluation acc result: {}".format(round(result[0], 3)))
    final_res = {
        "model_name": FLAGS.model_name,
        "precision": FLAGS.precision,
        "batch_size": FLAGS.batch_size,
        "Accuracy": {
            "value": result[0],
            "unit": "acc",
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


def main(FLAGS):
    """
    main func
    """
    
    predictor = PaddleLiteEngine(
        model_dir=FLAGS.model_dir,
        platform_place=[Place(TargetType.XPU, PrecisionType.FP16),
                        Place(TargetType.XPU, PrecisionType.FP32),
                        Place(TargetType.X86, PrecisionType.FP32)]
    )
    
    eval(predictor, FLAGS)
    
    return


if __name__ == "__main__":
    parser = argsparser()
    FLAGS = parser.parse_args()
    main(FLAGS)
