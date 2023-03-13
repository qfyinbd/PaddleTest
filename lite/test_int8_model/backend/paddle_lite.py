"""
# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddlelite.lite import *


class PaddleLiteEngine(object):
    """
    Paddle Lite instance
    """

    def __init__(
        self,
        model_dir,
        platform_place=[
            Place(TargetType.XPU, PrecisionType.FP32),
            Place(TargetType.X86, PrecisionType.FP32)
        ],
    ):
        """set AnalysisConfig, generate AnalysisPredictor
        Args:
            model_dir (str): root path of model.pdmodel and model.pdiparams.
            precision (str): mode of running(fp32/fp16/int8).
            use_trt (bool): whether use TensorRT or not.
            use_mkldnn (bool): whether use MKLDNN or not in CPU.
            batch_size (int): Batch size of infer sample.
            device (str): Choose the device you want to run, it can be: CPU/GPU, default is CPU.
            min_subgraph_size (int): min subgraph size in trt.
            use_dynamic_shape (bool): use dynamic shape or not.
            cpu_threads (int): num of thread when use CPU.
        """

        # Create paddle predictor using CxxConfig.
        config = CxxConfig()
        config.set_model_dir(model_dir)
        config.set_valid_places(platform_place)

        self.predictor = create_paddle_predictor(config)

        self.input_tensors = [
            self.predictor.get_input_by_name(name)
            for name in self.predictor.get_input_names()
        ]
        self.output_tensors = [
            self.predictor.get_output_by_name(name)
            for name in self.predictor.get_output_names()
        ]

        print("[Paddle Lite Backend] Completed PaddleLiteEngine init ...")

    def prepare_data(self, input_data):
        """
        Prepare data
        input_data(list): [data_all['image']] or [data_all['image'], data_all['scale_factor']])
        """
        for input_field, input_tensor in zip(input_data, self.input_tensors):
            input_tensor.from_numpy(input_field)

    def run(self):
        """ 
        Run inference.
        """
        # self.predictor.save_optimized_pb_model('./optimized_models/')

        # run in KUNLUNXIN XPU
        self.predictor.run()
        output = [
            output_tensor.numpy() for output_tensor in self.output_tensors
        ]

        return output
