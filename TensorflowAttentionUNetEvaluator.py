# Copyright 2023 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# TensorflowAttentionUNetBrainTumorEvaluator.py
# 2023/05/30 to-arai

import os
import sys

import shutil

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"]="false"

import traceback

from ConfigParser import ConfigParser
from ImageMaskDataset import ImageMaskDataset
from TensorflowAttentionUNet import TensorflowAttentionUNet

MODEL  = "model"
TRAIN  = "train"
EVAL   = "eval"


if __name__ == "__main__":
  try:
    config_file    = "./train_eval_infer.config"
    if len(sys.argv) == 2:
      config_file = sys.argv[1]

    config     = ConfigParser(config_file)

    width      = config.get(MODEL, "image_width")
    height     = config.get(MODEL, "image_height")
    channels   = config.get(MODEL, "image_channels")
    image_datapath = config.get(EVAL, "image_datapath")
    mask_datapath  = config.get(EVAL, "mask_datapath")
  
    if not (width == height and  height % 128 == 0 and width % 128 == 0):
      raise Exception("Image width should be a multiple of 128. For example 128, 256, 512")
    
    # Create a UNetMolde and compile
    model          = TensorflowAttentionUNet(config_file)

    resized_image  = (height, width, channels)
    dataset        = ImageMaskDataset(resized_image)
    x_test, y_test = dataset.create(image_datapath, mask_datapath)
  
    model.evaluate(x_test, y_test)

  except:
    traceback.print_exc()
    
