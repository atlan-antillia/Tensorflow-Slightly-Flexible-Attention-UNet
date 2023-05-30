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

# ImageMaskDataset.py
# 2023/05/31 to-arai

import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
from matplotlib import pyplot as plt
from skimage.io import imread, imshow
import traceback

class ImageMaskDataset:

  def __init__(self, resized_image, threshold=160, binarize=True, blur_mask=True):
    self.img_width, self.img_height, self.img_channels = resized_image
    self.binarize  = binarize
    self.threshold = threshold
    self.blur_size = (3, 3)
    self.blur_mask = blur_mask

  # If needed, please override this method in a subclass derived from this class.
  def create(self, image_datapath, mask_datapath,  debug=False):
    image_files  = glob.glob(image_datapath + "/*.jpg")
    image_files += glob.glob(image_datapath + "/*.png")
    image_files += glob.glob(image_datapath + "/*.bmp")
    image_files += glob.glob(image_datapath + "/*.tif")
    image_files  = sorted(image_files)

    mask_files   = None
    if os.path.exists(mask_datapath):
      mask_files  = glob.glob(mask_datapath + "/*.jpg")
      mask_files += glob.glob(mask_datapath + "/*.png")
      mask_files += glob.glob(mask_datapath + "/*.bmp")
      mask_files += glob.glob(mask_datapath + "/*.tif")
      mask_files  = sorted(mask_files)
      
      if len(image_files) != len(mask_files):
        raise Exception("FATAL: Images and masks unmatched")
      
    num_images  = len(image_files)
    if num_images == 0:
      raise Exception("FATAL: Not found image files")
    
    X = np.zeros((num_images, self.img_height, self.img_width, self.img_channels), dtype=np.uint8)

    Y = np.zeros((num_images, self.img_height, self.img_width, 1                ), dtype=np.bool)

    for n, image_file in tqdm(enumerate(image_files), total=len(image_files)):
  
      image = cv2.imread(image_file)
      
      image = cv2.resize(image, dsize= (self.img_height, self.img_width), interpolation=cv2.INTER_NEAREST)
      X[n]  = image

      if mask_files != None:

        mask  = cv2.imread(mask_files[n])
        mask  = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask  = cv2.resize(mask, dsize= (self.img_height, self.img_width),   interpolation=cv2.INTER_NEAREST)

        # Binarize mask
        if self.binarize:
          mask[mask< self.threshold] =   0  
          mask[mask>=self.threshold] = 255

        # Blur mask 
        if self.blur_mask:
          mask = cv2.blur(mask, self.blur_size)
  
        mask  = np.expand_dims(mask, axis=-1)
        Y[n]  = mask

        if debug:
          imshow(mask)
          plt.show()
          input("XX")   
  
    return X, Y


    
if __name__ == "__main__":
  try:
    resized_image = (256, 256, 3)
    threshold = 128
    binarize  = True
    blur_mask = True
    dataset = ImageMaskDataset(resized_image, threshold=threshold, binarize=binarize, blur_mask=blur_mask)

    # train dataset
    original_data_path  = "./BrainTumor/train/image/"
    segmented_data_path = "./BrainTumor/train/mask/"
    x_train, y_train = dataset.create(original_data_path, segmented_data_path)
    print(" len x_train {}".format(len(x_train)))
    print(" len y_train {}".format(len(y_train)))

    # test dataset
    original_data_path  = "./BrainTumor/test/image/"
    segmented_data_path = "./BrainTumor/test/mask/"

    x_test, y_test = dataset.create(original_data_path, segmented_data_path)
    print(" len x_test {}".format(len(x_test)))
    print(" len y_test {}".format(len(y_test)))

  except:
    traceback.print_exc()

