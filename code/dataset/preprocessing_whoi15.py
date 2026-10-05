
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import copy
import numpy as np
import albumentations as A
import cv2

from code.dataset.whoi15_padded import PLANKTON_WHOI15_PATH, PLANKTON_PATH


FLAGS = flags.FLAGS
flags.DEFINE_list("classes", ["Asterionellopsis", "Chaetoceros",
                              "Cylindrotheca", "Dactyliosolen", "detritus",
                              "Dinobryon", "Ditylum", "Licmophora",
                              "pennate", "Phaeocystis", "Pleurosigma",
                              "Pseudonitzschia", "Rhizosolenia", "Skeletonema", "Thalassiosira"], "Name of the classes")

flags.DEFINE_string("output_directory", "whoi15_padded_224", "Output directory of processed images.")
flags.DEFINE_string("output_year", "2007", "Output directory of processed images.")

flags.DEFINE_integer("output_size", 224, "Width and Height of processed images.")










def get_max_width_height(directory):
    # first bigger image in dir
    max_width = 0
    max_height = 0

    images = os.listdir(directory)

    for file in images:
        img_path = os.path.join(directory , file)

        image = cv2.imread(img_path)

        h, w, _ = image.shape

        if w > max_width and h > max_height:
            max_width = w
            max_height = h

    return max_width, max_height


def preprocess_class(output_directory, split, c, transform):

    # process the original image
    output_image_directory = os.path.join(output_directory, c)
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)

    # list files in img directory
    original_image_directory = os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split,c)

    images = os.listdir(original_image_directory)

    for file in images:
        # make sure file is an image
        if file.endswith('.png'):
            img_path = os.path.join(original_image_directory , file)

            image = cv2.imread(img_path)

            augmented = transform(image=image)
            image_padded = augmented['image']

            cv2.imwrite(os.path.join(output_image_directory, file), image_padded)





def preprocess_split(split="train"):

    # now create processed dataset with selected classes
    output_directory = os.path.join(PLANKTON_PATH, FLAGS.output_directory, FLAGS.output_year, split)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    for c in FLAGS.classes:
        max_width, max_height = get_max_width_height(os.path.join(PLANKTON_WHOI15_PATH, FLAGS.output_year, split, c))



        # trasform images and masks
        pad_if_needed = A.Compose([A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode= cv2.BORDER_CONSTANT, value =0, mask_value=0,  p=1), # value =(100, 100, 100)
                               A.Resize(FLAGS.output_size, FLAGS.output_size, p=1)], is_check_shapes=False, p=1)

        preprocess_class(output_directory, split, c, pad_if_needed)


def main(unused_args):
    preprocess_split("train")




if __name__ == "__main__":
    app.run(main)

