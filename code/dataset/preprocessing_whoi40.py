
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

from code.dataset.whoi40_padded import PLANKTON_WHOI40_PATH, PLANKTON_PATH


FLAGS = flags.FLAGS


flags.DEFINE_string("output_directory", "whoi40_padded_224", "Output directory of processed images.")

flags.DEFINE_integer("output_size", 224, "Width and Height of processed images.")


def get_max_width_height(directory):
    # first bigger image in dir
    max_width = 0
    max_height = 0

    images = os.listdir(directory)

    for file in images:

        if ".db" in file:
            continue

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
    original_image_directory = os.path.join(PLANKTON_WHOI40_PATH, split,c)

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
    output_directory = os.path.join(PLANKTON_PATH, FLAGS.output_directory, split)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    # list of classes
    path_to_classes = os.path.join(PLANKTON_WHOI40_PATH, split)
    classes = [f for f in os.listdir(path_to_classes) if not os.path.isfile(os.path.join(path_to_classes, f))]

    for c in classes:
        max_width, max_height = get_max_width_height(os.path.join(path_to_classes, c))



        # trasform images and masks
        pad_if_needed = A.Compose([A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode= cv2.BORDER_CONSTANT, value =0, mask_value=0,  p=1), # value =(100, 100, 100)
                               A.Resize(FLAGS.output_size, FLAGS.output_size, p=1)], is_check_shapes=False, p=1)

        preprocess_class(output_directory, split, c, pad_if_needed)


def main(unused_args):
    preprocess_split("train")
    #preprocess_split("TEST_IMAGE")




if __name__ == "__main__":
    app.run(main)
