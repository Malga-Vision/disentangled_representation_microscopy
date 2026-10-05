import os

from absl import app
from absl import flags

import albumentations as A
import cv2

from code.dataset.plakton_padded import PLANKTON_ORIGINAL_PATH, PLANKTON_PATH


FLAGS = flags.FLAGS
flags.DEFINE_list("classes", ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"], "Name of the classes")

flags.DEFINE_string("output_directory", "padded_224", "Output directory of processed images.")
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
    output_image_directory = os.path.join(output_directory, split, c)
    if not os.path.exists(output_image_directory):
        os.makedirs(output_image_directory)

    # process the binary image
    output_mask_directory = os.path.join(output_directory, split + "BIN", c)
    if not os.path.exists(output_mask_directory):
        os.makedirs(output_mask_directory)

    # list files in img directory
    original_image_directory = os.path.join(PLANKTON_ORIGINAL_PATH, split, split,c)
    original_mask_directory = os.path.join(PLANKTON_ORIGINAL_PATH, split, split+ "BIN", c)
    images = os.listdir(original_image_directory)

    for file in images:
        # make sure file is an image
        if file.endswith('.jpg'):
            img_path = os.path.join(original_image_directory , file)
            mask_path = os.path.join(original_mask_directory , file)

            image = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            augmented = transform(image=image, mask=mask)

            image_padded = augmented['image']
            mask_padded = augmented['mask']

            cv2.imwrite(os.path.join(output_image_directory, file), image_padded)
            cv2.imwrite(os.path.join(output_mask_directory, file), mask_padded)



def preprocess_split(split="TRAIN_IMAGE"):

    # first find bigger image
    max_width = 0
    max_height = 0

    for c in FLAGS.classes:
        width, height = get_max_width_height(os.path.join(PLANKTON_ORIGINAL_PATH, split, split, c))

        if width > max_width and height > max_height:
            max_width = width
            max_height = height

    # now create processed dataset with selected classes
    output_directory = os.path.join(PLANKTON_PATH, FLAGS.output_directory, split)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)


    # trasform images and masks
    pad_if_needed = A.Compose([A.PadIfNeeded(min_height=max_height, min_width=max_width, border_mode= cv2.BORDER_CONSTANT, value =0, mask_value=0,  p=1),
                               A.Resize(FLAGS.output_size, FLAGS.output_size, p=1)], is_check_shapes=False, p=1)
    for c in FLAGS.classes:
        preprocess_class(output_directory, split, c, pad_if_needed)


def main(unused_args):
    preprocess_split("TRAIN_IMAGE")
    preprocess_split("TEST_IMAGE")



if __name__ == "__main__":
    app.run(main)