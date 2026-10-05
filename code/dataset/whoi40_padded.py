import copy
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import torch
import cv2
import numpy as np
import random
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as T

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


PLANKTON_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "plankton",
    )

PLANKTON_WHOI40_PADDED224_PATH = os.path.join(PLANKTON_PATH, "whoi40_padded_224")
PLANKTON_WHOI40_NOPADDED224_PATH = os.path.join(PLANKTON_PATH, "whoi40_no_padded_224")

PLANKTON_WHOI40_PATH = os.path.join(PLANKTON_PATH, "WHOI40")

def loader_binary(file):
    mask = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    return mask

def loader_rgb(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def find_classes(directory: str, not_drop_classes) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    classes = [c for c in classes if c in not_drop_classes]
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def repeat_3channels(x):

    return x.repeat(3,1,1)


class WHOI40Padded224(ImageFolder):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "train"
        path = os.path.join(PLANKTON_WHOI40_PADDED224_PATH, split_folder)

        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])

        super(WHOI40Padded224, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        # split into train and test samples BALANCED
        train_set, val_set = train_test_split(self.samples, test_size= 0.2, random_state=42, stratify=[ y for x, y in self.samples])
        self.samples = train_set if split == "train" else val_set


    def __getitem__(self, item):

        # load mask
        path, target = self.samples[item]

        sample = self.loader(path)


        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if not self.get_filname:
            return sample, target

        #_, file = os.path.split(path)
        return sample, target, path



