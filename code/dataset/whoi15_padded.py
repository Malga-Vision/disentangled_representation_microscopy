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

PLANKTON_WHOI15_PADDED64_PATH = os.path.join(PLANKTON_PATH, "whoi15_padded_64")
PLANKTON_WHOI15_PADDED224_PATH = os.path.join(PLANKTON_PATH, "whoi15_padded_224")

PLANKTON_WHOI15_PATH = os.path.join(PLANKTON_PATH, "WHOI15_2007-2010")








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



class WHOI15Padded64(ImageFolder):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "train"
        year = "2007"
        path = os.path.join(PLANKTON_WHOI15_PADDED64_PATH, year, split_folder)

        self.data_shape = [64, 64, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["Asterionellopsis", "Chaetoceros",
                              "Cylindrotheca", "Dactyliosolen", "detritus",
                              "Dinobryon", "Ditylum", "Licmophora",
                              "pennate", "Phaeocystis", "Pleurosigma",
                              "Pseudonitzschia", "Rhizosolenia", "Skeletonema", "Thalassiosira"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([T.ToTensor(), T.Lambda(repeat_3channels)])

        super(WHOI15Padded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        # split into train and test samples
        train_set, val_set = train_test_split(self.samples, test_size= 0.2, random_state=42, stratify=[ y for x, y in self.samples])
        self.samples = train_set if split == "train" else val_set


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory, self.only_classes)


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



class WHOI15Padded224(ImageFolder):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "train"
        year = "2007"
        path = os.path.join(PLANKTON_WHOI15_PADDED224_PATH, year, split_folder)

        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["Asterionellopsis", "Chaetoceros",
                              "Cylindrotheca", "Dactyliosolen", "detritus",
                              "Dinobryon", "Ditylum", "Licmophora",
                              "pennate", "Phaeocystis", "Pleurosigma",
                              "Pseudonitzschia", "Rhizosolenia", "Skeletonema", "Thalassiosira"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([T.ToTensor(), T.Lambda(repeat_3channels)])

        super(WHOI15Padded224, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        # split into train and test samples BALANCED
        train_set, val_set = train_test_split(self.samples, test_size= 0.2, random_state=42, stratify=[ y for x, y in self.samples])
        self.samples = train_set if split == "train" else val_set


    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory, self.only_classes)


    def __getitem__(self, item):

        # load mask
        path, target = self.samples[item]
        bin_path = path2maskpath(path)

        sample = self.loader(path)


        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)



        if not self.get_filname:
            return sample, target

        #_, file = os.path.split(path)
        return sample, target, path












