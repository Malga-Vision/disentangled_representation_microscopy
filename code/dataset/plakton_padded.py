
import os
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import random
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as T

from sklearn.preprocessing import LabelEncoder



PLANKTON_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "plankton",
    )


PLANKTON_PADDED224_PATH = os.path.join(PLANKTON_PATH, "padded_224")
PLANKTON_PADDED64_PATH = os.path.join(PLANKTON_PATH, "padded_64")
PLANKTON_ORIGINAL_PATH = os.path.join(PLANKTON_PATH, "original")


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



class PlanktonBinaryPadded64(ImageFolder):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        path = os.path.join(PLANKTON_PADDED64_PATH, split_folder, split_folder + "BIN")

        self.data_shape = [64, 64, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])

        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_binary, transform= self.transform)

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

        if not self.get_filname:
            return super(PlanktonBinaryPadded64, self).__getitem__(item)

        path, target = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

         #_, file = os.path.split(path)
        file = path
        return sample, target, file


class PlanktonMaskedPadded64(PlanktonBinaryPadded64):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED64_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED64_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [64, 64, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)



    def __getitem__(self, item):

        # load mask
        path, target = self.mask_samples[item]
        mask = loader_binary(path)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)


        if not self.get_filname:
            sample, target = super(PlanktonMaskedPadded64, self).__getitem__(item)


            return sample, target

        else:
            sample, target, file = super(PlanktonMaskedPadded64, self).__getitem__(item)


            return sample, target, file



class PlanktonMaskedPadded224(PlanktonMaskedPadded64):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)




class PlanktonMaskedPadded224_TestArcella(PlanktonMaskedPadded64):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:

            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        if split=="train":
            self.only_classes.remove("ARCELLA VULGARIS")

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)




class PlanktonMaskedPadded224_TestBursaria(PlanktonMaskedPadded64):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:

            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        if split=="train":
            self.only_classes.remove("PARAMECIUM  BURSARIA")

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)


class PlanktonMaskedPadded224_TestDileptus(PlanktonMaskedPadded64):

    def __init__(self, split, classes=None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split == "train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder)

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                                 "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                                 "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                                 "STENTOR COERULEUS", "VOLVOX"]

        if split == "train":
            self.only_classes.remove("DILEPTUS")

        # expand to 3 channels instead of 1
        self.transform = T.Compose([T.ToTensor()])
        self.mask_transform = T.Compose([T.ToTensor(), T.Lambda(repeat_3channels)])

        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform=self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)


class LenslessMaskPadded224(PlanktonMaskedPadded224):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)


    def __getitem__(self, item):

        # load mask
        path, target = self.mask_samples[item]
        mask = loader_binary(path)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)


        if not self.get_filname:
            sample, target = super(LenslessMaskPadded224, self).__getitem__(item)


            return mask, target

        else:
            sample, target, file = super(LenslessMaskPadded224, self).__getitem__(item)


            return mask, target, file




class LenslessImageMaskPadded224(PlanktonMaskedPadded224):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED224_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)


    def __getitem__(self, item):

        # load mask
        path, target = self.mask_samples[item]
        mask = loader_binary(path)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)


        if not self.get_filname:
            sample, target = super(LenslessImageMaskPadded224, self).__getitem__(item)


            return sample, mask, target

        else:
            sample, target, file = super(LenslessImageMaskPadded224, self).__getitem__(item)


            return sample, mask, target, file




class LenslessMaskPadded64(PlanktonMaskedPadded64):

    def __init__(self, split, classes = None, get_filename=False):

        split_folder = "TRAIN_IMAGE" if split=="train" else "TEST_IMAGE"
        bin_path = os.path.join(PLANKTON_PADDED64_PATH, split_folder, split_folder + "BIN")
        path = os.path.join(PLANKTON_PADDED64_PATH, split_folder, split_folder )

        self.bin_path = bin_path
        self.data_shape = [64, 64, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["ACTINOSPHAERIUM NUCLEOFILUM", "ARCELLA VULGARIS",
                              "BLEPHARISMA AMERICANUM", "DIDINIUM NASUTUM", "DILEPTUS",
                              "EUPLOTES EURYSTOMUS", "PARAMECIUM  BURSARIA", "SPIROSTOMUM AMBIGUUM",
                              "STENTOR COERULEUS", "VOLVOX"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([ T.ToTensor()])
        self.mask_transform = T.Compose([ T.ToTensor(), T.Lambda( repeat_3channels )])


        super(PlanktonBinaryPadded64, self).__init__(root=path, loader=loader_rgb, transform= self.transform)

        is_valid_file = None
        classes, class_to_idx = self.find_classes(self.root)
        self.mask_samples = self.make_dataset(bin_path, class_to_idx, self.extensions, is_valid_file)


    def __getitem__(self, item):

        # load mask
        path, target = self.mask_samples[item]
        mask = loader_binary(path)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)


        if not self.get_filname:
            _, target = super(LenslessMaskPadded64, self).__getitem__(item)


            return mask, target

        else:
            _, target, file = super(LenslessMaskPadded64, self).__getitem__(item)


            return mask, target, file















