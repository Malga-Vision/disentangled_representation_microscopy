import os
import cv2
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import numpy as np
from torchvision.datasets import DatasetFolder, ImageFolder
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


from code.dataset.utils import find_classes

VACUOLES_PATH = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "vacuoles", "images")


def loader_rgb(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


class VACUOLES(ImageFolder):

    def __init__(self, split, classes=None, get_filename=False, random_seed=0):


        path = os.path.join(VACUOLES_PATH, "train_annotated") if split=="train" else os.path.join(VACUOLES_PATH, "test")

        self.data_shape = [224, 224, 3]
        self.get_filname = get_filename

        self.only_classes = classes
        # pick all classes if not specified
        if classes is None:
            self.only_classes = ["condensed", "multiple",
                                 "negative", "positive"]

        # expand to 3 channels instead of 1
        self.transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True)])

        super(VACUOLES, self).__init__(root=path, loader=loader_rgb, transform=self.transform)



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
            return super(VACUOLES, self).__getitem__(item)

        path, target = self.samples[item]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # _, file = os.path.split(path)
        file = path
        return sample, target, file






if __name__ == "__main__":
    import torch
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    import random
    from code.visualization.visualize_model import save_reconstruction

    # set fixed seed
    random.seed(42)
    np.random.seed(42)


    dataset = VACUOLES(split="test" , get_filename=True)
    dl = DataLoader(dataset,  batch_size=1 , shuffle=True, num_workers=2, drop_last=False, pin_memory=True)

    print(len(dl))

    print(dataset.classes)

    train_set, val_set = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))


    print(len(train_set), len(val_set))

    # Create an iterator for the DataLoader
    data_iter = iter(dl)


    for i in range(40):
        image, target, _ = next(data_iter)

        image = image.cpu().detach().numpy()

        image = np.squeeze(image)

        print(image.shape)

        image = np.moveaxis(image, 0, -1)

        directory = f"./vacuoles_{i}_{target}.png"

        # Create a figure and axis without border or gridlines
        fig, ax = plt.subplots()

        # Show the image
        ax.imshow(image)

        # Remove axes, ticks, and frame
        ax.set_axis_off()

        # Save the image without border or axes
        plt.savefig(directory, bbox_inches='tight', pad_inches=0)

        # Close the figure to free up memory
        plt.close()
