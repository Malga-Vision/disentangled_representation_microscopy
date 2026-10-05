import os
import torch
import numpy as np
import torchvision.transforms as T
import cv2



PLANKTON_REPRESENTATION_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "plankton", "representations", "vit-b-1k-dino"  #"vit-b-1k-dino"
    )


PLANKTON_REPRESENTATION_PADDED224_PATH = os.path.join(PLANKTON_REPRESENTATION_PATH, "lensless_masked_padded_224")
PLANKTON_REPRESENTATION_PADDED224_TEST_ARGELLA_PATH = os.path.join(PLANKTON_REPRESENTATION_PATH, "plankton_masked_padded_224_test_arcella")



WHOI15_2007_REPRESENTATION_PADDED224_PATH = os.path.join(PLANKTON_REPRESENTATION_PATH, "whoi15_2007_padded_224")
WHOI40_REPRESENTATION_PADDED224_PATH = os.path.join(PLANKTON_REPRESENTATION_PATH, "whoi40_padded_224")


VACUOLES_REPRESENTATION_PATH = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "vacuoles", "representation", "vit-b-1k-dino")

SIPAKMED_REPRESENTATION_PATH = os.path.join(os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "sipakmed", "representation", "vit-b-1k-dino")





def loader_rgb(file):
    image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


class RepresentationDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, **kwargs):
    super().__init__( **kwargs)


    # select split
    self.x = x #min_max_normalization(x)

    # select split
    y = y

    self.num_classes = np.unique(y).size

    self.y = torch.nn.functional.one_hot(torch.from_numpy(y).to(torch.int64), num_classes=self.num_classes)
    self.y = self.y.float()


  def __getitem__(self, idx):
    return torch.from_numpy(self.x[idx]), self.y[idx]

  def __len__(self):
    return self.x.shape[0]


def min_max_normalization(data):
    data_min = np.min(data) # -14.5106125 # precomputed
    data_max = np.max(data) # 19.817413 # precomputed
    #print("Min: ", data_min, " Max: ", data_max)
    data = (data - data_min)/(data_max-data_min)
    return data


def dsprites_min_max_normalization(data):
    data_min = -14.5106125 #np.min(data) # -14.5106125 # precomputed
    data_max = 19.817413 #np.max(data) # 19.817413 # precomputed
    #print("Min: ", data_min, " Max: ", data_max)
    data = (data - data_min)/(data_max-data_min)
    return data


class NumpyDataset(torch.utils.data.Dataset):

  def __init__(self, x, y, **kwargs):
    super().__init__( **kwargs)


    # select split
    self.x = x

    # select split
    y = y

    self.num_classes = np.unique(y).size

    self.y = torch.from_numpy(y).float()


  def __getitem__(self, idx):
    return torch.from_numpy(self.x[idx]), self.y[idx]

  def __len__(self):
    return self.x.shape[0]




class SIPAKMEDDataset(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False, random_seed=0):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(SIPAKMED_REPRESENTATION_PATH, "representations.npz"))
        csv = np.load(os.path.join(SIPAKMED_REPRESENTATION_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(SIPAKMED_REPRESENTATION_PATH, "filenames.npz"))[split]


        if split == "train" or split == "val":
            filename = 'TRAIN_labels_name.txt'

        else:
            filename = 'TEST_labels_name.txt'
        labels_name = []
        with open(os.path.join(SIPAKMED_REPRESENTATION_PATH, filename)) as file:
          for line in file:
            labels_name.append(line[:-1])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization
        x = dsprites_min_max_normalization(x)

        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name

        super(SIPAKMEDDataset, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(SIPAKMEDDataset, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y





class VACUOLESDataset(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False, random_seed=0):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(VACUOLES_REPRESENTATION_PATH, "representations.npz"))
        csv = np.load(os.path.join(VACUOLES_REPRESENTATION_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(VACUOLES_REPRESENTATION_PATH, "filenames.npz"))[split]


        if split == "train" or split == "val":
            filename = 'TRAIN_labels_name.txt'

        else:
            filename = 'TEST_labels_name.txt'
        labels_name = []
        with open(os.path.join(VACUOLES_REPRESENTATION_PATH, filename)) as file:
          for line in file:
            labels_name.append(line[:-1])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization
        x = dsprites_min_max_normalization(x)

        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name

        super(VACUOLESDataset, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(VACUOLESDataset, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y



class LENSLESSDataset(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_PATH, "representations.npz"))
        csv = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_PATH, "filenames.npz"))[split]





        labels_name = []
        with open(os.path.join(PLANKTON_REPRESENTATION_PADDED224_PATH, 'labels_name.txt')) as file:
          for line in file:
            labels_name.append(line[:-2])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization
        x = dsprites_min_max_normalization(x)

        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name

        super(LENSLESSDataset, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(LENSLESSDataset, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y





class LENSLESSDataset_TestArcella(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_TEST_ARGELLA_PATH, "representations.npz"))
        csv = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_TEST_ARGELLA_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(PLANKTON_REPRESENTATION_PADDED224_TEST_ARGELLA_PATH, "filenames.npz"))[split]


        if split == "train" or split == "val":
            filename = 'TRAIN_labels_name.txt'

        else:
            filename = 'TEST_labels_name.txt'

        labels_name = []
        with open(os.path.join(PLANKTON_REPRESENTATION_PADDED224_TEST_ARGELLA_PATH, filename)) as file:
          for line in file:
            labels_name.append(line[:-2])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization
        x = dsprites_min_max_normalization(x)

        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name

        super(LENSLESSDataset_TestArcella, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(LENSLESSDataset_TestArcella, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y







class WHOI152007Dataset(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(WHOI15_2007_REPRESENTATION_PADDED224_PATH, "representations.npz"))
        csv = np.load(os.path.join(WHOI15_2007_REPRESENTATION_PADDED224_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(WHOI15_2007_REPRESENTATION_PADDED224_PATH, "filenames.npz"))[split]

        labels_name = []
        with open(os.path.join(WHOI15_2007_REPRESENTATION_PADDED224_PATH, 'labels_name.txt')) as file:
          for line in file:
            labels_name.append(line[:-2])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization
        x = dsprites_min_max_normalization(x)


        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name


        super(WHOI152007Dataset, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(WHOI152007Dataset, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y





class WHOI40Dataset(NumpyDataset):

    def __init__(self, split, classes=None, get_filename=False):
        self.data_shape = [768]

        assert split in ["train", "val", "test"]

        self.get_filename = get_filename

        # load representation
        rep = np.load(os.path.join(WHOI40_REPRESENTATION_PADDED224_PATH, "representations.npz"))
        csv = np.load(os.path.join(WHOI40_REPRESENTATION_PADDED224_PATH, "classes.npz"))
        self.filenames = np.load(os.path.join(WHOI40_REPRESENTATION_PADDED224_PATH, "filenames.npz"))[split]

        labels_name = []
        with open(os.path.join(WHOI40_REPRESENTATION_PADDED224_PATH, 'labels_name.txt')) as file:
          for line in file:
            labels_name.append(line[:-2])  # storing everything in memory!


        x = rep[split]
        y = csv[split]

        # normalization


        x = dsprites_min_max_normalization(x)




        self.samples = [ (x[i], y[i]) for i in range(x.shape[0])]
        self.classes = labels_name


        super(WHOI40Dataset, self).__init__(x, y)



    def __getitem__(self, item):
        x, y = super(WHOI40Dataset, self).__getitem__(item)

        if self.get_filename:
            filename = self.filenames[item]

            return x, y, filename


        return x, y



