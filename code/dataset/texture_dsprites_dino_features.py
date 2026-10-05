
"""DSprites dataset and new variants with probabilistic decoders."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torch.utils.data as data

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor





REPRESENTATIONDSPRITES_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "dsprites",
    "representation", "vit-b-1k-dino")


def min_max_normalization(data):
    data_min = -14.5106125 # precomputed
    data_max = 19.817413 # precomputed
    data = (data - data_min)/(data_max-data_min)
    return data


def normalize_data(data):
    #data = data / (np.linalg.norm(data, axis=1, ord=2)[:, np.newaxis])  # normalize with norm 2
    data = min_max_normalization(data)
    return data

def factor_to_index(factors, factors_size):
    """
    Inverts the behavior of the original function.
    """
    idx = 0
    for factor, size in zip(factors, factors_size):
        idx = idx * size + factor
    return idx

def index_to_factor(idx, factors_size):

    factors = []
    for i, size in enumerate(factors_size):

        factor_idx = idx % size
        idx = idx // size

        factors.append(factor_idx)

    return factors


def classes_to_index(sizes, labels):
    """
    Given a list of class sizes and a corresponding list of labels, return the index of the image.

    :param sizes: List of sizes representing the number of classes for each attribute.
    :param labels: List of class labels (one for each attribute).
    :return: Index of the image corresponding to the class labels.
    """
    assert len(sizes) == len(labels), "Sizes and labels must have the same length."

    index = 0
    # Multiply class labels with the product of all subsequent sizes
    product = 1
    for i in reversed(range(len(sizes))):
        index += labels[i] * product
        product *= sizes[i]

    return index


def index_to_classes(sizes, index):
    """
    Given an index, return the corresponding class labels.

    :param sizes: List of sizes representing the number of classes for each attribute.
    :param index: Index of the image.
    :return: List of class labels corresponding to the given index.
    """

    labels = []

    for size in reversed(sizes):
        labels.append(index % size)
        index //= size

    return labels[::-1]  # Reverse the list to get the correct order



def load_chunk( path, chunck_id):
        representation = np.load(os.path.join(path, f"representation_{chunck_id}.npz"))["after_pooling"]
        classes = np.load(os.path.join(path, f"classes_{chunck_id}.npy"))
        return representation, classes



def load_representation(path):

    # compute number of chunkcs
    list_file = os.listdir(path)
    list_file = [f for f in list_file if os.path.isfile(os.path.join(path, f)) ] #remove subdirectories
    n_chunks = len(list_file) // 2

    representation = None
    classes = None

    for i in range(n_chunks):

        r, c = load_chunk(path, i)

        # concat data
        representation = r if representation is None else np.append(representation, r, axis=0)
        classes = c if classes is None else np.append(classes, c, axis=0)

        print("Loaded ", i, "-th chunck")


    return representation, classes


def load_representation_parallel(path, threads=2):

    def load_representation( chunck_id):
        representation = np.load(os.path.join(path, f"representation_{chunck_id}.npz"))["after_pooling"]
        representation = normalize_data(representation)
        return representation


    def load_classes( chunck_id):
        classes = np.load(os.path.join(path, f"classes_{chunck_id}.npy"))
        return classes

    # compute number of chunkcs
    list_file = os.listdir(path)
    list_file = [f for f in list_file if os.path.isfile(os.path.join(path, f)) ] #remove subdirectories
    n_chunks = len(list_file) // 2

    representation = None
    classes = None

    with ThreadPoolExecutor(threads) as pool:


        r = pool.map(load_representation, [i for i in range(n_chunks)])
        c = pool.map(load_classes, [i for i in range(n_chunks)])



    # concat data
    representation = np.vstack(list(r)) #r if representation is None else np.append(representation, r, axis=0)
    classes = np.vstack(list(c)) #c if classes is None else np.append(classes, c, axis=0)



    return representation, classes



class dSpritesDino(data.Dataset): # data.Dataset


  def __init__(self, latent_factor_indices=None, batch_size=64, random_state=0, resize=None, center_crop=None, path_to_features=None, **kwargs):
    # By default, all factors (including shape) are considered ground truth
    # factors.

    super(dSpritesDino, self).__init__(latent_factor_indices, batch_size, random_state)

    if path_to_features is None:
        path_to_features = REPRESENTATIONDSPRITES_PATH

    if latent_factor_indices is None:
      latent_factor_indices = list(range(7)) # default is dsprites

    self.latent_factor_indices = latent_factor_indices

    # Load the data so that we can sample from it.
    try:

        print("Start loading data and then normalize!")

        # Data was saved originally using python2, so we need to set the encoding.
        representation, classes = load_representation_parallel(path_to_features, threads=35) # load_representation(path_to_features)
    except:
        raise ValueError("Representation dataset not found.")

    print("Data Loaded!")

    # normalize data
    #representation = normalize_data(representation)


    self.images = representation
    self.latents_classes = np.squeeze(classes)
    

    self.full_factor_sizes = np.array([5, 7, 3, 6, 40, 32, 32]) # [np.unique(self.latents_classes[:, i], axis=0).shape[0] for i in latent_factor_indices]
    self.factor_names = ["Texture", "Color", "Shape", "Scale", "Orientation", "PosX", "PosY"]


    self.data_shape = [representation.shape[1]]

    print("Batch size of the Dataset is ignored...Dataloader will batch!")
    self.batch_size = 1



  def num_factors(self):
      return self.latents_classes.shape[1]


  def num_channels(self):
      return 0


  def get_shape(self):
      return self.data_shape


  def __len__(self):
    return np.prod(self.full_factor_sizes)


  def __getitem__(self, idx):


      #factors= index_to_factor(idx, self.full_factor_sizes)


      image, classes = self.images[idx], self.latents_classes[idx]

      image = torch.from_numpy(image)

      factors = np.expand_dims(classes, axis=0)
      factors = torch.from_numpy(factors)

      return factors, torch.squeeze(image).float(), classes


  def sample_observations_from_factors(self, factors):

      factors = factors[0]
      idx = classes_to_index(self.full_factor_sizes, factors)
      _ , image, classes = self.__getitem__(idx)
      return image, classes


  def get_images(self, idx):
      return self.__getitem__(idx)


  @property
  def factors_sizes(self):
      return self.full_factor_sizes


