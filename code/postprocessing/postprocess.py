'''
Implementation of postprocessing model
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch


import os
import pickle
import time

import numpy as np
import torch
import random

import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.empty_cache()



import os
import pandas as pd
import numpy as np

from code.choose_model import get_named_model
from code.choose_dataset import get_named_dataset


def get_representation_dataloader(args, dl, model, mode="mu"):

    # dict of representation to save
    representation_to_save = None
    classes_to_save = []
    filename_to_save = []

    print("Saving representation of {} samples (batches) ".format(len(dl)))

    # iterate over the dataset, with sampling
    for i, (images, labels, filename) in enumerate(dl):


        # move data to GPU
        images = images.to(device)

        labels = labels.numpy()

        dict_representation = model.encode(images)

        # update representation list
        old = representation_to_save
        new = dict_representation[mode].cpu().detach().numpy()[:, args["postprocess_dims"]]
        representation_to_save = new if old is None else np.vstack((old, new))

        # update classes list
        classes_to_save.extend(labels)

        # update filename list
        filename_to_save.extend(filename)

    return representation_to_save, classes_to_save, filename_to_save


def load_model(directory, model, quantized=False):

    if quantized:
        checkpoint_dir = os.path.join(directory, "lq_model", "checkpoint", "model.pth")
    else:

        checkpoint_dir = os.path.join(directory, "model", "checkpoint", "model.pth")

    model.load_state(checkpoint_dir)

    return model


def create_preprocessing_directory(directory, quantized=False):

    if not quantized:
        process_dir = os.path.join(directory, "postprocess")
    else:
        process_dir = os.path.join(directory, "postprocess_quantized")


    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)
    else:
        raise FileExistsError("Preprocessing folder exists")

    return process_dir


def postprocess_model(directory, args):

    # set fixed seed
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    torch.manual_seed(args["random_seed"])
    torch.cuda.manual_seed_all(args["random_seed"])

    train_dataset = get_named_dataset(args["postprocess_dataset"])(split="train", get_filename=True)

    val_dataset = get_named_dataset(args["postprocess_dataset"])(split="val", get_filename=True)

    test_dataset = get_named_dataset(args["postprocess_dataset"])(split="test", get_filename=True)

    # split train and validation BALANCED
    #train_dataset, val_dataset = train_test_split(train_dataset, test_size= args["perc_val_set"], random_state=args["split_random_seed"], stratify=[ y for x, y in train_dataset.samples])


    # get model
    model = get_named_model(args["method"])(data_shape=test_dataset.data_shape,  **args)

    model = load_model(directory, model, args["lq"])

    # create the folder devoted to the postprocessing
    directory = create_preprocessing_directory(directory, args["lq"])

    # move model to gpu
    model.to(device)

    print("===============START PREPROCESSING===============")
    model.eval()

    n_accumulation = args["grad_acc_steps"]  # steps for gradient accumulation

    if args["multithread"]:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                              num_workers=16, drop_last=False, pin_memory=True)

        val_dl = DataLoader(val_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                            num_workers=16, drop_last=False, pin_memory=True)

        test_dl = DataLoader(test_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                            num_workers=16, drop_last=False, pin_memory=True)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                              num_workers=0, drop_last=False, pin_memory=False)

        val_dl = DataLoader(val_dataset, batch_size=args["val_batch_size"] // n_accumulation, shuffle=False,
                            num_workers=0, drop_last=False, pin_memory=False)

        test_dl = DataLoader(test_dataset, batch_size=args["val_batch_size"] // n_accumulation, shuffle=False,
                            num_workers=0, drop_last=False, pin_memory=False)

        print("Not using Dataloader multithreading!")


    # extract representation for all splits
    representation_to_save = {}
    classes_to_save = {}
    filename_to_save = {}
    for split, dl in zip(["train", "val", "test"], [train_dl, val_dl, test_dl]):
        rep, cls, file = get_representation_dataloader(args, dl, model, mode="mu")
        representation_to_save[split] = rep
        classes_to_save[split] = cls
        filename_to_save[split] = file

    # save representation
    np.savez_compressed(os.path.join(directory, "representations"), **representation_to_save)
    np.savez_compressed(os.path.join(directory, "classes"), **classes_to_save)
    np.savez_compressed(os.path.join(directory, "filenames"), **filename_to_save)

    print(representation_to_save["train"].shape)
    print(representation_to_save["val"].shape)
    print(representation_to_save["test"].shape)

    # save ordered labels name
    file = open(os.path.join(directory, 'TEST_labels_name.txt'), 'w')
    for item in test_dataset.classes:
        file.write(item + "\n")
    file.close()

    # save ordered labels name
    file = open(os.path.join(directory, 'TRAIN_labels_name.txt'), 'w')
    for item in train_dataset.classes:
        file.write(item + "\n")
    file.close()


    print("===============END PREPROCESSING===============")








