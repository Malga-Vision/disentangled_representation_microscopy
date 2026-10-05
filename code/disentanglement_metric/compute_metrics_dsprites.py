

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time
import json
import numpy as np
import torch
import random
import pandas as pd

import wandb
from torch.utils.data import Dataset, DataLoader, ConcatDataset



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_type = "cuda" if use_cuda else "cpu"
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


from code.models.backbone_utils import get_named_backbone

from code.choose_model import get_named_model
from code.dataset.texture_dsprites import SimpleTextureDSprites
from code.disentanglement_metric.omes import OMES
from code.disentanglement_metric.dci import DCI_disentanglement
from code.disentanglement_metric.mig import MIG


def get_dataset(args):

    # load entire dataset since the task is to learn a representation
    train_dataset = SimpleTextureDSprites(latent_factor_indices=args["factor_idx"], batch_size=1,
                                                    random_state=args["data_seed"],
                                                    resize=args["resize"],
                                                    center_crop=args["center_crop"])

    # prepare arguments for next functions
    args["n_channel"] = train_dataset.num_channels()
    args["data_shape"] = train_dataset.get_shape()
    return train_dataset, args



def load_model(directory, model):
    checkpoint_dir = os.path.join(directory, "model", "checkpoint", "model.pth")

    model.load_state(checkpoint_dir)

    return model




def get_representation_dataloader(args, train_dl, model, factor_idx_process, num_samples=0.01):

    min_to_sample = 15000

    # 30000 is about 4% of 737280
    if num_samples == "all":
        num_samples = 1.0

    num_images = len(train_dl) + args["batch_size"]
    to_sample = num_samples * num_images

    # dict of representation to save
    representation_to_save = {mode: None for mode in args["mode"]}
    classes_to_save = None


    # at least 10000 samples, if any
    if to_sample < min_to_sample:
        to_sample = min_to_sample

    # prepare backbone
    if args["backbone"] is not None:
        backbone = get_named_backbone(args["backbone"])  # "vit-b-1k-dino"
        backbone.to(device)

    model.to(device)
    print("Saving representation of {} samples".format(int(to_sample)))

    # iterate over the dataset, with sampling
    for i, (_, images, classes) in enumerate(train_dl):

        # move data to GPU
        images = images.to(device)

        #print(classes)
        classes = classes.numpy().squeeze()

        classes = classes[:, factor_idx_process]

        # extract if backbone is defined
        if args["backbone"] is not None:
            images = backbone(images)

        dict_representation = model.encode(images)

        # update representation list
        for mode in args["mode"]:
            old = representation_to_save[mode]
            new = dict_representation[mode].cpu().detach().numpy()
            representation_to_save[mode] = new if old is None else np.vstack((old, new))

        # update classes list
        classes_to_save = classes if classes_to_save is None else np.vstack((classes_to_save, classes))

        # break the loop if samples are enough
        if (i + 1) * args["batch_size"] >= to_sample:
            break

    return representation_to_save, classes_to_save





def postprocess_dsprites(directory, args):


    dataset, args = get_dataset(args)

    # get model

    datashape = [768] if args["backbone"] is not None else [224, 224, 3]

    model = get_named_model(args["method"])(data_shape=datashape, latent_dim=args["latent_dim"],
                                            n_filters=args["n_filters"], dim_to_freeze=args["dim_to_freeze"])
    # load pretrained weights
    model = load_model(directory, model)


    train_dl = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True, num_workers=0, drop_last=False,
                          pin_memory=False)




    output_directory = os.path.join(directory, "postprocess_dsprites")

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

        representation_to_save, classes_to_save = get_representation_dataloader(args, train_dl, model,
                                                                                args["factor_idx_process"])

        # save representation
        np.savez_compressed(os.path.join(output_directory, "representations"), **representation_to_save)

        # save classes
        pd.DataFrame(classes_to_save, columns=[dataset.factor_names[i] for i in args["factor_idx_process"]]).to_csv(os.path.join(output_directory, "classes.csv"), index=False)


def evaluate_model_with_dsprites(directory, args):

    # set fixed seed
    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    torch.manual_seed(args["random_seed"])
    torch.cuda.manual_seed_all(args["random_seed"])

    args["factor_idx"] = list(range(0, 7))
    args["data_seed"] = 42
    args["resize"] = 224
    args["center_crop"] = None
    args["factor_idx_process"] = list(range(0, 7))
    args["mode"] = ["mu"]
    args["batch_size"] = 224

    print("EXTRACT REPRESENTATION START")
    # extract representation
    postprocess_dsprites(directory, args)
    print("EXTRACT REPRESENTATION END")

    output_dir = os.path.join(directory, "postprocess_dsprites")

    representation_path = os.path.join(output_dir, "representations")
    classes_path = os.path.join(output_dir, "classes")

    score = {}
    # evaluate MIG
    print("COMPUTE MIG START")
    mig = MIG(mode="mu", representation_path = representation_path, classes_path = classes_path)
    score = mig.get_score()
    print("COMPUTE MIG END")

    with open(os.path.join(output_dir, 'mig.json'), 'w') as fp:
        json.dump(score, fp)


    # evaluate DCI
    print("COMPUTE DCI START")
    if not os.path.exists(os.path.join(output_dir, 'dci.json')):
        dci = DCI_disentanglement(mode="mu", representation_path = representation_path, classes_path = classes_path)
        score = dci.get_score()
        with open(os.path.join(output_dir, 'dci.json'), 'w') as fp:
            json.dump(score, fp)
    print("COMPUTE DCI END")



    # evaluate OMES
    print("COMPUTE OMES START")
    omes = OMES(mode="mu", representation_path = representation_path, classes_path = classes_path)
    score = omes.get_score()
    print("COMPUTE OMES START")

    with open(os.path.join(output_dir, 'omes.json'), 'w') as fp:
        json.dump(score, fp)







