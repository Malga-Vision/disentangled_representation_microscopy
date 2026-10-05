'''
Implementation of general training scheme
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import numpy as np
import torch
import random

import wandb
from torch.utils.data import Dataset, DataLoader


from code.choose_model import get_named_model
from code.choose_criterion import get_named_criterion
from code.choose_dataset import get_named_dataset


from code.training.scheduler.cyclical_annealing_schedule import CycleLinearBeta, ConstantBeta
from code.models.utils import *


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_type = "cuda" if use_cuda else "cpu"
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_model(directory, model):
    checkpoint_dir = os.path.join(directory, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model.pth')  # _checkpoint_{iteration + 1}
    model.save_state(checkpoint_path)
    return checkpoint_path


def create_model_directory(directory):

    model_dir = os.path.join(directory, "model")
    # make experiment directory
    if not os.path.exists(model_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(model_dir)
    else:
        raise FileExistsError("Model folder exists")

    return model_dir


def train_model(directory, args):

    # set fixed seed

    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    torch.manual_seed(args["random_seed"])
    torch.cuda.manual_seed_all(args["random_seed"])


    # create the folder devoted to the model
    directory = create_model_directory(directory)

    # get dataset
    train_dataset = get_named_dataset(args["dataset"])(split="train", get_filename=False)

    # get model
    model = get_named_model(args["method"])(data_shape=train_dataset.data_shape, latent_dim=args["latent_dim"],
                                            n_filters=args["n_filters"])

    # split train and validation
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset,
                                                               [1. - args["perc_val_set"], args["perc_val_set"]],
                                                               generator=torch.Generator().manual_seed(
                                                                   args["split_random_seed"]))

    optimizer = torch.optim.Adam(lr=args["lr"], params=model.parameters())

    # get beta_scheduler
    beta_scheduler = ConstantBeta(b_max=args["beta"] )

    # get criterion
    reconstruction_criterion = get_named_criterion(args["criterion"])
    kl_criterion = get_named_criterion("normal_kullback")

    # move model to gpu
    model.to(device)

    n_accumulation = args["grad_acc_steps"]  # steps for gradient accumulation

    if args["multithread"]:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=True,
                              num_workers=16, drop_last=True, pin_memory=True)

        val_dl = DataLoader(val_dataset, batch_size=args["val_batch_size"] // n_accumulation, shuffle=False,
                              num_workers=4, drop_last=True, pin_memory=True)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=True,
                              num_workers=0, drop_last=True, pin_memory=False)

        val_dl = DataLoader(val_dataset, batch_size=args["val_batch_size"] // n_accumulation, shuffle=False,
                              num_workers=0, drop_last=True, pin_memory=False)

        print("Not using Dataloader multithreading!")


    print("Number of total parameters of the model: {:,}".format(num_params(model)))
    print("Number of trainable parameters of the model: {:,}".format(num_trainable_params(model)))
    print("Using device: ", device, "| Device type: ", device_type)


    print("===============START TRAINING===============")


    start_time = time.time()

    best_val_loss = np.inf

    epochs = args["epochs"]

    for epoch in range(epochs):

        train_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0

        val_loss = 0.0
        val_rec_loss = 0.0
        val_kl_loss = 0.0


        print("-" * 20, f"Epoch {epoch}!", "-" * 20)

        # training set
        for i, (inputs, labels) in enumerate(train_dl):

            model.train()

            # separate data
            inputs1, inputs2 = inputs
            labels1, labels2 = labels

            # move data to GPU
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)


            # encode image
            representations = model.encode_couple(inputs1, inputs2)

            # reconstruct back image
            outputs1, outputs2 = model.decode_couple(representations["sampled1"], representations["sampled2"])

            # compute the model output calculate loss for data
            reconstruction_loss = -(reconstruction_criterion(inputs1, outputs1) + reconstruction_criterion(inputs2, outputs2))/2.
            kl_loss = (kl_criterion(representations["mu1"], representations["log_var1"]) + kl_criterion(representations["mu2"], representations["log_var2"])) /2.

            elbo = reconstruction_loss - beta_scheduler.beta * kl_loss

            loss = -elbo

            # Backpropagation and optimization
            loss.backward()

            train_loss += loss.item()
            train_rec_loss += reconstruction_loss.item()
            train_kl_loss += kl_loss.item()


            if (i + 1) % n_accumulation == 0:

                # update weights
                optimizer.step()

                # zero the parameter gradients
                optimizer.zero_grad()

        # training set
        for i, (inputs, labels) in enumerate(val_dl):
            model.eval()

            # separate data
            inputs1, inputs2 = inputs
            labels1, labels2 = labels

            # move data to GPU
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)

            # encode image
            representations = model.encode_couple(inputs1, inputs2)

            # reconstruct back image
            outputs1, outputs2 = model.decode_couple(representations["sampled1"], representations["sampled2"])

            # compute the model output calculate loss for data
            reconstruction_loss = -(reconstruction_criterion(inputs1, outputs1) + reconstruction_criterion(inputs2, outputs2)) / 2.
            kl_loss = (kl_criterion(representations["mu1"], representations["log_var1"]) + kl_criterion(representations["mu2"], representations["log_var2"])) / 2.

            elbo = reconstruction_loss - beta_scheduler.beta * kl_loss

            loss = -elbo

            val_loss += loss.item()
            val_rec_loss += reconstruction_loss.item()
            val_kl_loss += kl_loss.item()


        # avg losses
        train_loss /= len(train_dl)
        train_rec_loss /= len(train_dl)
        train_kl_loss /= len(train_dl)

        val_loss /= len(val_dl)
        val_rec_loss /= len(train_dl)
        val_kl_loss /= len(train_dl)

        if val_loss < best_val_loss:

            checkpoint_path = save_model(directory, model)
            print(f"Checkpoint saved at {checkpoint_path}!")

            print(f"Loss improvement! New best validation loss: {val_loss:.2f}, Old best loss: {best_val_loss:.2f}")
            best_val_loss = val_loss


        else:
            print("Loss not improved")

        # Calculate and print the time taken for this checkpoint
        elapsed_time = time.time() - start_time
        print(f"Time elapsed for last epoch: {elapsed_time:.2f} seconds")
        remainig_time = (epochs - (epoch +1)) * elapsed_time
        print(f"Time to complete {epochs} epochs: {remainig_time:.2f} seconds")

        start_time = time.time()


        # log on wandb
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_reconstruction": train_rec_loss,
                "train_kl": train_kl_loss,

                "val_loss": val_loss,
                "val_reconstruction": val_rec_loss,
                "val_kl": val_kl_loss,

                "beta":beta_scheduler.beta,
            }
        )

        # update  beta and lr
        beta_scheduler.step()


    print("===============END TRAINING===============")

    # log info about model
    wandb.log({"num_params": num_params(model), "num_trainable_params": num_trainable_params(model)})


    # save model
    print("Loading model on wandb...")
    wandb.log_artifact(checkpoint_path, name="adagvae_model.pt", type='model')



    # free gpu memory
    del model
    del train_dataset
    del train_dl
    torch.cuda.empty_cache()

