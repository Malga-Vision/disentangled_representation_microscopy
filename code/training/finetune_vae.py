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
from torch.utils.data import Dataset, DataLoader, ConcatDataset


from code.choose_model import get_named_model
from code.choose_criterion import get_named_criterion
from code.choose_dataset import get_named_dataset


from code.training.scheduler.cyclical_annealing_schedule import ConstantBeta, CycleLinearBeta
from code.models.utils import *


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_type = "cuda" if use_cuda else "cpu"
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def save_model(directory, model):
    checkpoint_dir = os.path.join(directory, "model", "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Save the model checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model.pth')  # _checkpoint_{iteration + 1}
    model.save_state(checkpoint_path)
    return checkpoint_path



def load_model(directory, model):
    checkpoint_dir = os.path.join(directory, "model", "checkpoint", "model.pth")

    model.load_state(checkpoint_dir)

    return model


def create_model_directory(directory):

    print("Model folder already exists!")

    return directory


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
    #test_dataset = get_named_dataset(args["dataset"])(split="test", get_filename=False)

    # use both training and test data
    #train_dataset = ConcatDataset([train_dataset, test_dataset])

    # get model
    model = get_named_model(args["method"])(data_shape=train_dataset.data_shape, latent_dim=args["latent_dim"],
                                            n_filters=args["n_filters"], dim_to_freeze =args["dim_to_freeze"] )

    # load pretrained weights
    model = load_model(directory, model)

    optimizer = torch.optim.AdamW(lr=args["lr"], params=model.parameters())

    # get beta_scheduler
    beta_scheduler =  ConstantBeta(b_max=args["beta"] ) #CycleLinearBeta(b_max=args["beta"], n_cycle=3, steps=50)  # ConstantBeta(b_max=args["beta"] )

    # get criterion
    reconstruction_criterion = get_named_criterion(args["criterion"])
    kl_criterion = get_named_criterion("normal_kullback")

    # move model to gpu
    model.to(device)

    n_accumulation = args["grad_acc_steps"]  # steps for gradient accumulation


    if args["multithread"]:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=True,
                              num_workers=16, drop_last=False, pin_memory=True)

        print("Using Dataloader multithreading!")
    else:
        train_dl = DataLoader(train_dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=True,
                              num_workers=0, drop_last=False, pin_memory=False)

        print("Not using Dataloader multithreading!")


    print("Number of total parameters of the model: {:,}".format(num_params(model)))
    print("Number of trainable parameters of the model: {:,}".format(num_trainable_params(model)))
    print("Using device: ", device, "| Device type: ", device_type)


    print("===============START TRAINING===============")


    start_time = time.time()

    best_train_loss = np.inf

    epochs = args["epochs"]

    for epoch in range(epochs):

        train_loss = 0.0
        train_rec_loss = 0.0
        train_kl_loss = 0.0

        print("-" * 20, f"Epoch {epoch}!", "-" * 20)

        # training set
        for i, (inputs, labels) in enumerate(train_dl):

            model.train()

            # move data to GPU
            inputs = inputs.to(device)
            labels.to(device)

            # encode image
            representation = model.encode(inputs)

            # reconstruct back image
            outputs = model.decode(representation["sampled"])

            # compute the model output calculate loss for data
            reconstruction_loss = -reconstruction_criterion(inputs, outputs)
            kl_loss = beta_scheduler.beta * kl_criterion(representation["mu"], representation["log_var"])
            elbo = reconstruction_loss - kl_loss

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


        # avg losses
        train_loss /= len(train_dl)
        train_rec_loss /= len(train_dl)
        train_kl_loss /= len(train_dl)

        checkpoint_path = save_model(directory, model)
        print(f"Checkpoint saved at {checkpoint_path}!")

        print(f"Train Loss : {train_loss:.2f} (Rec: {train_rec_loss:.2f}; Kl: {train_kl_loss:.2f}), Old best loss: {best_train_loss:.2f}")
        best_train_loss = train_loss



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

                "beta":beta_scheduler.beta,
                #"lr":optimizer.param_groups[0]['lr'],
            }
        )

        # update  beta and lr
        beta_scheduler.step()
        #lr_scheduler.step()


    print("===============END TRAINING===============")

    # log info about model
    wandb.log({"num_params": num_params(model), "num_trainable_params": num_trainable_params(model)})


    # save model
    print("Loading model on wandb...")
    wandb.log_artifact(checkpoint_path, name="vae_model.pt", type='model')



    # free gpu memory
    del model
    del train_dataset
    del train_dl
    torch.cuda.empty_cache()

