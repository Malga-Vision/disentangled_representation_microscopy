'''
Implementation of visualizations
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
from torch.utils.data import DataLoader

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

import random
import numpy as np
import wandb


from code.choose_model import get_named_model
from code.choose_dataset import get_named_dataset
from code.visualization.utils import *

def save_reconstruction(id, images, reconstruction, log_wandb=True):



    if id is None:
        id = "reconstruction"

    images = np.moveaxis(images, 1, -1)
    reconstruction = np.moveaxis(reconstruction, 1, -1)

    paired_pics = np.concatenate((images, reconstruction), axis=2)
    paired_pics = [paired_pics[i, :, :, :] for i in range(paired_pics.shape[0])]

    grid_save_images(id, paired_pics, log_wandb)




def save_random_samples(id, random_samples, log_wandb=True):

    random_samples = np.moveaxis(random_samples, 1, -1)

    grid_save_images("random_samples", random_samples, log_wandb)




def save_traversal(directory, mode, representations, model, device, num_frames=20, fps=10):
    results_dir = os.path.join(directory, "traversals")


    if not os.path.exists(results_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(results_dir)

    for i, representation in enumerate(representations):

        frames =[]
        for j in range(representation.shape[0]):
            code = np.repeat( np.expand_dims(representation, axis=0), num_frames, axis=0)

            code[:, j] = traver_interval(representation[j], num_frames,
                                                       np.min(representations[:, j]),
                                                       np.max(representations[:, j]))

            code = torch.from_numpy(code).to(device)
            frames.append(model.decode(code).cpu().detach().numpy())


        # reshape to put channel last
        frames = np.moveaxis(np.array(frames), 2, -1) * 255.
        filename = os.path.join(results_dir, "minmax_interval_cycle{}.gif".format(i))
        save_animation(frames.astype(np.uint8), filename, fps)


def create_visualization_directory(directory):

    process_dir = os.path.join(directory, "visualization")

    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)


    return process_dir


def visualize_model(directory, args):
    # set fixed seed

    random.seed(args["random_seed"])
    np.random.seed(args["random_seed"])
    torch.manual_seed(args["random_seed"])
    torch.cuda.manual_seed_all(args["random_seed"])

    # load dataset
    dataset = get_named_dataset(args["postprocess_dataset"])(split="test", get_filename=False)

    # get model
    model = get_named_model(args["method"])(data_shape=dataset.data_shape, latent_dim=args["latent_dim"],
                                            n_filters=args["n_filters"])

    # load pretrained weights
    model.load_state(os.path.join(directory, "model", "checkpoint", "model.pth"))

    # move model to gpu
    model.to(device)

    n_accumulation = args["grad_acc_steps"]  # steps for gradient accumulation

    directory = create_visualization_directory(directory)
    with torch.no_grad():
        # the dataset requires a dataloader
        if args["multithread"]:
            val_dl = DataLoader(dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                                num_workers=4, drop_last=False, pin_memory=True)

            print("Using Dataloader multithreading!")

        else:
            val_dl = DataLoader(dataset, batch_size=args["batch_size"] // n_accumulation, shuffle=False,
                                num_workers=0, drop_last=False, pin_memory=False)

            print("Not using Dataloader multithreading!")

        print("===============START VISUALIZING===============")

        #  save reconsructions
        for i, (inputs, labels) in enumerate(val_dl):

            if i>args["n_reconstruction"]:
                break


            # move data to GPU
            inputs = inputs.to(device)
            outputs = model(inputs)

            inputs = inputs.cpu().detach().numpy()
            outputs = outputs.cpu().detach().numpy()


            if not os.path.exists(os.path.join(directory, "reconstructions")):
                # if the demo_folder directory is not present then create it.
                os.makedirs(os.path.join(directory, "reconstructions"))

            save_reconstruction(os.path.join(directory, "reconstructions", f"{i}_reconstructions.png"), inputs, outputs, args["log_wandb"])

        # save latent traversals
        for i, (inputs, labels) in enumerate(val_dl):

            if i > args["n_animations"]:
                break

            # move data to GPU
            inputs = inputs.to(device)

            dict_representation = {k: r.cpu().detach().numpy() for k, r in model.encode(inputs).items()}

            save_traversal(directory,  "mu", dict_representation["mu"], model, device)



        # save random samples
        #for i in range(args["n_samples"]):
            #random_samples = model.sample(args["n_samples"])
            #save_random_samples(i, random_samples.cpu().detach().numpy())



        print("===============END VISUALIZING===============")
