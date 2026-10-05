"""Downstream classification task."""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
device_type = "cuda" if use_cuda else "cpu"
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


import random
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import wandb
import json


from code.downstream_task.metric import Metric
from code.training.train_classifier import train, test
from code.dataset.representation_dataset import RepresentationDataset
from code.models.mlp_classifier import Classifier



def compute_prediction(x, y, model):

    dataset = RepresentationDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Initialize lists to store true labels and predictions
    all_preds = []
    all_targets = []

    # Disable gradient calculation for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs.to(device))
            _, preds = torch.max(outputs, 1)
            _, labels = torch.max(labels, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)


    return all_targets, all_preds



def make_confusion_matrix(y_true, y_pred, labels, title, path, wandb=True):
    matrix = confusion_matrix(y_true, y_pred)
    matrix = matrix.astype(np.float64)

    n_labels = len(labels)


    # normalize confusion matrix
    for i in range(int(n_labels)):
        # M[i,j] stands for Element of real class i was classified as j
        sum = np.sum(matrix[i, :])
        matrix[i, :] = matrix[i, :] / sum


    df_cm = pd.DataFrame(matrix, labels, labels)

    # matrix of bool values, True if different from Zero
    annot_pd = df_cm.applymap(lambda x: "{:.2%}".format(
        x) if round(x, 3) != 0.000 else '')

    mean_acc = np.array([matrix[i, i]
                         for i in range(n_labels)]).sum() / n_labels

    std_acc = np.std(np.array([matrix[i, i]
                               for i in range(n_labels)]))

    fig = plt.figure(figsize=(10, 7))
    fig.tight_layout()

    sn.set(font_scale=0.55)  # label size
    sn.heatmap(df_cm, annot=annot_pd, annot_kws={
        "size": 9}, fmt='s', vmin=0, vmax=1, cmap="Blues", cbar=False)  # font size

    plt.title(
        "Mean acc: {:.2%} - Std: {:.2} - {}".format(mean_acc, std_acc, title))

    plt.subplots_adjust(bottom=0.28)

    if not wandb:
        plt.savefig(path, dpi=300)

        print("Save plot in " + path)
        # clear the matplotlib figure to free memory
        #plt.close(fig)
        return fig, mean_acc

    return fig, mean_acc



def compute_downstream_task(representation, classes, path):

    # Create a numpy random state. We will sample the random seeds for training
    random_state = np.random.RandomState(0)
    data_seed = random_state.randint(2 ** 31)

    # set fixed seed
    np.random.seed(data_seed)
    random.seed(data_seed)
    torch.manual_seed(data_seed)
    torch.cuda.manual_seed_all(data_seed)

    scores = {}

    x_train, x_val, x_test = representation["train"], representation["val"], representation["test"]
    y_train, y_val, y_test = classes["train"].astype(np.int32), classes["val"].astype(np.int32), classes["test"].astype(np.int32)

    train_dataset = RepresentationDataset(x_train, y_train)
    val_dataset = RepresentationDataset(x_val, y_val)
    test_dataset = RepresentationDataset(x_test, y_test)


    predictor_model = Classifier(input_dim=x_train.shape[1], n_classes=train_dataset.num_classes)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(predictor_model.parameters(), lr=0.0001, weight_decay=0.1) # , weight_decay=1

    # Train model
    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = train(
        predictor_model, train_loader, val_loader, criterion, optimizer, num_epochs=1000, patience=30, device=device,
        save_path=path)


    # load trained model
    predictor_model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))

    # Test the model
    test_loss, test_accuracy = test(predictor_model, test_loader, criterion, device=device)


    scores["train"], scores["val"]  = train_accuracy_history[-1], val_accuracy_history[-1]
    scores["test"] = test_accuracy
    return scores, predictor_model




class MLP_classifier(Metric):


    def __init__(self,  **kwargs):

        super(MLP_classifier, self).__init__(**kwargs)
        self.predictor = "mlp"
        self.name = "mlp_classifier"


    def get_score(self, feature_names, **kwargs):
        ''' Return the score '''


        # load representation
        rep = np.load(os.path.join(self.path, "representations.npz") )
        csv = np.load(os.path.join(self.path, "classes.npz") )

        test_labels_name = []
        with open(os.path.join(self.path, 'TEST_labels_name.txt')) as file:
            for line in file:
                test_labels_name.append(line)  # storing everything in memory!

        train_labels_name = []
        with open(os.path.join(self.path, 'TRAIN_labels_name.txt')) as file:
            for line in file:
                train_labels_name.append(line)  # storing everything in memory!

        eval_directory = os.path.join(self.path, self.predictor)
        # make experiment directory
        if not os.path.exists(eval_directory):
            # if the demo_folder directory is not present then create it.
            os.makedirs(eval_directory)


        scores, model = compute_downstream_task(rep, csv, path=eval_directory)


        with open(os.path.join(eval_directory, 'accuracy.json'), 'w') as fp:
            json.dump(scores, fp)


        # compute confusion matrix
        x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
        y_train, y_val, y_test = csv["train"].astype(np.int32), csv["val"].astype(np.int32), csv["test"].astype(np.int32)

        target, pred = compute_prediction(x_train, y_train, model)
        fig_train, train_balanced_acc = make_confusion_matrix(target, pred, train_labels_name, "Training set",
                                          os.path.join(eval_directory, self.predictor + "_train_confusion.png"), self.wandb)

        target, pred = compute_prediction(x_val, y_val, model)
        fig_val, val_balanced_acc = make_confusion_matrix(target, pred, train_labels_name, "Validation set",
                                        os.path.join(eval_directory, self.predictor + "_val_confusion.png"), self.wandb)

        target, pred = compute_prediction(x_test, y_test, model)
        fig_test, test_balanced_acc = make_confusion_matrix(target, pred, test_labels_name, "Test set",
                                         os.path.join(eval_directory, self.predictor + "_test_confusion.png"), self.wandb)


        with open(os.path.join(eval_directory, 'balanced_accuracy.json'), 'w') as fp:
            json.dump({"train": train_balanced_acc, "val": val_balanced_acc, "test": test_balanced_acc}, fp)


        # clear the matplotlib figure to free memory
        plt.close(fig_train)
        plt.close(fig_val)
        plt.close(fig_test)




        if self.wandb:
            wandb.log({k+"_accuracy_" + self.predictor :v for k, v in scores.items()})

        return scores









