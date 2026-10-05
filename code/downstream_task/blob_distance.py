"""Downstream classification task."""

import os
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import json

import wandb
import random

from code.downstream_task.metric import Metric


def min_max_normalization(data):
    data_min = np.min(data)
    data_max = np.max(data)
    #print("Min: ", data_min, " Max: ", data_max)
    data = (data - data_min)/(data_max-data_min)
    return data


def compute_centroid_sklearn(points):
    """
    Compute the centroid of a set of N-dimensional sample points using numpy.

    Parameters:
    points (list of lists or numpy array): A list or array of N-dimensional coordinates.

    Returns:
    numpy array: The N-dimensional coordinates of the centroid.
    """
    # Convert the list of points to a numpy array if it isn't already
    points_array = np.array(points)

    # Compute the centroid
    centroid = np.mean(points_array, axis=0)

    return centroid


def predict(model, threshold, x):

    distances, _ = model.kneighbors(x)

    avg_distances = distances.mean(axis=1)
    preds = (avg_distances > threshold).astype(np.int32)

    return preds


def save_scatterplot(x, y, namex, namey, classes, centroid, path, title=""):
    data = {namex: [elem for elem in x], namey: [elem for elem in y]}


    with sns.axes_style("whitegrid"):
        sns.scatterplot(data=data, x=namex, y=namey, hue=classes,  palette = "deep", alpha=0.75,
                            linewidth=0)  # , kde=True palette=["red", "blue"],

    # add centroid
    sns.scatterplot(x=centroid[0], y=centroid[1], hue=["centroid"], palette=["black"], alpha=0.75,
                    linewidth=0)  # , kde=True

    # add some connections with the centroid
    # Select random pairs of points
    num_lines = 10  # Number of lines to draw
    indices = list(range(len(data[namex])))
    random_pairs = random.choices(indices, k=num_lines)

    # Draw lines between the random pairs
    for i in random_pairs:
        plt.plot([data[namex][i], centroid[0]], [data[namey][i], centroid[1]], 'r-')

    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()
    print("Save plot in " + path)




def make_predictor(name):

    if name == "knn":
        return NearestNeighbors(n_neighbors=5)
    else:
        raise "Name of predictor not exists!"


def _compute_model(x_train, x_test, predictor):
    """Compute average accuracy for train and test set."""
    train_acc = 0.0
    test_acc = 0.0

    # concat train and val
    x_train = np.concatenate((x_train, x_test), axis=0)

    model = predictor
    model.fit(x_train)


    return  model





class Blob_distance(Metric):

    def __init__(self,  mode, **kwargs):

        super(Blob_distance, self).__init__(**kwargs)
        self.mode = mode
        self.predictor = "blob_distance"
        self.name = "blob_distance"


    def get_score(self, class_to_evaluate, class_name, class_to_compare, class_to_compare_name, feature_names ):
        ''' Return the score '''

        output_directory = os.path.join(self.path, self.predictor)

        # make experiment directory
        if not os.path.exists(output_directory):
            # if the demo_folder directory is not present then create it.
            os.makedirs(output_directory)


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


        x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
        y_train, y_val, y_test = csv["train"].astype(np.int32), csv["val"].astype(np.int32), csv["test"].astype(np.int32)


        # compute centroid
        idx_A = np.where(y_test == class_to_compare)
        class_A = x_test[idx_A, ...]
        class_A = class_A.squeeze()
        centroid = compute_centroid_sklearn(class_A)

        # compute distance from the centroid of blob A
        A_distances = {f: 0.0 for f in feature_names}

        for i, feature in zip(range(x_test.shape[1]), feature_names):
            distances = np.abs(class_A[:, i] - centroid[i])
            A_distances[feature] = float(np.mean(distances))


        idx_B = np.where(y_test == class_to_evaluate)
        class_B = x_test[idx_B, ...]
        class_B = class_B.squeeze()
        # compute distance from the centroid of blob A
        B_distances = {f: 0.0 for f in feature_names}

        for i, feature in zip(range(x_test.shape[1]), feature_names):
            distances = np.abs(class_B[:, i] - centroid[i])
            B_distances[feature] = float(np.mean(distances))


        # compute distance sample x sample
        distances = np.zeros((class_B.shape[0], class_A.shape[0], class_A.shape[1]))

        # Compute distances
        for i in range(class_B.shape[0]):
            for j in range(class_A.shape[0]):
                distances[i, j, :] = np.abs(class_A[j,:] - class_B[i, :])



        with open(os.path.join(output_directory, 'distances_of_compare_class.json'), 'w') as fp:
            json.dump(A_distances, fp)


        with open(os.path.join(output_directory, 'distances_of_evaluate_class.json'), 'w') as fp:
            json.dump(B_distances, fp)


        with open(os.path.join(output_directory, 'distances_of_evaluate_class_sample_wise.json'), 'w') as fp:
            print(distances[..., 0].mean(axis=(0, 1)).shape)
            distances = { name: distances[..., i].mean(axis=(0,1)) for i, name in enumerate(feature_names)}
            json.dump(distances, fp)

        blobs = np.concatenate((class_A, class_B), axis=0)
        classes_blob = y_test[np.concatenate((idx_A, idx_B))]

        classes_blob = np.where(classes_blob == class_to_evaluate, class_name, class_to_compare)

        for i, name in enumerate(feature_names):
            for j, namey in enumerate(feature_names):

                save_scatterplot(blobs[:, i], blobs[:,j], namex=name, namey=namey, classes=classes_blob.flatten(), centroid=centroid, path=os.path.join(output_directory, f'{name}_{namey}.png'), title="Distance with centroid")

        return Blob_distance









