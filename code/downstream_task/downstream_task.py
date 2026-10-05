"""Downstream classification task."""

import os
import numpy as np
import pickle
from sklearn import ensemble
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import json

import wandb

from code.downstream_task.metric import Metric



def make_confusion_matrix_for_test_arcella(y_true, y_pred, labels, title, path, wandb=True):
    # adjust labels for arcella

    arcella_samples = np.where(y_true == 1) # it's the second class
    to_shift_samples = np.where(y_true > 1)  # the second  class on

    # shift from the second class
    y_true[to_shift_samples] = y_true[to_shift_samples] - 1
    # now arcella is the last
    y_true[arcella_samples] = len(labels)-1

    # change position of the class
    labels.remove("ARCELLA VULGARI\n")
    labels.append("ARCELLA VULGARI\n")


    return  make_confusion_matrix(y_true, y_pred, labels, title, path, wandb)


def make_confusion_matrix_for_test_dileptus(y_true, y_pred, labels, title, path, wandb=True):
    # adjust labels for arcella

    arcella_samples = np.where(y_true == 4) # it's the second class
    to_shift_samples = np.where(y_true > 4)  # the second  class on

    # shift from the second class
    y_true[to_shift_samples] = y_true[to_shift_samples] - 1
    # now arcella is the last
    y_true[arcella_samples] = len(labels)-1

    # change position of the class
    labels.remove("DILEPTUS\n")
    labels.append("DILEPTUS\n")


    return  make_confusion_matrix(y_true, y_pred, labels, title, path, wandb)


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


def make_predictor(name):

    if name == "gbt":

        return ensemble.GradientBoostingClassifier(validation_fraction=0.2, learning_rate=0.1, max_depth=4 )
    elif name == "mlp_sklearn":
        return MLPClassifier(hidden_layer_sizes=[256, 256], learning_rate="adaptive", learning_rate_init=0.0001, max_iter=1000000, tol=1e-7, alpha=0.0000001, shuffle=True, validation_fraction=0.2)

    else:
        raise "Name of predictor not exists!"


def _compute_loss(x_train, y_train, x_test, y_test, predictor):
    """Compute average accuracy for train and test set."""
    train_acc = 0.0
    test_acc = 0.0

    # concat train and val
    #x_train = np.concatenate((x_train, x_test), axis=0)
    #y_train = np.concatenate((y_train, y_test), axis=0)



    model = predictor
    model.fit(x_train, y_train)
    train_acc = np.mean(model.predict(x_train) == y_train)
    test_acc = np.mean(model.predict(x_test) == y_test) #model.best_validation_score_

    return train_acc, test_acc, model


def compute_downstream_task(representation, classes, predictor):

    scores = {}

    x_train, x_val, x_test = representation["train"], representation["val"], representation["test"]
    y_train, y_val, y_test = classes["train"].astype(np.int32), classes["val"].astype(np.int32), classes["test"].astype(np.int32)

    predictor_model = make_predictor(predictor)

    train_acc, val_acc, model = _compute_loss(x_train, y_train, x_val, y_val, predictor_model)
    test_acc = np.mean(model.predict(x_test) == y_test)

    scores["train"], scores["val"], scores["test"]  = train_acc, val_acc, test_acc
    return scores, model




class GBT_regressor(Metric):

    def __init__(self,  mode, **kwargs):

        super(GBT_regressor, self).__init__(**kwargs)
        self.mode = mode
        self.predictor = "gbt"
        self.name = "gbt_regressor"


    def get_score(self, feature_names ):
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

        scores, model = compute_downstream_task(rep, csv, predictor=self.predictor)


        with open(os.path.join(output_directory, 'accuracy.json'), 'w') as fp:
            json.dump(scores, fp)

        x_train, x_val, x_test = rep["train"], rep["val"], rep["test"]
        y_train, y_val, y_test = csv["train"].astype(np.int32), csv["val"].astype(np.int32), csv["test"].astype(np.int32)

        fig_train, train_balanced_acc = make_confusion_matrix(y_train, model.predict(x_train), train_labels_name, "Training set", os.path.join(output_directory, self.predictor + "_train_confusion.png"), self.wandb)
        fig_val, val_balanced_acc = make_confusion_matrix(y_val, model.predict(x_val), train_labels_name, "Validation set", os.path.join(output_directory, self.predictor +"_val_confusion.png"), self.wandb)
        fig_test, test_balanced_acc = make_confusion_matrix(y_test, model.predict(x_test), test_labels_name, "Test set", os.path.join(output_directory, self.predictor +"_test_confusion.png"), self.wandb)

        #fig_test, test_balanced_acc = make_confusion_matrix_for_test_dileptus(y_test, model.predict(x_test), test_labels_name, "Test set", os.path.join(output_directory, self.predictor +"_test_confusion.png"), self.wandb)


        with open(os.path.join(output_directory, 'balanced_accuracy.json'), 'w') as fp:
            json.dump({"train": train_balanced_acc, "val": val_balanced_acc, "test": test_balanced_acc}, fp)


        # save model
        with open(os.path.join(output_directory, self.predictor + 'model.pkl'), 'wb') as f:
            pickle.dump(model, f)


        # save csv with predictions

        if self.wandb:

            wandb.log({"confusion_matrix_matplotlib_" + self.predictor: [wandb.Image(fig_train, caption="Training set"),
                                                                         wandb.Image(fig_val, caption="Validation set"),
                                                                         wandb.Image(fig_test, caption="Test set")]})

            wandb.log({"confusion_matrix_test_" + self.predictor: wandb.plot.confusion_matrix(probs=None,
                                                                                                  y_true=y_test,
                                                                                                  preds=model.predict(
                                                                                                      x_test),
                                                                                                  class_names=test_labels_name)})


            # load model to wandb
            artifact = wandb.Artifact(self.predictor, type='model',
                                  description='Sklearn model for plankton classification',
                                  metadata={ "accuracy": scores})

            # Add the model file to the artifact
            artifact.add_file(os.path.join(output_directory, self.predictor + 'model.pkl'))

            wandb.log({k+"_accuracy_" + self.predictor :v for k, v in scores.items()})

        # clear the matplotlib figure to free memory
        plt.close(fig_train)
        plt.close(fig_val)
        plt.close(fig_test)

        if self.predictor == "gbt":
            # only Decision Trees have model.feature_importances_
            wandb.sklearn.plot_feature_importances(model, feature_names = feature_names )

            # Get feature importance
            feature_importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
            importance_df.sort_values(by='Importance', ascending=False, inplace=True)

            # Save feature importance to a file
            importance_df.to_csv(os.path.join(output_directory, 'feature_importance.csv'), index=False)


        return scores



class MLP_regressor(GBT_regressor):

    def __init__(self, **kwargs):

        super(MLP_regressor, self).__init__(**kwargs)

        self.predictor = "mlp_sklearn"
        self.name = "mlp_sklearn_regressor"





