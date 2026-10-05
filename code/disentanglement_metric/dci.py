"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""

import numpy as np
import pandas as pd
import scipy
from sklearn import ensemble
#import xgboost as xgb

from code.disentanglement_metric.metric import Metric


def split_train_test(observations, observations2=None, train_percentage=0.66):
  """Splits observations into a train and test_unsupervised set.

  Args:
    observations: Observations to split in train and test_unsupervised. They can be the
      representation or the observed factors of variation. The shape is
      (num_dimensions, num_points) and the split is over the points.
    train_percentage: Fraction of observations to be used for training.

  Returns:
    observations_train: Observations to be used for training.
    observations_test: Observations to be used for testing.
  """
  num_labelled_samples = observations.shape[1]
  num_labelled_samples_train = int(
      np.ceil(num_labelled_samples * train_percentage))
  num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
  observations_train = observations[:, :num_labelled_samples_train]
  observations_test = observations[:, num_labelled_samples_train:]


  if observations2 is not None:
      observations2_train = observations2[:, :num_labelled_samples_train]
      observations2_test = observations2[:, num_labelled_samples_train:]
      return observations_train, observations_test, observations2_train, observations2_test


  assert observations_test.shape[1] == num_labelled_samples_test, \
      "Wrong size of the test_unsupervised set."
  return observations_train, observations_test




def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation."""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code."""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1])


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation."""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees."""
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors],
                                 dtype=np.float64)
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = ensemble.GradientBoostingClassifier( n_iter_no_change=10, warm_start=True) #xgb.XGBClassifier(use_label_encoder=False, tree_method='gpu_hist')  # Use GPU
        model.fit(x_train.T, y_train[i, :])
        importance_matrix[:, i] = np.abs(model.feature_importances_)
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    return importance_matrix, np.mean(train_loss), np.mean(test_loss)


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = train_err
    scores["informativeness_test"] = test_err
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)
    return scores


def compute_dci(representation, factors, perc_train, perc_test, batch_size=16):
    # mus_train are of shape [ num_codes, num_train], while ys_train are of shape [num_factors, num_train  ].

    representation = representation.T
    factors = factors.T

    mus_train, mus_test, ys_train, ys_test = split_train_test(representation, observations2=factors,
                                                              train_percentage=perc_train)

    scores = _compute_dci(mus_train, ys_train, mus_test, ys_test)
    return scores


class DCI_disentanglement(Metric):

    def __init__(self, mode, **kwargs):
        super(DCI_disentanglement, self).__init__(**kwargs)
        self.mode = mode

    def get_score(self):
        ''' Return the score '''

        # load representation
        rep = np.load(self.representation_path + ".npz")
        data = rep[self.mode]

        csv = pd.read_csv(self.classes_path + '.csv')
        classes = csv.values
        scores = {}
        scores["dci-disentanglement"] = compute_dci(data, classes, perc_train=2 / 3, perc_test=1 / 3)["disentanglement"]
        return scores
