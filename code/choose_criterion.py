
from code.criterion.mse import mse, mse_features
from code.criterion.bce import bce, lower_bce, bce_features
from code.criterion.kl import normal_kl_divergence



def get_named_criterion(name):

    if name == "mse":
        return mse

    if name == "mse_features":
        return mse_features



    if name == "bce":
        return bce

    if name == "bce_features":
        return bce_features

    if name == "lower_bce":
        return lower_bce

    if name == "normal_kullback":
        return normal_kl_divergence

    raise "Criterion does not exist"