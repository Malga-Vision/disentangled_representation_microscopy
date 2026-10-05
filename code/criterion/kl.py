
import torch


def normal_kl_divergence(mu, log_var):

    """
    KL Divergence with a Normal distribution
    """

    return (-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1)).mean()




