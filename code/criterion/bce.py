

import torch


def bce(x, x_hat):

    x_hat_clamp = torch.clamp(x_hat, min=1e-6, max=1 - 1e-6)

    loss = -(x * torch.log(x_hat_clamp) + (1 - x) * torch.log(1 - x_hat_clamp)).sum(dim=(1, 2, 3))
    return loss.mean()


def bce_features(x, x_hat):

    x_hat_clamp = torch.clamp(x_hat, min=1e-6, max=1 - 1e-6)

    loss = -(x * torch.log(x_hat_clamp) + (1 - x) * torch.log(1 - x_hat_clamp)).sum(dim=1)
    return loss.mean() #* 768.


def lower_bce(x, x_hat):
    # Because true images are not binary, the lower bound in the x_hat is not zero:
    # the lower bound in the x_hat is the entropy of the true images.

    x_clamp = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    dist = torch.distributions.bernoulli.Bernoulli(probs=x_clamp)
    loss_lower_bound = torch.sum(dist.entropy(), dim=(1, 2, 3))


    x_hat_clamp = torch.clamp(x_hat, min=1e-6, max=1 - 1e-6)
    loss = -(x * torch.log(x_hat_clamp) + (1 - x) * torch.log(1 - x_hat_clamp)).sum(dim=(1, 2, 3))

    loss = (loss - loss_lower_bound)
    return loss.mean()