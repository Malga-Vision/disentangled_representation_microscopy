

from torch.nn.functional import mse_loss


def mse(x, x_hat):

    return mse_loss(x_hat, x, reduction="none").sum(dim=(1, 2, 3)).mean()


def mse_features(x, x_hat):

    return (mse_loss(x_hat, x, reduction="none").sum(dim=1) * 768.).mean()
