

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import torch
import numpy as np
from torch import nn
import torch.nn.functional as F




from code.models.vae import VAE



# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.cuda.set_device(device)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class View(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def __repr__(self):
        return f'View{self.shape}'

    def forward(self, x):
        batch_size = x.size(0)
        shape = (batch_size, *self.shape)
        out = x.view(shape)
        return out


class NormalizeTanh(nn.Module):
    r"""Applies the Hyperbolic Tangent (Tanh) function element-wise.

    Tanh is defined as:

    .. math::
        \text{Tanh}(x) = \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/Tanh.png

    Examples::

        >>> m = nn.Tanh()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return (torch.tanh(input) + 1)/2


def z_sample(mu, log_var):
    # sample z from q
    std = torch.exp(0.5 * log_var)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return z


class BACKBONEVAE(VAE):

    def __init__(self, data_shape=768, latent_dim=10, n_filters = 64 ,**kwargs):
        super(BACKBONEVAE, self).__init__(data_shape, latent_dim, n_filters, **kwargs)

        # encoded feature's size and volume
        self.feature_size = n_filters

        input_dim = data_shape[0]

        self.encoder = nn.Sequential(
            self._linear(input_dim, self.feature_size, relu=True),
            self._linear(self.feature_size, self.feature_size * 2, relu=True),
            self._linear(self.feature_size * 2, self.feature_size * 3, relu=True),
        )

        self.decoder = nn.Sequential(
            self._linear(latent_dim, self.feature_size * 3, relu=True),
            self._linear(self.feature_size * 3, self.feature_size * 2, relu=True),
            self._linear(self.feature_size * 2, self.feature_size, relu=True),
            self._linear(self.feature_size, input_dim, relu=True),
            NormalizeTanh()
        )

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)

        # distribution parameters
        self.fc_mu = self._linear(self.feature_size * 3, latent_dim, relu=False)
        self.fc_var = self._linear(self.feature_size * 3, latent_dim, relu=False)

        # for the gaussian likelihood
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))
        self.one = torch.Tensor([1.])


    def _linear(self, in_size, out_size, relu=True):
        return nn.Sequential(
            nn.Linear(in_size, out_size),
            nn.LeakyReLU(0.2),
        ) if relu else nn.Linear(in_size, out_size)



    def encode(self, x):
        ''' Return representation given a sample as only a point in the latent space'''
        x = self.encoder(x)

        mu, log_var = self.fc_mu(x), self.fc_var(x)


        # clamp log var to avoid NaN
        log_var = torch.clamp(log_var, min=-20, max=10)


        z = z_sample(mu, log_var)


        return {"mu" : mu, "log_var":log_var, "std": torch.exp(0.5 * log_var ), "sampled":z}


    def decode(self, code):
        c = self.decoder(code)
        return c


    def forward(self, x):
        y = self.encoder(x)

        mu, log_var = self.fc_mu(y), self.fc_var(y)


        z = z_sample(mu, log_var)


        x = self.decoder(z)
        return x


    def load_state(self, path):
        ''' Load model state, including criterion and optimiizer '''

        # load the model checkpoint
        checkpoint = torch.load(path)

        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.decoder.load_state_dict(checkpoint["decoder_state_dict"])
        self.fc_mu.load_state_dict(checkpoint["fc_mu_state_dict"])
        self.fc_var.load_state_dict(checkpoint["fc_var_state_dict"])



    def save_state(self, path):
        ''' Save model state'''
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'decoder_state_dict': self.decoder.state_dict(),
            'fc_mu_state_dict': self.fc_mu.state_dict(),
            'fc_var_state_dict': self.fc_var.state_dict(),
        }, path)




