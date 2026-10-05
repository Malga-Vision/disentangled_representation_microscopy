

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import overload, Optional, Union

import torch
import numpy as np
from torch import nn, device, dtype, Tensor
import torch.nn.functional as F
from torch.nn.modules.module import T

from code.models.baseline_model import Baseline
from code.models.utils import  _conv, _deconv, _linear, View, NormalizeTanh





class AE(Baseline):

    def __init__(self, data_shape, latent_dim, n_filters, dim_to_freeze=None, **kwargs):
        super(AE, self).__init__()

        n_channel = data_shape[-1]

        # encoded feature's size and volume
        self.feature_size = data_shape[0] // 8
        self.feature_volume = n_filters * (self.feature_size ** 2)

        self.encoder = nn.Sequential(
            _conv(n_channel, n_filters // 4),
            _conv(n_filters // 4, n_filters // 2),
            _conv(n_filters // 2, n_filters),
            View([self.feature_volume]),
            nn.LayerNorm(self.feature_volume)
        )

        self.decoder = nn.Sequential(

            View((n_filters, self.feature_size, self.feature_size)),
            _deconv(n_filters, n_filters // 2),
            _deconv(n_filters // 2, n_filters // 4),
            _deconv(n_filters // 4, n_channel),
            NormalizeTanh()
        )

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)


        # unused param
        self.log_scale = nn.Parameter(torch.Tensor([0.0]))

        # save latent dimension
        self.latent_dim = latent_dim

        # freeze latetn codes
        self.dim_to_freeze = dim_to_freeze
        self.latent_mask = torch.ones(latent_dim) # 1 for dimensions to be used, 0 for dimensions to freeze

        if dim_to_freeze is not None:
            self.latent_mask[dim_to_freeze] = 0.

        # define as Parameters but not required grad
        self.latent_mask = torch.nn.Parameter(self.latent_mask, requires_grad=False)


    def forward(self, x):
        x = self.encoder.forward(x)

        # mask latent code
        x = x * self.latent_mask

        x = self.decoder.forward(x)
        return x

    def encode(self, x):
        ''' Return representation given a sample as only a point in the latent space'''
        c = self.encoder.forward(x)

        return {"mu" : c * self.latent_mask }

    def sample(self, num=25):
        # mean and variance of latent code (better to estimate them with a test set)
        mean = 0.
        std = 1.

        # sample latent vectors from the normal distribution
        latents = torch.randn(num, self.latent_dim) * std + mean
        imgs = self.decoder.forward(latents * self.latent_mask)
        return imgs

    def decode(self, code):
        c = self.decoder.forward(code)
        return c
