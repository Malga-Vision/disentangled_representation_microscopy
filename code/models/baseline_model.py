from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
import numpy as np



class Baseline(nn.Module):

    def num_params(self):
        num_params = sum([np.prod(p.shape) for p in self.parameters()])
        return num_params

    def num_trainable_params(self):
        num_params = sum([np.prod(p.shape) for p in self.parameters() if p.requires_grad] )
        return num_params

    def freeze_module(self, module_name):

        module = getattr(self, module_name)
        for param in module.parameters():
            param.requires_grad = False

    def encode(self, x):
        ''' Return representation given a sample '''
        raise NotImplementedError()

    def sample(self, num=25):
        ''' Random sample of num images from distribution of the representation '''
        raise NotImplementedError()

    def decode(self, code):
        ''' Random sample of num images from distribution of the representation '''
        raise NotImplementedError()


    def forward(self, x):
        return super(Baseline, self).forward(x)


    def init_weights(self, m):

        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

