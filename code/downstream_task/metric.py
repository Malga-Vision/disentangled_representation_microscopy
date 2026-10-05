from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Metric(object):
    """Abstract class for metric."""

    def __init__(self, path, wandb=True, **kwargs):

        super(Metric, self).__init__()

        self.path = path
        self.wandb = wandb


    def get_score(self):
        ''' Return the score '''
        raise NotImplementedError()