from torch import nn
import torch




def freeze_all_params(module, freeze=True):
    freeze = not freeze
    for param in module.parameters():
        param.requires_grad = freeze

    return module

def num_params(model):
    total_params = sum(
        param.numel() for param in model.parameters()
    )

    return total_params

def num_trainable_params(model):
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    return trainable_params




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


def _conv( channel_size, kernel_num, batch_norm=False):

    layers_list = [nn.Conv2d(
            channel_size, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )]

    if batch_norm:
        layers_list.append(nn.BatchNorm2d(kernel_num))

    layers_list.append(nn.LeakyReLU(0.02)) # nn.LeakyReLU(0.02)

    return nn.Sequential(*layers_list)


def _deconv( channel_num, kernel_num, batch_norm=False):

    layers_list =[nn.ConvTranspose2d(
            channel_num, kernel_num,
            kernel_size=4, stride=2, padding=1,
        )]

    if batch_norm:
        layers_list.append(nn.BatchNorm2d(kernel_num))

    layers_list.append(nn.LeakyReLU(0.02))  # nn.LeakyReLU(0.02) nn.PReLU(num_parameters=kernel_num)
    return nn.Sequential(*layers_list)


def _linear( in_size, out_size, relu=True):
    return nn.Sequential(
        nn.Linear(in_size, out_size),
        nn.LeakyReLU(0.02),
    ) if relu else nn.Linear(in_size, out_size)