
import torch
from torch import nn

from code.models.vae import VAE




def z_sample(mu, log_var):
    # sample z from q
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std, requires_grad = True)
    z = mu + std * eps
    return z


def histogram_fixed_width_bins( values, min, max, nbins):
    """
    Given the tensor values, this operation returns a rank 1 Tensor representing the indices of a histogram into which each element of values would be binned. The bins are equal width and determined by the arguments value_range and nbins.
    """

    value_range = [min, max]

    # Calculate the width of each bin
    bin_width = (value_range[1] - value_range[0]) / nbins

    # Create the bin edges
    #bin_edges = torch.linspace(value_range[0], value_range[1], nbins + 1)

    # Compute the indices of bin placement for each value
    indices = ((values - value_range[0]) / bin_width).floor().clamp(0, nbins - 1).long()
    #print(idx.size())
    return indices


def discretize_in_bins( x):
    """Discretize a vector in two bins."""
    return histogram_fixed_width_bins(x, torch.min(x).item(), torch.max(x).item(), nbins=2)



def gaussian_kl_divergence(mu1, sigma1, mu2, sigma2, eps = 1e-6):
    # Ensure non-zero standard deviations to avoid division by zero

    sigma1 += eps
    sigma2 += eps

    # Compute KL Divergence
    kl_divergence = torch.log(sigma2 / sigma1) + 0.5 * ( ((sigma1**2.) + (mu1 - mu2)**2.) / (sigma2**2.) - 1. )

    return kl_divergence


def aggregate_argmax(z_mean, z_logvar, new_mean, new_log_var, kl_per_point):
    """Argmax aggregation with adaptive k.

             The bottom k dimensions in terms of distance are not averaged. K is
             estimated adaptively by binning the distance into two bins of equal width.

             Args:
               z_mean: Mean of the encoder distribution for the original image.
               z_logvar: Logvar of the encoder distribution for the original image.
               new_mean: Average mean of the encoder distribution of the pair of images.
               new_log_var: Average logvar of the encoder distribution of the pair of
                 images.
               labels: One-hot-encoding with the position of the dimension that should not
                 be shared.
               kl_per_point: Distance between the two encoder distributions.

             Returns:
               Mean and logvariance for the new observation.
             """



    one = torch.ones_like(kl_per_point)
    mask = discretize_in_bins(kl_per_point).eq(one)


    z_mean_averaged = torch.where(mask, z_mean, new_mean)
    z_logvar_averaged = torch.where(mask, z_logvar, new_log_var)

    return z_mean_averaged, z_logvar_averaged



class AdaGVAE(VAE):
    def __init__(self, data_shape, latent_dim=64, n_filters = 16):
        super(AdaGVAE, self ).__init__(data_shape, latent_dim, n_filters)

    def encode_couple(self, x1, x2):
        ''' Return representation given a sample as only a point in the latent space'''
        # encode couple
        y1 = self.encoder(x1)
        mu1, log_var1 = self.fc_mu(y1), self.fc_var(y1)

        y2 = self.encoder(x2)
        mu2, log_var2 = self.fc_mu(y2), self.fc_var(y2)

        # averaging parameters
        kl_per_point = 0.5 * gaussian_kl_divergence(mu1, mu2, log_var1, log_var2) + 0.5 * gaussian_kl_divergence(mu2, mu1, log_var2,
                                                                                                   log_var1)

        new_mean = 0.5 * mu1 + 0.5 * mu2
        var_1 = log_var1.exp()
        var_2 = log_var2.exp()
        new_log_var = torch.log(0.5 * var_1 + 0.5 * var_2)

        mu1, log_var1 = aggregate_argmax(mu1, log_var1, new_mean, new_log_var, kl_per_point)
        mu2, log_var2 = aggregate_argmax(mu2, log_var2, new_mean, new_log_var, kl_per_point)

        # sample points
        z1 = z_sample(mu1, log_var1)
        z2 = z_sample(mu2, log_var2)

        return {"mu1" : mu1, "mu2" : mu2, "log_var1":log_var1, "log_var2":log_var2,
                "std1": torch.exp(0.5 * log_var1 ), "std2": torch.exp(0.5 * log_var2 ),
                "sampled1":z1, "sampled2":z2}

    def decode_couple(self, code1, code2):
        c1 = self.decoder(code1)
        c2 = self.decoder(code2)
        return c1, c2

    def forward_couple(self, x1, x2):

        representation = self.encode_couple(x1, x2)

        # reconstruct
        x1_, x2_ = self.decode_couple(representation["sampled1"], representation["sampled2"])

        return x1_, x2_





