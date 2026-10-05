
import torch
from torch import nn

from code.models.autoencoder import AE
from code.models.utils import  _conv, _deconv, _linear, View, NormalizeTanh



def z_sample(mu, log_var):
    # sample z from q
    std = torch.exp(0.5 * log_var)
    q = torch.distributions.Normal(mu, std)
    z = q.rsample()
    return z


"""

def z_sample(mu, log_var):
    # sample z from q
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std, requires_grad = False)
    z = mu + std * eps
    return z
"""


class VAE(AE):
    def __init__(self, data_shape, latent_dim=64, n_filters = 16, dim_to_freeze=None, **kwargs):
        super(VAE, self).__init__(data_shape, latent_dim, n_filters, dim_to_freeze=dim_to_freeze)

        n_channel = data_shape[-1]

        # projection
        self.project = _linear(latent_dim, self.feature_volume, relu=True)

        # encoded feature's size and volume
        self.feature_size = data_shape[0] // 8
        self.feature_volume = n_filters * (self.feature_size ** 2)

        self.encoder = nn.Sequential(
            _conv(n_channel, n_filters // 4, batch_norm=True),
            _conv(n_filters // 4, n_filters // 2, batch_norm=True),
            _conv(n_filters // 2, n_filters, batch_norm=True),
            View([self.feature_volume]),
            nn.LayerNorm(self.feature_volume)
        )

        self.decoder = nn.Sequential(
            self.project,
            View((n_filters, self.feature_size, self.feature_size)),
            _deconv(n_filters, n_filters // 2, batch_norm=False),
            _deconv(n_filters // 2, n_filters // 4, batch_norm=False),
            _deconv(n_filters // 4, n_channel, batch_norm=False),
            NormalizeTanh()
        )

        # init weights
        self.encoder.apply(self.init_weights)
        self.decoder.apply(self.init_weights)


        # distribution parameters
        self.fc_mu = _linear(self.feature_volume, latent_dim, relu=False)
        self.fc_var = _linear(self.feature_volume, latent_dim, relu=False)



    def encode(self, x):
        ''' Return representation given a sample as only a point in the latent space'''
        x = self.encoder(x)

        mu, log_var = self.fc_mu(x), self.fc_var(x)


        # clamp log var to avoid NaN
        log_var = torch.clamp(log_var, min=-20, max=10)


        z = z_sample(mu, log_var)

        # mask latent code
        z = z * self.latent_mask

        return {"mu" : mu, "log_var":log_var, "std": torch.exp(0.5 * log_var ), "sampled":z}


    def decode(self, code):
        c = self.decoder(code)
        return c


    def forward(self, x):
        y = self.encoder(x)

        mu, log_var = self.fc_mu(y), self.fc_var(y)


        z = z_sample(mu, log_var)

        # mask latent code
        z = z * self.latent_mask

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





