
from code.models.autoencoder import AE
from code.models.vae import VAE
from code.models.backbone_vae import BACKBONEVAE
from code.models.ada_gvae import AdaGVAE


def get_named_model(name):

    if name == "ae":
        return AE

    if name == "vae":
        return VAE

    if name == "adagvae":
        return AdaGVAE


    if name== "backbone_vae":
        return BACKBONEVAE






    raise ValueError("Method does not exist")