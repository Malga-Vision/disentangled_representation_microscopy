



import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights,  resnet152,  ResNet152_Weights

import timm
from timm.models.vision_transformer import VisionTransformer


def classifier_to_backbone(model, layer = -1):

    return  nn.Sequential(*nn.ModuleList(model.children())[:layer])


def load_pretrained_vitl16(model_key):


    class VisionTransformerFex(torch.nn.Module):
        def __init__(self, base_model: VisionTransformer) -> None:
            super().__init__()
            self.base_model = base_model
            self.base_model.eval()
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.base_model.forward_features(x)
            return x[:, 0]


    #model_key = "vit_large_patch16_224.augreg_in21k"
    #model_key = "vit_base_patch16_224.dino"
    base_model: torch.nn.Module = timm.create_model(
        model_name=model_key,
        pretrained=True
    )

    feature_extractor = VisionTransformerFex(base_model)
    for param in feature_extractor.parameters():
        param.requires_grad = False
    feature_extractor.eval()
    #feature_extractor.to(device)
    return feature_extractor



def get_named_backbone(name):

    if name == "resnet50":
        bk = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        bk = classifier_to_backbone(bk)
        return bk



    if name == "resnet152":
        bk = resnet152(weights=ResNet152_Weights.IMAGENET1K_V1)
        bk = classifier_to_backbone(bk)

        return bk

    if name == "vit-l-21k":
        return  load_pretrained_vitl16(model_key="vit_large_patch16_224.augreg_in21k")

    if name == "vit-b-21k":
        return load_pretrained_vitl16(model_key="vit_base_patch16_224.augreg_in21k")

    if name == "vit-b-1k-dino":
        return  load_pretrained_vitl16(model_key="vit_base_patch16_224.dino")


    raise ValueError("Backbone name not supported!")


def get_named_backbone_dim(name):

    if name == "resnet50":

        return 2048



    if name == "resnet152":
        return 2048

    if name == "vit-l-21k":
        return  1024

    if name == "vit-b-21k":
        return  768

    if name == "vit-b-1k-dino":
        return  768


    raise ValueError("Backbone name not supported!")








