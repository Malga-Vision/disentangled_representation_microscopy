

from code.dataset.plakton_padded import PlanktonBinaryPadded64, PlanktonMaskedPadded64, PlanktonMaskedPadded224, LenslessMaskPadded224, LenslessMaskPadded64, LenslessImageMaskPadded224, PlanktonMaskedPadded224_TestArcella

from code.dataset.whoi15_padded import WHOI15Padded64, WHOI15Padded224
from code.dataset.whoi40_padded import WHOI40Padded224
from code.dataset.representation_dataset import LENSLESSDataset, WHOI152007Dataset, WHOI40Dataset, LENSLESSDataset_TestArcella


from code.dataset.vacuoles import VACUOLES
from code.dataset.representation_dataset import VACUOLESDataset

from code.dataset.sipakmed import SIPAKMED
from code.dataset.representation_dataset import SIPAKMEDDataset


from code.dataset.texture_dsprites import TextureDSprites
from code.dataset.texture_dsprites_dino_features import dSpritesDino

def get_named_dataset(name):

    if name == "texture-dsprites-dino":
        return dSpritesDino

    if name == "texture-dsprites":
        return TextureDSprites

    if name == "sipakmed":
        return SIPAKMED

    if name == "sipakmed_representation":
        return SIPAKMEDDataset

    if name == "vacuoles":
        return VACUOLES


    if name == "vacuoles_representation":
        return VACUOLESDataset



    if name == "whoi15_2007_padded_224":
        return WHOI15Padded224

    if name == "whoi40_padded_224":
        return WHOI40Padded224



    if name == "whoi15_2007_padded_64":
        return WHOI15Padded64



    if name == "plankton_binary_padded_64":
        return PlanktonBinaryPadded64

    if name == "plankton_masked_padded_224":
        return PlanktonMaskedPadded224


    if name == "plankton_masked_padded_224_test_arcella":
        return PlanktonMaskedPadded224_TestArcella


    if name == "plankton_mask_padded_224":
        return LenslessMaskPadded224

    if name == "plankton_image_mask_padded_224":
        return LenslessImageMaskPadded224


    if name == "plankton_mask_padded_64":
        return LenslessMaskPadded64

    if name == "plankton_masked_padded_64":
        return PlanktonMaskedPadded64


    if name == "lensless_representation":
        return LENSLESSDataset

    if name == "lensless_representation_test_arcella":
        return LENSLESSDataset_TestArcella


    if name == "whoi15_2007_representation":
        return WHOI152007Dataset

    if name == "whoi40_representation":
        return WHOI40Dataset



    raise ValueError("Dataset does not exist")
