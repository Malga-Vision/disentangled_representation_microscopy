

from code.dataset.plakton_padded import PLANKTON_PADDED224_PATH

from code.postprocessing.extract_simple_features import postprocess_model

def main(args):
    postprocess_model(PLANKTON_PADDED224_PATH, args)


if __name__ == "__main__":

    args = {"batch_size":1, "val_batch_size":1, "perc_val_set": 0.2, "split_random_seed": 42, "random_seed":42, "postprocess_dataset": "plankton_image_mask_padded_224"}
    main(args)
