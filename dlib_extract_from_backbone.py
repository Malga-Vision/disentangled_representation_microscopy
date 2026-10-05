
from absl import app
from absl import flags

import os
import yaml
import shutil


from code.postprocessing.extract_representation_from_backbone import postprocess_model

FLAGS = flags.FLAGS
flags.DEFINE_string("output_directory", None, "Output directory.")
flags.DEFINE_string("backbone", "vit-b-1k-dino", "Name of the backbone.")

flags.DEFINE_string("dataset", "plankton_masked_padded_224", "Name of the dataset.")

flags.DEFINE_integer("random_seed", 42, "Random seed.")



def main(unsued_args):

    config = {"postprocess_dataset": FLAGS.dataset,
                "perc_val_set" : 0.2,
                "backbone" : FLAGS.backbone,
                "split_random_seed" :  FLAGS.random_seed,
                "random_seed":42,
                "batch_size":64,
                "val_batch_size":64,
                "multithread":True,
                "grad_acc_steps":1}

    output_directory = FLAGS.output_directory

    if output_directory is None:
        output_directory = os.path.join("output", "representations", FLAGS.backbone, FLAGS.dataset)


    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        os.makedirs(output_directory)


    # extract representation
    postprocess_model(output_directory, config)




if __name__ == "__main__":
    app.run(main)
