
from absl import app
from absl import flags

import os
import yaml
import shutil

from code.visualization.visualize_latent_space_with_simple_features import visualize_latent_space

FLAGS = flags.FLAGS
flags.DEFINE_string("experiment", "BACKBONE_to_maskedplankton", "Name of the experiment to run")
flags.DEFINE_string("config", "vit_dino-brodatz_texture_dsprites_to_maskedplankton", "Name of the config to run")
flags.DEFINE_string("model_num", "beta-1+random_seed-0", "Model number to evaluate")
flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")
flags.DEFINE_boolean("quantized", False, "Whether are using a quatized representation")


def get_input_directory(input_experiment, model_num):
    # Set correct output directory. and check they already exist
    input_directory = os.path.join("input", "{experiment}", "{model_num}")

    # Insert model number and study name into path if necessary.
    input_directory = input_directory.format(model_num=str(model_num),
                                               experiment=str(input_experiment))

    if not os.path.exists(input_directory):
        raise FileExistsError("Input experiment folder do not exists")

    return input_directory


def read_config():
    # Read YAML file
    with open(f"configs/{FLAGS.config}/config.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    return data_loaded




def main(unsued_args):


    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{experiment}", "{model_num}")
    else:
        output_directory = FLAGS.output_directory

    model_num = FLAGS.model_num

    config = {"split_random_seed" :  42,
              "postprocess_dims": [0, 1, 2, 3, 4,],
                "dim_to_factor":["Texture", "Color", "Shape", "Scale","Orientation"],
              "lq": FLAGS.quantized}


    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=model_num,
                                               experiment=str(FLAGS.experiment))



    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        raise FileExistsError("Experiment folder does not exists")

    postprocess_dir ="postprocess"

    if FLAGS.quantized:
        postprocess_dir += "_quantized"

    print("Start  BEFORE\n")
    # BEFORE-TRANSFER
    pre_output_directory = os.path.join(output_directory, "before")

    path_to_representation = os.path.join(pre_output_directory, postprocess_dir)
    config["path_to_representation"] = path_to_representation

    # lower-dimensional visualization
    visualize_latent_space(pre_output_directory, config)

    print("Start  AFTER\n")
    # AFTER-TRANSFER
    post_output_directory = os.path.join(output_directory, "after")

    path_to_representation = os.path.join(post_output_directory, postprocess_dir)
    config["path_to_representation"] = path_to_representation

    # lower-dimensional visualization
    visualize_latent_space(post_output_directory, config)




if __name__ == "__main__":
    app.run(main)