
"""  Test wandb integration  """
from absl import app
from absl import flags

import os
import yaml
import wandb
import shutil

from code.training.finetune_vae import train_model
from code.postprocessing.postprocess import postprocess_model
from code.downstream_task.downstream_task import GBT_regressor, MLP_regressor


FLAGS = flags.FLAGS


flags.DEFINE_string("experiment", "BACKBONE_to_lenseless_test_dileptus", "Name of the experiment to run")
flags.DEFINE_string("config", "vit_dino-brodatz_texture_dsprites_to_maskedplankton", "Name of the config to run")

flags.DEFINE_integer("n_sweep", 20, "Number of sweeps to run")


flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")


def copytree(src, dst, symlinks=False, ignore=None):

    if not os.path.exists(dst):
        # if the demo_folder directory is not present then create it.
        os.makedirs(dst)

    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def get_model_num(config):
    return "beta-" + str(config["beta"]) + "+" +  "random_seed-" + str(config["random_seed"])


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


def sweep(unused_args):

    # Initialize sweep by passing in configs.
    # (Optional) Provide a name of the project.
    sweep_id = wandb.sweep(sweep=read_config(), project="plankton_finetuning")

    # Start sweep job.
    wandb.agent(sweep_id, function=main, count=FLAGS.n_sweep)


def main():

    run = wandb.init(project="plankton_finetuning",
                     notes="transfer learning from dsprites to plankton",
                     tags=["DRL: dsprites --> plankton"]
                     )


    assert wandb.run is not None


    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{experiment}", "{model_num}")
    else:
        output_directory = FLAGS.output_directory

    model_num = get_model_num(run.config)
    run.name = model_num
    run.config.update({"lq": False}, allow_val_change=True)


    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=str(run.name),
                                               experiment=str(FLAGS.experiment))


    input_directory = get_input_directory(run.config["input_experiment"], model_num)

    print("Transfer from ", input_directory)


    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        os.makedirs(output_directory)
    else:

        raise FileExistsError("Experiment folder exists")


    # train model
    print(f"SWEEP: {run.id}")
    print(run.config)

    with open(os.path.join(output_directory, 'config.yaml'), 'w') as f:
        _ = yaml.dump(run.config, f)

    # BEFORE-TRANSFER
    pre_output_directory = os.path.join(output_directory, "before")

    # copy trained model into new experiment
    copytree(os.path.join(input_directory, "model"), os.path.join(pre_output_directory, "model"))

    postprocess_model(pre_output_directory, run.config)
    run.config["path_to_representation"] = os.path.join(pre_output_directory, "postprocess")

    # run plankton classification
    gbt_classifier = GBT_regressor(mode=None, path=run.config["path_to_representation"], wandb=False)
    mlp_regressor = MLP_regressor(mode=None, path=run.config["path_to_representation"], wandb=False)


    scores = gbt_classifier.get_score(feature_names=run.config["dim_to_factor"])
    scores = mlp_regressor.get_score(feature_names=run.config["dim_to_factor"])



    # AFTER-TRANSFER
    post_output_directory = os.path.join(output_directory, "after")

    # make experiment directory
    if not os.path.exists(post_output_directory):
        # if the demo_folder directory is not present then create it.
        os.makedirs(post_output_directory)
    else:

        raise FileExistsError("Experiment folder exists")

    with open(os.path.join(post_output_directory, 'config.yaml'), 'w') as f:
        _ = yaml.dump(run.config, f)

    copytree(os.path.join(input_directory, "model"), os.path.join(post_output_directory, "model"))

    train_model(post_output_directory, run.config)
    postprocess_model(post_output_directory, run.config)
    run.config.update({"path_to_representation":os.path.join(post_output_directory, "postprocess")}, allow_val_change=True)

    # run plankton classification
    gbt_classifier = GBT_regressor(mode=None, path=run.config["path_to_representation"], wandb=False)
    mlp_regressor = MLP_regressor(mode=None, path=run.config["path_to_representation"], wandb=False)


    scores = gbt_classifier.get_score(feature_names=run.config["dim_to_factor"])
    scores = mlp_regressor.get_score(feature_names=run.config["dim_to_factor"])

    run.finish()


if __name__ == "__main__":
    app.run(sweep)
