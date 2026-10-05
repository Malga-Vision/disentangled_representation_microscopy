
"""  Test wandb integration  """
from absl import app
from absl import flags

import os
import yaml
import wandb
import shutil


from code.downstream_task.blob_distance import Blob_distance



FLAGS = flags.FLAGS


flags.DEFINE_string("experiment", "BACKBONE_to_lensless_test_arcella", "Name of the experiment to run")
flags.DEFINE_string("config", "vit_dino-brodatz_texture_dsprites_to_maskedplankton_testarcella", "Name of the config to run")

flags.DEFINE_integer("n_sweep", 20, "Number of sweeps to run")


flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")




def get_model_num(config):
    return "beta-" + str(config["beta"]) + "+" +  "random_seed-" + str(config["random_seed"])




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



    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.

        raise FileExistsError("Experiment folder doean not exist")


    # train model
    print(f"SWEEP: {run.id}")
    print(run.config)

    with open(os.path.join(output_directory, 'config.yaml'), 'w') as f:
        _ = yaml.dump(run.config, f)

    # BEFORE-TRANSFER
    pre_output_directory = os.path.join(output_directory, "before")
    run.config["path_to_representation"] = os.path.join(pre_output_directory, "postprocess")

    # make experiment directory
    if not os.path.exists(pre_output_directory):
        raise FileExistsError("Experiment folder does exist")




    blob = Blob_distance(mode=None, path=run.config["path_to_representation"], wandb=False)
    scores = blob.get_score(class_to_evaluate=run.config["class_to_evaluate"], class_name=run.config["class_name"],
                            class_to_compare=run.config["class_to_compare"],
                            class_to_compare_name=run.config["class_to_compare_name"],
                            feature_names=run.config["dim_to_factor"])

    # AFTER-TRANSFER
    post_output_directory = os.path.join(output_directory, "after")
    run.config.update({"path_to_representation": os.path.join(post_output_directory, "postprocess")},
                      allow_val_change=True)

    # make experiment directory
    if not os.path.exists(post_output_directory):
        raise FileExistsError("Experiment folder does exist")




    blob = Blob_distance(mode=None, path=run.config["path_to_representation"], wandb=False)
    scores = blob.get_score(class_to_evaluate=run.config["class_to_evaluate"], class_name=run.config["class_name"], class_to_compare=run.config["class_to_compare"], class_to_compare_name = run.config["class_to_compare_name"], feature_names=run.config["dim_to_factor"])

    run.finish()


if __name__ == "__main__":
    app.run(sweep)
