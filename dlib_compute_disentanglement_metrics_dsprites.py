

from absl import app
from absl import flags

import os
import yaml
import wandb
import shutil

from code.training.finetune_vae import train_model
from code.disentanglement_metric.compute_metrics_dsprites import evaluate_model_with_dsprites

FLAGS = flags.FLAGS


flags.DEFINE_string("experiment", "BRODATZTEXTURE_to_maskedplankton_224", "Name of the experiment to run")
flags.DEFINE_string("config", "brodatz_texture_dsprites_to_maskedplankton_224", "Name of the config to run")

flags.DEFINE_integer("n_sweep", 20, "Number of sweeps to run")


flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")

flags.DEFINE_string("backbone", None, "Name of the backbone to run")




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



def get_input_directory(input_experiment, model_num):
    # Set correct output directory. and check they already exist
    input_directory = os.path.join("input", "{experiment}", "{model_num}")

    # Insert model number and study name into path if necessary.
    input_directory = input_directory.format(model_num=str(model_num),
                                               experiment=str(input_experiment))

    if not os.path.exists(input_directory):
        raise FileExistsError("Input experiment folder do not exists")

    return input_directory




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


    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(model_num=str(run.name),
                                               experiment=str(FLAGS.experiment))


    input_directory = get_input_directory(run.config["input_experiment"], model_num)

    print("Transfer from ", input_directory)


    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.
        raise FileExistsError("Experiment folder does not exist")
        #raise FileExistsError("Experiment folder exists")


    # train model
    print(f"SWEEP: {run.id}")
    print(run.config)

    config = {k:v for k,v in  run.config.items()}
    config["backbone"] = FLAGS.backbone

    # BEFORE-TRANSFER
    pre_output_directory = os.path.join(output_directory, "before")

    if os.path.exists(pre_output_directory):
        evaluate_model_with_dsprites(pre_output_directory, config)


    # AFTER-TRANSFER
    post_output_directory = os.path.join(output_directory, "after")



    #if os.path.exists(post_output_directory):
        #evaluate_model_with_dsprites(post_output_directory, config)



    run.finish()


if __name__ == "__main__":
    app.run(sweep)







