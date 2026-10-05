
"""  Test wandb integration  """
from absl import app
from absl import flags

import os
import json
import pandas as pd
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment", "BRODATZTEXTUREDSPRITES_to_whoi15_2007_224", "Name of the experiment to run")


flags.DEFINE_string("output_directory", None, "Output directory of experiments ('{model_num}' will be"
                                                " replaced with the model index  and '{experiment}' will be"
                                                " replaced with the study name if present).")

def aggregate_json(data, aggregator):
    for key, value in data.items():
        aggregator[key].append(value)

    return aggregator


def aggregate_csv(csv, aggregator):

    data = { row["Feature"]:row["Importance"] for index, row in csv.iterrows()}

    for key, value in data.items():

        if key not in aggregator:
            aggregator[key] = []

        aggregator[key].append(value)

    return aggregator





def aggregate_results(directory, before=True):
    aggregated_mlp = {"train":[], "val":[], "test":[]}


    aggregated_mlp_sk = {"train":[], "val":[], "test":[]}
    aggregated_gbt = {"train":[], "val":[], "test":[]}

    aggregated_balanced_mlp = {"train":[], "val":[], "test":[]}
    aggregated_balanced_mlp_sk = {"train":[], "val":[], "test":[]}


    aggregated_balanced_gbt = {"train":[], "val":[], "test":[]}

    features_importace = {}

    # List all entries in the directory
    for entry in os.listdir(directory):
        # Create the full path
        path = os.path.join(directory, entry)
        # Check if it's a directory
        if os.path.isdir(path) and "before" not in path and "after" not in path:

            if before:
                full_path = os.path.join(path, "before")
            else:
                full_path = os.path.join(path, "after")


            # aggregate
            postprocess_path = os.path.join(full_path, "postprocess")


            # aggregate GBT
            gbt_path = os.path.join(postprocess_path, "gbt")



            with open(os.path.join(gbt_path, "accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_gbt)

            with open(os.path.join(gbt_path, "balanced_accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_balanced_gbt)

            data = pd.read_csv(os.path.join(gbt_path, "feature_importance.csv"))
            aggregate_csv(data, features_importace)


            # aggregate MLP
            mlp_path = os.path.join(postprocess_path, "mlp")


            with open(os.path.join(mlp_path, "accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_mlp)


            with open(os.path.join(mlp_path, "balanced_accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_balanced_mlp)




            # aggregate MLP SKLEARN
            mlp_sk_path = os.path.join(postprocess_path, "mlp_sklearn")


            with open(os.path.join(mlp_sk_path, "accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_mlp_sk)


            with open(os.path.join(mlp_sk_path, "balanced_accuracy.json"), 'r') as json_file:
                data = json.load(json_file)
                if isinstance(data, dict):
                    aggregate_json(data, aggregated_balanced_mlp_sk)

    if before:
        save_directory = os.path.join(directory, "before")
    else:
        save_directory = os.path.join(directory, "after")



    # create folder
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    # SAVE FILES
    with open(os.path.join(save_directory, 'aggregated_balanced_gbt.json'), 'w') as fp:
        json.dump(aggregated_balanced_gbt, fp)

    with open(os.path.join(save_directory, 'aggregated_gbt.json'), 'w') as fp:
        json.dump(aggregated_gbt, fp)

    with open(os.path.join(save_directory, 'features_importace.json'), 'w') as fp:
        json.dump(features_importace, fp)

    with open(os.path.join(save_directory, 'aggregated_balanced_mlp.json'), 'w') as fp:
        json.dump(aggregated_balanced_mlp, fp)

    with open(os.path.join(save_directory, 'aggregated_mlp.json'), 'w') as fp:
        json.dump(aggregated_mlp, fp)



    with open(os.path.join(save_directory, 'aggregated_balanced_mlp_sk.json'), 'w') as fp:
        json.dump(aggregated_balanced_mlp_sk, fp)

    with open(os.path.join(save_directory, 'aggregated_mlp_sk.json'), 'w') as fp:
        json.dump(aggregated_mlp_sk, fp)



def compute_statistics(directory):

    # compute statistics of each key of each json file

    for root, _, files in os.walk(directory):
        for file in files:

            if file.endswith('.json'):
                file_path = os.path.join(root, file)

                stats = {}
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)

                    for k, v in data.items():

                        if k not in stats:
                            stats[k] = {}

                        # statistics
                        stats[k]["mean"] = np.mean(v)
                        stats[k]["std"] = np.std(v)

                        stats[k]["min"] = np.min(v)
                        stats[k]["max"] = np.max(v)

                with open(os.path.join(root, "STATS_"+file) , 'w') as fp:
                    json.dump(stats, fp)






def main(args):





    # Set correct output directory. and check they already exist
    if FLAGS.output_directory is None:
        output_directory = os.path.join("output", "{experiment}")
    else:
        output_directory = FLAGS.output_directory


    # Insert model number and study name into path if necessary.
    output_directory = output_directory.format(experiment=str(FLAGS.experiment))




    # make experiment directory
    if not os.path.exists(output_directory):
        # if the demo_folder directory is not present then create it.

        raise FileExistsError("Experiment does folder exist!")



    # BEFORE FINETUNING

    aggregate_results(output_directory, before=True)

    compute_statistics(os.path.join(output_directory, "before"))

    # AFTER FINETUNING

    aggregate_results(output_directory, before=False)
    compute_statistics(os.path.join(output_directory, "after"))





if __name__ == "__main__":
    app.run(main)
