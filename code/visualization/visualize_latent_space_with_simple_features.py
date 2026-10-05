
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import colorcet as cc
from sklearn.metrics import r2_score

import wandb

from sklearn.preprocessing import LabelEncoder

from code.dataset.plakton_padded import PLANKTON_PADDED224_PATH


def save_scatterplot(x, y, namex, namey, path, title=""):
    data = {namex: np.array([elem for elem in x]), namey: np.array([elem for elem in y])}



    with sns.axes_style("whitegrid"):
        sns.regplot(data=data, x=namex, y=namey, robust=True, line_kws=dict(color="r"))  # , kde=True,alpha=0.90, ,  linewidth=0

    plt.xlabel(namex)
    plt.ylabel(namey)
    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()



def plot_hist(y, split, path):

    df = {"Classes": [elem for elem in y], "Split":[elem for elem in split]}

    unique_z = np.unique(df['Split'])
    # Get the 'deep' color palette
    palette = sns.color_palette('deep', len(unique_z))

    # Create a dictionary mapping each unique value in 'z' to a color
    color_mapping = {value: palette[i] for i, value in enumerate(unique_z)}

    # Plot bar plot
    # plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.45)  # font size 2

    with sns.axes_style("whitegrid"):
        sns.histplot(data= df, x="Classes", hue="Split", palette=color_mapping)

    plt.title(r'Distribution splits')

    plt.xlabel('Classes')
    plt.ylabel('Counts')

    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.clf()



def save_scatterplot_with_simple_features(x, y, namex, namey, classes, path, title=""):
    data = {namex: [elem for elem in x], namey: [elem for elem in y]}
    #classes= [str(elem) for elem in classes]



    #le = LabelEncoder()
    #le.fit(classes)
    #classes = le.transform(classes)
    #print(classes)

    #palette = sns.color_palette("tab20", 10)
    palette = "viridis"

    with sns.axes_style("whitegrid"):
        ax = sns.scatterplot(data=data, x=namex, y=namey, hue=classes, palette=palette, alpha=0.75,
                        linewidth=0)  # , kde=True

    norm = plt.Normalize(classes.min(), classes.max())
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    ax.get_legend().remove()
    ax.figure.colorbar(sm)

    #plt.legend([], [], frameon=False)

    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()


def create_visualization_directory(directory, quantized):

    if not quantized:
        process_dir = os.path.join(directory, "visualization_with_simple")
    else:
        process_dir = os.path.join(directory, "visualization_with_simple_quantized")
    # make experiment directory
    if not os.path.exists(process_dir):
        # if the demo_folder directory is not present then create it.
        os.makedirs(process_dir)


    return process_dir


def visualize_latent_space(directory, args):

    directory = create_visualization_directory(directory, args["lq"])

    # load representation
    path_to_representation = args["path_to_representation"]
    rep = np.load(os.path.join(path_to_representation, "representations.npz"))
    #csv = np.load(os.path.join(path_to_representation, "classes.npz"))



    # load simple features
    simple_rep = np.load(os.path.join(PLANKTON_PADDED224_PATH, "SIMPLE_FEATURES_only_color", "representations.npz"), allow_pickle=True) # [avg_color, dominant_color, area, orientation]

    print(simple_rep.files)
    for split in ["test"]:  # "train", "val", # adjust perc of validation




        split_path = os.path.join(directory, split)

        # make experiment directory
        if not os.path.exists(split_path):
            # if the demo_folder directory is not present then create it.
            os.makedirs(split_path)

        x = rep[split]
        #y = csv[split].astype(np.int32)

        simple_x = simple_rep[split]



        postprocess_dims = args["postprocess_dims"]
        dim_to_factor = args["dim_to_factor"]

        for i, name in zip(postprocess_dims, dim_to_factor):


            for j, namej in zip(postprocess_dims, dim_to_factor):

                if name in  ["Color", "Scale", "Orientation", "Texture", "Shape"] and namej in ["Color", "Scale", "Orientation", "Texture", "Shape"]:
                    print(simple_x.shape)


                    namez = "Solidity"
                    scale = simple_x[:, 0]
                    print(scale.shape)
                    score = np.corrcoef(scale, x[:, i])[0, 1]


                    save_scatterplot(x[:, i], scale, name + " (Z2)", namez + f" (handcrafted)",
                                                             path=os.path.join(split_path, f"{name}_scatter_{namez}.png"),
                                                             title=f"Pearson correlation: {score:.2}")


                    namez = "Scale"
                    scale = simple_x[:, 0]
                    print(scale.shape)
                    score = np.corrcoef(scale, x[:, i])[0, 1]

                    save_scatterplot(x[:, i], scale, name + " (Z3)", namez + f" (handcrafted)",
                                         path=os.path.join(split_path, f"{name}_scatter_{namez}.png"),
                                         title=f"Pearson correlation: {score:.2}")


                    namez = "Color AVG"
                    for i_channel, channel in zip([0, 1, 2], ["Red", "Green", "Blue"]):
                        color = simple_x[:, i_channel]
                        print(color.shape)
                        #color = color[:, i_channel]
                        score = np.corrcoef(color, x[:, i])[0, 1]

                        save_scatterplot(x[:, i], color, name + " (Z1)", namez + f" (handcrafted): {channel}",
                                         path=os.path.join(split_path, f"{name}_scatter_{namez}_{channel}.png"),
                                         title=f"Pearson correlation: {score:.2}")

