
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import colorcet as cc

import wandb


def find_best_perplexity(args, x_train, n_components=2):

    perplexity_list = [30] #[100] # [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 150,200, 250, 300, 400, 500]
    model_list = []
    divergence_list = []
    reduced_list = []

    for perplexity in perplexity_list:

        tsne_model = TSNE(n_components=n_components, perplexity=perplexity, random_state=args["split_random_seed"])

        x_reduced = tsne_model.fit_transform(x_train)


        model_list.append(tsne_model)
        divergence_list.append(tsne_model.kl_divergence_)
        reduced_list.append(x_reduced)

    i = np.argmin(divergence_list)

    #fig = px.line(x=perplexity_list, y=divergence_list)
    #fig.show()

    return model_list[i], reduced_list[i], perplexity_list[i]


def fit_transform_pca(x_train, x_test, n_components=2):
    pca = PCA(n_components=n_components)

    pca.fit(x_train)

    return pca.transform(x_test)


def save_plotly( x_reduced, y, title, path, log_key="", use_wandb=False):

    if x_reduced.shape[1] == 3:
        fig = px.scatter_3d(x=x_reduced[:, 0], y= x_reduced[:, 1], z= x_reduced[:, 2],color=y.astype(str))

    if x_reduced.shape[1] == 2:
        fig = px.scatter(x=x_reduced[:, 0], y=x_reduced[:, 1], color=y.astype(str))

    fig.update_layout(title=title)

    if use_wandb:
        wandb.log({log_key: wandb.Plotly(fig)})
    else:
        fig.write_html(path)



def save_distplot(x, name, path):
    data = {name: [elem for elem in x]}

    with sns.axes_style("whitegrid"):
        sns.displot(data=data, x=name) # , kde=True


    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()


def save_scatterplot(x, y, namex, namey, classes, path, title=""):
        data = {namex: [elem for elem in x], namey:[elem for elem in y]}


        num_colors = len(np.unique(classes).tolist())

        palette = sns.color_palette("tab20", num_colors)

        classes_name = ["Actinospaerium", "Arcella", "Blepharisma", "Didinium", "Dileptus", "Euplotes", "Paramecium", "Spirostomum", "Stentor", "Volvox"]
        #classes_name = ["Asterionellopsis", "Chaetoceros", "Cylindrotheca", "Dactyliosolen", "Detritus", "Dinobryon", "Ditylum","Licmophora", "Pennate", "Phaeocystis", "Pleurosigma", "Pseudonitzschia", "Rhizosolenia", "Skeletonema", "Thalassiosira"]

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.20)  # font size 2

        with sns.axes_style("whitegrid"):

            ax = sns.scatterplot(data=data, x=namex, y = namey, hue=classes, palette=palette, alpha=0.75, linewidth=0, s=50)  # , kde=True

        leg = ax.axes.get_legend()

        for t, l in zip(leg.texts, classes_name):
            t.set_text(l)

        # Create a custom legend
        handles, labels = ax.get_legend_handles_labels()

        plt.legend(handles=handles[:],labels=classes_name, bbox_to_anchor=(0.5, 1.20), loc='upper center',
                   ncol=3)  # ,bbox_to_anchor=(0.4, 1.), loc='upper center', loc='upper center', borderaxespad=0,


        #plt.title(title)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.clf()

def create_visualization_directory(directory, quantized):

    if not quantized:
        process_dir = os.path.join(directory, "visualization")
    else:
        process_dir = os.path.join(directory, "visualization_quantized")
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
    csv = np.load(os.path.join(path_to_representation, "classes.npz"))


    for split in ["test"]: # "train", "val",



        split_path = os.path.join(directory, split)

        # make experiment directory
        if not os.path.exists(split_path):
            # if the demo_folder directory is not present then create it.
            os.makedirs(split_path)

        x = rep[split]
        y = csv[split].astype(np.int32)


        # compute t-SNE
        tsne_2, x_reduced2, perplexity2 = find_best_perplexity(args, x, n_components=2)
        tsne_3, x_reduced3, perplexity3 = find_best_perplexity(args, x, n_components=3)

        save_plotly(x_reduced2, y, title="2D t-SNE (Test set): " + str(perplexity2), path = os.path.join(split_path, "tsne2d.html") )
        save_plotly(x_reduced3, y, title="3D t-SNE (Test set): " + str(perplexity3), path = os.path.join(split_path, "tsne3d.html") )
        save_scatterplot(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y, path=os.path.join(split_path, "tsne2d.png"))


        # compute PCA
        x_reduced2 = fit_transform_pca(x, x, n_components=2)
        x_reduced3 = fit_transform_pca(x, x, n_components=3)


        save_plotly(x_reduced2, y, title="2D PCA (Test set)",  path=os.path.join(split_path, "pca2d.html"))
        save_plotly(x_reduced3, y, title="3D PCA (Test set): ", path=os.path.join(split_path, "pca3d.html"))
        save_scatterplot(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                         path=os.path.join(split_path, "pca2d.png"))

        postprocess_dims = args["postprocess_dims"]
        dim_to_factor = args["dim_to_factor"]

        for i, name in zip(postprocess_dims, dim_to_factor):
            save_distplot(x[:, i], name, path=os.path.join(split_path, f"{name}_dist.png"))

            save_scatterplot(x[:, i], [elem for elem in x[:, i]], name, "y", y,  path=os.path.join(split_path, f"{name}_scatter.png"))

            for j, namej in zip(postprocess_dims, dim_to_factor):

                if name == namej:
                    continue
                #save_distplot(x[:, i], x[:, j], name, namej, path=os.path.join(split_path, f"{name}_{namej}_dist.png"))

                save_scatterplot(x[:, i], x[:, j], name, namej, y,
                                 path=os.path.join(split_path, f"{name}_{namej}_scatter.png"), title=f"{name}-{namej}")

                """
                for z, namez in zip(postprocess_dims, dim_to_factor):

                    if namez == namej or namez == name:
                        continue

                    tsne_2, x_reduced2, perplexity2 = find_best_perplexity(args, x[:, [i, j, z]], n_components=2)
                    save_scatterplot(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                                    path=os.path.join(split_path, f"{name}_{namej}_{namez}_tsne2d.png"), title=f"T-SNE of {name}-{namej}-{namez}")





                    tsne_2, x_reduced2, perplexity2 = find_best_perplexity(args, x[:, [i, j, z]], n_components=2)
                    save_scatterplot(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                                    path=os.path.join(split_path, f"{name}_{namej}_{namez}_tsne2d.png"), title=f"T-SNE of {name}-{namej}-{namez}")

                """




