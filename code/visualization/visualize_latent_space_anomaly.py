
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import colorcet as cc
import pandas as pd
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
        sns.displot(data=data, x=name, ) # , kde=True


    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()



def plot_hist_dileptus(x, classes, namex, path):

    df = {"x": [elem for elem in x], "classes": [ "others" if elem!=4 else "dileptus" for elem in classes]}



    # Plot bar plot
    # plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.45)  # font size 2

    with sns.axes_style("whitegrid"):
        sns.histplot(data= df, x="x", hue="classes", palette=["gray", "red"], stat="density", common_norm=False)

    plt.title(r'Distribution classes')

    plt.xlabel(namex)
    plt.ylabel('Counts')

    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.clf()



def save_scatterplot_dileptus(x, y, namex, namey, classes, path, title=""):
        data = {namex: [elem for elem in x], namey:[elem for elem in y], "classes": [ "others" if elem!=4 else "dileptus" for elem in classes]}


        num_colors = len(np.unique(classes).tolist())

        #palette = sns.color_palette("tab20", num_colors)

        with sns.axes_style("whitegrid"):
            sns.scatterplot(data=data, x=namex, y = namey, hue="classes", alpha=0.75, linewidth=0, palette=["gray", "red"])  # , kde=True

        plt.title(title)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.clf()



def plot_hist_arcella(x, classes, namex, path):
    classes = ["Arcella" if elem == 1 else elem for elem in classes]
    classes = ["Euplotes" if elem == 4 else elem for elem in classes]
    classes = ["Others" if (elem != "Arcella" and elem != "Euplotes") else elem for elem in classes]
    df = {"x": [elem for elem in x], "classes": classes}



    # Plot bar plot
    # plt.figure(figsize=(14, 10))
    sns.set(font_scale=1.45)  # font size 2

    with sns.axes_style("whitegrid"):
        sns.histplot(data= df, x="x", hue="classes", palette=["gray", "red", "blue"], stat="density", common_norm=False)

    plt.title(r'Distribution classes')

    plt.xlabel(namex)
    plt.ylabel('Counts')

    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.clf()



def save_scatterplot_arcella(x, y, namex, namey, classes, path, title=""):
        classes = ["Arcella" if elem == 1 else elem for elem in classes]
        classes = ["Euplotes" if elem == 4 else elem for elem in classes]
        classes = [ "Others" if (elem != "Arcella" and elem != "Euplotes") else elem for elem in classes]

        data = {namex: [elem for elem in x], namey:[elem for elem in y], "classes": classes}
        print(np.unique(classes))

        df = pd.DataFrame(data)
        class_order = ['Others' , 'Euplotes', "Arcella" ]  # 'C' will be in the background, 'A' will be in the foreground

        # Sort the DataFrame by class
        df['classes'] = pd.Categorical(df['classes'], categories=class_order, ordered=True)
        df = df.sort_values('classes')

        num_colors = len(np.unique(classes).tolist())

        #palette = sns.color_palette("tab20", num_colors)

        plt.figure(figsize=(8, 6))
        sns.set(font_scale=1.75)  # font size 2

        with sns.axes_style("whitegrid"):
            sns.scatterplot(data=df, x=namex, y = namey, hue="classes", alpha=0.75, linewidth=0, palette=[ "silver",  "blue","red" ], hue_order=class_order, s=50)  # , kde=True

        plt.title(title)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(path)
        plt.clf()


def create_visualization_directory(directory, quantized):

    if not quantized:
        process_dir = os.path.join(directory, "visualization_anomaly")
    else:
        process_dir = os.path.join(directory, "visualization_anomaly_quantized")
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


    for split in ["test"]: # only test


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
        save_scatterplot_arcella(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y, path=os.path.join(split_path, "tsne2d.png"))


        # compute PCA
        x_reduced2 = fit_transform_pca(x, x, n_components=2)
        x_reduced3 = fit_transform_pca(x, x, n_components=3)


        save_plotly(x_reduced2, y, title="2D PCA (Test set)",  path=os.path.join(split_path, "pca2d.html"))
        save_plotly(x_reduced3, y, title="3D PCA (Test set): ", path=os.path.join(split_path, "pca3d.html"))
        save_scatterplot_arcella(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                         path=os.path.join(split_path, "pca2d.png"))

        postprocess_dims = args["postprocess_dims"]
        dim_to_factor = args["dim_to_factor"]

        for i, name in zip(postprocess_dims, dim_to_factor):
            plot_hist_arcella(x[:, i], y, name, path=os.path.join(split_path, f"{name}_dist_euplotes.png"))

            save_scatterplot_arcella(x[:, i], [elem for elem in x[:, i]], name, "y", y,  path=os.path.join(split_path, f"{name}_scatter_euplotes.png"))

            for j, namej in zip(postprocess_dims, dim_to_factor):
                #save_distplot(x[:, i], x[:, j], name, namej, path=os.path.join(split_path, f"{name}_{namej}_dist.png"))

                save_scatterplot_arcella(x[:, i], x[:, j], name, namej, y,
                                 path=os.path.join(split_path, f"{name}_{namej}_scatter_euplotes.png"), title=f"{name}-{namej}")

                tsne_2, x_reduced2, perplexity2 = find_best_perplexity(args, x[:, [i, j]], n_components=2)
                save_scatterplot_arcella(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                                 path=os.path.join(split_path, f"{name}_{namej}_tsne2d.png"), title=f"T-SNE of {name}-{namej}")

                tsne_1, x_reduced1, perplexity1 = find_best_perplexity(args, x[:, [i, j]], n_components=1)
                save_scatterplot_arcella(x_reduced1[:, 0], [0 for _ in x_reduced1[:, 0]], namex="t-sne 1", namey="t-sne 2", classes=y,
                                 path=os.path.join(split_path, f"{name}_{namej}_tsne1d.png"))


                save_distplot(x_reduced1[:, 0], "tsne1d", path=os.path.join(split_path, f"{name}_tsne1d_dist.png"))

                tsne_2, x_reduced2, perplexity2 = find_best_perplexity(args, x[:, [i, j]], n_components=2)
                save_scatterplot_arcella(x_reduced2[:, 0], x_reduced2[:, 1], namex="t-sne 1", namey="t-sne 2", classes=y,
                                 path=os.path.join(split_path, f"{name}_{namej}_tsne2d.png"), title=f"T-SNE of {name}-{namej}")

                tsne_1, x_reduced1, perplexity1 = find_best_perplexity(args, x[:, [i, j]], n_components=1)
                save_scatterplot_arcella(x_reduced1[:, 0], [0 for _ in x_reduced1[:, 0]], namex="t-sne 1", namey="t-sne 2", classes=y,
                                 path=os.path.join(split_path, f"{name}_{namej}_tsne1d.png"))


                save_distplot(x_reduced1[:, 0], "tsne1d", path=os.path.join(split_path, f"{name}_tsne1d_dist.png"))


