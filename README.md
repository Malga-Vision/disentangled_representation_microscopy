# Disentangled representations of microscopy images
This is official code repository of the paper **"Disentangled representations of microscopy images"** ([IJCNN 2025](https://2025.ijcnn.org/))

**Jacopo Dapueto**, Vito Paolo Pastore, Nicoletta Noceti, Francesca Odone

[![Paper](https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green)](https://doi.org/10.1109/IJCNN64981.2025.11228114)
[![Arxiv](https://img.shields.io/badge/arXiv-2508.09925-b31b1b.svg)](https://arxiv.org/abs/2506.20649) <!--- [[Dataset🤗]()] -->

<img src="https://github.com/JacopoDapueto/disentangled_plankton/blob/main/assets/pipeline.png" width="50%" align="center">


## Set up
### Install
The code was developed and tested with Python 3.10 and the main dependencies are Pytorch 2.0.0 and Cuda 11.7. We use Wandb to log scores and models.

Set up the environment and then install the dependencies with
```
pip install -r requirements.txt
```

### Download and prepare datasets
#### Microscopy datasets

1. Set the environment variable `DISENTANGLEMENT_LIB_DATA` to this path, for example, by adding

```
export DISENTANGLEMENT_LIB_DATA=<path to the data directory>
```
2. Download all the necessary datasets that can be found in the following links
   [Lensless](https://ibm.ent.box.com/v/PlanktonData),  [WHOI15](https://github.com/Malga-Vision/Anomaly-detection-in-feature-space-for-detecting-changes-in-phytoplankton-populations), [Vacuoles](https://github.com/CCCofficial/Vacuoles-dataset-unsupervised-learning), [Sipakmed](https://www.cs.uoi.gr/~marina/sipakmed.html) 

4. Unzip the compressed files and put in DISENTANGLEMENT_LIB_DATA

5. Postprocess the dataset with the commands
```
python code/dataset/preprocessing_lensless.py
python code/dataset/preprocessing_whoi15.py
```
#### Texture-dSprites
1. Download the .npz file for [dSprites](https://github.com/google-deepmind/dsprites-dataset) and put it in folder `DISENTANGLEMENT_LIB_DATA/dsprites/`

2. Download the textures from the [database](https://multibandtexture.recherche.usherbrooke.ca/normalized_brodatz.html) and put them in folder `DISENTANGLEMENT_LIB_DATA/texture/`
   
## How to reproduce 

`./config` folder contains the .yaml files to reproduce the experiments


### Extract deep features from the backbone
To extract the features $\Phi$ executes:
```
python dlib_extract_from_backbone.py --dataset <name of the dataset>
```
The representation is saved in DISENTANGLEMENT_LIB_DATA as .npz file.

The name of the datasets to be used can be found in `./code/choose_dataset.py`

<!--- **Alternatively** the representation can be downloaded from [Hugging Face 🤗]() -->

### Compute hand-crafted features
To extract the hand-crafted features of Lensless:
```
python dlib_extract_simple_features.py
```
The features are saved in DISENTANGLEMENT_LIB_DATA as `SIMPLE_FEATURES/representation.npz` file.


### Train Source models
**RGB Input:** We trained the Source models on RGB input using the implementation of Ada-GVAE and dSprites in [transfer_disentanglement](https://github.com/JacopoDapueto/transfer_disentanglement).

Deep DINO features **$\Phi$ Input:** Train the Source models trained on the $\Phi$ with the command
```
python dlib_training_adagvae.py
```

### Transfer on Target dataset
**Once** you have trained the source models, put them in `./input` folder and run the scripts to execute the transfer experiments.
To transfer (without and with fine-tuning) and to evaluate the representation, run the following script:
```
python dlib_transfer_dsprites_to_plankton.py --config <name of the config folder> --experiment <name of the output directory>
```
The results will be saved in `./output` directory, organized by experiment name:
*experiment/sweeps* and each sweep is further divided into `before` and `after` folders (meaning w/o or w/ finetuning ).

**Once** one experiment folder is completed aggregate the results of all random seeds with the scripts
```
python dlib_group_results_scores.py --experiment experiment_name 
```
For each score of interest (e.g. accuracy) a .json file is created inside the experiment folder reporting mean and std.

### Compute Disentanglement metrics OMES, DCI and MIG for a given dataset
```
python dlib_compute_disentanglement_metrics_dsprites.py --experiment experiment_name --config <name of the config folder>
```

### Open set experiment
To evaluate the model on the open-set setting, run the following script with default parameters:

```
python dlib_evaluate_open_set.py
```


### Without disentanglement
To evaluate the classifiers directly on $\Phi$, run the script with default parameters:

```
python dlib_compute_baseline.py
```


### Visualizations on Target dataset

To reproduce the visualizations run the following scripts
```
python dlib_visualize_latent_space.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
python dlib_visualize_latent_space_with_handcrafted.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
python dlib_visualize_latent_space_open_set.py --config <name of the config folder> --experiment <name of the output directory> --model_num <name of the sweep to visualize>
```



## 📧 Contacts
If you have any questions, you are very welcome to email jacopo.dapueto@gmail.com

## 📚 Bibtex citation
If you use our dataset or code, please give the repository a star ⭐ and cite our paper:

```BibTeX
@inproceedings{DBLP:conf/ijcnn/DapuetoPNO25,
  author       = {Jacopo Dapueto and
                  Vito Paolo Pastore and
                  Nicoletta Noceti and
                  Francesca Odone},
  title        = {Disentangled representations of microscopy images},
  booktitle    = {International Joint Conference on Neural Networks, {IJCNN} 2025, Rome,
                  Italy, June 30 - July 5, 2025},
  pages        = {1--8},
  publisher    = {{IEEE}},
  year         = {2025},
  url          = {https://doi.org/10.1109/IJCNN64981.2025.11228114},
  doi          = {10.1109/IJCNN64981.2025.11228114},
  timestamp    = {Fri, 21 Nov 2025 20:23:55 +0100},
  biburl       = {https://dblp.org/rec/conf/ijcnn/DapuetoPNO25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
