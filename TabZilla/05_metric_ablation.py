"""
Prototype the work flow to run ablation over a range of n_samples and 5 repeats.
reformulating the ablation example from:
https://github.com/capitalone/ablation/blob/main/examples/example_model_agnostic.ipynb

"""

import argparse
import os
import pickle

# import sys
# import time
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rbo

# import seaborn as sns
# import shapreg
# from captum.attr import KernelShap
# from fastshap import KernelExplainer
import torch
import torch.nn as nn
from ablation.ablation import Ablation
from ablation.dataset import NumpyDataset
from ablation.perturb import generate_perturbation_distribution

# from fastshap import FastSHAP, KLDivLoss, Surrogate
# from fastshap.utils import MaskLayer1d
# from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything
from scipy.stats import spearmanr

# from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
# from scipy.spatial.distance import cosine
# from shap import KernelExplainer as ke
# from models.basemodel import BaseModel
# from models.tree_models import XGBoost
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from tabzilla_utils import get_experiment_parser
from utils.io_utils import get_output_path, get_sample_list
from utils.metric_utils import ranking_array


def prepare_ablation_dataset(dataset):
    """Prepare the first fold from the openml datasets as a NumpyDataset for ablation

    Returns:
        NumpyDataset: dataset
    """

    isplit = 0
    train_idx = dataset.split_indeces[isplit]["train"]
    val_idx = dataset.split_indeces[isplit]["val"]
    test_idx = dataset.split_indeces[isplit]["test"]

    # X_train = dataset.X[train_idx, :]
    # y_train = dataset.y[train_idx]
    # X_test = dataset.X[test_idx, :]
    # y_test = dataset.y[test_idx]
    # X_val = dataset.X[val_idx, :]

    data_processed = process_data(dataset, train_idx, val_idx, test_idx)
    X_train, y_train = data_processed["data_train"]
    X_test, y_test = data_processed["data_test"]
    X_val, y_val = data_processed["data_val"]

    # fix object type
    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    # X_val = np.array(X_val, dtype=float)

    # provide indices for categorical and numerical features
    # cat_ix = dataset.cat_idx
    # idx = np.arange(dataset.num_features)
    # num_ix = [i for i in idx if i not in dataset.cat_idx]

    # we don't get feature names in the meta-data - so we'll just use the column indices
    feature_names = [f"feature_{i}" for i in range(dataset.num_features)]

    return NumpyDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        n_classes=2,
        feature_names=feature_names,
        original_feature_names=feature_names,
    )


# some setup for the experiment to save XAI results
output_dir = "output/"
directory = "xai/"
if not os.path.isdir(output_dir + directory):
    os.makedirs(output_dir + directory)

parser = argparse.ArgumentParser(description="parser for tabzilla experiments")

parser.add_argument(
    "--experiment_config",
    required=True,
    type=str,
    help="config file for parameter experiment args",
)

parser.add_argument(
    "--dataset_dir",
    required=True,
    type=str,
    help="directory containing pre-processed dataset.",
)
parser.add_argument(
    "--model_name",
    required=True,
    type=str,
    choices=ALL_MODELS,
    help="name of the algorithm",
)


args = parser.parse_args()
# args.use_gpu = False
# print(f"ARGS: {args}")


# now parse the dataset and search config files
experiment_parser = get_experiment_parser()

experiment_args = experiment_parser.parse_args(
    args="-experiment_config " + args.experiment_config
)
# print(f"EXPERIMENT ARGS: {experiment_args}")


# set random seed for repeatability
seed_everything(experiment_args.subset_random_seed, workers=True)
np.random.seed(seed=experiment_args.subset_random_seed)
repeats = 5

# load dataset
dataset = TabularDataset.read(Path(args.dataset_dir).resolve())
abltn_dataset = prepare_ablation_dataset(dataset)

perturbation = generate_perturbation_distribution(
    method="marginal", X=abltn_dataset.X_train, X_obs=abltn_dataset.X_train
)

# load the model
arg_namespace = namedtuple(
    "args",
    [
        "model_name",
        "batch_size",
        "scale_numerical_features",
        "val_batch_size",
        "objective",
        "gpu_ids",
        "use_gpu",
        "epochs",
        "data_parallel",
        "early_stopping_rounds",
        "dataset",
        "cat_idx",
        "num_features",
        "subset_features",
        "subset_rows",
        "subset_features_method",
        "subset_rows_method",
        "cat_dims",
        "num_classes",
        "logging_period",
    ],
)

model_args = arg_namespace(
    model_name=args.model_name,
    batch_size=experiment_args.batch_size,
    val_batch_size=experiment_args.val_batch_size,
    scale_numerical_features=experiment_args.scale_numerical_features,
    epochs=experiment_args.epochs,
    gpu_ids=experiment_args.gpu_ids,
    use_gpu=experiment_args.use_gpu,
    data_parallel=experiment_args.data_parallel,
    early_stopping_rounds=experiment_args.early_stopping_rounds,
    logging_period=experiment_args.logging_period,
    objective=dataset.target_type,
    dataset=dataset.name,
    cat_idx=dataset.cat_idx,
    num_features=dataset.num_features,
    subset_features=experiment_args.subset_features,
    subset_rows=experiment_args.subset_rows,
    subset_features_method=experiment_args.subset_features_method,
    subset_rows_method=experiment_args.subset_rows_method,
    cat_dims=dataset.cat_dims,
    num_classes=dataset.num_classes,
)

model_handle = get_model(args.model_name)
my_model = model_handle(model_handle.default_parameters(), model_args)
my_model.load_model(extension="best", directory="models")


# generate 5 sets of random explanations - and their ablation curves
random_curves = pd.DataFrame()
for a_repeat in range(repeats):
    # generate the explanations
    random_explanations = np.random.rand(
        abltn_dataset.X_train.shape[0], abltn_dataset.X_train.shape[1]
    )

    # run ablation
    abtn = Ablation(
        perturbation,
        my_model.model,
        abltn_dataset,
        abltn_dataset.X_train,
        abltn_dataset.y_train,
        explanation_values=random_explanations,
        explanation_values_dense=random_explanations,
        random_feat_idx=None,
        scoring_methods=["auroc"],
        local=True,
    )
    random_result = abtn.ablate_features()
    tmp_curve = random_result[["scores"]].transpose()

    random_curves = pd.concat([random_curves, tmp_curve])
avg_random_curve = random_curves.mean(axis=0)

# load our explanations
model_name = args.model_name
sample_list = get_sample_list(abltn_dataset.X_train)
ablation_results = pd.DataFrame()

abc_list = []
for a_sample in sample_list:
    exp_curve = pd.DataFrame()
    for a_repeat in range(repeats):
        # load the explanations
        fastshap_file = get_output_path(
            model_args,
            directory=directory,
            filename="fastshap",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pkl",
        )

        with open(fastshap_file, "rb") as f:
            explanation_list = pickle.load(f)

        # reshape the explanations into an ndarray - with explanations for class 1
        tmp = []
        for i in range(len(explanation_list)):
            tmp.append(explanation_list[i][:, 1])
        explanations = np.stack(tmp)

        # run ablation
        abtn = Ablation(
            perturbation,
            my_model.model,
            abltn_dataset,
            abltn_dataset.X_train,
            abltn_dataset.y_train,
            explanation_values=explanations,
            explanation_values_dense=explanations,
            random_feat_idx=None,
            scoring_methods=["auroc"],
            local=True,
        )

        result = abtn.ablate_features()

        # let's extract the ablation curves
        tmp_curve = result[["scores"]].transpose()
        exp_curve = pd.concat([exp_curve, tmp_curve])
    avg_exp_curve = exp_curve.mean(axis=0)

    # this is the number we need - random curve should be less informative/higher than exp curve
    abc_metric = np.trapz(avg_random_curve) - np.trapz(avg_exp_curve)
    abc_list.append(abc_metric)

    print(f"{a_sample}: {abc_metric}")

    # plot the ablation curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(avg_exp_curve, label="FastSHAP")
    ax.plot(avg_random_curve, label="Random")
    ax.set_xlabel("Number of features")
    ax.set_ylabel("AUROC")
    ax.set_title(f"Sample {a_sample}")
    ax.legend()
    plt.show()
    # save the plot as a png in output/model/data/xai directory
    plot_file = get_output_path(
        model_args,
        directory=directory,
        filename="ablation_curves",
        extension=f"sample_{a_sample}",
        file_type="png",
    )
    fig.savefig(plot_file, bbox_inches="tight")

plt.close("all")

# plot the curve as a function of the number of samples
plt.plot(sample_list, abc_list, "ko-")
# set x-axis to have log scale
plt.xscale("log")
plt.xlabel("Number of samples")
plt.ylabel("ABC")
plot_file = get_output_path(
    model_args,
    directory=directory,
    filename="abc",
    extension=f"_{dataset.name}",
    file_type="png",
)
plt.savefig(plot_file, bbox_inches="tight")

plt.close("all")


# rank the ABC metric and stash all in a dataframe for later use
rank = ranking_array(np.array(abc_list), reverse=True)
abc_df = pd.DataFrame(
    {
        "sample": sample_list,
        "abc": abc_list,
        "rank": rank,
    }
)

# save the results
result_file = get_output_path(
    model_args,
    directory=directory,
    filename="ablation_curves",
    extension="",
    file_type="csv",
)
abc_df.to_csv(result_file, index=False)

# calculate the correlation between the ABC metric rank and the inverse of the sample size
# (Larger sample size is higher fidelity - hence higher known rank)
known_rank = np.flip(abc_df.index.values)
# abc_df['rank'].values.T
rho = spearmanr(known_rank, abc_df["rank"])
rbo = rbo.RankingSimilarity(known_rank, abc_df["rank"].values.T).rbo()

with open("ablation_correlation.txt", "w") as f:
    f.write(f"RBO: {rbo}\n")
    f.write(f"RHO: {rho.correlation}\n")
