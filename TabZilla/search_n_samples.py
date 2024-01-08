"""
figure out how to find the minimum number of samples needed to get a good
set of KernelSHAP explanations - when compared to treeSHAp explanations

1. simplest - create a stratfiied sample and use Wasserstein distance to compare to the training set

2. more complex - check a stratified sample of the data, generate KernelSHAP 
    explanations and compare to treeSHAP explanations using cosine distance
    
"""
import argparse
import pickle
from collections import namedtuple
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cosine

# from scipy.special import kl_div
from scipy.stats import wasserstein_distance
from shap import KernelExplainer as ke
from sklearn.model_selection import train_test_split

from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import get_experiment_parser

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

parser.add_argument(
    "--xai_config",
    required=True,
    type=str,
    # choices=ALL_MODELS,
    help="config file for the XAI method",
)


args = parser.parse_args()
# args.use_gpu = False
print(f"ARGS: {args}")

# now parse the dataset and search config files
experiment_parser = get_experiment_parser()

experiment_args = experiment_parser.parse_args(
    args="-experiment_config " + args.experiment_config
)
print(f"EXPERIMENT ARGS: {experiment_args}")


# load dataset
dataset = TabularDataset.read(Path(args.dataset_dir).resolve())
# split into train test split using indices

# pick one of the CV splits
isplit = 0
train_idx = dataset.split_indeces[isplit]["train"]
val_idx = dataset.split_indeces[isplit]["val"]
test_idx = dataset.split_indeces[isplit]["test"]

X_train = dataset.X[train_idx, :]
y_train = dataset.y[train_idx]
X_test = dataset.X[test_idx, :]
# y_test = dataset.y[test_idx]
X_val = dataset.X[val_idx, :]


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

# my_model = XGBoost(XGBoost.default_parameters(), model_args)
model_handle = get_model(args.model_name)
my_model = model_handle(model_handle.default_parameters(), model_args)
my_model.model.load_model("output/XGBoost/openml__adult__7592/models/m_best.json")

#
# simple approach first - generate a stratified sample and compare to the training set
#
# generate a stratified sample of the training set

for a_frac in [0.01, 0.02, 0.03]:  # , 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]:
    _, X_train_sample, _, _ = train_test_split(
        X_train,
        y_train,
        test_size=a_frac,
        random_state=42,
        stratify=y_train,
    )
    dist = 0
    for i in range(X_train.shape[1]):
        tmp_dist = wasserstein_distance(X_train[:, i], X_train_sample[:, i])
        dist += tmp_dist

    print(f"Fraction: {a_frac}, avg Wasserstein distance: {dist/X_train.shape[1]}")

#     # generate treeSHAP explanations for the sample
#     explainer = shap.TreeExplainer(my_model.model)
#     shap_values = explainer.shap_values(X_train_sample)

#     # generate treeSHAP explanations for the training set
#     explainer = shap.TreeExplainer(my_model.model)
#     shap_values = explainer.shap_values(X_train)

#     # compare the two sets of explanations using Wasserstein distance
#     wasserstein_distance(shap_values, shap_values_sample)

# # Generate shap values using predict_contribs
# shap_values = model.predict(data, pred_contribs=True)

# # Generate kernel shap explanations
# explainer = shap.KernelExplainer(model.predict, data)
# kernel_shap_values = explainer.shap_values(data)

# # Calculate the cosine distance between attributions and kernel shap values
# cosine_distance = cosine(shap_values, kernel_shap_values)

# cosine_distance
