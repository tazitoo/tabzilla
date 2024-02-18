"""
Prototype the work flow to run kernelSHAP over a range of n_samples.
    - load model
    - load dataset
    - train explainer
    - generate and save explanations

TODO:
- reformat this script to the style of tabzialla_experiment.py (... create a main() ...)
- generalize XAI method
- model_args should get put into a utility function (and reused in cross_validation)
- fixed seed for repeatability?

"""

# import torch
# torch.cuda.empty_cache()

# import gc
# gc.collect()

# attempting to minimize "out of memory" errors in ipython
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse
import os
import pickle

# import sys
import time
from collections import namedtuple
from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np

# import shapreg
# from captum.attr import KernelShap
# from fastshap import KernelExplainer
import torch
import torch.nn as nn

# from fastshap.utils import MaskLayer1d
from captum.attr import KernelShap

# from sklearn.model_selection import train_test_split
from pytorch_lightning import seed_everything

# from scipy.spatial.distance import cosine
# from shap import KernelExplainer as ke
# from models.basemodel import BaseModel
# from models.tree_models import XGBoost
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_data_processing import process_data
from tabzilla_datasets import TabularDataset
from tabzilla_utils import get_experiment_parser
from utils.io_utils import get_output_path, get_sample_list

# some setup for the experiment to save XAI results
output_dir = "output/"
directory = "kernel_shap/"
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
print(f"EXPERIMENT ARGS: {experiment_args}")

# sys.exit()

# set random seed for repeatability
seed_everything(experiment_args.subset_random_seed, workers=True)
np.random.seed(seed=experiment_args.subset_random_seed)
repeats = 1

# load dataset
dataset = TabularDataset.read(Path(args.dataset_dir).resolve())

# pick one of the CV splits
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

# fix object type for X_train
X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)
X_val = np.array(X_val, dtype=float)


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
my_model.load_model(extension="best", directory="models")


num_features = X_train.shape[1]
# sample_list = get_sample_list(X_train)
sample_list = get_sample_list(X_test)
max_sample = max(sample_list)

if hasattr(my_model, "device"):
    device = my_model.device
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


x = torch.tensor(X_test, dtype=torch.float32, device=device)
for a_sample in sample_list:
    idx = np.random.randint(0, X_test.shape[0], max_sample)
    # idx = np.random.randint(0, X_train.shape[0], max_sample)

    # sample_list = [4]
    for a_repeat in range(repeats):
        # a_frac = a_sample / X_train.shape[0]
        start = time.time()

        idx_sub = np.random.choice(idx, a_sample, replace=False)
        # X_train_subset = X_train[idx_sub, :]
        X_test_subset = X_test[idx_sub, :]
        ks = KernelShap(my_model.model)

        # Computes attribution
        # features as a separate interpretable feature
        attr = ks.attribute(x, target=0, n_samples=200)
        # attribute(inputs, baselines=None, target=None, additional_forward_args=None,
        #           feature_mask=None, n_samples=25, perturbations_per_eval=1, return_input_shape=True,
        #           show_progress=False)

        # fastshap_list = []
        # # for idx_sample in range(X_val.shape[0]):
        # #     x = X_val[idx_sample, :].reshape(1, -1)
        # for idx_sample in range(X_train.shape[0]):
        #     x = X_train[idx_sample, :].reshape(1, -1)

        #     # compute fastshap explanations
        #     fastshap_values = fastshap.shap_values(x)[0]
        #     fastshap_list.append(fastshap_values)

        # save the explanations
        ks_file = get_output_path(
            model_args,
            directory=directory,
            filename="ks",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pkl",
        )

        with open(ks_file, "wb") as f:
            pickle.dump(attr, f)
