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
from fastshap import FastSHAP, KLDivLoss, Surrogate
from fastshap.utils import MaskLayer1d

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
print(f"EXPERIMENT ARGS: {experiment_args}")

# sys.exit()

# set random seed for repeatability
seed_everything(experiment_args.subset_random_seed, workers=True)
np.random.seed(seed=experiment_args.subset_random_seed)
repeats = 5

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


# Set up original model - pytorch
def original_model_pytorch(x):
    """
    PYTORCH VERSION - for binary classification, the model should output a 2-vector of probabilities

    TODO: we are wasting time moving data from GPU to CPU and back again...
    """
    pred = my_model.predict_proba(x.cpu().numpy())
    # pred = my_model.predict_proba(x)
    # pred = np.stack([1 - pred, pred]).T  # if the model is just outputting the probability of the positive class
    return torch.tensor(pred, dtype=torch.float32, device=x.device)


# XGBoost version
def original_model_xgb(x):
    """
    XGBoost version - for binary classification, the model should output a 2-vector of probabilities

    x - torch tensor, needs to be brought back to CPU for xgb model
    """
    pred = my_model.alt_predict_proba(x.cpu().numpy())
    # pred = my_model.predict_proba(x)
    # pred = np.stack([1 - pred, pred]).T  # if the model is just outputting the probability of the positive class
    return torch.tensor(pred, dtype=torch.float32, device=x.device)


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


# figure out which "original model" function to use
if hasattr(my_model, "alt_predict_proba"):
    original_model = original_model_xgb
else:
    original_model = original_model_pytorch


num_features = X_train.shape[1]
sample_list = get_sample_list(X_train)
max_sample = max(sample_list)

for a_sample in sample_list:
    # idx = np.random.randint(0, X_test.shape[0], max_sample)
    idx = np.random.randint(0, X_train.shape[0], max_sample)

    # sample_list = [4]
    for a_repeat in range(repeats):
        # a_frac = a_sample / X_train.shape[0]
        start = time.time()
        # subset the training data for use as a background dataset for KernelSHAP
        # _, data_subset, _, _ = train_test_split(
        #     X_train,
        #     y_train,
        #     test_size=a_frac,
        #     random_state=42,
        #     stratify=y_train,
        # )
        idx_sub = np.random.choice(idx, a_sample, replace=False)
        X_train_subset = X_train[idx_sub, :]

        if hasattr(my_model, "device"):
            device = my_model.device
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        surrogate_file = get_output_path(
            model_args,
            directory=directory,
            filename="surr",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pt",
        )
        # Check for model
        if os.path.isfile(surrogate_file):
            print(
                f"Loading saved surrogate model- sample {a_sample}, repeat {a_repeat}"
            )
            surr = torch.load(surrogate_file).to(device)
            surr.eval()
            surrogate = Surrogate(surr, num_features)

        else:
            # Create surrogate model
            surr = nn.Sequential(
                MaskLayer1d(value=0, append=True),
                nn.Linear(2 * num_features, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 128),
                nn.ELU(inplace=True),
                nn.Linear(128, 2),
            ).to(device)

            # Set up surrogate object
            surrogate = Surrogate(surr, num_features)

            # Train
            print(f"Training surrogate...a_sample={a_sample}, a_repeat={a_repeat}")
            # print(X_train_subset.shape, X_val_torch.shape)
            surrogate.train_original_model(
                X_train_subset,
                X_test,  # X_test_subset,
                original_model,
                batch_size=64,
                max_epochs=200,
                loss_fn=KLDivLoss(),
                validation_samples=10,
                validation_batch_size=10000,
                verbose=False,
            )
            # Save surrogate
            surr.cpu()
            torch.save(surr, surrogate_file)
            surr.to(device)

        # Create explainer model
        exp_file = get_output_path(
            model_args,
            directory=directory,
            filename="explainer",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pt",
        )
        if os.path.isfile(exp_file):
            print(
                f"Loading saved explainer model- sample {a_sample}, repeat {a_repeat}"
            )
            explainer = torch.load(exp_file).to(device)
            explainer.eval()
            fastshap = FastSHAP(
                explainer, surrogate, normalization="additive", link=nn.Softmax(dim=-1)
            )
        else:
            explainer = nn.Sequential(
                nn.Linear(num_features, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2 * num_features),
            ).to(device)

            # Set up FastSHAP object
            fastshap = FastSHAP(
                explainer, surrogate, normalization="additive", link=nn.Softmax(dim=-1)
            )

            # Train
            fastshap.train(
                X_train_subset,
                X_test,  # _subset,
                batch_size=32,
                # batch_size=min(64, X_train_subset.shape[0]),
                num_samples=32,
                # num_samples = X_train_subset.shape[0],
                max_epochs=200,
                validation_samples=10,
                paired_sampling=True,
                verbose=True,
            )
            # Save explainer
            explainer.cpu()
            torch.save(explainer, exp_file)
            explainer.to(device)

        fastshap_list = []
        # for idx_sample in range(X_val.shape[0]):
        #     x = X_val[idx_sample, :].reshape(1, -1)
        for idx_sample in range(X_train.shape[0]):
            x = X_train[idx_sample, :].reshape(1, -1)

            # compute fastshap explanations
            fastshap_values = fastshap.shap_values(x)[0]
            fastshap_list.append(fastshap_values)

        # save the explanations
        fastshap_file = get_output_path(
            model_args,
            directory=directory,
            filename="fastshap",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pkl",
        )

        with open(fastshap_file, "wb") as f:
            pickle.dump(fastshap_list, f)

        # # Setup for KernelSHAP
        # def imputer(x, S):
        #     x = torch.tensor(x, dtype=torch.float32, device=device)
        #     S = torch.tensor(S, dtype=torch.float32, device=device)
        #     pred = surrogate(x, S).softmax(dim=-1)
        #     return pred.cpu().data.numpy()

        # kernelshap_list = []
        # for idx in range(X_val.shape[0]):
        #     game = shapreg.games.PredictionGame(imputer, x)
        #     shap_values, _ = shapreg.shapley.ShapleyRegression(
        #         game,
        #         batch_size=128,
        #         paired_sampling=False,
        #         detect_convergence=True,
        #         bar=False,
        #         return_all=True,
        #     )
        #     kernelshap_list.append(shap_values)

        # kernelshap_file = get_output_path(
        #     smodel_args,
        #     directory=directory,
        #     filename="kernelshap",
        #     extension=f"_{a_sample}",
        #     file_type="pkl",
        # )

        # with open(kernelshap_file, "wb") as f:
        #     pickle.dump(kernelshap_list, f)

        # Calculate the average cosine distance between ts and sv (agreement)
        # dist = []
        # for i in range(ts.shape[0]):
        #     tmp_dist = cosine(ts[i, :], sv_no_bias[i, :])
        #     dist.append(tmp_dist)

        # elapsed = time.time() - start
        # print(
        #     f"{a_frac:.3f}, {data_subset.shape[0]:5d}, {elapsed:.2f} s, {outer_batch_size:5d}, {np.array(dist).mean():.3f}, {max(dist):.3f}, {min(dist):.3f}"
        # )
