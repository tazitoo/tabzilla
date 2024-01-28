"""
loop through `repeats` of explanations and compare the average agreement between them 
sample sizes are powers of 2, with max based on the size of the training set
uses: avg cosine similarity 
all subsamples are compared to the max sample size
output - text to the screen and a plot

"""
import argparse
import os
import pickle
import sys
import time
from collections import namedtuple
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning import seed_everything
from scipy.spatial.distance import cosine

from tabzilla_alg_handler import ALL_MODELS  # , get_model
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

# parser.add_argument(
#     "--xai_config",
#     required=True,
#     type=str,
#     # choices=ALL_MODELS,
#     help="config file for the XAI method",
# )

args = parser.parse_args()
# args.use_gpu = False
print(f"ARGS: {args}")


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

X_train = dataset.X[train_idx, :]
y_train = dataset.y[train_idx]
X_test = dataset.X[test_idx, :]
y_test = dataset.y[test_idx]
X_val = dataset.X[val_idx, :]

# fix object type for X_train
X_train = np.array(X_train, dtype=float)
X_test = np.array(X_test, dtype=float)
X_val = np.array(X_val, dtype=float)

# cast numpy to torch
# X_train_torch = torch.tensor(X_train).float().cuda()
# X_test_torch = torch.tensor(X_test).float().cuda()
# X_val_torch = torch.tensor(X_val).float().cuda()


#def get_sample_list(x):
#    """
#    We'd like to select samples sizes that are powers of 2 for KernelSHAP
#    to demonstrate the convergence properties of the algorithm.
#    """
#    n = x.shape[0]
#    max_exp = int(np.floor(np.log2(n)))
#    min_exp = max(max_exp - 10, 2)
#    sample_list = [2**i for i in range(min_exp, max_exp + 1)]
#    return sample_list
#

# # Set up original model - pytorch
# def original_model_pytorch(x):
#     """
#     for binary classification, the model should output a 2-vector of probabilities

#     TODO: we are wasting time moving data from GPU to CPU and back again...
#     """
#     pred = my_model.predict_proba(x.cpu().numpy())
#     # pred = my_model.predict_proba(x)
#     # pred = np.stack([1 - pred, pred]).T  # if the model is just outputting the probability of the positive class
#     return torch.tensor(pred, dtype=torch.float32, device=x.device)


# # XGBoost version
# def original_model_xgb(x):
#     """
#     for binary classification, the model should output a 2-vector of probabilities

#     x - torch tensor, needs to be brought back to CPU for xgb model
#     """
#     pred = my_model.alt_predict_proba(x.cpu().numpy())
#     # pred = my_model.predict_proba(x)
#     # pred = np.stack([1 - pred, pred]).T  # if the model is just outputting the probability of the positive class
#     return torch.tensor(pred, dtype=torch.float32, device=x.device)


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


# load out the "best" explanations as well
tree_shap_file = get_output_path(
    model_args,
    directory=directory,
    filename="tree_shap",
    extension="",
    file_type="pkl",
)
with open(tree_shap_file, "rb") as f:
    ts = pickle.load(f)

num_features = X_train.shape[1]
sample_list = get_sample_list(X_train)
max_sample = max(sample_list)


# for a_sample in sample_list:
#     dist = []
#     for a_repeat in range(repeats):
#         # start = time.time()

#         # load the explanations
#         fastshap_file = get_output_path(
#             model_args,
#             directory=directory,
#             filename="fastshap",
#             extension=f"sample_{a_sample}_repeat_{a_repeat}",
#             file_type="pkl",
#         )

#         with open(fastshap_file, "rb") as f:
#             fastshap_list = pickle.load(f)

#         # Calculate the average cosine distance between ts and sv (agreement)

#         for i in range(ts.shape[0]):
#             tmp_dist = cosine(ts[i, :], fastshap_list[i][:, 1])
#             dist.append(tmp_dist)

#     # elapsed = time.time() - start
#     print(
#         f"{a_sample:5d}, {len(dist):3d}  {np.array(dist).mean():.3f}, {np.array(dist).std():.3f}"
#     )


# load the explanations


# Calculate the average cosine distance between best_shap (using max_samples) and fast_shap (agreement)
avg_cosine_list = []
avg_cosine_std = []
for a_sample in sample_list:
    dist = []
    for a_repeat in range(repeats):
        # start = time.time()

        # load the explanations
        fastshap_file = get_output_path(
            model_args,
            directory=directory,
            filename="fastshap",
            extension=f"sample_{a_sample}_repeat_{a_repeat}",
            file_type="pkl",
        )

        with open(fastshap_file, "rb") as f:
            fastshap_list = pickle.load(f)

        # Calculate the average cosine similarity between ts and sv (agreement)
        for another_repeat in range(repeats):
            best_shap_file = get_output_path(
                model_args,
                directory=directory,
                filename="fastshap",
                extension=f"sample_{max_sample}_repeat_{another_repeat}",
                file_type="pkl",
            )
            with open(best_shap_file, "rb") as f:
                best_shap_list = pickle.load(f)

            for i in range(ts.shape[0]):
                tmp_dist = 1 - cosine(best_shap_list[i][:, 1], fastshap_list[i][:, 1])
                # tmp_dist = 1 - cosine(ts[i, :], fastTshap_list[i][:, 1])

                dist.append(tmp_dist)

    avg_cosine_list.append(np.array(dist).mean())
    avg_cosine_std.append(np.array(dist).std())
    print(
        f"{a_sample:5d}, {len(dist):3d}  {np.array(dist).mean():.3f}, {np.array(dist).std():.3f}"
    )

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(sample_list, avg_cosine_list, "o-")
ax.errorbar(sample_list, avg_cosine_list, avg_cosine_std, linestyle=None, color='grey', capsize=3, marker="o")

ax.set_xlabel("Number of samples")
ax.set_ylabel("Average cosine similarity")
ax.set_xscale("log")
ax.grid(True)
plt.show()
dataset_str = args.dataset_dir.split("/")[1]
fig_file = get_output_path(
    model_args,
    directory=directory,
    filename="avg_cos_sim",
    extension=f"{args.model_name}_{dataset_str}",
    file_type="png",
)
fig.savefig(fig_file, bbox_inches="tight")
