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
import argparse
import pickle
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from fastshap import KernelExplainer
from scipy.spatial.distance import cosine
from sklearn.model_selection import train_test_split

# from shap import KernelExplainer as ke
# from models.basemodel import BaseModel
# from models.tree_models import XGBoost
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
y_test = dataset.y[test_idx]
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

# explain the (subsetted) validation data
idx_val = np.random.randint(0, X_test.shape[0], max(int(0.01 * X_test.shape[0]), 1))
val_subset = X_val[idx_val, :]

ts = my_model.attribute(val_subset, remove_bias=True)


# load the XAI method and train explainer
# fractions = [0.25, 0.5, 0.75, 1.0]
fractions = [0.001, 0.005, 0.01, 0.1, 0.2, 0.3, 0.4]
for a_frac in fractions:
    start = time.time()
    # subset the training data for use as a background dataset for KernelSHAP
    _, data_subset, _, _ = train_test_split(
        X_train,
        y_train,
        test_size=a_frac,
        random_state=42,
        stratify=y_train,
    )

    # train the explainer
    ke = KernelExplainer(my_model.alt_predict_proba, data_subset)
    ke.stratify_background_set(1)
    # set batch size we could use
    # outer_batch_size = 256  # min(data_subset.shape[0], 5000) uses 4 cpus at 0.02...<grumble>
    outer_batch_size = data_subset.shape[0]  # min(data_subset.shape[0], 5000)

    sv = ke.calculate_shap_values(
        val_subset,
        outer_batch_size=outer_batch_size,
        inner_batch_size=None,
        n_coalition_sizes=5,
        background_fold_to_use=None,
        verbose=False,
    )

    # remove bias from KernelSHAP explanations
    sv_no_bias = sv[:, :-1, 1]

    # Calculate the average cosine distance between ts and sv (agreement)
    dist = []
    for i in range(ts.shape[0]):
        tmp_dist = cosine(ts[i, :], sv_no_bias[i, :])
        dist.append(tmp_dist)

    elapsed = time.time() - start
    print(
        f"{a_frac:.3f}, {data_subset.shape[0]}, {elapsed:.2f} s, {outer_batch_size}, {np.array(dist).mean():.5f}, {max(dist):.5f}, {min(dist):.4f}"
    )
