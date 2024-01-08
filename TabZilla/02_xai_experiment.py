"""
This script is used to run an XAI experiment on a given dataset and model.  An experiment
is intended to create a series of local attributions of varying faithfulness.

The prototype experiment is to use kernelSHAP with increasing number of n_samples to better
represent the marginal distributions of the inputs (10% of the test set is low, 100% of the test
set is best - it should agree with DeepSHAP or treeSHAP).

Directory strucuture is:
output
    model
        dataset
            xai method
                trial

The experiment is defined by the XAI config file, which is a yaml file with the following structure:
    name - name of the method (will also be the directory name where attributions are stored)
    seed - for reproducibility
    trial - number of times to repeat the xai loop to account for random effects
    subsample_percentage - 1.0 (default)
    fractions - [0.25, 0.5, 0.75, 1.0] (default) - how to split the baseline/samples
    overwrite - whether to overwrite the directory if it (already) exists
"""
import argparse
import logging
import os
import sys
import traceback
from collections import namedtuple
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from models.basemodel import BaseModel
from tabzilla_alg_handler import ALL_MODELS, get_model
from tabzilla_datasets import TabularDataset
from tabzilla_utils import (  # ExperimentResult,; cross_validation,
    get_experiment_parser,
    get_scorer,
)


def load_dataset(dataset_dir):
    """
    Load X, y, and indices from the given dataset directory.

    Parameters:
    dataset_dir (str): The directory where the dataset files are stored.

    Returns:
    X (numpy.ndarray): The feature matrix.
    y (numpy.ndarray): The target vector.
    indices (numpy.ndarray): The indices of the data points.
    """
    X = np.load(os.path.join(dataset_dir, "X.npy.gz"))
    y = np.load(os.path.join(dataset_dir, "y.npy.gz"))
    indices = np.load(os.path.join(dataset_dir, "split_indeces.npy.gz"))

    print(X.shape, y.shape, indices.shape)
    return X, y, indices


def main(model_name, dataset_dir, xai_config, xai_method):
    # read dataset from folder
    dataset = TabularDataset.read(Path(dataset_dir).resolve())

    model_handle = get_model(model_name)

    # output directory parent should exist - we just need to create the directory for this XAI method
    dataset_root = dataset_dir.split("/")[1]
    output_parent = Path("output/{model_name}/{dataset_root}/").resolve()
    if output_parent.is_file():
        output_path = output_parent + xai_method
        if output_path.is_file() and xai_config["overwrite"] is True:
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            print(
                "XAI directory exists - not overwriting, stopping.  Change XAI config to continue."
            )
            sys.exit()

    # split X

    # make X work with the model (e.g. dmatrix)

    # loop over fractions
    # split X according to fraction
    # train explainer (new class?  with BaseExplainer & subclasses?)
    # make local attributions
    # save attributions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="parser for using a dataset and model to create local explanations"
    )

    # parser.add_argument(
    #     "--experiment_config",
    #     required=True,
    #     type=str,
    #     help="config file for parameter experiment args",
    # )

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
    print(f"ARGS: {args}")

    # load XAI config
    config = yaml.safe_load(open(args.xai_config, "r"))
    print(config)

    xai_method = args.xai_config.split("/")[1].split(".")[0]

    # now parse the dataset and search config files
    # experiment_parser = get_experiment_parser()

    # experiment_args = experiment_parser.parse_args(
    # args="-experiment_config " + args.experiment_config
    # )
    print(f"CURRENT {args}")

    main(args.model_name, args.dataset_dir, config, xai_method)
