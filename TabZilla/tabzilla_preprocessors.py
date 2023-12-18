'''
This file contains the preprocessor functions for the local (think SD) datasets.

Each preprocessor function must be decorated with the @dataset_preprocessor decorator.  The decorator takes two arguments:
    - preprocessor_dict: a dictionary of preprocessor functions.  This is used to build the preprocessors dictionary in tabzilla_data_preprocessing.py
    - dataset_name: the name of the dataset.  This is used to build the dataset_name list in tabzilla_data_preprocessing.py

The preprocessor function must return a dictionary with the following keys:
    - X: the data matrix
    - y: the target vector
    - cat_idx: a list of indices of the categorical and binary features. all other features are assumed to be numerical.
    - target_type: can be "binary", "classification", or "regression
    - num_classes: is '1', or ">2" (...so off by one compared to my instinct)
'''
import functools

import numpy as np
import openml
import pandas as pd
from tabzilla_preprocessor_utils import dataset_preprocessor

preprocessor_dict_local = {}


@dataset_preprocessor(preprocessor_dict_local, "ExampleDataset", target_encode=True)
def preprocess_foo():

    n_samples = 15
    n_features = 2
    n_classes = 2
    X = np.random.rand(n_samples,n_features)
    y = np.random.randint(0, high=n_classes, size=n_samples, dtype=int)


    # a list of indices of the categorical and binary features. all other features are assumed to be numerical.
    cat_idx = []

    # can be "binary", "classification", or "regression
    # num_classes is '1', or ">2" (...so off by one compared to my instinct)
    target_type = "binary"
    

    return {
        "X": X,
        "y": y,
        "cat_idx": [],
        "target_type": "binary",
        "num_classes": 1,
    }
