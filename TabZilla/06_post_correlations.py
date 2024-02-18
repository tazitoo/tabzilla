"""
We have calculated both spearmans rho and rbo for 23 datasets,  3 model architectures and a handful of metrics. see 05_metric_ablation.py for an example.

Now we want to read the values off disk and plot them as heatmaps.

We will store the values in numpy arrays of shape (23, 5, 3) where
    - the first dimension (horizontal) is the dataset,  (i)
    - the second dimension (vertical) is the metric, and (j)
    - the third dimension (depth) is the model architecture. (k)
This let's us compute the averages and standard deviations for each metric and model architecture across all datasets - 
this will allow a global comparison of the metrics - using a box and whisker plot.

We can also split them out by architecture, and metafeatures of the dataset/explanations.

we can think about the ordering of the datasets according to metafeatures (TBD)

"""

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# set aside storage for the results
SIZE_OF_DATASET = 23
SIZE_OF_METRICS = 5
SIZE_OF_MODELS = 3
rho_array = np.zeros((SIZE_OF_DATASET, SIZE_OF_METRICS, SIZE_OF_MODELS))
rbo_array = np.zeros((SIZE_OF_DATASET, SIZE_OF_METRICS, SIZE_OF_MODELS))

# #  get list of hard datasets
# read in the names of the datasets
with open("../scripts/HARD_DATASETS_BENCHMARK.sh", "r") as f:
    lines = f.readlines()

# clean up header, footer and whitespace/newline from bash script
lines = lines[2:-1]
dataset_names = []
[dataset_names.append(item.strip()) for item in lines]


# dataset_names = dataset_names[0:1]

# get list of model architectures
models = ['XGBoost', 'rtdl_MLP', 'rtdl_ResNet']

# get list of metrics
metric_list = ['ablation']

for i, a_dataset in enumerate(dataset_names):
    for j, a_metric in enumerate(metric_list):
        for k, a_model in enumerate(models):
        
            correlation_file = f"output/{a_model}/{a_dataset}/xai/corelation.txt"
            # read in the correlation file
            with open(correlation_file, "r") as f:
                lines = f.readlines()
            # there are two lines in the file, one for rho and one for rbo of the form:
            # rho: 0.1234
            # rbo: 0.1234
            # we want to extract the float value from each line
            rho = float(lines[0].split(":")[1].strip())
            rbo = float(lines[1].split(":")[1].strip())

            rho_array[i, j, k] = rho
            rbo_array[i, j, k] = rbo


# plot rho and rbo pltos from the arrays.
# first the average across all model types
avg_model_rho = np.mean(rho_array, axis=2)
avg_model_rbo = np.mean(rbo_array, axis=2)

fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(avg_model_rho, annot=True, ax=ax, cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
plt.title("Average Spearman's Rho across all models")
fig.savefig("output/average_spearman_rho.png")

fig, ax = plt.subplots(figsize=(10, 2.5))
sns.heatmap(avg_model_rbo, annot=True, ax=ax, cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
plt.title("Average RBO across all models")
fig.savefig("output/average_rbo.png")

# now let's plot separate plots for each model type
fig, ax = plt.subplots(3, 1, figsize=(10, 7.5))
sns.heatmap(rho_array[:, :, 0], annot=True, ax=ax[0], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[0].set_title("XGBoost Spearman's Rho")
sns.heatmap(rho_array[:, :, 1], annot=True, ax=ax[1], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[1].set_title("MLP Spearman's Rho")
sns.heatmap(rho_array[:, :, 2], annot=True, ax=ax[2], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[2].set_title("ResNet Spearman's Rho")
fig.savefig("output/spearman_rho_by_model.png")

fig, ax = plt.subplots(3, 1, figsize=(10, 7.5))
sns.heatmap(rbo_array[:, :, 0], annot=True, ax=ax[0], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[0].set_title("XGBoost RBO")
sns.heatmap(rbo_array[:, :, 1], annot=True, ax=ax[1], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[1].set_title("MLP RBO")
sns.heatmap(rbo_array[:, :, 2], annot=True, ax=ax[2], cbar=False, cmap="Blues", vmin=0, vmax=1, xticklabels=dataset_names,
            yticklabels=metric_list, fmt=".2f")
ax[2].set_title("ResNet RBO")
fig.savefig("output/rbo_by_model.png")