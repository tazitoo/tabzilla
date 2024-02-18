import os
import sys

# we have three strings that vary for running 05_metric_ablation.py:
# 1. the dataset name
# 2. the model architecture
# 3. the experiment configuration / yaml file

# TODO: filter out multiclass 

#  get list of hard datasets
# read in the names of the datasets
with open("../scripts/HARD_DATASETS_BENCHMARK.sh", "r") as f:
    lines = f.readlines()

# clean up header, footer and whitespace/newline from bash script
lines = lines[2:-1]
dataset_names = []
[dataset_names.append(item.strip()) for item in lines]

for i, a_dataset in enumerate(dataset_names):
    if
    print(i, a_dataset)
sys.exit()


dataset_names = dataset_names[0:6]

# get list of model architectures
models = ['XGBoost', 'rtdl_MLP', 'rtdl_ResNet']

for i, a_dataset in enumerate(dataset_names):
    for a_model in models:
        # get the yaml file
        # patch in substitute model name
        if a_model == 'XGBoost':
            yaml_file = "model_config/30_random_trials_xgb.yml"
        elif a_model == 'rtdl_ResNet':
            yaml_file = "model_config/30_random_trials_ResNet.yml"
        elif a_model == 'rtdl_MLP':
            yaml_file = "model_config/30_random_trials_MLP.yml"

        # run the script
        print(i, a_dataset, a_model)
        os.system(f"python 05_metric_ablation.py --dataset_dir datasets/{a_dataset} --model_name {a_model} --experiment_config {yaml_file}")