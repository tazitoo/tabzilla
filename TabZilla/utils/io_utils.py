import datetime
import json
import os
import pickle
import pprint

import numpy as np

output_dir = "output/"

def get_sample_list(x):
    """
    We'd like to select samples sizes that are powers of 2 for KernelSHAP
    to demonstrate the convergence properties of the algorithm.
    """
    n = x.shape[0]
    max_exp = int(np.floor(np.log2(n)))
    min_exp = max(max_exp - 10, 2)
    sample_list = [2**i for i in range(min_exp, max_exp + 1)]

    if len(sample_list) < 10:
        sample_list.append(x.shape[0])  # if it's a short dataset, let's use all samples
    return sample_list

def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(
        args, directory="logging", filename=name, extension=extension, file_type="txt"
    )
    np.savetxt(filename, arr)


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(
        args,
        directory="predictions",
        filename="p",
        extension=extension,
        file_type="npy",
    )
    np.save(filename, arr)


def save_model_to_file(model, args, extension="", file_type="pkl"):
    # filename = get_output_path(
    #     args,
    #     "m",
    #     file_type,
    #     directory="models",
    #     extension=extension,
    # )

    if hasattr(model, "save_model"):
        try:
            model.save_model(extension=extension)
        # except:
        except Exception as e:
            print(e)
            # print(f"error saving model - {filename}")
    else:
        filename = get_output_path(
            args,
            directory="models",
            filename="m",
            extension=extension,
            file_type="pkl",
        )
        # print("DeBUG 55 - in save_model--- ", filename)

        with open(filename, "wb") as f:
            pickle.dump(model, f)


def load_model_from_file(model, args, extension="", file_type="pkl"):
    filename = get_output_path(
        args, directory="models", filename="m", extension=extension, file_type="pkl"
    )
    with open(filename, "rb") as f:
        x = pickle.load(f)
    return x


def save_results_to_json_file(args, jsondict, resultsname, append=True):
    """Write the results to a json file.
    jsondict: A dictionary with results that will be serialized.
    If append=True, the results will be appended to the original file.
    If not, they will be overwritten if the file already exists.
    """
    filename = get_output_path(args, filename=resultsname, file_type="json")
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"))


def save_results_to_file(
    args, results, train_time=None, test_time=None, best_params=None
):
    filename = get_output_path(args, filename="results", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        # we have a list of scores, so it would need to be avg, or something else...
        for key, value in results.items():
            #     text_file.write("%s: %.5f\n" % (key, value))

            # output_s = pprint.pformat(results)
            # print("DEBUG --- ")
            # print(filename)
            # print(output_s)
            # text_file.write(output_s)
            text_file.write(f"{key}: {value}\n")

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)


def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)

        for key, value in results.items():
            text_file.write("%s: %.5f\n" % (key, value))

        if time:
            text_file.write("\nTrain time: %f\n" % time[0])
            text_file.write("Test time: %f\n" % time[1])

        text_file.write("\n---------------------------------------\n")


def get_output_path(args, filename, file_type, directory=None, extension=None):
    # For example: output/LinearModel/Covertype

    # dataset_name = str.split(args.dataset_dir, "/")[1]
    # dir_path = output_dir + args.model_name + "/" + dataset_name

    dir_path = output_dir + args.model_name + "/" + args.dataset

    if directory:
        # For example: .../models
        dir_path = dir_path + "/" + directory

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_path = dir_path + "/" + filename

    if extension is not None:
        # file_path += "_" + str(extension)
        file_path = file_path + "_" + str(extension)

    if file_type is not None:
        # file_path += "." + file_type
        file_path = file_path + "." + file_type

    # For example: .../m_3.pkl

    return file_path


def get_predictions_from_file(args):
    dir_path = output_dir + args.model_name + "/" + args.dataset + "/predictions"

    files = os.listdir(dir_path)
    content = []

    for file in files:
        content.append(np.load(dir_path + "/" + file))

    return content
