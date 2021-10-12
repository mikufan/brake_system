import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit
from scipy.stats import gaussian_kde

import random


def data_loader(dir_path, scaler=None):
    files = os.listdir(dir_path)
    sample_list = []
    file_map = {}
    file_id = 0
    scale_mean, scale_std = 0, 0
    for file in files:
        file_path = dir_path + '/' + file
        try:
            lines = pd.read_csv(file_path).values
            file_map[file_id] = file
            file_id += 1
        except pd.errors.EmptyDataError:
            lines = np.array([])
        if len(lines) > 0:
            sample_list.append(lines)
    if scaler is None:
        samples = sample_list
    else:
        samples = []
        all_samples = np.vstack(sample_list)
        scaler.fit(all_samples)
        for i in range(len(sample_list)):
            one_file_samples = scaler.transform(sample_list[i])
            samples.append(one_file_samples)
    return samples, file_map


def data_mfs_loader(dir_path, mfs_idx, scaler=None):
    files = os.listdir(dir_path)
    sample_list = []
    file_map = {}
    file_id = 0
    for file in files:
        file_path = dir_path + '/' + file
        try:
            lines = pd.read_csv(file_path).values
            file_map[file_id] = file
            file_id += 1
        except pd.errors.EmptyDataError:
            lines = np.array([])
        if len(lines) > 0:
            new_lines = lines[:, :mfs_idx]
            mfs_lines = lines[:, mfs_idx:]
            mfs_lines = np.sum(mfs_lines, axis=1).reshape(len(mfs_lines), 1)
            new_lines = np.concatenate((new_lines, mfs_lines), axis=1)
            sample_list.append(new_lines)
    if scaler is None:
        samples = sample_list
    else:
        samples = []
        all_samples = np.vstack(sample_list)
        scaler.fit(all_samples)
        for i in range(len(sample_list)):
            one_file_samples = scaler.transform(sample_list[i])
            samples.append(one_file_samples)
    return samples, file_map


def construct_batch_data(input_list, label_list, batch_size):
    batch_data = []
    group_data = []
    for i in range(len(input_list)):
        group_data.append([input_list[i], label_list[i]])
    for i in range(len(group_data)):
        one_file_batch_data = []
        one_file_data = group_data[i]
        num_batch = len(one_file_data[0]) // batch_size
        if not len(one_file_data[0]) % batch_size == 0:
            num_batch += 1
        for j in range(num_batch):
            start_idx = j * batch_size
            end_idx = min(len(one_file_data[0]), start_idx + batch_size)
            one_file_batch_data.append([one_file_data[0][start_idx:end_idx], one_file_data[1][start_idx:end_idx]])
        batch_data.append(one_file_batch_data)
    return batch_data


def write_file(write_list, file):
    f = open(file, 'w')
    for i in write_list:
        f.write(str(i))
        f.write("\n")


def fig_drawing(draw_list, save_path):
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1)
    for i, d in enumerate(draw_list):
        c = 'red'
        if i % 3 == 1:
            c = 'blue'
        elif i % 3 == 2:
            c = 'green'
        d = np.array(d)
        ax.plot(d, label=str(i), color=c)
    plt.savefig(save_path)
    return


def k_split(sample_list, k_num, random_state):
    k_train_index_list = []
    k_test_index_list = []
    if k_num > 1:
        kf = KFold(n_splits=k_num, shuffle=True, random_state=random_state)
        for train, test in kf.split(sample_list):
            k_train_index_list.append(train)
            k_test_index_list.append(test)
    else:
        rs = ShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
        for train_index, test_index in rs.split(sample_list):
            k_train_index_list = [train_index]
            k_test_index_list = [test_index]
    return k_train_index_list, k_test_index_list


def process_mfs(sample_input, sample_label):
    new_sample_input = []
    new_sample_label = []
    length = len(sample_label[0][0])
    for i in range(len(sample_label)):
        one_file_sample_input = sample_input[i]
        one_file_label_input = sample_label[i][:, length - 2:]
        one_file_sample_label = sample_label[i][:, :length - 2]
        #one_file_sample_input = one_file_sample_input[:, :10]
        one_file_sample_input = np.concatenate((one_file_sample_input, one_file_label_input), axis=1)
        new_sample_input.append(one_file_sample_input)
        new_sample_label.append(one_file_sample_label)
    return new_sample_input, new_sample_label


def model_plot(prediction, label, save_path, model_type, output_id=-1):
    model_name = ""
    if model_type == "nn_model":
        model_name = "MLP neural network"
    elif model_type == "ridge":
        model_name = "ridge regression"
    elif model_type == "rf":
        model_name = "random forest"
    fig = plt.figure(figsize=(10, 5))
    xy = np.vstack((label, prediction))
    z = gaussian_kde(xy)(xy)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(label, prediction, label='predictions', s=0.5, c=z, cmap='jet')
    ax.plot(label, label, label='labels', color='red')
    if output_id >= 0:
        plt.gca().set(aspect='auto', title="2-dimensional predictions vs labels plot of output " + str(output_id) +
                                           " for " + model_name, xlabel="labels", ylabel="predictions")
    else:
        plt.gca().set(aspect='auto', title="2-dimensional predictions vs labels plot for " + model_name,
                      xlabel="labels", ylabel="predictions")
    plt.legend()
    plt.savefig(save_path)

    return


if __name__ == '__main__':
    data = 'data/x'
    data_loader(data)
