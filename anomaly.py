from sklearn import svm
import utils
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np


def anomaly_detect_ridge(input_train, label_train, input_test, label_test, alpha, score_threshold):
    ridge = Ridge(alpha=alpha)
    ridge.fit(input_train, label_train)
    score = ridge.score(input_test, label_test)
    print("Score in this file " + str(score))
    if score > score_threshold:
        sample_state = True
    else:
        sample_state = False
    return sample_state


def check_anomaly(input_data, label_data, alpha, threshold, file_map):
    anomaly_list = []
    anomaly_num = 0
    for file_id in range(len(input_data)):
        train_file_input = input_data.copy()
        train_file_input.pop(file_id)
        train_file_input = np.vstack(train_file_input)
        train_file_label = label_data.copy()
        train_file_label.pop(file_id)
        train_file_label = np.vstack(train_file_label)
        train_file_label = train_file_label[:, 1:]
        test_file_input = input_data[file_id]
        test_file_label = label_data[file_id]
        test_file_label = test_file_label[:, 1:]
        state = anomaly_detect_ridge(train_file_input, train_file_label, test_file_input, test_file_label, alpha,
                                     threshold)
        anomaly_list.append(state)
        if state is False:
            anomaly_num += 1
            file = file_map[file_id]
            print("Anomaly found in file " + file)
    print("There are " + str(anomaly_num) + " anomalies with threshold " + str(threshold))
    return anomaly_list


def set_anomaly(input_data):
    anomaly_list = []
    return anomaly_list
