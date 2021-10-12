from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import utils
import pickle


def train(input_labels, k_train_index_list, model_list, model_path, model_type, load_model):
    r2_list = []
    k_num = len(k_train_index_list)
    for i in range(k_num):
        train_list = []
        train_index_list = k_train_index_list[i]
        for j in range(len(train_index_list)):
            train_list.append(input_labels[train_index_list[j]])
        train_input, train_label = [s[0] for s in train_list], [s[1] for s in train_list]
        train_input = np.vstack(train_input)
        train_label = np.vstack(train_label)
        model = model_list[i]
        if not load_model:
            model.fit(train_input, train_label)
        train_pred = model.predict(train_input)
        r2_train = r2_score(train_label, train_pred)
        r2_list.append(r2_train)
        with open(model_path + "/models_" + model_type, 'wb') as fw:
            pickle.dump(model_list, fw)
    avg_r2 = np.mean(np.array(r2_list))

    return avg_r2


def test(input_labels, k_test_index_list, model_list, fig_path, model_type, separate_display):
    test_r2_list = []
    k_num = len(k_test_index_list)
    for i in range(k_num):
        model = model_list[i]
        test_list = []
        test_index_list = k_test_index_list[i]
        for j in range(len(test_index_list)):
            test_list.append(input_labels[test_index_list[j]])
        test_input, test_label = [s[0] for s in test_list], [s[1] for s in test_list]
        test_input = np.vstack(test_input)
        test_label = np.vstack(test_label)
        test_pred = model.predict(test_input)
        test_r2_score = r2_score(test_label, test_pred)
        if k_num > 1:
            print("r2 score " + str(test_r2_score) + " for model " + str(i))
        save_path = fig_path + "/figure_" + str(i) + "_" + model_type + ".jpg"
        if not separate_display:
            utils.model_plot(test_pred.flatten(), test_label.flatten(), save_path, model_type)
        else:
            for j in range(len(test_label[0])):
                display_label = test_label[:, j]
                display_pred = test_pred[:, j]
                print("Test r2 of output " + str(j) + " for model " + str(j) + " "
                      + str(r2_score(display_label, display_pred)))
                save_path = fig_path + "/figure_" + str(i) + "_" + str(j) + "_output_" \
                            + str(j) + " " + model_type + ".jpg"
                utils.model_plot(display_pred, display_label, save_path, model_type, output_id=j)
        test_r2_list.append(test_r2_score)
    avg_test_r2 = np.mean(np.array(test_r2_list))
    return avg_test_r2


def one_d_train(input_labels, k_train_index_list, model_list):
    r2_list = []
    k_num = len(k_train_index_list)
    for i in range(k_num):
        train_list = []
        train_index_list = k_train_index_list[i]
        for j in range(len(train_index_list)):
            train_list.append(input_labels[train_index_list[j]])
        train_input, train_label = [s[0] for s in train_list], [s[1] for s in train_list]
        train_input = np.vstack(train_input)
        train_label = np.vstack(train_label)
        train_pred_list = []
        for j in range(len(train_label[0])):
            model = model_list[i * k_num + j]
            one_d_train_label = train_label[:, j]
            model.fit(train_input, one_d_train_label)
            print(" Complete training SVR for output " + str(j))
            one_d_train_pred = model.predict(train_input)
            train_pred_list.append(one_d_train_pred)
            one_d_r2 = r2_score(one_d_train_label, one_d_train_pred)
        train_pred = np.vstack(train_pred_list)
        train_pred = np.swapaxes(train_pred, 0, 1)
        r2_train = r2_score(train_label, train_pred)
        r2_list.append(r2_train)
    avg_r2 = np.mean(np.array(r2_list))
    return avg_r2


def one_d_test(input_labels, k_test_index_list, model_list, fig_path, model_type):
    test_r2_list = []
    k_num = len(k_test_index_list)
    for i in range(k_num):
        test_list = []
        test_index_list = k_test_index_list[i]
        for j in range(len(test_index_list)):
            test_list.append(input_labels[test_index_list[j]])
        test_input, test_label = [s[0] for s in test_list], [s[1] for s in test_list]
        test_input = np.vstack(test_input)
        test_label = np.vstack(test_label)
        test_pred_list = []
        for j in range(len(test_label[0])):
            model = model_list[i * k_num + j]
            one_d_test_pred = model.predict(test_input)
            test_pred_list.append(one_d_test_pred)
        test_pred = np.vstack(test_pred_list)
        test_pred = np.swapaxes(test_pred, 0, 1)
        test_r2_score = r2_score(test_label, test_pred)
        if k_num > 1:
            print("r2 score " + str(test_r2_score) + " for model " + str(i))
        save_path = fig_path + "/figure_" + str(i) + " " + model_type + ".jpg"
        utils.model_plot(test_pred.flatten(), test_label.flatten(), save_path, model_type)
        test_r2_list.append(test_r2_score)
        if k_num > 1:
            print("r2 score " + str(test_r2_score) + " for model " + str(i))
        save_path = fig_path + "/figure_" + str(i) + " " + model_type + ".jpg"
        utils.model_plot(test_pred.flatten(), test_label.flatten(), save_path, model_type)
        test_r2_list.append(test_r2_score)
    avg_test_r2 = np.mean(np.array(test_r2_list))
    return avg_test_r2
