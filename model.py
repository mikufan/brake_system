import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from sklearn.metrics import r2_score
import utils


class NNModel(nn.Module):
    def __init__(self, input_dim, target_dim, options):
        super(NNModel, self).__init__()

        # Parameter setting
        self.input_dim = input_dim
        self.output_dim = target_dim
        self.hidden_dim = options.hidden_dim
        self.lr = options.lr
        self.activation = options.activation

        self.hidden = nn.Linear(self.input_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

        self.hidden_2 = nn.Linear(self.hidden_dim, self.hidden_dim)

        if options.optim == 'adam':
            self.trainer = optim.Adam(self.parameters(), self.lr)
        else:
            self.trainer = optim.SGD(self.parameters(), self.lr)

    def forward(self, input_v, target_v, is_test=True):
        # Transform to tensors
        input_t = torch.FloatTensor(input_v)
        target_t = torch.FloatTensor(target_v)

        # Through hidden layers
        hidden_t = self.hidden(input_t)
        if self.activation == "sig":
            hidden_t = torch.sigmoid(hidden_t)
        elif self.activation == "tanh":
            hidden_t = torch.tanh(hidden_t)
        else:
            hidden_t = F.relu(hidden_t)
        hidden_t = self.hidden_2(hidden_t)
        hidden_t = F.relu(hidden_t)
        output_t = self.output_layer(hidden_t)

        # Loss computation
        loss_f = nn.MSELoss()
        loss = loss_f(output_t, target_t)
        if not is_test:
            return loss, output_t.detach().numpy()
        else:
            return loss.detach().numpy(), output_t.detach().numpy()


class NNModelRunner(object):
    def __init__(self, input_dim, output_dim, options):
        self.model_list = []
        self.batch_size = options.batch_size
        self.k_num = options.k_num
        self.epoch = options.epoch

        self.separate_display = options.separate_display
        self.fig_path = options.fig_path

        self.model_path = options.model_path
        self.load_model = options.load_model

        for i in range(self.k_num):
            self.model_list.append(NNModel(input_dim, output_dim, options))

    def train(self, batch_sample_list, k_train_index_list):
        loss_value_list = []
        r2_list = []

        # Training for each model
        for i in range(self.k_num):
            one_model = self.model_list[i]
            if self.load_model:
                one_model.load_state_dict(torch.load(self.model_path+"/models_nn_model_"+str(i)))
            one_model.train()
            loss_value = 0.0
            batch_train_list = []
            train_index_list = k_train_index_list[i]

            for j in range(len(train_index_list)):
                one_batch_one_file = batch_sample_list[train_index_list[j]]
                for s in one_batch_one_file:
                    batch_train_list.append(s)
            batch_total = len(batch_train_list)
            train_label = []
            train_pred = []
            # Compute loss for each batch

            for one_batch in batch_train_list:
                one_model.trainer.zero_grad()
                batch_input, batch_label = one_batch[0], one_batch[1]
                loss, pred = one_model.forward(batch_input, batch_label, False)
                loss_value += loss
                # Update model parameters
                if not self.load_model:
                    loss.backward()
                    one_model.trainer.step()
                train_pred.append(pred)
                train_label.append(batch_label)
            train_pred = np.vstack(train_pred)
            train_label = np.vstack(train_label)
            r2_train = r2_score(train_label, train_pred)
            r2_list.append(r2_train)
            # print("Loss in this epoch " + str(loss_value.detach().numpy() / batch_total) + " for model " + str(i))
            # print("Training r2 in this epoch " + str(r2_train) + " for model " + str(i))
            loss_value_list.append(loss_value.detach().numpy() / batch_total)
        avg_loss = np.mean(np.array(loss_value_list))
        avg_r2 = np.mean(np.array(r2_list))
        return avg_loss, avg_r2

    def test(self, batch_sample_list, k_test_index_list):
        test_r2_list = []
        all_prediction_list = []
        all_label_list = []
        for i in range(self.k_num):
            one_model = self.model_list[i]
            one_model.eval()
            batch_test_list = []
            test_index_list = k_test_index_list[i]
            for j in range(len(test_index_list)):
                one_batch_one_file = batch_sample_list[test_index_list[j]]
                for s in one_batch_one_file:
                    batch_test_list.append(s)
            label_list = []
            prediction_list = []
            for one_batch in batch_test_list:
                batch_input, batch_label = one_batch[0], one_batch[1]
                mse, prediction = one_model.forward(batch_input, batch_label)
                prediction_list.append(prediction)
                label_list.append(batch_label)
            prediction_list = np.vstack(prediction_list)
            label_list = np.vstack(label_list)
            test_r2 = r2_score(label_list, prediction_list)
            test_r2_list.append(test_r2)
            all_prediction_list.append(prediction_list)
            all_label_list.append(label_list)
            # print("r2 score in this epoch " + str(test_r2) + " for model " + str(i))
        avg_test_r2 = np.mean(np.array(test_r2_list))
        return avg_test_r2, all_prediction_list, all_label_list

    def display_results(self, prediction_list, label_list, epoch):
        model_type = 'nn_model'
        for i in range(self.k_num):
            if not self.separate_display:
                print("Test r2 for model " + str(i) + " " + str(r2_score(label_list[i], prediction_list[i])))
                save_path = self.fig_path + "/figure_nn_" + str(epoch) + "_" + str(i) + ".jpg"
                utils.model_plot(prediction_list[i].flatten(), label_list[i].flatten(), save_path, model_type)
            else:
                for j in range(len(prediction_list[0][0])):
                    display_label_list = label_list[i]
                    display_label = display_label_list[:, j]
                    display_prediction_list = prediction_list[i]
                    display_prediction = display_prediction_list[:, j]
                    print("Test r2 of output " + str(j) + " for model " + str(i) + " "
                          + str(r2_score(display_label, display_prediction)))
                    save_path = self.fig_path + "/figure_nn_" + str(i) + "_" + str(j) + "_output_" \
                                + str(j) + ".jpg"
                    #utils.model_plot(display_prediction, display_label, save_path, model_type, output_id=j)
        return
