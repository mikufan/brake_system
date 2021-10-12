from optparse import OptionParser
import torch
import simple_models
import model
import utils
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import anomaly
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pickle

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--input_data", metavar="FILE", default="data/x")
    parser.add_option("--label_data", metavar="FILE", default="data/y")
    parser.add_option("--batch_size", type="int", default=1000)
    parser.add_option("--hidden_dim", type="int", default=6)
    parser.add_option("--epoch", type="int", default=1200)
    parser.add_option("--optim", type="string", default="adam")
    parser.add_option("--lr", type="float", default=0.001)
    parser.add_option("--activation", type="string", default="sig")
    parser.add_option("--k_num", type="int", default=5)
    parser.add_option("--fig_path", metavar="FILE", default="data/output_figures")
    parser.add_option("--method", type="string", default="nn_model")
    parser.add_option("--alpha", type="float", default=0.1)
    parser.add_option("--check_anomaly", action="store_true", default=False)
    parser.add_option("--threshold", type="float", default=0.0)
    parser.add_option("--n_estimators", type="int", default=10)
    parser.add_option("--random_state", type="int", default=0)
    parser.add_option("--kernel", type="string", default="poly")
    parser.add_option("--scale_data", action="store_true", default=False)
    parser.add_option("--separate_display", action="store_true", default=False)
    parser.add_option("--mfs_data", action="store_true", default=False)
    parser.add_option("--mfs_idx", type="int", default=16)
    parser.add_option("--model_path", type="string", default="data/model")
    parser.add_option("--load_model", action="store_true", default=False)

    options, args = parser.parse_args()

    # Read data
    print("Reading data")
    scaler = None
    if options.scale_data:
        scaler = StandardScaler()
    if not options.mfs_data:
        sample_input, input_map = utils.data_loader(options.input_data, scaler)
    else:
        sample_input, input_map = utils.data_mfs_loader(options.input_data, options.mfs_idx, scaler)
    sample_label, test_map = utils.data_loader(options.label_data)

    if options.mfs_data:
        sample_input, sample_label = utils.process_mfs(sample_input, sample_label)

    if options.check_anomaly:
        anomaly_list = anomaly.check_anomaly(sample_input, sample_label, options.alpha, options.threshold,
                                             input_map)
        sample_input = []
        sample_label = []
        for i in range(len(sample_input)):
            if anomaly_list[i]:
                sample_input.append(sample_input[i])
                sample_label.append(sample_label[i])
        sample_input = np.vstack(sample_input)
        sample_label = np.vstack(sample_label)

    if options.method == "nn_model":
        # Batchfy training data
        batch_sample_list = utils.construct_batch_data(sample_input, sample_label, options.batch_size)
        input_dim = len(batch_sample_list[0][0][0][0])
        output_dim = len(batch_sample_list[0][0][1][0])
        # Split training and test data with k-fold index
        k_train_index_list, k_test_index_list = utils.k_split(batch_sample_list, options.k_num, options.random_state)
        # Initialize models
        nn_runner = model.NNModelRunner(input_dim, output_dim, options)
        if not options.load_model:
            epoch_num = options.epoch
        else:
            epoch_num = 1
        for i in range(epoch_num):
            print("Start epoch " + str(i))
            # Start training
            avg_loss, avg_r2 = nn_runner.train(batch_sample_list, k_train_index_list)
            print("Average loss in this epoch " + str(avg_loss))
            print("Average r2 in this epoch " + str(avg_r2))
            # Test for each model
            avg_test_r2, all_prediction_list, all_label_list = nn_runner.test(batch_sample_list, k_test_index_list)
            print("Test r2 score in this epoch " + str(avg_test_r2))
            # Visualize results
            if i == epoch_num - 1:
                for j in range(options.k_num):
                    torch.save(nn_runner.model_list[j].state_dict(), options.model_path+"/models_nn_model_"+str(j))
                nn_runner.display_results(all_prediction_list, all_label_list, i)

    else:
        model_list = []
        if not options.load_model:
            for i in range(options.k_num):
                if options.method == 'ridge':
                    model_list.append(Ridge(alpha=options.alpha))
                elif options.method == 'rf':
                    model_list.append(RandomForestRegressor(n_estimators=options.n_estimators))
                elif options.method == 'svr':
                    for j in range(len(sample_label[0][0])):
                        model_list.append(SVR(kernel=options.kernel))
        else:
            with open(options.model_path+"/models_"+options.method, 'rb') as fr:
                model_list = pickle.load(fr)

        sample_input_label = []
        for i in range(len(sample_input)):
            sample_input_label.append([sample_input[i], sample_label[i]])
        k_train_index_list, k_test_index_list = utils.k_split(sample_input_label, options.k_num, options.random_state)
        if options.method == 'svr':
            avg_r2 = simple_models.one_d_train(sample_input_label, k_train_index_list, model_list)
        else:
            avg_r2 = simple_models.train(sample_input_label, k_train_index_list, model_list, options.model_path,
                                         options.method, options.load_model)
        print("Average train r2 is " + str(avg_r2))
        if options.method == 'svr':
            avg_test_r2 = simple_models.one_d_test(sample_input_label, k_test_index_list, model_list, options.fig_path,
                                                   options.method)
        else:
            avg_test_r2 = simple_models.test(sample_input_label, k_test_index_list, model_list, options.fig_path,
                                             options.method, options.separate_display)
        print("Average test r2 is " + str(avg_test_r2))
