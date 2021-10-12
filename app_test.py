from optparse import OptionParser
from sklearn.preprocessing import StandardScaler
import utils
import model
import pickle
import simple_models
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--input_data", metavar="FILE", default="data/mfs_data/mfs/x")
    parser.add_option("--label_data", metavar="FILE", default="data/mfs_data/mfs/y")
    parser.add_option("--model_path", type="string", default="data/model")
    parser.add_option("--method", type="string", default="nn_model")

    options, args = parser.parse_args()
    options.k_num = 1
    options.epoch = 1000
    options.batch_size = 1000
    options.random_state = 0
    options.separate_display = True
    options.fig_path = "data/output_figures"
    options.load_model = True
    options.mfs_idx = 16
    options.hidden_dim = 6
    options.lr = 0.001
    options.activation = 'sig'
    options.optim = 'adam'

    scaler = StandardScaler()
    sample_input, input_map = utils.data_mfs_loader(options.input_data, options.mfs_idx, scaler)
    sample_label, test_map = utils.data_loader(options.label_data)
    sample_input, sample_label = utils.process_mfs(sample_input, sample_label)
    k_train_index_list = []
    one_train_index_list = []
    for i in range(len(sample_input)):
        one_train_index_list.append(i)
    k_train_index_list.append(one_train_index_list)
    if options.method == "nn_model":
        # Batchfy training data
        batch_sample_list = utils.construct_batch_data(sample_input, sample_label, options.batch_size)
        input_dim = len(batch_sample_list[0][0][0][0])
        output_dim = len(batch_sample_list[0][0][1][0])

        # Initialize models
        nn_runner = model.NNModelRunner(input_dim, output_dim, options)
        avg_loss, avg_r2 = nn_runner.train(batch_sample_list, k_train_index_list)
        print("Average loss " + str(avg_loss))
        print("Average r2 " + str(avg_r2))
    else:
        model_list = []
        with open(options.model_path + "/models_" + options.method, 'rb') as fr:
            model_list = pickle.load(fr)

        sample_input_label = []
        for i in range(len(sample_input)):
            sample_input_label.append([sample_input[i], sample_label[i]])

        avg_r2 = simple_models.train(sample_input_label, k_train_index_list, model_list, options.model_path,
                                     options.method, options.load_model)
        print("Average train r2 is " + str(avg_r2))


