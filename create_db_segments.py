import itertools

import datasets
from models.config import SearchSpace, HyperParameter
from search_algorithms.evolution_search import regularized_evolution, create_segment
from tqdm import tqdm
import numpy as np

from models.config import ModelConfiguration, SearchSpace
from models.models import AlexNet, TargetNetwork

from utils import TargetNetworkType, HyperParameterType

NUMBER_FILTERS = [8, 16, 32, 48, 64]
FILTER_HEIGHT = [3,5,7,9]
FILTER_WIDTH = [3,5,7,9]

EXPERIMENT_ID = "STL_AlexNet"
DATASET = 5
SAMPLE_SIZE = 1
TARGET_NETWORK_TYPE = TargetNetworkType.AlexNet
EPOCHS = 20
BATCH_SIZE = 4112

if __name__ == '__main__':
    batch_size = BATCH_SIZE
    epochs = EPOCHS
    experiment_id = EXPERIMENT_ID
    (X_train, y_train), (X_val, y_val) = datasets.load_data(DATASET)
    num_classes = datasets.getNumClasses(DATASET)
    input_shape = X_train.shape[1:4]

    # todo why do we use this data preprocessing?
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    X_val -= x_train_mean

    hyper_parameters = []
    hyper_parameters.append(HyperParameter(HyperParameterType.NUMBER_OF_FILTERS, values=NUMBER_FILTERS).get_all)
    hyper_parameters.append(HyperParameter(HyperParameterType.FILTER_WIDTH, values=FILTER_WIDTH).get_all)
    hyper_parameters.append(HyperParameter(HyperParameterType.FILTER_HEIGHT, values=FILTER_HEIGHT).get_all)
    search_space = SearchSpace(hyper_parameters=hyper_parameters)
    config_space = [1, 2]

    model_configs = list(itertools.product(*hyper_parameters))

    part_model_config = []
    for config in model_configs:
        config_dict = dict()
        for hyper_parameter in config:
            config_dict[hyper_parameter[0]] =hyper_parameter[1]
        part_model_config.append(config_dict)

    model_configurations = list(itertools.product(*[part_model_config,part_model_config]))

    model_configurations = [ModelConfiguration(hyper_parameter_list=list(l)) for l in model_configurations]

    for population_count, model_configuration in enumerate(tqdm(model_configurations)):
        if TARGET_NETWORK_TYPE == TargetNetworkType.AlexNet:
            try:
                model = AlexNet(id= population_count, model_configuration=model_configuration, num_classes=num_classes, input_shape=input_shape)
            except Exception as e:
                print(e)
                print("Configuration" +str(population_count)+ " not valid")
        else:
            raise ValueError('target_network_type is not valid')

        history = model.train(X_train=X_train,
                              y_train=y_train,
                              X_val=X_val,
                              y_val=y_val,
                              epochs=epochs,
                              batch_size=batch_size)
        population_count += 1
        create_segment(history=history, model=model, X_val=X_val, y_val=y_val, class_indices=list(range(num_classes))
                                 ,layer_name="conv_2", experiment_id=experiment_id)