import datasets
import numpy as np

from models.config import SearchSpace, HyperParameter
from search_algorithms.evolution_search import regularized_evolution
from utils import TargetNetworkType, HyperParameterType

NUMBER_FILTERS = [8, 16, 32, 48, 64]
FILTER_HEIGHT = [3,5,7,9]
FILTER_WIDTH = [3,5,7,9]

EXPERIMENT_ID = "Cifar10_AlexNet"
DATASET = 1
CYCLES = 1000
POPULATION_SIZE = 100
SAMPLE_SIZE = 80
TARGET_NETWORK_TYPE = TargetNetworkType.AlexNet
EPOCHS = 50
BATCH_SIZE = 4056

PRELOAD_POPULATION = True
RANKING_MODE = 2 #1: human , 2: synthetic, 3: None
PREFERENCES = {
    "accuracy" : .5,
    'epoch_training_time': .5,
    #"val_accuracy": 0.5,
    #"gpu_consumption": 1.0
}



"""
Data sets:
            # 1: cifar10
            # 2: cifar100
            # 3: MNIST
            # 4: MNIST-Fashion 
            # 5: stl10 data set
"""

if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = datasets.load_data(DATASET)
    num_classes = datasets.getNumClasses(DATASET)
    x_train_mean = np.mean(X_train, axis=0)
    X_train -= x_train_mean
    X_test -= x_train_mean

    hyper_parameters = []
    hyper_parameters.append(HyperParameter(HyperParameterType.NUMBER_OF_FILTERS,values=NUMBER_FILTERS))
    hyper_parameters.append(HyperParameter(HyperParameterType.FILTER_WIDTH,values=FILTER_WIDTH))
    hyper_parameters.append(HyperParameter(HyperParameterType.FILTER_HEIGHT,values=FILTER_HEIGHT))
    search_space = SearchSpace(hyper_parameters=hyper_parameters)

    regularized_evolution(
        experiment_id=EXPERIMENT_ID,
        cycles=CYCLES,
        population_size=POPULATION_SIZE,
        sample_size = SAMPLE_SIZE,
        target_network_type=TARGET_NETWORK_TYPE,
        search_space=search_space,
        X_train=X_train,
        y_train=y_train,
        X_val=X_test,
        y_val=y_test,
        epochs=EPOCHS,
        num_classes=num_classes,
        batch_size=BATCH_SIZE,
        ranking_mode=RANKING_MODE,
        preferences= PREFERENCES,
        preload_database=PRELOAD_POPULATION)