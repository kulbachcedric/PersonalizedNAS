import collections
from time import sleep
from typing import Dict

from human_app.views import get_rated_comparisons, get_not_rated_comparisons, get_segments
from tqdm import tqdm

import datasets
import numpy as np

from models.config import SearchSpace, HyperParameter, ModelConfiguration
from models.models import AlexNet
from preference_controller.preference_controller import PreferenceController
from ranking.ranknet.ranknet import RanknetScorer
from ranking.ranknet.ranknet2 import RanknetScorer2
from ranking.synthetic.agent import LinearAgent
from search_algorithms.evolution_search import regularized_evolution, segments_to_df, create_segment
from utils import TargetNetworkType, HyperParameterType

NUMBER_FILTERS = [8, 16, 32, 48, 64]
FILTER_HEIGHT = [3,5,7,9]
FILTER_WIDTH = [3,5,7,9]

EXPERIMENT_ID = "Cifar100_AlexNet"
DATASET = 2
CYCLES = 5
POPULATION_SIZE = 100
SAMPLE_SIZE = 1
TARGET_NETWORK_TYPE = TargetNetworkType.AlexNet
EPOCHS = 1
BATCH_SIZE = 32

PRELOAD_POPULATION = True
RANKING_MODE = 3 #1: human , 2: synthetic, 3: None
PREFERENCES = {
    #"accuracy" : 0.5,
    'epoch_training_time': 0.5,
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





def ranker_test(experiment_id:str,
                        training_samples:int,
                          cycles:int,
                          population_size:int,
                          sample_size:int,
                          target_network_type: TargetNetworkType,
                          search_space:SearchSpace,
                          X_train,
                          y_train,
                          X_val,
                          y_val,
                          num_classes:int,
                          epochs:int,
                          batch_size:int,
                          ranking_mode:int,
                          preferences: Dict,
                          preload_database:bool,
                          comparison_validation_split:int= .2):

    population = collections.deque()
    evaluation_history = []
    population_count = 1


    segments = get_segments(experiment_name=experiment_id)
    agent = LinearAgent(segments=segments[:100], features=list(segments[0].features.all()), preferences=preferences)

    if ranking_mode ==2:
        comparisons = PreferenceController.create_comparisons(segments=segments[:100], amount=700, experiment_id=experiment_id)
        assert preferences is not None
        [agent.score_comparison(comp) for comp in tqdm(comparisons)]  # scoring comparisons on agent based preferences

    comparisons = get_rated_comparisons(experiment_name=experiment_id)

    idx = 200#int(comparison_validation_split * len(comparisons))
    test_comparisons = comparisons[:idx]
    train_comparisons = comparisons[idx:idx + training_samples]

    ranker = RanknetScorer2(comparisons=train_comparisons, validation_comparisons=test_comparisons)
    return ranker.validation_accuracy


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

    training_samples = [5,10,50,100,250,500]

    res= []
    comparisons = get_rated_comparisons(experiment_name=EXPERIMENT_ID)
    for comparison in comparisons:
        comparison.delete()

    rank = True
    for training_sample in training_samples:
        if rank:
            mode = 2
        else:
            mode = 3
        val_acc = ranker_test(
            training_samples=training_sample,
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
            ranking_mode=mode,
            preferences= PREFERENCES,
            preload_database=PRELOAD_POPULATION)
        res.append(val_acc)
        rank = False

    print("===========")
    print(res)




