import base64
import collections
import random
from time import sleep
from typing import List, Dict
import cv2
import numpy as np
from django.core.files import File
from human_app.views import get_not_rated_comparisons, get_rated_comparisons, get_segments
from tf_explain.utils.display import heatmap_display, grid_display
from tqdm import tqdm
import pandas as pd

from models.config import ModelConfiguration, SearchSpace
from models.models import AlexNet, TargetNetwork
from human_app.models import Image, Segment, Feature

from preference_controller.preference_controller import PreferenceController
from ranking.ranknet.ranknet import RanknetScorer
from ranking.ranknet.ranknet2 import RanknetScorer2
from ranking.synthetic.agent import LinearAgent
from utils import TargetNetworkType, HyperParameterType, GradCAM


def create_features(history, segment:Segment, model_parameter_config:ModelConfiguration, save=True):
    for key in history.history:
        # todo if loss return min
        val = max(history.history[key])
        f = Feature(name=key, value=val, segment=segment)
        if save:
            f.save()
    for idx, hyper_parameter_dict in enumerate(model_parameter_config.hyper_parameter_list):
        for key in hyper_parameter_dict:
            val = hyper_parameter_dict[key]
            f = Feature(name = str(key)+"_"+str(idx), value = val, segment=segment)
            if save:
                f.save()


def create_images(X_val,y_val, model:TargetNetwork, class_index:int,layer_name:str, segment:Segment, amount:int=10, save=True):
    validation_data = (X_val, y_val)
    explainer = GradCAM()

    heatmaps = explainer.explain(
        validation_data,
        model.model,
        class_index=class_index,
        layer_name=layer_name,
        image_weight=0.8
    )
    images = random.choices(heatmaps, k=amount)
    images = [cv2.resize(image, (400, 400)) for image in images]
    #files = [File(image) for image in images]
    files = [base64.b64encode(cv2.imencode('.jpg', img)[1]).decode() for img in images]
    if save:
        [Image(image=file, segment=segment, ground_truth= class_index).save() for file in files]


def create_segment(experiment_id:str,history,model:TargetNetwork,X_val, y_val, class_indices:List[int], layer_name:str, save:bool = True) -> Segment:
    segment = Segment(experiment_name=experiment_id)
    segment.full_clean()
    if save:
        segment.save()
    for class_index in class_indices:
        create_images(X_val=X_val,y_val=y_val, model=model, class_index=class_index, layer_name=layer_name,segment=segment, amount=1, save=save)
    create_features(history=history, segment=segment, model_parameter_config=model.model_parameter_configuration, save=save)
    return segment

def segments_to_df(segments:List[Segment]):
    d={}
    for feature in list(segments[0].features.all()):
        d[feature.name] = []
    for segment in segments:
        for feature in list(segment.features.all()):
            d[feature.name].append(feature.value)
    return pd.DataFrame(data=d)


def get_architecture(segment:Segment) -> ModelConfiguration:
    hyperparamenter_list= [dict()]
    features = segment[0].features.all()
    counter = 0
    for feature in features:
        if HyperParameterType.from_str(feature.name) is not None:
            number = int(feature.name.split("_")[-1])
            if number >= len(hyperparamenter_list):
                hyperparamenter_list.append(dict())
            type = HyperParameterType.from_str(feature.name)
            val = feature.value
            hyperparamenter_list[number][type] = val
    return ModelConfiguration(hyper_parameter_list=hyperparamenter_list)


def regularized_evolution(experiment_id:str,
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

    if preload_database:
        population = get_segments(experiment_name=experiment_id)[:population_size]
        evaluation_history = population

    ## Initialize Population
    while len(population) < population_size:
        if target_network_type == TargetNetworkType.AlexNet:
            valid = False
            while not valid:
                try:
                    model_configuration = ModelConfiguration(hyper_parameter_list=[
                            {HyperParameterType.NUMBER_OF_FILTERS: search_space.get_random_value(HyperParameterType.NUMBER_OF_FILTERS),
                             HyperParameterType.FILTER_HEIGHT: search_space.get_random_value(HyperParameterType.FILTER_HEIGHT),
                             HyperParameterType.FILTER_WIDTH: search_space.get_random_value(HyperParameterType.FILTER_WIDTH)},
                            {HyperParameterType.NUMBER_OF_FILTERS: search_space.get_random_value(
                                HyperParameterType.NUMBER_OF_FILTERS),
                             HyperParameterType.FILTER_HEIGHT: search_space.get_random_value(
                                 HyperParameterType.FILTER_HEIGHT),
                             HyperParameterType.FILTER_WIDTH: search_space.get_random_value(
                                 HyperParameterType.FILTER_WIDTH)}
                        ])
                    model = AlexNet(id= population_count, model_configuration=model_configuration, num_classes=num_classes, input_shape=X_train.shape[1:])
                    valid = True
                except:
                    print("Configuration not valid")
        else:
            raise ValueError('target_network_type is not valid')

        history = model.train(X_train=X_train,
                              y_train=y_train,
                              X_val=X_val,
                              y_val=y_val,
                              epochs=epochs,
                              batch_size=batch_size)
        population_count += 1
        segment = create_segment(history=history, model=model, X_val=X_val, y_val=y_val, class_indices=list(range(num_classes)),layer_name="conv_2", experiment_id=experiment_id)
        population.append(segment)
        evaluation_history.append(segment)


    segments = get_segments(experiment_name=experiment_id)
    agent = LinearAgent(segments=segments[:100], features=list(segments[0].features.all()), preferences=preferences)

    if ranking_mode ==2:
        comparisons = PreferenceController.create_comparisons(segments=segments[:100], amount=500, experiment_id=experiment_id)
        assert preferences is not None
        [agent.score_comparison(comp) for comp in tqdm(comparisons)]  # scoring comparisons on agent based preferences


    while len(get_not_rated_comparisons(experiment_name=experiment_id)) > 0:
        sleep(10)
    comparisons = get_rated_comparisons(experiment_name=experiment_id)

    ranker = RanknetScorer2(comparisons=comparisons, validation_comparisons=None)

    ranker_rankings = [ranker.score(s) for s in evaluation_history]
    agent_scores = [agent.score(s) for s in evaluation_history]

    while len(evaluation_history) < cycles:
        sample = []
        population_count += 1
        buffer = random.sample(list(population), sample_size)
        for candidate in buffer:
            sample.append((candidate,ranker.score(candidate)))
        parent = max(sample, key=lambda i: i[1])

        if target_network_type == TargetNetworkType.AlexNet:
            valid = False
            while not valid:
                try:
                    model_configuration = get_architecture(segment=parent)
                    model_configuration.mutate(search_space=search_space)
                    model = AlexNet(id= population_count, model_configuration=model_configuration, num_classes=num_classes, input_shape=X_train.shape[1:])
                    valid = True
                except:
                    print("Configuration not valid")
        else:
            raise ValueError('target_network_type is not valid')
        history = model.train(X_train=X_train,
                              y_train=y_train,
                              X_val=X_val,
                              y_val=y_val,
                              epochs=epochs*2,
                              #epochs=1,
                              batch_size=batch_size)
        population_count += 1
        segment = create_segment(history=history, model=model, X_val=X_val, y_val=y_val,
                                 class_indices=list(range(num_classes)), layer_name="conv_2",
                                 experiment_id=experiment_id, save=True)
        population.append(segment)
        evaluation_history.append(segment)
        population.pop(0)
        ranker_rankings.append(ranker.score(segment))
        agent_scores.append(agent.score(segment))
        df = segments_to_df(evaluation_history)
        df['ranker_rankings'] = ranker_rankings
        df['agent_scores'] = agent_scores
        df.to_csv('scores.csv')





