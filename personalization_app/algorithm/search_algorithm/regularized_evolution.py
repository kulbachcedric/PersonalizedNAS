import collections
from random import random
from typing import List

from personalization_app.algorithm.comparison_creation.random import create_unranked_comparisons
from personalization_app.algorithm.models import Individual
from personalization_app.models import Dataset, Experiment, DbModel, RatedComparison, UnratedComparison


class RegularizedEvolution():
    def __init__(self):
        pass

    @staticmethod
    def initialize_for_personalization(experiment:Experiment, model_class:Individual):
        population = collections.deque()
        population_count = 1

        ## Initialize Population
        while len(population) < experiment.population_size:
            individual = model_class()
            individual.instantiate_random(experiment=experiment)
            population_count += 1
            population.append(individual)




    def personalize(self):
        rated_comparisons = RatedComparison.objects.filter(experiment=Experiment)





    def optimize(self):
        population = DbModel.objects.filter(initial_tag=True)
        population_count = len(population)






def regularized_evolution(experiment:Experiment, dataset:Dataset, model_class:Individual):

    population = collections.deque()
    evaluation_history = []
    population_count = 1


    ## Initialize Population
    while len(population) < experiment.population_size:
        individual = model_class()
        individual.instantiate_random(experiment=experiment)

        population_count += 1

        population.append(individual)


    segments = get_segments(experiment_name=experiment_id)
    agent = LinearAgent(segments=segments[:100], features=list(segments[0].features.all()), preferences=preferences)

    if ranking_mode ==2:
        #comparisons = PreferenceController.create_unranked_comparisons(segments=segments[:100], amount=500, experiment_id=experiment_id)
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