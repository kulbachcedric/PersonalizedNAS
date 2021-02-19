import itertools
import random

import numpy as np

def get_metrics(api, index:int, dataset:str):
    res = []
    result = api.query_by_index(index, dataset)[111]
    res.append(result.get_latency())
    res.append(result.flop)
    res.append(result.get_train()['accuracy'])
    res.append(result.get_train()['all_time'])
    res.append(result.get_eval('x-valid')['accuracy'])
    res.append(result.get_eval('x-valid')['all_time'])
    res.append(result.get_eval('x-test')['accuracy'])
    res.append(result.get_eval('x-test')['all_time'])
    return np.array(res)

class AccuracyScorer():
    def __init__(self, api, dataset):
        self.api = api
        self.dataset = dataset

    def score(self,idx):
        X = get_metrics(api=self.api, index=idx, dataset=self.dataset)
        return X[4]

def create_dataset(api,scorer, number_archs=100,number_comparisons=None, dataset='cifar100'):
    architectures = np.random.randint(0,len(api)-1,(number_archs))
    print(architectures)
    return create_dataset_from_list(architectures=architectures, scorer=scorer, dataset=dataset, number=number_comparisons)

def create_dataset_from_list(architectures, scorer, dataset='cifar100', number=None):
    combinations = list(itertools.combinations(architectures, 2))
    if number is not None:
        random.shuffle(combinations)
        combinations = combinations[:number]
    X_1 = []
    X_2 = []
    y= []
    for idx in combinations:
        X_1_temp = get_metrics(idx[0],dataset=dataset)
        X_2_temp = get_metrics(idx[1], dataset=dataset)
        score_1 = scorer.score(idx=idx[0])
        score_2 = scorer.score(idx=idx[1])

        if score_1 < score_2:
            X_1.append(X_2_temp)
            X_2.append(X_1_temp)
        else:
            X_1.append(X_1_temp)
            X_2.append(X_2_temp)

        y = np.ones((len(X_1),1))
    return np.array(X_1), np.array(X_2), y
