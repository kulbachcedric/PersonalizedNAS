import random
from typing import List, Dict, Tuple

from utils import HyperParameterType


class HyperParameter:
    def __init__(self, param_type:HyperParameterType, values: List[int]):
        self.param_type = param_type
        self.values = values

    @property
    def get_all(self)-> List[Tuple[HyperParameterType, int]]:
        return [(self.param_type, val) for val in self.values]

    @property
    def possible_values(self) -> List[int]:
        return self.values

    @property
    def first_value(self) -> int:
        return self.values[0]

    @property
    def get_random_value(self) -> int:
        return random.choice(self.values)




class SearchSpace:
    def __init__(self, hyper_parameters:List[HyperParameter]):
        self.hyper_parameters = hyper_parameters

    def get_random_value(self, type:HyperParameterType):
        for hyper_parameter in self.hyper_parameters:
            if hyper_parameter.param_type == type:
                return hyper_parameter.get_random_value

    def get_first_value(self, type:HyperParameterType):
        for hyper_parameter in self.hyper_parameters:
            if hyper_parameter.param_type == type:
                return hyper_parameter

    @property
    def num_possible_values(self):
        num_possible_values = 0
        for hyper_parameter in self.hyper_parameters:
            num_possible_values += len(hyper_parameter.possible_values)
        return num_possible_values

    def __len__(self):
        res = 1
        for hyperparameter in self.hyper_parameters:
            res = res * len(hyperparameter.values)


class ModelConfiguration:

    def __init__(self, hyper_parameter_list: List[Dict[HyperParameterType,int]]):
        self.hyper_parameter_list = hyper_parameter_list

    def mutate(self, search_space:SearchSpace):
        i = random.choice(range(len(self.hyper_parameter_list)))
        d = self.hyper_parameter_list[i] # get dict to change
        c = random.choice(list(d.keys())) # return random val
        v = search_space.get_random_value(type=c)
        self.hyper_parameter_list[i][c] = v