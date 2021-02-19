import collections
import random

import numpy as np

class RegularizedEvolutionarySearch:
    def __init__(self, P,C,S, api, scorer, dataset):
        self.P = P
        self.C = C
        self.S = S
        self.api = api
        self.dataset = dataset
        self.population = collections.deque()
        self.history = collections.deque()
        self.scorer = scorer

    def initialize(self):
        self.population =  collections.deque()
        self.history = collections.deque()

        ## Initialize Population
        while len(self.population) < self.P:
            model = np.random.randint(0,len(self.api)-1)
            self.population.append(model)
            self.population.append(model)

    def evolve(self):
        while len(self.history) < self.C:
            sample = []
            while len(sample) < self.S:
                canidate = random.choice(self.population)
                sample.append(canidate)
            parent = self.__get_best_arch(architectures=self.population)
            child = self.__mutate(parent)
            self.population.append(child)
            self.history.append(child)
            self.population.popleft()

    def __get_best_arch(self, architectures)->int:
        scores = [self.scorer.score(arch) for arch in architectures]
        return architectures[np.argmax(scores)]

    def __mutate(self, idx)->int:
        return idx + random.choice([-2,-1,1,2])
