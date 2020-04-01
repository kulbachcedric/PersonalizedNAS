from typing import List, Dict

from human_app import Comparison, Feature, Segment, get_features_to_array_from_segment
import numpy as np
from sklearn import preprocessing
from tqdm import tqdm

from ranking.scorer import Scorer

def get_preference_list(features:List[Feature],preferences:Dict)->List[float]:
    val = []
    for feature in features:
        if feature.name in preferences.keys():
            val.append(preferences[feature.name])
        else:
            val.append(.0)
    return val

class LinearAgent(Scorer):
    def __init__(self, segments:List[Segment], features: List[Feature], preferences:Dict):
        self.features = features
        self.order = segments[0].features.all()
        self.preferences = get_preference_list(features=self.order,preferences=preferences)
        #X = np.zeros((len(segments), len(features)))
        #for idx, segment in enumerate(tqdm(segments)):
        #    X[idx] = get_features_to_array_from_segment(segment=segment, order=self.features)
        #self.min_max_scaler = preprocessing.MinMaxScaler()
        #self.min_max_scaler.fit(X)

    def score(self, segment:Segment) -> float:
        vals = get_features_to_array_from_segment(segment=segment,order=self.features)
        #vals = self.min_max_scaler.transform([vals])

        return float(np.sum(np.multiply(self.preferences, vals)))

    def score_comparison(self, comparison:Comparison)-> Comparison:
        score_1 = self.score(segment=comparison.segments.all()[0])
        score_2 = self.score(segment=comparison.segments.all()[1])
        if score_1 <= score_2:
            comparison.winner = comparison.segments.all()[1].id
            comparison.rated = True
            comparison.full_clean()
            comparison.save()
        else:
            comparison.winner = comparison.segments.all()[0].id
            comparison.rated = True
            comparison.full_clean()
            comparison.save()
        return comparison
