import random
from typing import List

from personalization_app.algorithm.models import Individual
from personalization_app.models import UnratedComparison, Experiment, Personalization, DbModel


def create_unranked_comparisons(models: List[DbModel], experiment:Experiment, personalization:Personalization):
    res = []
    for i in range(experiment.num_comparisons):
        segment_1, segment_2 = random.sample(models, 2)
        comparison = UnratedComparison(segment_1=segment_1, segment_2=segment_2, personalization=personalization)
        comparison.full_clean()
        comparison.save()
        res.append(comparison)
    return res