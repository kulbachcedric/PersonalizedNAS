import random
from typing import List

from human_app import Segment, Comparison


class PreferenceController:
    def __init__(self):
        pass

    @staticmethod
    def create_comparisons(segments:List[Segment], amount, experiment_id:str) -> List[Comparison]:
        random.shuffle(segments)
        segment_pairs = []
        for idx in range(amount):
            seg_1, seg_2 = random.sample(segments, k=2)
            comparison = Comparison(experiment_name=experiment_id)
            comparison.save()
            comparison.segments.add(seg_1,seg_2)
            segment_pairs.append(comparison)
        return segment_pairs