from typing import List

from django.db import models

# Create your models here.

from django.db import models

class Segment(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    experiment_name = models.TextField('name of experiment')

    def full_clean(self, exclude=None, validate_unique=True):
        super(Segment, self).full_clean(exclude=exclude, validate_unique=validate_unique)


class Feature(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    segment = models.ForeignKey(Segment, related_name='features', on_delete=models.CASCADE)
    name = models.TextField('feature name', default="")
    value = models.FloatField('feature value', default=0.0)

    def full_clean(self, exclude=None, validate_unique=True):
        super(Feature, self).full_clean(exclude=exclude, validate_unique=validate_unique)




class Image(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    prediction = models.TextField(max_length=100, default='Not available')
    ground_truth = models.TextField(max_length=100, default='No available')
    segment = models.ForeignKey(Segment, related_name='images', on_delete=models.CASCADE)
    image = models.TextField(max_length=240)

    def full_clean(self, exclude=None, validate_unique=True):
        super(Image, self).full_clean(exclude=exclude, validate_unique=validate_unique)


class Comparison(models.Model):
    created_at = models.DateTimeField('date created', auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)
    rated = models.BooleanField(default=False)
    winner = models.IntegerField(default=99999999)
    experiment_name = models.TextField('name of experiment')
    segments = models.ManyToManyField(Segment)


def get_features_to_array_from_segment(segment:Segment, order:List[Feature])->List[float]:
    res = []
    for feature in order:
        for sfeature in list(segment.features.all()):
            if feature.name == sfeature.name:
                res.append(sfeature.value)
            #if sfeature.name == 'val_accuracy':
            #    res = [sfeature.value]
    return res



