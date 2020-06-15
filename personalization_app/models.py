import zipfile
from typing import List
import datetime
import yaml
from django import forms
from django.db import models

import pandas as pd
from django.db import models
from django.db.models import FileField


DATASET_CHOICES = [
        ('TS', 'time series'),
        ('IC', 'image classification')
    ]

SEARCH_ALGORITHM_CHOICES = [
    ('REA', 'regularized evolution')
]

ACTIVE_LEARNER_CHOICES= [
    ('RDM', 'random'),
]
MODEL_TYPE_CHOICES=[
    ('DAR', 'deep ar'),
    ('SPL', 'simple')
]

class Dataset(models.Model):
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)

    type = models.CharField(
        max_length=2,
        choices=DATASET_CHOICES,
        default='TS',
    )
    name = models.CharField('Name', max_length=30, unique=True)
    data = FileField('data', upload_to='data/')

    def full_clean(self, exclude=None, validate_unique=True):
        super(Dataset, self).full_clean(exclude=exclude, validate_unique=validate_unique)

    def get_data(self):
        '''
        returns X_train, y_train, X_test, y_test as pandas dataframe
        '''
        archive = zipfile.ZipFile(self.data.path, 'r')
        if self.type == DATASET_CHOICES[0][0]:
            df = [pd.read_csv(archive.open(f.filename), sep=';') for f in archive.filelist if
                  f.filename.endswith('data.csv')][0]
            config = [yaml.load(archive.open(f.filename)) for f in archive.filelist if f.filename.endswith('config.yml')][0]
            return df, config


class Experiment(models.Model):
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)
    name = models.CharField(max_length=40)
    search_algorithm = models.CharField(max_length=3,
                                        choices=SEARCH_ALGORITHM_CHOICES,
                                        default='REA')
    active_learner = models.CharField(max_length=3,
                                      choices=ACTIVE_LEARNER_CHOICES,
                                      default='RDM')
    model_type = models.CharField(max_length=3,
                                  choices=MODEL_TYPE_CHOICES,
                                  default='NN')


    dataset = models.ForeignKey(Dataset, related_name='experiments', on_delete=models.CASCADE)


    # regularized evolution
    population_size = models.IntegerField(default=100)
    cycles = models.IntegerField(default=100)
    sample_size = models.IntegerField(default=80)
    # dataset
    train_test_split = models.FloatField(default=0.8)
    # Neural Network
    batch_size = models.IntegerField(default=10)
    # active learner
    num_comparisons = models.IntegerField(default=50)

    def full_clean(self, exclude=None, validate_unique=True):
        super(Experiment, self).full_clean(exclude=exclude, validate_unique=validate_unique)


class DbModel(models.Model):
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)
    experiment = models.ForeignKey(Experiment, related_name='models', on_delete=models.CASCADE)
    initial_tag = models.BooleanField('initial_tag',default=False)
    model = FileField('data', upload_to='models/')

    def full_clean(self, exclude=None, validate_unique=True):
        super(DbModel, self).full_clean(exclude=exclude, validate_unique=validate_unique)

class Personalization(models.Model):
    experiment = models.ForeignKey(Experiment, related_name='personalizations', on_delete=models.CASCADE)
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)

    def full_clean(self, exclude=None, validate_unique=True):
        super(Personalization, self).full_clean(exclude=exclude, validate_unique=validate_unique)


class RatedComparison(models.Model):
    personalization = models.ForeignKey(Personalization,related_name='rated_comparisons', on_delete=models.CASCADE)
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)
    winner = models.ForeignKey(DbModel, related_name='winner', on_delete=models.CASCADE)
    looser = models.ForeignKey(DbModel, related_name='looser', on_delete=models.CASCADE)

    def full_clean(self, exclude=None, validate_unique=True):
        super(RatedComparison, self).full_clean(exclude=exclude, validate_unique=validate_unique)

class UnratedComparison(models.Model):
    personalization = models.ForeignKey(Personalization, related_name='unrated_comparisons', on_delete=models.CASCADE)
    created_at = models.DateTimeField('Date Created', auto_now_add=True, db_index=True)
    segment_1 = models.ForeignKey('DbModel', related_name='segment_1', on_delete=models.CASCADE)
    segment_2 = models.ForeignKey('DbModel', related_name='segment_2', on_delete=models.CASCADE)

    def full_clean(self, exclude=None, validate_unique=True):
        super(UnratedComparison, self).full_clean(exclude=exclude, validate_unique=validate_unique)


