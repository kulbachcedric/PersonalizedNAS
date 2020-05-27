from django import forms

from personalization_app.models import DATASET_CHOICES, SEARCH_ALGORITHM_CHOICES, ACTIVE_LEARNER_CHOICES, \
    MODEL_TYPE_CHOICES


class DatasetForm(forms.Form):
    name = forms.CharField(label='Name', max_length=30)
    type = forms.ChoiceField(label='Dataset type',choices=DATASET_CHOICES)
    data = forms.FileField(label='Select a file')


class ExperimentForm(forms.Form):
    name = forms.CharField(label='Name', max_length=40)
    search_algorithm = forms.ChoiceField(label='Search Algorithm', choices=SEARCH_ALGORITHM_CHOICES)
    active_learner = forms.ChoiceField(label='Active Learner', choices=ACTIVE_LEARNER_CHOICES)
    model_type = forms.ChoiceField(label='Model Type',
                                  choices=MODEL_TYPE_CHOICES)


class ExperimentDatasetTimeSeriesForm(forms.Form):
    train_test_split = forms.FloatField(label='Train Test Split',max_value=1, min_value=0)


class ExperimentDatasetImageClassificationForm(forms.Form):
    train_test_split = forms.FloatField(label='Train Test Split', max_value=1, min_value=0)


class ExperimentSearchAlgorithmRegularizedEvolutionForm(forms.Form):
    population_size = forms.IntegerField(label='Population Size')
    cycles = forms.IntegerField(label='Cycles')
    sample_size = forms.IntegerField(label='Sample Size')


class ExperimentModelTypeDeepArForm(forms.Form):
    batch_size = forms.IntegerField(label='Batch Size')


class ExperimentModelTypeSimpleNNForm(forms.Form):
    batch_size = forms.IntegerField(label='Batch Size')


class ExperimentActiveLearnerRandomForm(forms.Form):
    num_comparisons = forms.IntegerField(label='Number of Comparisons')


