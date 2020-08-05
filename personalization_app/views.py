import yaml
from django.forms import forms
from django.http import HttpResponseRedirect
from django.shortcuts import render
from django.shortcuts import render
import random
from typing import List
import zipfile
import pandas as pd
from django.db.models import Q, Count
from django.urls import reverse
from django.views.generic import CreateView, FormView

from personalization_app.algorithm.models import Individual
from personalization_app.algorithm.models.SimpleNN import SimpleNN
from personalization_app.dao import get_dataset_view
from personalization_app.experiment_handler import ExperimentExecutor
from personalization_app.forms import *
from personalization_app.models import Dataset, DATASET_CHOICES, Experiment, Personalization, UnratedComparison, \
    RatedComparison


def upload(request):
    if request.method == 'POST':
        form = DatasetForm(request.POST, request.FILES)
        if form.is_valid():
            new_dataset = Dataset(name=request.POST['name'],
                                  type=request.POST['type'],
                                  data=request.FILES['data'])
            new_dataset.full_clean()
            new_dataset.save()
            return index(request)
    else:
        form = DatasetForm()
    return render(request, 'upload.html', context= {'form' : form})

def datasets(request):
    datasets = Dataset.objects.all()
    return render(request, 'datasets.html', context={'datasets' : datasets})

def index(request):
    return render(request, 'index.html')

def dataset(request, dataset_id:str):
    dataset = Dataset.objects.get(pk=dataset_id)
    view = get_dataset_view(dataset=dataset)
    return render(request, 'dataset.html', context={'id'   : dataset.id,
                                                    'name' : dataset.name,
                                                    'type' : dataset.type,
                                                    'experiment' : dataset.experiments.all(),
                                                    'view' : view})

def experiment(request, experiment_id):
    experiment = Experiment.objects.get(pk=experiment_id)
    settings = [
        {'name':'cycles','value': experiment.cycles},
    ]


    return render(request, 'experiment.html', context={
        'experiment': experiment,
        'settings': settings
    })

def new_experiment(request, dataset_id:str):
    dataset = Dataset.objects.get(pk=dataset_id)
    if request.method == 'POST':
        if request.POST["experiment_id"] != "":
            experiment = Experiment.objects.get(pk=request.POST['experiment_id'])
            # DATASET timeseries
            if 'train_test_split' in request.POST: experiment.train_test_split = request.POST['train_test_split']
            # SEARCH ALGORITHM regularized evolution
            if 'population_size' in request.POST: experiment.population_size = request.POST['population_size']
            if 'cycles' in request.POST: experiment.cycles = request.POST['cycles']
            if 'sample_size' in request.POST: experiment.sample_size = request.POST['sample_size']
            # MODEL TYPE neural network
            if 'batch_size' in request.POST: experiment.batch_size = request.POST['batch_size']
            # ACTIVE LEARNER random
            if 'num_comparisons' in request.POST: experiment.num_comparisons = request.POST['num_comparisons']

            experiment.full_clean()
            experiment.save()
            experiment_executor = ExperimentExecutor(experiment=experiment)
            experiment_executor.start()
            return index(request)
        else:
            new_experiment = Experiment(name=request.POST['name'],
                             search_algorithm=request.POST['search_algorithm'],
                             active_learner=request.POST['active_learner'],
                             model_type = request.POST['model_type'],
                             dataset = dataset)
            new_experiment.full_clean()
            new_experiment.save()
            forms = []
            #DATASET
            if new_experiment.dataset.type == 'TS':
                forms.append(ExperimentDatasetTimeSeriesForm())
            elif new_experiment.dataset.type == 'IC':
                forms.append(ExperimentDatasetImageClassificationForm())
            #ACTIVE LEARNER:
            if new_experiment.active_learner == 'RDM':
                forms.append(ExperimentActiveLearnerRandomForm())
            #SEARCH ALGORITHM
            if new_experiment.search_algorithm =='REA':
                forms.append(ExperimentSearchAlgorithmRegularizedEvolutionForm())
            #MODEL TYPE
            if new_experiment.model_type=='DAR':
                forms.append(ExperimentModelTypeDeepArForm())
            elif new_experiment.model_type=='SPL':
                forms.append(ExperimentModelTypeSimpleNNForm())

            return render(request, 'new_experiment.html', context= {
                'experiement_id' : new_experiment.id,
                'forms' : forms})
    else:
        forms = [ExperimentForm()]
    return render(request, 'new_experiment.html', context= {'forms' : forms})

def new_personalization(request, experiment_id):
    experiment = Experiment.objects.get(pk=experiment_id)
    experiment_executor = ExperimentExecutor(experiment=experiment)
    db_personalization = Personalization(experiment=experiment)
    db_personalization.full_clean()
    db_personalization.save()
    unrated_comparisons = experiment_executor.create_unranked_comparisons(db_personalization=db_personalization)
    experiment_executor.personalize(db_personalization=db_personalization)
    if experiment.model_type == 'SPL':
        segment_1 = SimpleNN()
        segment_1.load(unrated_comparisons[0].segment_1)
        segment_2 = SimpleNN()
        segment_2.load(unrated_comparisons[0].segment_2)
    elif experiment.model_type == 'DAR':
        segment_1 = SimpleNN()
        segment_1.load(unrated_comparisons[0].segment_1)
        segment_2 = SimpleNN()
        segment_2.load(unrated_comparisons[0].segment_2)
    else:
        segment_1 = Individual()
        segment_1.load(unrated_comparisons[0].segment_1)
        segment_2 = Individual()
        segment_2.load(unrated_comparisons[0].segment_2)

    return render(request,'personalization.html', context={
        'personalization'  : db_personalization,
        'comparison_id' : unrated_comparisons[0].id,
        'unrated_comparisons' : unrated_comparisons,
        'segment_1' : segment_1.get_div(),
        'segment_2' : segment_2.get_div()
    })

def response(request, personalization_id):
    db_personalization = Personalization.objects.get(pk=personalization_id)
    experiment = db_personalization.experiment
    if request.method == "POST":
        response = request.POST["response"]
        unrated_comparison = None
        try:
            unrated_comparison = UnratedComparison.objects.get(pk=request.POST["comparison_id"])
        except:
            print('Unrated Comparison was already deleted')
        if unrated_comparison is not None:
            if response == "segment_1":
                winner = unrated_comparison.segment_1
                looser = unrated_comparison.segment_2
            elif response == "segment_2":
                winner = unrated_comparison.segment_2
                looser = unrated_comparison.segment_1
            else:
                print("No correct response was given")
                winner , looser = None, None
            if winner is not None and looser is not None:
                unrated_comparison.delete()
                rated_comparison = RatedComparison(winner=winner, looser=looser, personalization=db_personalization)
                rated_comparison.full_clean()
                rated_comparison.save()

    unrated_comparisons = db_personalization.unrated_comparisons.all()
    if len(unrated_comparisons) > 0:
        # If there are unrated_comparisons return the next comparison
        if experiment.model_type == 'SPL':
            segment_1 = SimpleNN()
            segment_1.load(unrated_comparisons[0].segment_1)
            segment_2 = SimpleNN()
            segment_2.load(unrated_comparisons[0].segment_2)
        elif experiment.model_type == 'DAR':
            segment_1 = SimpleNN()
            segment_1.load(unrated_comparisons[0].segment_1)
            segment_2 = SimpleNN()
            segment_2.load(unrated_comparisons[0].segment_2)
        else:
            segment_1 = Individual()
            segment_1.load(unrated_comparisons[0].segment_1)
            segment_2 = Individual()
            segment_2.load(unrated_comparisons[0].segment_2)

        return render(request, 'personalization.html', context={
            'personalization': db_personalization,
            'comparison_id': unrated_comparisons[0].id,
            'unrated_comparisons': unrated_comparisons,
            'segment_1': segment_1.get_div(),
            'segment_2': segment_2.get_div()
        })
    else:
        pass# calculate personalized metric








