from django.shortcuts import render
from human_app.models import Comparison, Segment
import random
from typing import List

from django.db.models import Q, Count


def get_not_rated_comparisons(experiment_name:str) -> List[Comparison]:

    not_rated = Q(rated=False)
    # not_responded = Q(responded_at__isnull=True)
    # cutoff_time = timezone.now() - timedelta(minutes=2)
    # not_in_progress = Q(shown_to_tasker_at__isnull=True) | Q(shown_to_tasker_at__lte=cutoff_time)
    # finished_uploading_media = Q(created_at__lte=datetime.now() - timedelta(seconds=2)) # Give time for upload

    # ready = not_responded & not_in_progress & finished_uploading_media
    comparisons = Comparison.objects.filter(not_rated,experiment_name=experiment_name).order_by('-created_at')
    print(str(len(comparisons))+" left in experiment "+experiment_name)
    # Sort by priority, then put newest labels first
    return list(comparisons)

def get_rated_comparisons(experiment_name:str) -> List[Comparison]:

    not_rated = Q(rated=True)
    # not_responded = Q(responded_at__isnull=True)
    # cutoff_time = timezone.now() - timedelta(minutes=2)
    # not_in_progress = Q(shown_to_tasker_at__isnull=True) | Q(shown_to_tasker_at__lte=cutoff_time)
    # finished_uploading_media = Q(created_at__lte=datetime.now() - timedelta(seconds=2)) # Give time for upload

    # ready = not_responded & not_in_progress & finished_uploading_media
    comparisons = Comparison.objects.filter(not_rated,experiment_name=experiment_name).order_by('-created_at')
    # Sort by priority, then put newest labels first
    return list(comparisons)

def get_segments(experiment_name:str) -> List[Segment]:
    has_features = Q(features__isnull=False)
    segments = Segment.objects.filter(has_features,experiment_name=experiment_name).distinct()
    return list(segments)

def get_random_not_rated_comparison(experiment_name:str)-> Comparison:
    comparison = random.choice(get_not_rated_comparisons(experiment_name=experiment_name))
    return comparison

# Create your views here.



def index(request):
    return render(request, 'index.html', context=dict(
        experiment_names=[exp for exp in
                          Segment.objects.order_by().values_list('experiment_name', flat=True).distinct()]
    ))


def respond(request, experiment_id:str):

    comparison = get_random_not_rated_comparison(experiment_name=experiment_id)
    print("Comparison ID: " + str(comparison.id))
    return render(request, 'experiment.html', context=dict(
        comparison=comparison
    ))


def next(request, experiment_id:str):
    post = request.POST
    comparison_id = post.get("comparison_id")
    print("Comparison ID: "+comparison_id)
    comparison = Comparison.objects.get(pk=comparison_id)
    print(post)
    response = post.get('response')
    if response == 'left':
        print('Left was chosen')
        comparison.winner = comparison.segments.all()[0].id
    elif response == 'right':
        print('Right was chosen')
        comparison.winner = comparison.segments.all()[1].id
    else:
        pass
    comparison.rated = True
    comparison.full_clean()
    comparison.save()
    comparison = get_random_not_rated_comparison(experiment_id)
    return render(request, 'next.html', context=dict(
        comparison=comparison
    ))