
import os

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "personalized_neural_architecture_search.settings")
import django
django.setup()
from personalization_app.experiment_handler import ExperimentExecutor

from personalization_app.models import Experiment

if __name__ == '__main__':

    experiment = Experiment.objects.get(pk=1)
    experiment_executor = ExperimentExecutor(experiment=experiment)
    #experiment_executor.start()
    experiment_executor.create_unranked_comparisons()