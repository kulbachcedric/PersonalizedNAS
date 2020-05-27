from personalization_app.algorithm.comparison_creation import random
from personalization_app.algorithm.models.DeepAR import DeepAR
from personalization_app.algorithm.models.SimpleNN import SimpleNN
from personalization_app.algorithm.search_algorithm.regularized_evolution import regularized_evolution, \
    RegularizedEvolution
from personalization_app.models import Experiment, DbModel


class ExperimentExecutor():

    def __init__(self, experiment:Experiment):
        self.experiment = experiment
        self.dataset = experiment.dataset
        if self.experiment.model_type == 'DAR':
            self.model_class = DeepAR
        elif self.experiment.model_type == 'SPL':
            self.model_class = SimpleNN

    def start(self):
        if self.experiment.search_algorithm == 'REA':
            re = RegularizedEvolution()
            re.initialize_for_personalization(experiment=self.experiment, model_class=self.model_class)
            self.create_unranked_comparisons()

    def create_unranked_comparisons(self):
        db_models = list(DbModel.objects.filter(experiment=self.experiment))
        if self.experiment.active_learner == 'RDM':
            random.create_unranked_comparisons(models=db_models, experiment=self.experiment)

    def personalize(self):
        #todo check if models are not empty
        pass
