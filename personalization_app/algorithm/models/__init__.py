from abc import ABC

from personalization_app.models import DbModel, Experiment


class Individual(ABC):
    def __init__(self):
        self.model = None
        self.config = None
        super().__init__()

    def load(self, db_model:DbModel):
        pass

    def instantiate_random(self, experiment:Experiment):
        pass

    def instantiate_mutation(self):
        pass

    def predict(self,X):
        return self.model.predict(X)

    def get_div(self):
        pass
