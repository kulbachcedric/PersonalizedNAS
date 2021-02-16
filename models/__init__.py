from tensorflow_core.python.keras.wrappers.scikit_learn import KerasClassifier


class TargetNetworkInterface:

    def build_nn(self):
        raise NotImplementedError('Needs to be implemented and returns a Keras Model')


    def get_model(self):
        return self.build_nn

    def get_classifier(self):
        return KerasClassifier(self.build_nn)


class AlexNetClassifier(TargetNetworkInterface):

    def build_nn(self):


