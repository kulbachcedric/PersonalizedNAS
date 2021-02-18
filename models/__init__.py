import copy
import types

from sklearn.utils.multiclass import _check_partial_fit_first_call
from tensorflow.keras import Sequential
from tensorflow.python.keras import losses
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Sequential, save_model, load_model
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras.wrappers.scikit_learn import KerasClassifier
import pickle
import tempfile
import numpy as np

class VariableKerasClassifier(KerasClassifier):

    def __init__(self, build_fn=None, **sk_params):
        super().__init__(build_fn, **sk_params)

    @staticmethod
    def make_keras_picklable():
        def __getstate__(self):
            model_str = ""
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                save_model(self, fd.name, overwrite=True)
                model_str = fd.read()
            d = {'model_str': model_str}
            return d

        def __setstate__(self, state):
            with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
                fd.write(state['model_str'])
                fd.flush()
                model = load_model(fd.name)
            self.__dict__ = model.__dict__

    def fit(self, x, y, **kwargs):
        """Constructs a new model with `build_fn` & fit the model to `(x, y)`.

        Arguments:
            x : array-like, shape `(n_samples, n_features)`
                Training samples where `n_samples` is the number of samples
                and `n_features` is the number of features.
            y : array-like, shape `(n_samples,)` or `(n_samples, n_outputs)`
                True labels for `x`.
            **kwargs: dictionary arguments
                Legal arguments are the arguments of `Sequential.fit`

        Returns:
            history : object
                details about the training history at each epoch.
        """
        y = np.array(y)
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)

        VariableKerasClassifier.make_keras_picklable()
        self.sk_params['input_shape'] = x.shape[1:]
        if self.build_fn is None:
            self.model = self.__call__(**self.filter_sk_params(self.__call__))
        elif (not isinstance(self.build_fn, types.FunctionType) and
              not isinstance(self.build_fn, types.MethodType)):
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn.__call__))
        else:
            self.model = self.build_fn(**self.filter_sk_params(self.build_fn))

        if (losses.is_categorical_crossentropy(self.model.loss) and
                len(y.shape) != 2):
            y = to_categorical(y)

        fit_args = copy.deepcopy(self.filter_sk_params(Sequential.fit))
        fit_args.update(kwargs)
        self.history = self.model.fit(x, y, ** fit_args)

        return self
