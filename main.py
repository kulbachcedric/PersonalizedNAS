from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.callbacks import CSVLogger
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.keras.metrics import Precision, Recall, TruePositives, TrueNegatives

from data import datasets

import numpy as np

from models import VariableKerasClassifier
from models.build_functions import build_AlexNetSimple



if __name__ == '__main__':

    """
    Data sets:
                # 1: cifar10
                # 2: cifar100
                # 3: MNIST
                # 4: MNIST-Fashion
                # 5: stl10 data set
    """

    # %%
    losses = [
        #categorical_crossentropy,
        LossFunctionWrapper(accuracy_score),
        LossFunctionWrapper(make_scorer(precision_score, greated_is_better=False)),
        LossFunctionWrapper(make_scorer(recall_score, greated_is_better=False)),
        LossFunctionWrapper(make_scorer(f1_score, greated_is_better=False)),
    ]
    losses_names = [
        #'categorical_crossentropy',
        'accuracy',
        'precision',
        'recall',
        'f1'
    ]

    metrics = [
        categorical_crossentropy,
        'accuracy',
        Precision(),
        Recall(),
        TrueNegatives(),
        TruePositives(),
    ]
    dataset_numbers = [1,2,3,4,5]
    for dataset_no in dataset_numbers:
        (X_train, y_train), (X_test, y_test) = datasets.load_data(dataset_no)
        num_classes = datasets.getNumClasses(dataset_no)
        x_train_mean = np.mean(X_train, axis=0)
        X_train -= x_train_mean
        X_test -= x_train_mean

        for idx, loss in enumerate(losses):
            classifier = VariableKerasClassifier(build_fn=build_AlexNetSimple,
                                                 input_shape=X_train.shape[1:],
                                                 num_classes=num_classes,
                                                 loss=loss,
                                                 metrics=metrics)
            history = classifier.fit(X_train, y_train,
                                     epochs=100,
                                     batch_size=25,
                                     validation_data = (X_test,y_test),
                                     callbacks=[CSVLogger(f'./results/dataset_{dataset_no}_{losses_names[idx]}.csv', append=True)])
