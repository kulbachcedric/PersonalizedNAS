from typing import List

from human_app import Comparison, Segment, get_features_to_array_from_segment
from sklearn.preprocessing import normalize
from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Subtract, Activation
from tensorflow.keras import backend
from sklearn import preprocessing
from tensorflow_core.python import get_default_graph
import numpy as np
from ranking.scorer import Scorer
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class RanknetScorer2(Scorer):

    def __init__(self, comparisons:List[Comparison],
                 batch_size=10,
                 epochs = 1000,
                 validation_comparisons:List[Comparison] = None):
        self.comparisons = comparisons
        self.order = list(self.comparisons[0].segments.all()[0].features.all())
        self.model = None
        self.base_network = None
        self.validation_comparisons = validation_comparisons

        X_1_train, X_2_train, y_train = self.create_dataset(self.comparisons)

        input_dim = X_1_train.shape[1]

        h_1 = Dense(128, activation='relu')
        h_2 = Dense(32, activation='relu')
        h_3 = Dense(16, activation='relu')
        s = Dense(1)

        # Relevant document score.
        rel_doc = Input(shape=(input_dim,), dtype="float32")
        h_1_rel = h_1(rel_doc)
        h_2_rel = h_2(h_1_rel)
        h_3_rel = h_3(h_2_rel)
        rel_score = s(h_3_rel)

        # Irrelevant document score.
        irr_doc = Input(shape=(input_dim,), dtype="float32")
        h_1_irr = h_1(irr_doc)
        h_2_irr = h_2(h_1_irr)
        h_3_irr = h_3(h_2_irr)
        irr_score = s(h_3_irr)

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)

        # Build model.
        self.model = Model(inputs=[rel_doc, irr_doc], outputs=prob)
        self.model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=['mae','acc','mse'])

        self.model.summary()
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0.0001,
                                           patience=2,
                                           verbose=1, mode='auto')

        if self.validation_comparisons is not None:
            X_1_test, X_2_test, y_test = self.create_dataset(self.validation_comparisons)
            hist = self.model.fit([X_1_train,X_2_train], y=y_train, callbacks=[es],  batch_size=batch_size, epochs=epochs, validation_data=([X_1_test,X_2_test],y_test))
            data = pd.DataFrame.from_dict(hist.history)
            sns.lineplot(x=data.index, y="val_acc",
                         data=data).set_title("RanKNet " + str(len(X_2_train)))
            plt.show()
            self.validation_accuracy = data['val_acc'].iloc[-1]
            print("val_acc: " + str(data['val_acc'].iloc[-1]))
            print("ready")
        else:
            hist = self.model.fit([X_1_train,X_2_train], y=y_train, callbacks=[es],  batch_size=batch_size, epochs=epochs)
        self.get_score = backend.function([rel_doc], [rel_score])

    def create_dataset(self, comparisons:List[Comparison], validation_split=.2):
        X_1 = np.zeros((len(comparisons),len(comparisons[0].segments.all()[0].features.all())))
        X_2 = np.zeros((len(comparisons),len(comparisons[0].segments.all()[0].features.all())))
        for idx, comparison in enumerate(comparisons):
            if comparison.winner == comparison.segments.all()[0].id:
                X_1[idx] = get_features_to_array_from_segment(segment=comparison.segments.all()[0], order=self.order)
                X_2[idx] = get_features_to_array_from_segment(segment=comparison.segments.all()[1], order=self.order)

            elif comparison.winner == comparison.segments.all()[1].id:
                X_1[idx] = get_features_to_array_from_segment(segment=comparison.segments.all()[1], order=self.order)
                X_2[idx] = get_features_to_array_from_segment(segment=comparison.segments.all()[0], order=self.order)
        y = np.ones((X_1.shape[0], 1))
        X_1_train = normalize(X_1, norm='l2', axis=1, copy=True, return_norm=True)[0]
        X_2_train = normalize(X_2, norm='l2', axis=1, copy=True, return_norm=True)[0]
        #X_1_train = X_1
        #X_2_train = X_2
        y_train = y
        return X_1_train, X_2_train, y_train

    def score(self, segment:Segment):
        features = get_features_to_array_from_segment(segment=segment, order=self.order)
        features = normalize([np.array(features)], norm='l2', axis=1, copy=True, return_norm=True)[0]
        score = self.get_score(np.array(features))
        return float(np.squeeze(score))