from typing import List

from human_app import Comparison, Segment, get_features_to_array_from_segment
from sklearn.preprocessing import normalize

from tensorflow.keras import Sequential, Input, Model
from tensorflow.keras.layers import Dense, Subtract, Activation, Dropout
from sklearn import preprocessing
import numpy as np
from ranking.scorer import Scorer
import tensorflow as tf
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class RanknetScorer(Scorer):

    def __init__(self, comparisons:List[Comparison],
                 batch_size=10,
                 epochs = 1000,
                 validation_comparisons:List[Comparison] = None):
        self.comparisons = comparisons
        self.order = list(self.comparisons[0].segments.all()[0].features.all())
        self.model = None
        self.base_network = None
        self.validation_comparisons = validation_comparisons
        def _create_base_network(input_dim):
            '''Base network to be shared (eq. to feature extraction).
            '''
            seq = Sequential()
            #seq.add(input)
            #seq.add(Input(shape=(input_dim,)))

            seq.add(Dense(input_dim, input_shape=(input_dim,)))
            # seq.add(Dropout(0.1))
            #seq.add(Dense(128, activation='relu'))
            #seq.add(Dropout(0.1))
            #seq.add(Dense(64, activation='relu'))
            seq.add(Dropout(0.1))
            seq.add(Dense(16, activation='relu'))
            seq.add(Dense(1))
            return seq

        def _create_meta_network(input_dim,base_network):
            input_a = Input(shape=(input_dim,))
            input_b = Input(shape=(input_dim,))

            rel_score = base_network(input_a)
            irr_score = base_network(input_b)

            # subtract scores
            diff = Subtract()([rel_score, irr_score])

            # Pass difference through sigmoid function.
            prob = Activation("sigmoid")(diff)


            # Build model.
            model = Model(inputs=[input_a, input_b], outputs=prob)
            model.compile(optimizer="adadelta", loss="binary_crossentropy", metrics=['mae','acc','mse']) #loss="binary_crossentropy"
            #model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['mae','acc','mse'])
            return model

        X_1_train, X_2_train, y_train = self.create_dataset(self.comparisons)

        self.min_max_scaler = preprocessing.MinMaxScaler()
        self.min_max_scaler.fit(X_1_train)
        #X_1_train = self.min_max_scaler.transform(X_1_train)
        #X_2_train = self.min_max_scaler.transform(X_2_train)

        INPUT_DIM = X_1_train.shape[1]
        self.base_network = _create_base_network(input_dim=INPUT_DIM)
        self.model = _create_meta_network(input_dim=INPUT_DIM, base_network=self.base_network)

        self.model.summary()
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                           min_delta=0,
                                           patience=2,
                                           verbose=1, mode='auto')

        if self.validation_comparisons is not None:
            X_1_test, X_2_test, y_test = self.create_dataset(self.validation_comparisons)
            X_1_test = normalize(X_1_test, norm='l2', axis=1, copy=True, return_norm=True)[0]
            X_2_test = normalize(X_2_test, norm='l2', axis=1, copy=True, return_norm=True)[0]


            hist = self.model.fit([X_1_train,X_2_train], y=y_train, callbacks=[es],  batch_size=batch_size, epochs=epochs, validation_data=([X_1_test,X_2_test],y_test))
        else:
            hist = self.model.fit([X_1_train,X_2_train], y=y_train, callbacks=[es],  batch_size=batch_size, epochs=epochs)
        data = pd.DataFrame.from_dict(hist.history)
        sns.lineplot(x=data.index, y="val_acc",
                     data=data).set_title("RanKNet "+str(len(X_2_train)))
        plt.show()
        print("val_acc: "+str(data['val_acc'].iloc[-1]))
        print("ready")
        self.validation_accuracy = data['val_acc'].iloc[-1]

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
        score = self.base_network.predict(x=features)
        return float(score)
