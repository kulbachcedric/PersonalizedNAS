from tensorflow.python.keras import Sequential, Input, Model
from tensorflow.python.keras.layers import Subtract, Activation, Dense

class RankNetClassifier():

    def __init__(self, batch_size=200, epochs=1000, input_shape=(8,)):
        self.__BATCH_SIZE = batch_size
        self.__NUM_EPOCHS = epochs
        self.__INPUT_SHAPE = input_shape

        def _create_base_network():
            '''Base network to be shared (eq. to feature extraction).
            '''
            seq = Sequential()
            # seq.add(Dense(input_dim, input_shape=(input_dim,)))
            # seq.add(Dropout(0.1))
            # seq.add(Dense(128, activation='relu'))
            # seq.add(Dropout(0.1))
            # seq.add(Dense(64, activation='relu'))
            # seq.add(Dropout(0.1))
            seq.add(Dense(16))
            seq.add(Dense(1))
            return seq

        def _create_meta_network(input_shape, base_network):
            input_a = Input(shape=input_shape)
            input_b = Input(shape=input_shape)

            rel_score = base_network(input_a)
            irr_score = base_network(input_b)

            # subtract scores
            diff = Subtract()([rel_score, irr_score])

            # Pass difference through sigmoid function.
            prob = Activation("sigmoid")(diff)

            # Build model.
            model = Model(inputs=[input_a, input_b], outputs=prob)
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])  # loss="binary_crossentropy"

            return model



        self.base_network = _create_base_network()
        self.model = _create_meta_network(self.__INPUT_SHAPE, self.base_network)

    def fit(self,X_1,X_2,y, **kwargs):

        self.history = self.model.fit([X_1, X_2], y,
                                # validation_data=([X_1_test, X_2_test], y_test),
                                batch_size=self.__BATCH_SIZE, epochs=self.__NUM_EPOCHS, verbose=3)

    def predict(self,X):
        return self.model.predict(X)
