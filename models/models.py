import os
from typing import List, Tuple

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from models.callbacks import CategoryCallback, ComputeMetricsLogger
from models.config import ModelConfiguration

from utils import HyperParameterType

class TargetNetwork:
    def __init__(self, id:int, model_configuration:ModelConfiguration, num_classes:int, input_shape:Tuple):
        self.id = id
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_parameter_configuration = model_configuration
        self.model = self._create_model()
        self.model.summary()


    def train(self, X_train,y_train,X_val, y_val, epochs:int, batch_size:int):
        callbacks = []
        callbacks.append(ComputeMetricsLogger('training.log', gpu=os.getenv('CUDA_VISIBLE_DEVICES'), id=self.id))
        #callbacks.append(BatchTimeCallback())
        callbacks.append(CategoryCallback(validation_data=(X_val,y_val)))
        return self.model.fit(x=X_train,y=y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(X_val,y_val),
                              callbacks=callbacks)

    def _create_model(self):
        return Sequential()



class AlexNet(TargetNetwork):


    def _create_model(self):
        # Instantiate an empty model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape,name='conv_0'))
        for idx,hyper_parameter_config in enumerate(self.model_parameter_configuration.hyper_parameter_list):
            # Max Pooling
            model.add(MaxPooling2D(pool_size=(2, 2)))
            filters = int(hyper_parameter_config[HyperParameterType.NUMBER_OF_FILTERS])
            filter_height = int(hyper_parameter_config[HyperParameterType.FILTER_HEIGHT])
            filter_width = int(hyper_parameter_config[HyperParameterType.FILTER_WIDTH])

            model.add(
                Conv2D(filters=filters, kernel_size=(filter_width, filter_height), activation='relu',name='conv_'+str(idx+1)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.num_classes, activation='softmax'))
        opt = keras.optimizers.RMSprop(decay=1e-6)

        metrics = [keras.metrics.categorical_accuracy]
        metrics.append("accuracy")

        model.compile(optimizer=opt,
                      loss='categorical_crossentropy', metrics=metrics)
        return model
