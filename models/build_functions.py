from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Activation, Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.python.keras.optimizer_v2.adam import Adam

NUMBER_FILTERS = [8, 16, 32, 48, 64]
FILTER_HEIGHT = [3,5,7,9]
FILTER_WIDTH = [3,5,7,9]


def build_AlexNet(
        n_filters_1=96,
        kernel_size_1 = (11, 11),
        strides_1 = (4,4),

        n_filters_2=256,
        kernel_size_2=(5, 5),
        strides_2=(1, 1),

        n_filters_3=384,
        kernel_size_3=(3,3),
        strides_3=(1,1),

        n_filters_4=384,
        kernel_size_4=(3, 3),
        strides_4=(1, 1),

        n_filters_5=256,
        kernel_size_5=(3,3),
        strides_5=(1, 1),

        pool_size_max_pooling=(2,2),
        strides_max_pooling=(2,2),

        n_dense_1 = 4096,
        n_dense_2 = 4096,
        n_dense_3 = 1000,
        n_dense_4 = 1,

        dropout = 0.4,
        learning_rate = 3e-3,
        loss = categorical_crossentropy

        ):

    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(input_shape=(32,32,3), filters=n_filters_1, kernel_size=kernel_size_1, strides=strides_1, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=n_filters_2, kernel_size=kernel_size_2, strides=strides_2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=n_filters_3, kernel_size=kernel_size_3, strides=strides_3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=n_filters_4, kernel_size=kernel_size_4, strides=strides_4, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=n_filters_5, kernel_size=kernel_size_5, strides=strides_5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(n_dense_1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(dropout))

    # 2nd Fully Connected Layer
    model.add(Dense(n_dense_2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(dropout))

    # 3rd Fully Connected Layer
    model.add(Dense(n_dense_3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(dropout))

    # Output Layer
    model.add(Dense(n_dense_4))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))


    optimizer = Adam(lr=learning_rate)
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model

def build_AlexNetSimple(
        learning_rate=3e-3,
        loss=categorical_crossentropy,
        metrics=[categorical_accuracy],
        input_shape=(3,2,1),
        num_classes=10,
        n_filters_1=96,
        kernel_size_1=(11, 11),
        strides_1=(4, 4),

        n_filters_2=256,
        kernel_size_2=(5, 5),
        strides_2=(1, 1),

        n_filters_3=384,
        kernel_size_3=(3, 3),
        strides_3=(1, 1),

        n_filters_4=384,
        kernel_size_4=(3, 3),
        strides_4=(1, 1),

        n_filters_5=256,
        kernel_size_5=(3, 3),
        strides_5=(1, 1),

        pool_size_max_pooling=(2, 2),
        strides_max_pooling=(2, 2),

        n_dense_1=4096,
        n_dense_2=4096,
        n_dense_3=1000,

        dropout=0.4,
    ):


    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(input_shape=input_shape, filters=n_filters_1, kernel_size=kernel_size_1, strides=strides_1,
                     padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=n_filters_2, kernel_size=kernel_size_2, strides=strides_2, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=n_filters_3, kernel_size=kernel_size_3, strides=strides_3, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=n_filters_4, kernel_size=kernel_size_4, strides=strides_4, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=n_filters_5, kernel_size=kernel_size_5, strides=strides_5, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size_max_pooling, strides=strides_max_pooling, padding='same'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(n_dense_1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(dropout))

    # 2nd Fully Connected Layer
    model.add(Dense(n_dense_2))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(dropout))

    # 3rd Fully Connected Layer
    model.add(Dense(n_dense_3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(dropout))

    # Output Layer
    model.add(Dense(num_classes))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))


    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=metrics)
    return model
