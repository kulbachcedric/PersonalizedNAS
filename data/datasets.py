from urllib.request import urlretrieve

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist

"""
the various data sets are loaded via this class and returned to the 
search algorithm as training and evaluation data sets

            # 1: cifar10
            # 2: cifar100
            # 3: MNIST
            # 4: MNIST-Fashion 
            # 5: stl10 data set
"""


def load_data(index):
    if (index == 1):
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        num_classes = 10
        # Convert class vectors to binary class matrices.
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif (index == 2):
        print("data set cifar100")
        (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
        num_classes = 100
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif (index == 3):
        print("data set MNIST")
        # input image dimensions
        img_rows, img_cols = 28, 28

        # the data, split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        num_classes = 10

        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        # print('x_train shape:', x_train.shape)
        # print(x_train.shape[0], 'train samples')
        # print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    elif (index == 4):
        print("data set MNIST-Fashion")
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
        num_classes = 10
        # Convert class vectors to binary class matrices.
        # if we are using "channels first" ordering, then reshape the design
        # matrix such that the matrix is:
        # 	num_samples x depth x rows x columns
        if K.image_data_format() == "channels_first":
            x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
            x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))

        # otherwise, we are using "channels last" ordering, so the design
        # matrix shape should be: num_samples x rows x columns x depth
        else:
            x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
            x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

    elif (index == 5):
        # print("probiere STL 10 data set")
        import os
        import urllib as urllib
        import tarfile
        import sys
        import numpy as np

        HEIGHT, WIDTH, DEPTH = 96, 96, 3
        num_classes = 10
        SIZE = HEIGHT * WIDTH * DEPTH
        DATA_DIR = './stl10_data'
        DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
        TRAIN_DATA_PATH = DATA_DIR + '/stl10_binary/train_X.bin'
        TEST_DATA_PATH = DATA_DIR + '/stl10_binary/test_X.bin'
        TRAIN_LABELS_PATH = DATA_DIR + '/stl10_binary/train_y.bin'
        TEST_LABELS_PATH = DATA_DIR + '/stl10_binary/test_y.bin'
        CLASS_NAMES_PATH = DATA_DIR + '/stl10_binary/class_names.txt'

        def read_labels(path_to_labels):
            with open(path_to_labels, 'rb') as f:
                labels = np.fromfile(f, dtype=np.uint8)
                return labels

        def read_all_images(path_to_data):
            with open(path_to_data, 'rb') as f:
                # read whole file in uint8 chunks
                everything = np.fromfile(f, dtype=np.uint8)
                images = np.reshape(everything, (-1, DEPTH, WIDTH, HEIGHT))

                images = np.transpose(images, (0, 3, 2, 1))
                return images

        def download_and_extract():
            # if the dataset already exists locally, no need to download it again.
            if all((
                    os.path.exists(TRAIN_DATA_PATH),
                    os.path.exists(TRAIN_LABELS_PATH),
                    os.path.exists(TEST_DATA_PATH),
                    os.path.exists(TEST_LABELS_PATH),
            )):
                return

            dest_directory = DATA_DIR
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)

            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(dest_directory, filename)
            if not os.path.exists(filepath):
                def _progress(count, block_size, total_size):
                    sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                                  float(count * block_size) / float(
                                                                      total_size) * 100.0))
                    sys.stdout.flush()

                filepath, _ = urlretrieve(DATA_URL, filepath)
                print('Downloaded', filename)
                tarfile.open(filepath, 'r:gz').extractall(dest_directory)

        def load_dataset():
            # download the extract the dataset.
            download_and_extract()

            # load the train and test data and labels.
            x_train = read_all_images(TRAIN_DATA_PATH)
            y_train = read_labels(TRAIN_LABELS_PATH)
            x_test = read_all_images(TEST_DATA_PATH)
            y_test = read_labels(TEST_LABELS_PATH)

            if K.image_data_format() == "channels_first":
                x_train = x_train.reshape((x_train.shape[0], DEPTH, HEIGHT, WIDTH))
                x_test = x_test.reshape((x_test.shape[0], DEPTH, HEIGHT, WIDTH))
            else:
                x_train = x_train.reshape((x_train.shape[0], HEIGHT, WIDTH, DEPTH))
                x_test = x_test.reshape((x_test.shape[0], HEIGHT, WIDTH, DEPTH))

            x_train = x_train.astype('float32')
            x_train = (x_train - 127.5) / 127.5
            x_test = x_test.astype('float32')
            x_test = (x_test - 127.5) / 127.5

            # convert the labels to be zero based.
            y_train -= 1
            y_test -= 1

            # convert labels to hot-one vectors.
            y_train = keras.utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.to_categorical(y_test, num_classes)

            return (x_train, y_train), (x_test, y_test)

        (x_train, y_train), (x_test, y_test) = load_dataset()

    else:
        print("data set not found")

    # gibt die geladenen Datensaetze an den Suchalgorithmus zurueck
    return (x_train, y_train), (x_test, y_test)


def getNumClasses(index):
    if (index == 1):
        return 10
    if (index == 2):
        return 100
    if (index == 3):
        return 10
    if (index == 4):
        return 10
    if (index == 5):
        return 10
