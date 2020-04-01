import csv
import os
import time
from collections import OrderedDict, Iterable

import gpuinfo
import six
from gpuinfo import GPUInfo
from gpuinfo.GPUInfo import gpu_usage
from tensorflow.keras.callbacks import Callback, CSVLogger
import numpy as np
from sklearn.metrics import classification_report


class ComputeMetricsLogger(Callback):
    """Callback that streams epoch results to a csv file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Example

    ```python
    csv_logger = CSVLogger('training.log')
    model.fit(X_train, Y_train, callbacks=[csv_logger])
    ```

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=True, id=None, gpu=None):
        self.id = id
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.file_flags = 'b' if six.PY2 and os.name == 'nt' else ''
        self.gpu = gpu
        super(ComputeMetricsLogger, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename, 'r' + self.file_flags) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a' + self.file_flags)
        else:
            self.csv_file = open(self.filename, 'w' + self.file_flags)

        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print('id: '+str(self.id))
        logs['id'] = self.id

        epoch_end_time = time.time()
        train_time = epoch_end_time - self.epoch_start_time
        logs['epoch_training_time'] = train_time

        if self.gpu is not None:
            gpu_u, gpu_m = gpu_usage()
            if gpu_u == 0.0:
                logs['gpu_consumption'] = 160 * train_time/60/60
            else:
                logs['gpu_consumption'] = gpu_m[int(self.gpu)] * train_time/60
        else:
            logs['gpu_consumption'] = train_time/60 * 250
        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, six.string_types):
                return k
            elif isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if self.model.stop_training:
            # We set NA so that csv parsers do not fail for this last epoch.
            logs = dict([(k, logs[k]) if k in logs else (k, 'NA') for k in self.keys])

        if not self.writer:
            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=['epoch'] + self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({'epoch': epoch})
        row_dict.update((key, handle_value(logs[key])) for key in self.keys)
        self.writer.writerow(row_dict)
        self.csv_file.flush()
        self.epoch_start_time = time.time()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None

class BatchTimeCallback(Callback):
    def on_train_begin(self, logs={}):
        self.batch_times = []

    def on_batch_end(self, batch, logs={}):
        self.batch_times.append(time.time())


class CategoryCallback(Callback):

    def __init__(self, validation_data):
        self.validation_data = validation_data
        self.model = None
        # Whether this Callback should only run on the chief worker in a
        # Multi-Worker setting.
        # TODO(omalleyt): Make this attr public once solution is stable.
        self._chief_worker_only = None
        self.history = {}


    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.validation_data
        Y_test = np.argmax(y_test, axis=1)  # Convert one-hot to index
        y_pred = self.model.predict_classes(x_test)
        classification_dict = classification_report(Y_test, y_pred,output_dict=True)
        del classification_dict['accuracy']
        del classification_dict['macro avg']
        del classification_dict['weighted avg']
        for k, v in classification_dict.items():
            name = 'precision'
            dict_name = name + '_' + str(k)
            self.history.setdefault(dict_name, []).append(v[name])
            logs[dict_name] = v[name]

            name = 'recall'
            dict_name = name+'_' + str(k)
            self.history.setdefault(dict_name, []).append(v[name])
            logs[dict_name] = v[name]

            name = 'f1-score'
            dict_name = name + '_' + str(k)
            self.history.setdefault(dict_name, []).append(v[name])
            logs[dict_name] = v[name]
