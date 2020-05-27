import datetime

from django.core.files import File
from tqdm import tqdm

from personalization_app.algorithm.models import Individual
from personalization_app.models import Experiment, DbModel
from tensorflow.keras.layers import Dense,Input,LSTM
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import glorot_normal
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class DeepAR(Individual):

    def __init__(self):
        self.loss = gaussian_likelihood
        self.optimizer='adam'
        self.get_intermediate = None
        self._output_layer_name = 'main_output'
        super(DeepAR, self).__init__()

    def load(self, db_model: DbModel):
        self.model = load_model(db_model.model.url)
        self.get_intermediate = K.function(inputs=[self.model.input],
                                           outputs=self.model.get_layer(self._output_layer_name).output)

    def instantiate_random(self, experiment: Experiment):
        df, config = experiment.dataset.get_data()

        time_regex = config['config']['timestamp_regex']
        df['timestamp'] = df['timestamp'].apply(lambda x: datetime.datetime.strptime(x, time_regex))
        df['year'] = df['timestamp'].apply(lambda x: x.year)
        df['month'] = df['timestamp'].apply(lambda x: x.month)
        df['day'] = df['timestamp'].apply(lambda x: x.day)
        df['hour'] = df['timestamp'].apply(lambda x: x.hour)
        df['minute'] = df['timestamp'].apply(lambda x: x.minute)
        df['second'] = df['timestamp'].apply(lambda x: x.second)
        df['microsecond'] = df['timestamp'].apply(lambda x: x.microsecond)
        df['weekday'] = df['timestamp'].apply(lambda x: x.weekday())

        cut_off = int(df.count(axis=0) * experiment.train_test_split)

        y = df.pop(['value'])
        X = df.drop(['timestamp'])
        y_train = y.head(cut_off)
        X_train = X.head(cut_off)
        y_test = y.tail(df.count(axis=0) - cut_off)
        X_test = X.tail(df.count(axis=0) - cut_off)

        PREDICTION_WINDOW = 1

        y = df.pop('value')[PREDICTION_WINDOW:]
        df.pop('timestamp')
        X = df[:-PREDICTION_WINDOW]
        X['category'] = ['1' for i in range(X.shape[0])]

        source_df = X
        source_df['target'] = y

        ts = TimeSeries(pandas_df=source_df, scaler=MinMaxScaler)

        input_shape = (self.batch_size, 8)
        inputs = Input(shape=input_shape)
        x = LSTM(4, return_sequences=True)(inputs)
        x = Dense(3, activation='relu')(x)
        loc, scale = GaussianLayer(1, name='main_output')(x)
        theta = [loc, scale]
        model = Model(inputs, theta[0])
        model.compile(loss=self.loss(theta[1]), optimizer=self.optimizer)
        model.fit_generator(ts_generator(self.ts_obj,
                                         input_shape[0]),
                            steps_per_epoch=self.steps_per_epoch,
                            epochs=self.epochs)
        model.save('test.h5')

        db_model = DbModel(experiment=experiment, model=File(open('test.h5','rb')))
        db_model.full_clean()
        db_model.save()
        self.get_intermediate = K.function(inputs=[self.model.input],
                                           outputs=self.model.get_layer(self._output_layer_name).output)


    def instantiate_mutation(self):
        super().instantiate_mutation()

    def predict(self, X):
        ress = []
        for i in tqdm(range(10)):
            sample = X
            sample = np.array(sample).reshape(1, self.model.experiment.batch_size, 8)
            output = self.predict_theta_from_input([sample])
            samples = []
            for mu, sigma in zip(output[0].reshape(self.model.experiment.batch_size), output[1].reshape(self.model.experiment.batch_size)):
                samples.append(np.random.normal(loc=mu, scale=np.sqrt(sigma), size=1)[0])
            pred = np.array(samples)
            ress.append(pred)

    def predict_theta_from_input(self,sample):
        """
                This function takes an input of size equal to the n_steps specified in 'Input' when building the
                network
                :param input_list:
                :return: [[]], a list of list. E.g. when using Gaussian layer this returns a list of two list,
                corresponding to [[mu_values], [sigma_values]]
                """
        if not self.get_intermediate:
            raise ValueError('TF model must be trained first!')
        return self.get_intermediate(sample)

def ts_generator(ts_obj, n_steps):
    """
    This is a util generator function for Keras
    :param ts_obj: a Dataset child class object that implements the 'next_batch' method
    :param n_steps: parameter that specifies the length of the net's input tensor
    :return:
    """
    while 1:
        batch = ts_obj.next_batch(1, n_steps)
        yield batch[0], batch[1]

def gaussian_likelihood(sigma):
    def gaussian_loss(y_true, y_pred):
        return tf.reduce_mean(input_tensor=0.5*tf.math.log(sigma) + 0.5*tf.math.truediv(tf.math.square(y_true - y_pred), sigma)) + 1e-6 + 6
    return gaussian_loss



class GaussianLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        self.kernel_1, self.kernel_2, self.bias_1, self.bias_2 = [], [], [], []
        super(GaussianLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        n_weight_rows = input_shape[2]
        self.kernel_1 = self.add_weight(name='kernel_1',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.kernel_2 = self.add_weight(name='kernel_2',
                                        shape=(n_weight_rows, self.output_dim),
                                        initializer=glorot_normal(),
                                        trainable=True)
        self.bias_1 = self.add_weight(name='bias_1',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        self.bias_2 = self.add_weight(name='bias_2',
                                      shape=(self.output_dim,),
                                      initializer=glorot_normal(),
                                      trainable=True)
        super(GaussianLayer, self).build(input_shape)
    def call(self, x):
        output_mu = K.dot(x, self.kernel_1) + self.bias_1
        output_sig = K.dot(x, self.kernel_2) + self.bias_2
        output_sig_pos = K.log(1 + K.exp(output_sig)) + 1e-06
        return [output_mu, output_sig_pos]
    def compute_output_shape(self, input_shape):
        """
        The assumption is the output ts is always one-dimensional
        """
        return [(input_shape[0], self.output_dim), (input_shape[0], self.output_dim)]

class TimeSeries():
    def __init__(self, pandas_df, one_hot_root_list=None, grouping_variable='category', scaler=None):
        super().__init__()
        self.data = pandas_df
        self.one_hot_root_list = one_hot_root_list
        self.grouping_variable = grouping_variable
        if self.data is None:
            raise ValueError('Must provide a Pandas df to instantiate this class')
        self.scaler = scaler
    def _one_hot_padding(self, pandas_df, padding_df):
        """
        Util padding function
        :param padding_df:
        :param one_hot_root_list:
        :return:
        """
        for one_hot_root in self.one_hot_root_list:
            one_hot_columns = [i for i in pandas_df.columns   # select columns equal to 1
                               if i.startswith(one_hot_root) and pandas_df[i].values[0] == 1]
            for col in one_hot_columns:
                padding_df[col] = 1
        return padding_df
    def _pad_ts(self, pandas_df, desired_len, padding_val=0):
        """
        Add padding int to the time series
        :param pandas_df:
        :param desired_len: (int)
        :param padding_val: (int)
        :return: X (feature_space), y
        """
        pad_length = desired_len - pandas_df.shape[0]
        padding_df = pd.concat([pd.DataFrame({col: padding_val for col in pandas_df.columns},
                                             index=[i for i in range(pad_length)])])
        if self.one_hot_root_list:
            padding_df = self._one_hot_padding(pandas_df, padding_df)
        return pd.concat([padding_df, pandas_df]).reset_index(drop=True)
    @staticmethod
    def _sample_ts(pandas_df, desired_len):
        """
        :param pandas_df: input pandas df with 'target' columns e features
        :param desired_len: desired sample length (number of rows)
        :param padding_val: default is 0
        :param initial_obs: how many observations to skip at the beginning
        :return: a pandas df (sample)
        """
        if pandas_df.shape[0] < desired_len:
            raise ValueError('Desired sample length is greater than df row len')
        if pandas_df.shape[0] == desired_len:
            return pandas_df
        start_index = np.random.choice([i for i in range(0, pandas_df.shape[0] - desired_len + 1)])
        return pandas_df.iloc[start_index: start_index+desired_len, ]
    def next_batch(self, batch_size, n_steps,
                   target_var='target', verbose=False,
                   padding_value=0):
        """
        :param batch_size: how many time series to be sampled in this batch (int)
        :param n_steps: how many RNN cells (int)
        :param target_var: (str)
        :param verbose: (boolean)
        :param padding_value: (float)
        :return: X (feature space), y
        """
        # Select n_batch time series
        groups_list = self.data[self.grouping_variable].unique()
        np.random.shuffle(groups_list)
        selected_groups = groups_list[:batch_size]
        input_data = self.data[self.data[self.grouping_variable].isin(set(selected_groups))]
        # Initial padding for each selected time series to reach n_steps
        sampled = []
        for cat, cat_data in input_data.groupby(self.grouping_variable):
                if cat_data.shape[0] < n_steps:
                    sampled_cat_data = self._pad_ts(pandas_df=cat_data,
                                                    desired_len=n_steps,
                                                    padding_val=padding_value)
                else:
                    sampled_cat_data = self._sample_ts(pandas_df=cat_data,
                                                       desired_len=n_steps)
                sampled.append(sampled_cat_data)
        rnn_output = pd.concat(sampled).drop(columns=self.grouping_variable).reset_index(drop=True)
        if self.scaler:
            batch_scaler = self.scaler()
            n_rows = rnn_output.shape[0]
            # Scaling must be extended to handle multivariate time series!
            #rnn_output['feature_1'] = rnn_output.feature_1.astype('float')
            #rnn_output[target_var] = rnn_output[target_var].astype('float')

            rnn_output.astype('float')
            rnn_output[rnn_output.columns.tolist()] = batch_scaler.fit_transform(rnn_output.values.reshape(n_rows,9)).reshape(n_rows,9)
            #rnn_output['feature_1'] = batch_scaler.fit_transform(rnn_output.feature_1.values.reshape(n_rows, 1)).reshape(n_rows)
            #rnn_output[target_var] = batch_scaler.fit_transform(rnn_output[target_var].values.reshape(n_rows, 1)).reshape(n_rows)
        return rnn_output.loc[:, rnn_output.columns != target_var].values.reshape(batch_size, n_steps, -1), \
               rnn_output[target_var].values.reshape(batch_size, n_steps, 1)