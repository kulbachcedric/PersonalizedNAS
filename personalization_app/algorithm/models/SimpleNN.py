import math
import os
import time
import json
import zipfile
from pathlib import Path
from zipfile import ZipFile
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import pandas as pd
import yaml
from django.core.files import File
from plotly.offline import plot
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import models
from personalization_app.algorithm.models import Individual
from personalization_app.models import Experiment, DbModel
import numpy as np
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

class SimpleNN(Individual):
    def __init__(self):
        self.optimizer='adam'
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.config = {}
        self.model = None
        self.plot_data = None
        self.db_model = None


    def load(self, db_model: DbModel):

        archive = zipfile.ZipFile(db_model.model.url, 'r')
        self.db_model = db_model
        self.plot_data = [pd.read_csv(archive.open(f.filename), sep=';') for f in archive.filelist if
                  f.filename.endswith('.csv')][0]
        self.model = [models.load(archive.open(f.filename)) for f in archive.filelist if
                  f.filename.endswith('.h5')][0]
        self.config = [json.load(archive.open(f.filename)) for f in archive.filelist if
                  f.filename.endswith('.json')][0]


    def get_div(self):
        assert self.plot_data is not None
        scatter1 = go.Scatter(x=self.plot_data['timestamp'], y=self.plot_data['data'])
        scatter2 = go.Scatter(x=self.plot_data['timestamp'], y=self.plot_data['train_prediction'])
        scatter3 = go.Scatter(x=self.plot_data['timestamp'], y=self.plot_data['test_prediction'])
        layout = go.Layout(title='Energy Plot', xaxis=dict(title='Date'),
                           yaxis=dict(title='(kWh)'))
        fig = go.Figure(data=[scatter1,scatter2,scatter3], layout=layout)
        return plot(fig, output_type='div', include_plotlyjs=False)

    def instantiate_random(self, experiment: Experiment):
        df, config = experiment.dataset.get_data()
        dataset = df[['value']]

        # normalize the dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size = int(len(dataset) * experiment.train_test_split)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]


        # reshape into X=t and Y=t+1
        look_back = 1
        trainX, trainY = self.create_dataset(dataset=train, look_back=look_back)
        testX, testY = self.create_dataset(test, look_back)

        # reshape input to be [samples, time steps, features]
        trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # create and fit the LSTM network
        self.model = Sequential()
        self.model.add(LSTM(4, input_shape=(1, look_back)))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.fit(trainX, trainY, epochs=10, batch_size=experiment.batch_size, verbose=2)

        timestr = time.strftime("%Y%m%d_%H%M%S")
        model_name = 'model'+timestr+'.h5'
        config_name = 'config'+timestr+'.json'
        zip_name = 'model'+timestr+'.zip'
        plot_name = 'plot_data'+timestr+".csv"

        model_path = Path(model_name)
        config_path = Path(config_name)
        zip_path = Path(zip_name)
        plot_data_path = Path(plot_name)

        json_string = json.dumps(self.config)
        f = open(config_name, "w+")
        f.write(json_string)
        f.close()
        self.model.save(str(model_path))

        # make predictions
        trainPredict = self.model.predict(trainX)
        testPredict = self.model.predict(testX)
        # invert predictions
        trainPredict = scaler.inverse_transform(trainPredict)
        trainY = scaler.inverse_transform([trainY])
        testPredict = scaler.inverse_transform(testPredict)
        testY = scaler.inverse_transform([testY])
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        print('Train Score: %.2f RMSE' % (trainScore))
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
        # plot baseline and predictions
        train_data = scaler.inverse_transform(dataset)
        self.plot_data = pd.DataFrame({'data': list(train_data.flat),
                          'train_prediction': list(trainPredictPlot.flat),
                          'test_prediction': list(testPredictPlot.flat)})
        self.plot_data['timestamp'] = df['timestamp']

        self.plot_data.to_csv(str(plot_data_path))

        ## Save files

        with ZipFile(zip_name, 'w') as zip:
            zip.write(str(plot_data_path))
            zip.write(str(model_path))
            zip.write(str(config_path))
            zip.close()

        zip_file = open(str(zip_path),'rb')
        db_file = File(zip_file)
        db_model = DbModel(experiment=experiment, model=db_file, initial_tag=True)
        db_model.full_clean()
        db_model.save()
        zip_file.close()

        os.remove(str(model_path))
        os.remove(str(config_path))
        os.remove(str(plot_data_path))
        os.remove(str(zip_path))

    def instantiate_mutation(self):
        super().instantiate_mutation()

    def predict(self, X):
        return super().predict(X)

    ## Helper functions

    def create_dataset(self,dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)