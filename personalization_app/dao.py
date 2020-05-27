import zipfile

import yaml

from personalization_app.models import Dataset, DATASET_CHOICES, DbModel, MODEL_TYPE_CHOICES
import pandas as pd
from plotly.offline import plot
import plotly.graph_objs as go

def get_dataset_view(dataset:Dataset):
    archive = zipfile.ZipFile(dataset.data.path, 'r')
    if dataset.type == DATASET_CHOICES[0][0]:
        df = [pd.read_csv(archive.open(f.filename),sep=';') for f in archive.filelist if f.filename.endswith('data.csv')][0]
        config = [yaml.load(archive.open(f.filename)) for f in archive.filelist if f.filename.endswith('config.yml')][0]
        scatter = go.Scatter(x=df['timestamp'], y=df['value'])
        layout = go.Layout(title='Energy Plot', xaxis=dict(title='Date'),
                           yaxis=dict(title='(kWh)'))
        fig = go.Figure(data=[scatter], layout=layout)
        return plot(fig, output_type='div',include_plotlyjs=False)

    else:
        pass

def get_model_view(model:DbModel):
    archive = zipfile.ZipFile(model.model.path,'r')
    if model.type == MODEL_TYPE_CHOICES[1][0]:
        df = [pd.read_csv(archive.open(f.filename),sep=';') for f in archive.filelist if f.filename.endswith('vis.csv')]
