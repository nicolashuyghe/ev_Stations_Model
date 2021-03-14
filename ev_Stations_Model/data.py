"""A module for loading and cleaning historical data
"""

import pandas as pd
from ev_Stations_Model.utils import simple_time_tracker
from google.cloud import storage

TRAIN_PATH = "raw_data/ytrain_raw.csv"
GS_PATH = "gs://ev-stations-bucket-202103/raw_data/ytrain_raw.csv"


@simple_time_tracker
def get_data_from_gcp(nrows=10000, local=False, **kwargs):
    """method to get the training data (or a portion of it) from google cloud bucket"""
    # Add Client() here
    client = storage.Client()
    if local:
        path = LOCAL_PATH
    else:
        path = GS_PATH
    df = pd.read_csv(path, nrows=nrows)
    return df


def get_train_data(nrows=None):
    '''returns a DataFrame with nrows from local path'''
    df = pd.read_csv(TRAIN_PATH, nrows=nrows)
    return df


def clean_data(df):
    df_clean = df.copy()
    df_clean = df_clean.fillna('Offline')
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    df_clean = df_clean.set_index('timestamp')
    return df_clean


if __name__ == '__main__':
    df = get_train_data()
    df_clean = clean_data(df)
