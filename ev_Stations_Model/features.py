"""A module for all features engineering functions
"""

import pandas as pd
import numpy as np

################
# FEATURES ENG #
################


def timeFeatures(X):
    assert isinstance(X, pd.DataFrame)

    X_fea = X.copy()

    if 'timestamp' not in X_fea.columns:
        X_fea.reset_index(inplace=True)

    X_fea['year'] = pd.to_datetime(X_fea['timestamp']).dt.year
    X_fea['month'] = pd.to_datetime(X_fea['timestamp']).dt.month
    X_fea['day'] = pd.to_datetime(X_fea['timestamp']).dt.day
    X_fea['hour'] = pd.to_datetime(X_fea['timestamp']).dt.hour
    X_fea['minute'] = pd.to_datetime(X_fea['timestamp']).dt.minute
    X_fea['weekday'] = pd.to_datetime(X_fea['timestamp']).dt.weekday

    return X_fea[['timestamp', 's_id', 'year', 'month', 'day', 'hour', 'minute', 'weekday', 'value']]


def lockdow_feat(df):
    # Define lockdown dates
    start_lock_1 = '2020-03-17'
    end_lock_1 = '2020-05-10'
    start_lock_2 = '2020-10-31'
    end_lock_2 = '2020-12-15'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['lockdown'] = np.where(\
                    ((df_fea['timestamp'] >= start_lock_1) & (df_fea['timestamp'] <= end_lock_1)|\
                    (df_fea['timestamp'] >= start_lock_2) & (df_fea['timestamp'] <= end_lock_2)), 1, 0)

    return df_fea


def car_free_feat(df):
    # Define lockdown dates
    car_free_day = '2020-09-27'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['car_free'] = np.where(df_fea['timestamp'] == car_free_day, 1, 0)

    return df_fea


def saints_holidays_feat(df):
    # Define lockdown dates
    start_holidays_1 = '2019-10-19'
    end_holidays_1 = '2019-11-3'
    start_holidays_2 = '2020-10-17'
    end_holidays_2 = '2020-11-01'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['saints_holidays'] = np.where(\
                    ((df_fea['timestamp'] >= start_holidays_1) & (df_fea['timestamp'] <= end_holidays_1)|\
                    (df_fea['timestamp'] >= start_holidays_2) & (df_fea['timestamp'] <= end_holidays_2)), 1, 0)

    return df_fea


def christmas_holidays_feat(df):
    # Define lockdown dates
    start_holidays = '2019-12-21'
    end_holidays = '2020-01-5'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['christmas_holidays'] = np.where(\
                    (df_fea['timestamp'] >= start_holidays) & (df_fea['timestamp'] <= end_holidays), 1, 0)

    return df_fea


def winter_holidays_feat(df):
    # Define lockdown dates
    start_holidays = '2020-02-08'
    end_holidays = '2020-02-23'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['winter_holidays'] = np.where(\
                    (df_fea['timestamp'] >= start_holidays) & (df_fea['timestamp'] <= end_holidays), 1, 0)

    return df_fea


def easter_holidays_feat(df):
    # Define lockdown dates
    start_holidays = '2020-04-04'
    end_holidays = '2020-04-19'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['easter_holidays'] = np.where(\
                    (df_fea['timestamp'] >= start_holidays) & (df_fea['timestamp'] <= end_holidays), 1, 0)

    return df_fea


def summer_holidays_feat(df):
    # Define lockdown dates
    start_holidays = '2020-07-04'
    end_holidays = '2020-09-01'

    # Create new column lockdown
    df_fea = df.copy()

    if 'timestamp' not in df_fea.columns:
        df_fea.reset_index(inplace=True)

    df_fea['summer_holidays'] = np.where(\
                    (df_fea['timestamp'] >= start_holidays) & (df_fea['timestamp'] <= end_holidays), 1, 0)

    return df_fea


def combine_event_feat(df, lockdown=True, car_free=True, saints=True, christmas=True, winter=True, easter=True, summer=True, drop_timestamp=True):
    '''Combine event features functions into one function'''

    # Create a copy of the image
    df_fea = df.copy()

    if lockdown:
        df_fea = lockdow_feat(df_fea)
    if car_free:
        df_fea = car_free_feat(df_fea)
    if saints:
        df_fea = saints_holidays_feat(df_fea)
    if christmas:
        df_fea = christmas_holidays_feat(df_fea)
    if winter:
        df_fea = winter_holidays_feat(df_fea)
    if easter:
        df_fea = easter_holidays_feat(df_fea)
    if summer:
        df_fea = summer_holidays_feat(df_fea)
    if drop_timestamp:
        df_fea.drop(columns='timestamp', inplace=True)

    return df_fea
