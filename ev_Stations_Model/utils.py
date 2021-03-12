"""A module for all utils functions and decorators
"""
import time
import numpy as np
import pandas as pd


################
#### METRICS ###
################

def compute_mae(y_pred, y_true):
    return (np.abs(y_pred - y_true)).mean()


################
# PREPROCESSING #
################

def unpivot(X):
    assert isinstance(X, pd.DataFrame)

    X_unpivot = X.melt(ignore_index=False).rename(columns={'variable':'s_id'})

    return X_unpivot


################
#  DECORATORS  #
################

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed
