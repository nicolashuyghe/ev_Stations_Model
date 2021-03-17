import pandas as pd
import numpy as np

def label_encoder(X):
    assert isinstance(X, pd.DataFrame)

    X_ = X.copy()

    # List terminals_name
    terminals_name = [col for col in X_.columns if col.startswith('S')]

    # Encode labels
    for col in terminals_name:
        X_[col] = np.where(X_[col]=='Available',1,0)

    # Rename columns
    stations_id = [int(col[1:-3]) for col in X_.columns]
    cols_dict = {terminals_name[i]: stations_id[i] for i in range(len(terminals_name))}
    X_ = X_.rename(columns=cols_dict)

    # Group columns by stations
    X_stations = X_.groupby(level=0, axis=1).sum()

    return X_stations
