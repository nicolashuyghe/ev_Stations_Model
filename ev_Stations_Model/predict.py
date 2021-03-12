import joblib
import pandas as pd
import numpy as np
from ev_Stations_Model.data import clean_data
from ev_Stations_Model.encoders import label_encoder
from ev_Stations_Model.utils import unpivot
from ev_Stations_Model.features import timeFeatures, combine_event_feat


TEST_PATH = "raw_data/ytrain_raw.csv"
PATH_TO_LOCAL_MODEL = "model.joblib"
PRED_PATH = "data/ypred.csv"

def predict(df, model, target_col, idx_cols, integer_output=True):
    """Predict target variable with a trained LightGBM model.
    Args:
        df (pandas.DataFrame): Dataframe including all needed features
        model (lightgbm.Booster): Trained LightGBM booster model
        target_col (str): Name of the target column
        idx_col (list[str]): List of the names of the index columns, e.g. ["store", "brand", "week"]
        integer_output (bool): It it is True, the forecast will be rounded to an integer

    Returns:
        pandas.DataFrame including the predictions of the target variable
    """
    if target_col in df.columns:
        df = df.drop(target_col, axis=1)
    predictions = pd.DataFrame({target_col: model.predict(df)})

    if integer_output:
        predictions[target_col] = predictions[target_col].apply(lambda x: round(x))

    return pd.concat([df[idx_cols].reset_index(drop=True), predictions], axis=1)


def get_test_data(nrows=None):
    '''returns a DataFrame with nrows from local path'''
    df = pd.read_csv(TEST_PATH, nrows=nrows)
    return df


def get_model(path_to_joblib):
    pipeline = joblib.load(path_to_joblib)
    return pipeline


def generate_csv(path, df_pred):
    df_pred['timestamp'] = pd.to_datetime(df_pred[['year', 'month', 'day', 'hour', 'minute']])
    ypred = df_pred[['timestamp','s_id','value']].rename(columns={'value':'value_pred'})
    ypred.to_csv(path, index=False)


if __name__ == '__main__':
    # Get and clean test data
    df_test = get_test_data()
    df_test_cleaned = clean_data(df_test)

    # Encode data
    df_test_encoded = label_encoder(df_test_cleaned)

    # Unpivot dataset
    df_test_unpivot = unpivot(df_test_encoded)

    # Add time features
    df_test_fea = timeFeatures(df_test_unpivot)

    # Add event features
    df_test_fea_augmented = combine_event_feat(df_test_fea)

    # Load model
    model = joblib.load(PATH_TO_LOCAL_MODEL)

    # Predict
    idx_cols = ['s_id', 'year', 'month', 'day', 'hour', 'minute']
    df_pred = predict(df_test_fea_augmented, model=model, target_col="value", idx_cols=idx_cols, integer_output=True)\
            .sort_values(by=idx_cols)\
            .reset_index(drop=True)

    # Generate csv
    generate_csv(PRED_PATH, df_pred)
