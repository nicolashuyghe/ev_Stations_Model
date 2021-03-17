from fastapi import FastAPI, Response, Request, Header
from fastapi.middleware.cors import CORSMiddleware
import os
import joblib
import pandas as pd
from ev_Stations_Model.features import combine_event_feat
from ev_Stations_Model.predict import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"]  # Allows all headers
    )


# ---------------------------------------------------------------------------
# - Save model in cache on api start up -
# ---------------------------------------------------------------------------

# Instanciate cache object to mem store models
cache_models = {}

@app.on_event("startup")
async def startup_event():
    # On api startup, load and store models in mem
    print("load model ...")
    dirname = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(dirname, 'model.joblib')
    model = joblib.load(model_path)
    cache_models["model_lgbm"] = model
    print("model #1 is ready ...")
    model_path_2 = os.path.join(dirname, 'model_2.joblib')
    model_2 = joblib.load(model_path_2)
    cache_models["model_lgbm_2"] = model_2
    print("model #2 is ready ...")


# ---------------------------------------------------------------------------
# - Handle a GET request with authentication -
# ---------------------------------------------------------------------------

@app.get("/predict")
def get_handler(request: Request, response: Response, station_id, year, month, day, hour, minute):

    params = {
        's_id': int(station_id),
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'minute': int(minute)
    }

    # Parse parameters into a dataframe
    df_pred_fea = pd.DataFrame.from_dict([params])

    # Add columns timestamp and weekdat to compute event features
    df_pred_fea['timestamp'] = pd.to_datetime(df_pred_fea[['year', 'month', 'day', 'hour', 'minute']])
    df_pred_fea['weekday'] = pd.to_datetime(df_pred_fea['timestamp']).dt.weekday

    # Add event features
    df_pred_fea_augmented = combine_event_feat(df_pred_fea)

    # Compute prediction
    idx_cols = ['s_id', 'year', 'month', 'day', 'hour', 'minute']
    y_pred = predict(df_pred_fea_augmented, model=cache_models["model_lgbm"], target_col="value", idx_cols=idx_cols, integer_output=True)
    y_pred = y_pred['value'][0]

    response_payload = {
        's_id': int(station_id),
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'minute': int(minute),
        "number_terminals_available": int(y_pred)
    }

    return response_payload

@app.get("/predict-type")
def get_handler(request: Request, response: Response, station_id, terminal_type, year, month, day, hour, minute):

    terminal_type_dict = {'normal':0,'fast':1}

    params = {
        's_id': int(station_id),
        't_type': terminal_type_dict[terminal_type],
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'minute': int(minute)
    }

    # Parse parameters into a dataframe
    df_pred_fea = pd.DataFrame.from_dict([params])

    # Add columns timestamp and weekdat to compute event features
    df_pred_fea['timestamp'] = pd.to_datetime(df_pred_fea[['year', 'month', 'day', 'hour', 'minute']])
    df_pred_fea['weekday'] = pd.to_datetime(df_pred_fea['timestamp']).dt.weekday

    # Add event features
    df_pred_fea_augmented = combine_event_feat(df_pred_fea)

    # Compute prediction
    idx_cols = ['s_id', 't_type', 'year', 'month', 'day', 'hour', 'minute']
    y_pred = predict(df_pred_fea_augmented, model=cache_models["model_lgbm_2"], target_col="value", idx_cols=idx_cols, integer_output=True)
    y_pred = y_pred['value'][0]

    response_payload = {
        's_id': int(station_id),
        't_type': terminal_type,
        'year': int(year),
        'month': int(month),
        'day': int(day),
        'hour': int(hour),
        'minute': int(minute),
        "number_terminals_available": int(y_pred)
    }

    return response_payload
