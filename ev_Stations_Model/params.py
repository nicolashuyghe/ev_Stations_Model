### GCP configuration - - - - - - - - - - - - - - - - - - -

# /!\ you should fill these according to your account

### GCP Project - - - - - - - - - - - - - - - - - - - - - -

# not required here

### GCP Storage - - - - - - - - - - - - - - - - - - - - - -

BUCKET_NAME = 'wagon-ev-scharging-stations'

##### Data  - - - - - - - - - - - - - - - - - - - - - - - -

# train data file location
BUCKET_TRAIN_DATA_PATH = 'data/y_train_raw.csv'

##### Training  - - - - - - - - - - - - - - - - - - - - - -

# not required here

##### Model - - - - - - - - - - - - - - - - - - - - - - - -

# model folder name (will contain the folders for all trained model versions)
MODEL_NAME = 'models'

# model version folder name (where the trained model.joblib file will be stored)
MODEL_VERSION = 'LightGbm'

### GCP AI Platform - - - - - - - - - - - - - - - - - - - -

# not required here

### Parameters of LightGBM model - - - - - - - - - - - - - -
LGBM_PARAMS = {
    "objective": "mape",
    "num_leaves": 124,
    "min_data_in_leaf": 340,
    "learning_rate": 0.1,
    "feature_fraction": 0.65,
    "bagging_fraction": 0.87,
    "bagging_freq": 19,
    "num_rounds": 500,
    "early_stopping_rounds": 125,
    "num_threads": 16,
    "seed": 1,
}
