import joblib
from termcolor import colored
from ev_Stations_Model.data import get_train_data, clean_data, get_data_from_gcp
from ev_Stations_Model.encoders import label_encoder
from ev_Stations_Model.utils import unpivot
from ev_Stations_Model.features import timeFeatures, combine_event_feat
from ev_Stations_Model.params import LGBM_PARAMS, BUCKET_NAME, MODEL_NAME, MODEL_VERSION
import lightgbm as lgb
from google.cloud import storage


class Trainer(object):
    def __init__(self, X_train, X_val, **kwargs):
        """
            X: pandas DataFrame preprocessing (i.e. including features engineering)
        """
        self.X_train = X_train
        self.X_val = X_val
        self.local = kwargs.get("local", True)  # if True training is done locally

    def create_dataset(self):
        print(colored("Creating dataset...", "green"))
        self.dataset_train = lgb.Dataset(self.X_train.drop(["value"], axis=1, inplace=False), label=self.X_train["value"])
        self.dataset_val = lgb.Dataset(self.X_val.drop(["value"], axis=1, inplace=False), label=self.X_val["value"])

    def run(self):
        # Create dataset
        self.create_dataset()

        # Train LightGBM model
        print(colored("Training LightGBM model...", "green"))
        categ_fea = ["s_id"]
        self.model = lgb.train(params=LGBM_PARAMS, train_set=self.dataset_train, valid_sets=[self.dataset_val], categorical_feature=categ_fea, verbose_eval=True)

    def save_model(self, local=True):
        """Save the model into a .joblib format"""
        # saving the trained model to disk
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        if not self.local:
            storage.storage_upload(model_version=MODEL_VERSION)

    def save_model_to_gcp(self):
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/model.joblib"
        blob = client.blob(storage_location)
        blob.upload_from_filename('model.joblib')
        print("uploaded model.joblib to gcp cloud storage under \n => {}".format(storage_location))


if __name__ == "__main__":
    # Get and clean data
    df_train = get_data_from_gcp()
    df_train_cleaned = clean_data(df_train)

    # Encode data
    df_train_encoded = label_encoder(df_train_cleaned)

    # Handout
    train_size = int(len(df_train_encoded)*0.8)
    val_size = len(df_train_encoded)-train_size

    df_train = df_train_encoded.iloc[0:train_size]
    df_val = df_train_encoded.iloc[train_size:len(df_train_encoded)]

    # Unpivot dataset
    df_train_unpivot = unpivot(df_train)
    df_val_unpivot = unpivot(df_val)

    # Add time features
    df_train_fea = timeFeatures(df_train_unpivot)
    df_val_fea = timeFeatures(df_val_unpivot)

    # Add event features
    df_train_fea_augmented = combine_event_feat(df_train_fea)
    df_val_fea_augmented = combine_event_feat(df_val_fea)

    # # Instantiate and train model
    trainer = Trainer(X_train=df_train_fea_augmented, X_val=df_val_fea_augmented, local=False)
    trainer.run()

    # Save model to gcp cloud storage
    trainer.save_model_to_gcp()
