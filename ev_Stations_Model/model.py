import joblib
from termcolor import colored
from ev_Stations_Model.data import get_train_data, clean_data
from ev_Stations_Model.encoders import label_encoder
from ev_Stations_Model.utils import unpivot
from ev_Stations_Model.features import timeFeatures, combine_event_feat
from ev_Stations_Model.params import LGBM_PARAMS
import lightgbm as lgb


class Trainer(object):
    def __init__(self, X_train, X_val):
        """
            X: pandas DataFrame preprocessing (i.e. including features engineering)
        """
        self.X_train = X_train
        self.X_val = X_val

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

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.model, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))


if __name__ == "__main__":
    # Get and clean data
    df_train = get_train_data()
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

    # Instantiate and train model
    trainer = Trainer(X_train=df_train_fea_augmented, X_val=df_val_fea_augmented)
    trainer.run()

    # Save model
    trainer.save_model()
