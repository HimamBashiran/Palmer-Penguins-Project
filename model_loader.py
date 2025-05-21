import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import DATASET_PATH, MODEL_PATH, TARGET_COLUMN, ENCODE_COLUMNS, TARGET_MAPPER

class ModelLoader:
    def __init__(self):
        self.model = None

    def build_model(self):
        df = pd.read_csv(DATASET_PATH)
        for col in ENCODE_COLUMNS:
            dummies = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummies], axis=1)
            df.drop(columns=[col], inplace=True)

        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAPPER)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        model = RandomForestClassifier()
        model.fit(X, y)

        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        return model

    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            self.model = self.build_model()
        else:
            with open(MODEL_PATH, 'rb') as f:
                self.model = pickle.load(f)
        return self.model