import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None

    def build_model(self):
        df = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")
        target = 'species'
        encode = ['sex', 'island']

        for col in encode:
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy], axis=1)
            del df[col]

        target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
        df['species'] = df['species'].apply(lambda x: target_mapper[x])

        fitur = df.drop('species', axis=1)
        label = df['species']

        model = RandomForestClassifier()
        model.fit(fitur, label)

        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

        return model

    def load_model(self):
        if not os.path.exists(self.model_path):
            print(f"Model tidak ditemukan. Membangun model baru di {self.model_path}...")
            self.model = self.build_model()
        else:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        return self.model