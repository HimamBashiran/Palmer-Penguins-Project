import pandas as pd

class Encoder:
    def __init__(self, input_df, dataset_path):
        self.input_df = input_df
        self.dataset_path = dataset_path
        self.encoded_df = None

    def encode_input(self):
        penguins_raw = pd.read_csv(self.dataset_path)
        penguins = penguins_raw.drop(columns=['species'])
        df = pd.concat([self.input_df, penguins], axis=0)

        encode = ['sex', 'island']
        for col in encode:
            dummy = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, dummy], axis=1)
            del df[col]

        self.encoded_df = df[:1]
        return self.encoded_df
