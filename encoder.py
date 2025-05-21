import pandas as pd
from config import DATASET_PATH, ENCODE_COLUMNS

class Encoder:
    def __init__(self, input_df):
        self.input_df = input_df.copy()
        self.df_raw = pd.read_csv(DATASET_PATH)

    def _drop_label_column(self, df):
        """Menghapus kolom target label jika ada (default: 'species')."""
        if 'species' in df.columns:
            return df.drop(columns=['species'])
        return df

    def _one_hot_encode(self, df):
        """Melakukan one-hot encoding untuk kolom kategorikal."""
        for col in ENCODE_COLUMNS:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    def encode_input(self):
        # Buang kolom label dari input pengguna juga
        input_clean = self._drop_label_column(self.input_df)

        # Buang kolom label dari data asli
        df_raw_clean = self._drop_label_column(self.df_raw)

        # Gabungkan input user dengan data asli
        df_combined = pd.concat([input_clean, df_raw_clean], axis=0)

        # One-hot encoding
        df_encoded = self._one_hot_encode(df_combined)

        # Ambil hanya baris pertama (baris user input)
        return df_encoded.iloc[:1]

