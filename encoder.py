# Import library yang dibutuhkan
import pandas as pd
from config import DATASET_PATH, ENCODE_COLUMNS

# Class untuk menangani encoding data input (One-Hot Encoding)
class Encoder:
    def __init__(self, input_df):
        self.input_df = input_df.copy()             # Salin data input dari pengguna
        self.df_raw = pd.read_csv(DATASET_PATH)     # Baca dataset asli dari file

    # Fungsi untuk menghapus kolom target/label jika ada (default: 'species')
    def _drop_label_column(self, df):
        """Menghapus kolom target label jika ada (default: 'species')."""
        if 'species' in df.columns:
            return df.drop(columns=['species'])     # Hapus kolom 'species'
        return df                                   # Jika tidak ada, kembalikan dataframe tanpa perubahan

    # Fungsi untuk melakukan one-hot encoding pada kolom kategorikal
    def _one_hot_encode(self, df):
        """Melakukan one-hot encoding untuk kolom kategorikal."""
        for col in ENCODE_COLUMNS:
            if col in df.columns:
                # Buat kolom dummy (one-hot encoding)
                dummies = pd.get_dummies(df[col], prefix=col)
                # Gabungkan kolom dummy ke dataframe dan hapus kolom aslinya
                df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
        return df

    # Fungsi utama untuk melakukan encoding data input
    def encode_input(self):
        # Buang kolom label dari data input pengguna
        input_clean = self._drop_label_column(self.input_df)

        # Buang kolom label dari dataset asli
        df_raw_clean = self._drop_label_column(self.df_raw)

        # Gabungkan input pengguna dan data asli agar struktur encoding sama
        df_combined = pd.concat([input_clean, df_raw_clean], axis=0)

        # Lakukan one-hot encoding pada gabungan data
        df_encoded = self._one_hot_encode(df_combined)

        # Ambil hanya baris pertama, yaitu input dari pengguna
        return df_encoded.iloc[:1]

