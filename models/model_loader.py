# Import library yang diperlukan
import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from config import DATASET_PATH, MODEL_PATH, TARGET_COLUMN, ENCODE_COLUMNS, TARGET_MAPPER

# Membuat class untuk memuat dan membangun model machine learning
class ModelLoader:
    def __init__(self):
        self.model = None # Tempat menyimpan model setelah dimuat atau dibuat

    # Fungsi untuk membangun model dari awal menggunakan dataset
    def build_model(self):
        df = pd.read_csv(DATASET_PATH) # Membaca dataset dari file CSV

        # Mengubah data kategorikal menjadi angka dengan one-hot encoding        
        for col in ENCODE_COLUMNS:
            dummies = pd.get_dummies(df[col], prefix=col) # Buat kolom baru dari kategori
            df = pd.concat([df, dummies], axis=1)         # Gabungkan dengan dataframe utama
            df.drop(columns=[col], inplace=True)          # Hapus kolom aslinya

        # Mengubah kolom target (label) menjadi angka menggunakan kamus TARGET_MAPPER
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(TARGET_MAPPER)

        # Pisahkan fitur (X) dan label (y)
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        # Buat dan latih model Random Forest
        model = RandomForestClassifier()
        model.fit(X, y)

        # Simpan model ke file agar bisa digunakan kembali tanpa dilatih ulang
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        return model    # Kembalikan model yang sudah dilatih
    
    # Fungsi untuk memuat model dari file jika sudah ada, atau membangun jika belum ada
    def load_model(self):
        if not os.path.exists(MODEL_PATH):    # Jika file model belum ada
            self.model = self.build_model()   # Bangun model baru
        else:
            with open(MODEL_PATH, 'rb') as f: # Jika file model sudah ada
                self.model = pickle.load(f)   # Muat model dari file
        return self.model   # Kembalikan model yang sudah dimuat atau dibangun