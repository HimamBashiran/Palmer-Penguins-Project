import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Kelas untuk membangun model machine learning
class ModelLoader:
    def __init__(self, model_path):
        self.model_path = model_path  # Lokasi pengimpanan file model pickle
        self.model = None  # Tempat menyimpan objek model setelah dimuat

    def build_model(self):
        # Membaca data dan membangun model jika file model tidak tersedia
        df = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")
        target = 'species' # Menentukan kolom target 
        encode = ['sex', 'island'] # Menentukan kolom-kolom yang akan di-encode

        for col in encode:
            # Mengubah kolom kategorikan menjadi one-hot encoding
            dummy = pd.get_dummies(df[col], prefix=col) # Membuat kolom dummy
            df = pd.concat([df, dummy], axis=1) # Menggabungkan dummy dengan dataframe utama
            del df[col] # Menghapus kolom asli

        # Mengubah nilai label (spesies) ke bentuk numerik
        target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
        df['species'] = df['species'].apply(lambda x: target_mapper[x])

        # Memisahkan fitur dan label
        fitur = df.drop('species', axis=1)
        label = df['species']

        # Melatih model Random Forest
        model = RandomForestClassifier()
        model.fit(fitur, label)

        # Menyimpan model ke file menggunakan pickle
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)

        return model # Mengembalikan model yang telah dilatih

    def load_model(self):
        # Memuat model dari file, atau membangun ulang jika belum ada
        if not os.path.exists(self.model_path):
            print(f"Model tidak ditemukan. Membangun model baru di {self.model_path}...")
            self.model = self.build_model()
        else:
            # Jika model sudah ada, buka dan muat dari file
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        return self.model # Mengembalikan objek model