# Membangun dan menyimpan model machine learning dengan logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import logging
import sys

# Konfigurasi logging: mencatat log ke file dan juga menampilkan ke console
logging.basicConfig(
    level=logging.INFO,  # Tampilkan info dan error
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),  # Menulis log ke file
        logging.StreamHandler(sys.stdout)           # Menampilkan log ke konsol
    ]
)

# Membaca dataset
try:
    penguins = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")
    logging.info("Dataset berhasil dibaca.")
except FileNotFoundError:
    # Jika file CSV tidak ditemukan, log kesalahan dan hentikan program
    logging.error("File penguins_cleaned.csv tidak ditemukan. Pastikan path-nya benar.")
    sys.exit(1)
except pd.errors.ParserError:
    # Jika ada kesalahan dalam format file CSV, log kesalahan dan hentikan program
    logging.error("Terjadi kesalahan saat membaca CSV. Periksa format file.")
    sys.exit(1)

df = penguins.copy()
target = 'species'
encode = ['sex', 'island']

# Mengubah kolom kategorikal menjadi numerik menggunakan one-hot encoding
for col in encode:
    if col not in df.columns:
        # Jika kolom yang ingin diencode tidak ada dalam dataset, log kesalahan dan hentikan program
        logging.error(f"Kolom '{col}' tidak ditemukan dalam dataset.")
        sys.exit(1)
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
    logging.info(f"Kolom '{col}' berhasil diencode.")

# Mapping target menjadi angka
target_mapper = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

def target_encode(val):
    if val not in target_mapper:
        # Jika ada label target yang tidak dikenali, log kesalahan dan hentikan program
        logging.error(f"Label target tidak dikenali: {val}")
        sys.exit(1)
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)
logging.info("Kolom target 'species' berhasil di-encode.")

# Memisahkan fitur dan label
if 'species' not in df.columns:
    # Jika kolom target 'species' hilang karena kesalahan sebelumnya, log kesalahan dan hentikan program
    logging.error("Kolom 'species' tidak ditemukan setelah proses encoding.")
    sys.exit(1)

fitur = df.drop('species', axis=1)
label = df['species']
logging.info("Fitur dan label berhasil dipisahkan.")

# Melatih model
try:
    random_forest_model = RandomForestClassifier()
    random_forest_model.fit(fitur, label)
    logging.info("Model berhasil dilatih.")
except Exception as e:
    # Jika terjadi kesalahan saat training model (misalnya data tidak valid), log kesalahan dan hentikan program
    logging.error(f"Terjadi kesalahan saat melatih model: {e}")
    sys.exit(1)

# Menyimpan model ke file
try:
    with open('penguins_clf.pkl', 'wb') as f:
        pickle.dump(random_forest_model, f)
    logging.info("Model berhasil disimpan ke file 'penguins_clf.pkl'.")
except Exception as e:
    # Jika gagal menyimpan model ke file (misal hak akses atau disk penuh), log kesalahan dan hentikan program
    logging.error(f"Terjadi kesalahan saat menyimpan model: {e}")
    sys.exit(1)
