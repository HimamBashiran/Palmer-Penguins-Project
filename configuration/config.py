# Lokasi file dataset yang sudah dibersihkan
DATASET_PATH = r'C:\Users\A C E R V E R O\OneDrive\Documents\GitHub\Palmer-Penguins-Project\Dataset\penguins_cleaned.csv' 

# Lokasi untuk menyimpan atau memuat model yang telah dilatih
MODEL_PATH = 'penguins_clf.pkl'

# Nama kolom target (label) pada dataset
TARGET_COLUMN = 'species'

# Daftar kolom kategorikal yang perlu di-encode (one-hot encoding)
ENCODE_COLUMNS = ['sex', 'island']

# Pemetaan label target dari teks ke angka agar bisa digunakan oleh model
TARGET_MAPPER = {
    'Adelie': 0, 
    'Chinstrap': 1, 
    'Gentoo': 2
}

# Label kelas untuk menampilkan hasil prediksi kembali dalam bentuk teks
CLASS_LABELS = ['Adelie', 'Chinstrap', 'Gentoo']