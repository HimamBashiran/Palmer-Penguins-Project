import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier  # Model yang digunakan di pipeline

# === Load pipeline dan label ===
# Memuat pipeline yang sudah dilatih (berisi model, scaler, dan nama-nama kolom)
pipeline = joblib.load("C:/Users/User/PASD/penguin_knn_pipeline.pkl")
# Memuat label asli (nama spesies) yang digunakan saat training
labels = np.load("C:/Users/User/PASD/labels.npy", allow_pickle=True)

# Ambil komponen-komponen dari pipeline
model = pipeline["model"]                    # Model machine learning
scaler = pipeline["scaler"]                  # Normalisasi untuk fitur numerik
numerical_cols = pipeline["numerical_columns"]    # Kolom numerik
categorical_cols = pipeline["categorical_columns"]  # Kolom kategori (string)
feature_cols = pipeline["feature_columns"]   # Semua kolom fitur yang digunakan saat training

# === Judul App ===
# Tampilkan judul di halaman web
st.title("🧪 Prediksi Spesies Penguin dengan LightGBM")

# === Form Input User ===
# Buat form untuk pengguna mengisi karakteristik penguin
st.header("Masukkan Karakteristik Penguin")

# Input kategori dari dropdown
island = st.selectbox("Pulau", ["Biscoe", "Dream", "Torgersen"])
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])

# Input numerik dengan batas minimum dan maksimum
bill_length = st.number_input("Panjang Paruh (mm)", min_value=20.0, max_value=60.0, value=45.0)
bill_depth = st.number_input("Kedalaman Paruh (mm)", min_value=10.0, max_value=25.0, value=15.0)
flipper_length = st.number_input("Panjang Sirip (mm)", min_value=170, max_value=240, value=200)
body_mass = st.number_input("Berat Badan (g)", min_value=2500, max_value=6500, value=4000)

# === Buat DataFrame dari input user ===
# Gabungkan semua input menjadi sebuah tabel (DataFrame)
user_input_df = pd.DataFrame({
    "island": [island],
    "bill_length_mm": [bill_length],
    "bill_depth_mm": [bill_depth],
    "flipper_length_mm": [flipper_length],
    "body_mass_g": [body_mass],
    "sex": [sex]
})

# === Tombol Prediksi ===
if st.button("Prediksi Spesies"):
    # Salin DataFrame untuk diproses
    df_scaled = user_input_df.copy()

    # Lakukan normalisasi (scaling) hanya pada kolom numerik
    df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
    
    # Lakukan one-hot encoding untuk kolom kategori
    df_encoded = pd.get_dummies(df_scaled)

    # Tambahkan kolom yang hilang agar sesuai dengan fitur saat training
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0  # Kolom tidak ada, beri nilai default 0

    # Urutkan kolom agar sama dengan urutan fitur saat training
    df_encoded = df_encoded[feature_cols]

    # Prediksi kelas dari data input
    pred_index = model.predict(df_encoded)[0]  # Ambil indeks kelas
    pred_label = labels[pred_index]           # Ambil label berdasarkan indeks

    # Tampilkan hasil prediksi ke pengguna
    st.success(f"🎉 Spesies penguin yang diprediksi adalah: **{pred_label}**")
