import streamlit as st
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMClassifier


# === Load pipeline dan label ===
pipeline = joblib.load("C:/Users/User/PASD/penguin_knn_pipeline.pkl")
labels = np.load("C:/Users/User/PASD/labels.npy", allow_pickle=True)

model = pipeline["model"]
scaler = pipeline["scaler"]
numerical_cols = pipeline["numerical_columns"]
categorical_cols = pipeline["categorical_columns"]
feature_cols = pipeline["feature_columns"]

# === Judul App ===
st.title("🧪 Prediksi Spesies Penguin dengan LightGBM")

# === Form Input User ===
st.header("Masukkan Karakteristik Penguin")

island = st.selectbox("Pulau", ["Biscoe", "Dream", "Torgersen"])
sex = st.selectbox("Jenis Kelamin", ["Male", "Female"])
bill_length = st.number_input("Panjang Paruh (mm)", min_value=20.0, max_value=60.0, value=45.0)
bill_depth = st.number_input("Kedalaman Paruh (mm)", min_value=10.0, max_value=25.0, value=15.0)
flipper_length = st.number_input("Panjang Sirip (mm)", min_value=170, max_value=240, value=200)
body_mass = st.number_input("Berat Badan (g)", min_value=2500, max_value=6500, value=4000)

# === Buat DataFrame dari input user ===
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
    # Preprocessing
    df_scaled = user_input_df.copy()
    df_scaled[numerical_cols] = scaler.transform(df_scaled[numerical_cols])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df_scaled)

    # Tambahkan kolom yang hilang (jika ada)
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Urutkan kolom agar sama dengan training
    df_encoded = df_encoded[feature_cols]

    # Prediksi
    pred_index = model.predict(df_encoded)[0]
    pred_label = labels[pred_index]

    # Tampilkan hasil
    st.success(f"🎉 Spesies penguin yang diprediksi adalah: **{pred_label}**")
