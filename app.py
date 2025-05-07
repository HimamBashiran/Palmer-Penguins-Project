import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.write("""
# Prediksi Spesies Penguin Palmer

🐧 Aplikasi Prediksi Spesies Penguin 🐧

Aplikasi yang akan membantu kamu memprediksi spesies penguin berdasarkan ciri-ciri fisiknya seperti panjang paruh, berat badan, dan lainnya.

Data yang digunakan berasal dari kaggle [Palmer Penguins](https://www.kaggle.com/datasets/ashkhagan/palmer-penguins-datasetalternative-iris-dataset/data) yang dikembangkan oleh Dr. Kristen Gorman
""")

st.sidebar.header('Fitur Input Variabel ')

st.sidebar.markdown("""
Silakan masukkan detail karakteristik penguin di bawah ini untuk memprediksi spesiesnya.
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island',('Biscoe','Dream','Torgersen'))
        sex = st.sidebar.selectbox('Sex',('male','female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1,59.6,43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1,21.5,17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0,231.0,201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Gabungkan input user dengan data asli untuk encoding
penguins_raw = pd.read_csv('C:/Users/User/PASD/Dataset/penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df,penguins],axis=0)

# Encoding kolom kategorikal
encode = ['sex','island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Ambil hanya data input user

# Tampilkan input pengguna
st.subheader('Fitur Input Pengguna')

if uploaded_file is not None:
    st.write(df)
else:
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# Hasil Prediksi
st.subheader('Prediksi')
penguins_species = np.array(['Adelie', 'Gentoo', 'Chinstrap'])
hasil_prediksi = penguins_species[prediction][0]
st.write(f"Spesies yang diprediksi: **{hasil_prediksi}**")

# Tampilkan gambar berdasarkan spesies yang diprediksi
if hasil_prediksi == 'Adelie':
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/e/e3/Hope_Bay-2016-Trinity_Peninsula%E2%80%93Ad%C3%A9lie_penguin_%28Pygoscelis_adeliae%29_04.jpg",
        caption='Penguin Adelie',
        use_container_width=True  
    )
    
elif hasil_prediksi == 'Gentoo':
    st.image(
        "https://birdlifedata.blob.core.windows.net/species-images/22697755.jpg",
        caption='Penguin Gentoo',
        use_container_width=True  
    )

elif hasil_prediksi == 'Chinstrap':
    st.image(
        "https://cdn.britannica.com/08/152708-050-23B255B3/Chinstrap-penguin.jpg",
        caption='Penguin Chinstrap',
        use_container_width=True  
    )

# Peluang prediksi
st.subheader('Probabilitas Prediksi')
st.write(prediction_proba)