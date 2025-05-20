import numpy as np
import streamlit as st

class Predictor:
    def __init__(self, model, input_df):
        self.model = model  # Model machine learning
        self.input_df = input_df  # Data input pengguna
        self.penguins_species = np.array(['Adelie', 'Gentoo', 'Chinstrap'])  # Label spesies

    def predict(self):
        # Menghasilkan prediksi dan probabilitas
        prediction = self.model.predict(self.input_df)
        prediction_proba = self.model.predict_proba(self.input_df)
        return prediction, prediction_proba

    def show_result(self, prediction, prediction_proba):
        # Menampilkan hasil prediksi dan gambar spesies
        hasil_prediksi = self.penguins_species[prediction][0]
        st.subheader('Prediksi')
        st.write(f"Spesies yang diprediksi: **{hasil_prediksi}**")

        # Menampilkan gambar sesuai hasil prediksi
        if hasil_prediksi == 'Adelie':
            st.image("https://i.pinimg.com/564x/e2/b3/cd/e2b3cd99d77364f751d8f42f81f1e78e.jpg", caption='Penguin Adelie', use_container_width=True)
        elif hasil_prediksi == 'Gentoo':
            st.image("https://birdlifedata.blob.core.windows.net/species-images/22697755.jpg", caption='Penguin Gentoo', use_container_width=True)
        elif hasil_prediksi == 'Chinstrap':
            st.image("https://encrypted-tbn2.gstatic.com/images?q=tbn:ANd9GcQI2Eyelti4suqlwyxZK0_xRYDa8aCQMl0C4Ue00NlqFDEBLaRmtBQ1mhgr-XHtLUHyP0J63IlhJNdaY8axdrLr4A", caption='Penguin Chinstrap', use_container_width=True)

        # Menampilkan probabilitas prediksi
        st.subheader('Probabilitas Prediksi')
        st.write(prediction_proba)