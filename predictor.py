import numpy as np
import streamlit as st

class Predictor:
    def __init__(self, model, input_df):
        self.model = model
        self.input_df = input_df
        self.penguins_species = np.array(['Adelie', 'Gentoo', 'Chinstrap'])

    def predict(self):
        prediction = self.model.predict(self.input_df)
        prediction_proba = self.model.predict_proba(self.input_df)
        return prediction, prediction_proba

    def show_result(self, prediction, prediction_proba):
        hasil_prediksi = self.penguins_species[prediction][0]
        st.subheader('Prediksi')
        st.write(f"Spesies yang diprediksi: **{hasil_prediksi}**")

        # Gambar
        if hasil_prediksi == 'Adelie':
            st.image("https://upload.wikimedia.org/...04.jpg", caption='Penguin Adelie', use_container_width=True)
        elif hasil_prediksi == 'Gentoo':
            st.image("https://birdlifedata.blob.core.windows.net/species-images/22697755.jpg", caption='Penguin Gentoo', use_container_width=True)
        elif hasil_prediksi == 'Chinstrap':
            st.image("https://cdn.britannica.com/...Chinstrap-penguin.jpg", caption='Penguin Chinstrap', use_container_width=True)

        st.subheader('Probabilitas Prediksi')
        st.write(prediction_proba)
