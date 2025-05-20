# Membangun dan menyimpan model machine learning
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Membaca dataset
penguins = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")
df = penguins.copy()
target = 'species'
encode = ['sex','island']

# Mengubah kolom kategorikal menjadi numerik menggunakan one-hot encoding
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Mapping target menjadi angka
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Memisahkan fitur dan label
fitur = df.drop('species', axis=1)
label = df['species']

# Melatih model dan menyimpannya ke file
random_forest_model = RandomForestClassifier()
random_forest_model.fit(fitur, label)

pickle.dump(random_forest_model, open('penguins_clf.pkl', 'wb'))