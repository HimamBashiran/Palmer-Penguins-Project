# Membangun dan menyimpan model machine learning
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Membaca dataset yang telah dibersihkan
penguins = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")
df = penguins.copy() # Membuat salinan data untuk diolah
target = 'species' # Menentukan kolom target
encode = ['sex','island'] # Kolom-kolom kategorikal yang akan diencoding

# Mengubah kolom kategorikal menjadi numerik menggunakan one-hot encoding
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col) # Membuat kolom dummy
    df = pd.concat([df,dummy], axis=1) # Menambahkan kolom dummy ke DataFrame
    del df[col] # Menghapus kolom asli karena sudah diganti

# Mapping target dari string menjadi angka
target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val): # Mengubah label menjadi angka
    return target_mapper[val]

# Menerapkan encoding pada kolom target
df['species'] = df['species'].apply(target_encode)

# Memisahkan fitur dan label
fitur = df.drop('species', axis=1) # Semua kolom selain target sebagai fitur
label = df['species'] # Target label

# Melatih model Random Forest dan melatihnya
random_forest_model = RandomForestClassifier()
random_forest_model.fit(fitur, label)

# Menyimpan model ke file menggunakan pickle
pickle.dump(random_forest_model, open('penguins_clf.pkl', 'wb'))