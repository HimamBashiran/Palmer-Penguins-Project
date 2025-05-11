# import Library
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# membaca dataset yang sudah dibersihkan
penguins = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")

df = penguins.copy() # duplikasi data untuk diproses
target = 'species' # menentukan kolom target dan kolom kategorikal yang akan diencoding
encode = ['sex','island']

# melakukan one-hot encoding pada kolom kategorikal
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2} # mapping label target ke angka
def target_encode(val): # encode nilai target
    return target_mapper[val]

# mengubah nilai target menjadi numerik
df['species'] = df['species'].apply(target_encode)

# memisahkan fitur dan label (x dan y)
fitur = df.drop('species', axis=1)
label = df['species']

# membuat dan melatih random forest
random_forest_model = RandomForestClassifier()
random_forest_model.fit(fitur, label)

# menyimpan model ke file pickle
pickle.dump(random_forest_model, open('penguins_clf.pkl', 'wb'))