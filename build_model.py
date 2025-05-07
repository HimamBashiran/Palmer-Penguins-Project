import pandas as pd
from sklearn.ensemble import RandomForestClassifier

penguins = pd.read_csv("C:/Users/User/PASD/Dataset/penguins_cleaned.csv")

df = penguins.copy()
target = 'species'
encode = ['sex','island']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}
def target_encode(val):
    return target_mapper[val]

df['species'] = df['species'].apply(target_encode)

# Separating X and y
fitur = df.drop('species', axis=1)
label = df['species']

# Build random forest model
clf = RandomForestClassifier()
clf.fit(fitur, label)

# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))