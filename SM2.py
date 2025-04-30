import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

df = pd.read_csv('mushrooms.csv')
print(df.head())
print(df.info())
print(df.describe())

plt.figure(figsize=(10, 6))
df['class'].value_counts().plot(kind='bar')
plt.suptitle('Before')

df2 = df

df2['class'] = df['class'].map({'e': 0, 'p': 1})
df['class'].value_counts().plot(kind='bar')
for col in df.columns:
    if df[col].dtype == 'O':
        lb = LabelEncoder()
        df[col] = lb.fit_transform(df[col])

xt, xte, yt, yte = train_test_split(df2.drop('class', axis=1), df2['class'])

dtc = DecisionTreeClassifier()
dtc.fit(xt, yt)
print("Before : ")
print(classification_report(yte, dtc.predict(xte)))

smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
x, y = smote.fit_resample(xt, yt)


print("After : ")
print(classification_report(yte, dtc.predict(xte)))


print(df2['class'].value_counts())
print(y.value_counts())
plt.figure(figsize=(10, 6))
y.value_counts().plot(kind='bar')
plt.suptitle('After')
plt.show()
