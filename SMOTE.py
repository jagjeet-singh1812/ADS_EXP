import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report

df = pd.read_csv('Churn_Modelling.csv')
print(df.head())
print(df.info())
print(df.describe())

nc = df.select_dtypes(include=[np.number])

print(nc.iloc[0])

for col in df.columns:
    if df[col].dtype == 'O':
        label_encode = LabelEncoder()
        df[col] = label_encode.fit_transform(df[col])

X_train, X_test, y_train, y_test = train_test_split(df.drop('Exited', axis=1), df['Exited'], test_size=0.2,
                                                    random_state=101)

#
# print(df['Age'].value_counts())
#
plt.figure(figsize=(10, 4))
df['Exited'].value_counts().plot(kind='bar')
#
# X = df.drop('class', axis=1)
# y = df['class']
#
# xtr, xte, ytr, yte = train_test_split(X, y, test_size=0.2)
#
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

print(classification_report(y_test, dtc.predict(X_test)))
smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
x_sam, y_sam = smote.fit_resample(X_train, y_train)

dtc2 = DecisionTreeClassifier()
dtc2.fit(x_sam, y_sam)

print(classification_report(y_test, dtc2.predict(X_test)))

plt.figure(figsize=(10, 4))
y_sam.value_counts().plot(kind='bar')
plt.show()
#
# plt.figure(figsize=(10, 4))
# y_sam.count_values().plot(kind='bar')
# plt.show()
