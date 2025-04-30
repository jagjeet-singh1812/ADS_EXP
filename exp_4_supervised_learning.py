import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot

df = pd.read_csv('supermarket_sales - Sheet1.csv')

print(df.head())
print(df.info())
X = ['Unit price', 'Quantity', 'Rating']
y = ['Gender']

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X_train, x_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=42)

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred = dtc.predict(x_test)

print(f"Classification repo: \n{classification_report(y_test, y_pred)}")
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(precision_score(y_test, y_pred))
print(recall_score(y_test, y_pred))
print(f1_score(y_test, y_pred))

print(roc_auc_score(y_test, y_pred))
fpt, tpr, t = roc_curve(y_test, y_pred)
#
plt.plot(fpt, tpr)
plt.plot([0,1],ls='--')
plt.plot([0,0],[0,1],c='.7')
plt.plot([1,1],c='.7')
plt.show()
