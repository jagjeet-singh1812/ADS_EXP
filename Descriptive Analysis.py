import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv('travel-times.csv')
# to get top 5 data rows
print(df.head(5))
print(df.info())
print(df.shape)

print(df.describe())

print(df['Distance'].value_counts())


def trim_mean(col, prop):
    lower = int(prop * int(len(col)))
    upper = int((1 - prop) * (len(col)))
    setx = col.sort_values().iloc[lower:upper]
    return setx.mean()


print(df['Distance'].mean())
print(trim_mean(df['Distance'], 0.1))

# all frquency Distribution
numeric_col = df.select_dtypes(include=[np.number])
print(numeric_col)
#
# numeric_col = df.drop(columns=['Id'])
#
# plt.figure(figsize=(12, 6))
# plt.suptitle("Distribution Numerical Columns")
# j = 1
# for col in numeric_col:
#     plt.subplot(1, 5, j)
#     j += 1
#     sns.histplot(df[col], kde=True, color='blue')
#     plt.xticks(rotation=45)
#
# all box plot
# plt.figure(figsize=(12, 6))
# plt.suptitle("Distribution Numerical Columns")
# for j, col in enumerate(numeric_col, 1):
#     # plt.subplot(1, 5, j)
#     sns.boxplot(data=df, x=col)
# numeric_col.plot(kind='box')
#     plt.xticks(rotation=45)
# plt.show()


plt.figure(figsize=(10, 6))
# plt.scatter(df['Species'], df['SepalLengthCm'], c="blue")
sns.heatmap(numeric_col.corr(), annot=True)
plt.show()

# # mean median mode
# df['Species'] = df['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
# print(df.median())
#
# plt.figure(figsize=(10, 6))
# # plt.scatter(df['Species'], df['SepalLengthCm'], c="blue")
# sns.scatterplot(data=df, x='Species', y='SepalLengthCm', hue='Species')
# plt.show()
#
# plt.figure(figsize=(10, 6))
# # plt.scatter(df['Species'], df['SepalLengthCm'], c="blue")
# sns.scatterplot(data=df, x='Species', y='SepalWidthCm', hue='Species')
# plt.show()

print(numeric_col.var())
print(numeric_col.std())
print(numeric_col.sem())
print(numeric_col.skew())
