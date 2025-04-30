import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('supermarket_sales - Sheet1.csv')

print(df.head())
print(df.info())
# print(df.isnull().sum())

numerical_col = df.select_dtypes(include=[np.number])
print(len(numerical_col))
#
# plt.figure(figsize=(14, 8))
# i = 1
# for col in numerical_col:
#     plt.subplot(1, 8, i)
#     sns.histplot(data=df, x=col, kde=True)
#     i += 1
#
# # plt.figure(figsize=(14, 8))
# # i = 1
# # for col in numerical_col:
# #     plt.subplot(1, 8, i)
# #     sns.boxplot(data=df, x=col)
# #     i += 1
# # plt.show()
#
# df[['gross income', 'Rating', 'gross margin percentage', 'cogs', 'Quantity']].plot(kind='box')
# plt.xticks(rotation=45)
#
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df, x='Total', y='Quantity', hue='Customer type')
#
#
#
# plt.figure(figsize=(10, 6))
# sns.displot(x=df['Total'])

# sns.pairplot(df)
# plt.yticks(rotation=45)

plt.figure(figsize=(10, 6))
# sns.jointplot(x='Total', y='Tax 5%', data=df)

# sns.violinplot(x='Gender', y='Total', data=df)

# plt.pie(df['Product line'].value_counts(), autopct='%1.1f%%', labels=df['Product line'].unique())
df1 = df[['Quantity', 'Total', 'Rating']]
df1 = df1.sample(n=50)
x = pd.plotting.andrews_curves(df1, 'Rating')
x.plot()
plt.title("Andrews Curves by Rating")
plt.show()
