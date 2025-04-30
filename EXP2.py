# data imputation
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv('loan_data_set.csv')
print(df.head(4))

print("Missing values count:")
print(df.isnull().sum())
na_columns = [i for i in df.columns if df[i].isnull().mean() > 0]

print(na_columns)
print(df.info())
# mean
# print(df['LoanAmount'])
# df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
#
# print(df['LoanAmount'])

# median
# df['Dependents'] = df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})
# df_mean = ceil(df['Dependents'].mean() * 1.0)
# print(df['Dependents'].isnull().sum())
# df['Dependents'].fillna(df_mean, inplace=True)
# print(df['Dependents'].isnull().sum())

#
# print(df['Loan_Amount_Term'].isnull().sum())
# df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mean(), inplace=True)
# print(df['Loan_Amount_Term'].isnull().sum())


# missing_col = ["LoanAmount"]
# print(df['LoanAmount'].isnull().sum())
# df['LoanAmount'].plot(kind='hist')
# # plt.show()
# df[missing_col[0]].fillna(df['LoanAmount'].mode()[0], inplace=True)
# print(df['LoanAmount'].mode()[0])
# print(df['LoanAmount'].isnull().sum())
# df['LoanAmount'].plot(kind='hist')
# plt.show()

# frequency data categories
# dfx = df
# mode_x = dfx['Gender'].mode()[0]
# print(dfx['Gender'].isnull().sum())
# # avoid warning
# dfx['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
# plt.show()
# dfx['Gender'] = dfx['Gender'].fillna(mode_x)
# dfx['Gender'].value_counts().plot(kind='pie', autopct='%1.1f%%')
# print(dfx['Gender'].isnull().sum())
# plt.show()


# regression Imputaition
# lr = LogisticRegression()
# df1 = df[["CoapplicantIncome", "LoanAmount"]]
# df1_train = df1[df['LoanAmount'].isnull() == False]
# df2_test = df1[df['LoanAmount'].isnull() == True]
#
# model = lr.fit(df1_train['LoanAmount'], df1_train['CoapplicantIncome'])
#
# print(lr.predict(df2_test))
from sklearn.linear_model import LinearRegression

# Step 1: Prepare data
df1 = df[["CoapplicantIncome", "LoanAmount"]]

# Step 2: Split into train/test based on LoanAmount missing
testdf = df1[df1['LoanAmount'].isnull()]
traindf = df1[df1['LoanAmount'].notnull()]

# Step 3: Fit regression model using CoapplicantIncome to predict LoanAmount
lr = LinearRegression()
lr.fit(traindf[['CoapplicantIncome']], traindf['LoanAmount'])

# Step 4: Predict missing LoanAmounts
pred = lr.predict(testdf[['CoapplicantIncome']])

# Step 5: (Optional) Impute predicted values
print(testdf)
testdf['LoanAmount'] = pred
print(testdf)
#
# from sklearn.preprocessing import OrdinalEncoder
#
# data = df[['Gender']]
# oe = OrdinalEncoder()
# result = oe.fit_transform(data)
# print(result)


from sklearn.preprocessing import LabelEncoder

data = df['Gender']

lb = LabelEncoder()
rep = lb.fit_transform(data)
t = pd.DataFrame({
    'Gender': data,
    'Label': rep
})
print(t.head(10))


# # random sampling
# df5 = df
# df5['LoanAmount'].dropna().sample(df5['LoanAmount'].isnull().sum(), random_state=0)
# print(df5['LoanAmount'].isnull().sum())


def random_impute(series):
    null_count = series.isnull().sum()
    non_null_values = series.dropna()
    random_values = np.random.choice(non_null_values, size=null_count, replace=True)
    series.loc[series.isnull()] = random_values
    return series


def rand(col):
    sizes = col.isnull().sum()
    cleaned_set = col.dropna()
    randx = np.random.choice(cleaned_set, size=sizes,replace=True)
    print(randx)
    col.loc[col.isnull()] = randx
    return col


df5 = df
print(df5['LoanAmount'].isnull().sum())
df5['LoanAmount'] = rand(df5['LoanAmount'])
print(df5['LoanAmount'].isnull().sum())
