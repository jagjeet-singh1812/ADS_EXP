import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('loan_data_set.csv')
print(df.head(5))
print(df.info())

df = df[['LoanAmount', 'CoapplicantIncome']]

testdf = df[df['LoanAmount'].isnull() == True]
traindf = df[df['LoanAmount'].isnull() == False]
print(df)

lr = LinearRegression()
lr.fit(traindf[['CoapplicantIncome']], traindf['LoanAmount'])
pred = lr.predict(testdf[['CoapplicantIncome']])
testdf = testdf.copy()  # Avoid SettingWithCopyWarning
testdf['LoanAmount'] = pred
full_df = pd.concat([traindf, testdf]).sort_index()

# Print combined DataFrame
print(full_df)
