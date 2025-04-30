import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import numpy as np

df = pd.read_csv('diamonds.csv', encoding='ansi')
print(df.head(10))

print(df.info())

df = df[df['price'] != 0]
# all zero data removal
df = df[(df != 0).all(axis=1)]

print("Descriptive Analysis of price : ")
print(df.describe()['price'])

mean_price = df['price'].mean()
print(f"Mean Price: {mean_price}")

median_price = df['price'].median()
print(f"Median Price: {median_price}")

mode_price = df['price'].mode()

for val in mode_price:
    freq = df['price'].value_counts().loc[val]
    print(f"Mode Value: {val} --> {freq} times")

q1 = df.describe()['depth']['25%']
q2 = df.describe()['depth']['50%']
q3 = df.describe()['depth']['75%']

iqr = 1.5 * (q3 - q1)
lower_bound = q1 - iqr
upper_bound = q3 + iqr

print(f"Q1 (25th percentile): {q1}")
print(f"Q2 (50th percentile / Median): {q2}")
print(f"Q3 (75th percentile): {q3}")
print(f"IQR (Interquartile Range): {iqr}")
print(f"Lower Bound: {lower_bound}")
print(f"Upper Bound: {upper_bound}")


def trimmed_mean(df, prop):
    lower = int(len(df) * prop)
    upper = int(len(df) * (1 - prop))
    df = df.sort_values().iloc[lower:upper]
    return df.mean()


trimmed_mean_value = trimmed_mean(df['carat'], 0.1)
mean_value = df['carat'].mean()
print(df['carat'].mean())
print(trimmed_mean_value)

if np.isclose(mean_value, trimmed_mean_value):
    print("The mean and trimmed mean are similar.")
    print(f"Mean: {mean_value}")
    print(f"Trimmed Mean: {trimmed_mean_value}")
    print("Conclusion: Data is symmetric or has no significant outliers.")
else:
    if mean_value > trimmed_mean_value:
        print(f"Mean: {mean_value}")
        print(f"Trimmed Mean: {trimmed_mean_value}")
        print("Conclusion: Data may be right-skewed (positively skewed) with larger outliers.")
    else:
        print(f"Mean: {mean_value}")
        print(f"Trimmed Mean: {trimmed_mean_value}")
        print("Conclusion: Data may be left-skewed (negatively skewed) with smaller outliers.")

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='depth')

plt.figure(figsize=(10, 6))

plt.scatter(df['x'], df['z'],
            s=df['y'] * 10,
            alpha=0.5,
            c='blue',
            edgecolors="w",
            linewidth=2,
            label='Bubble Data')

plt.xlabel('X values')
plt.ylabel('Y values')
plt.legend()
plt.title('Bubble Chart Example')
plt.grid(True)

plt.show()

corr_matrix = df.corr(numeric_only=True)

all_numeric_col = df.select_dtypes(include=[np.number])
all_numeric_col = all_numeric_col.drop(columns=['Unnamed: 0'])
plt.figure(figsize=(12, 8))
sns.heatmap(all_numeric_col.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Find features with correlation greater than 0.7 (excluding self-correlation)
threshold = 0.7
high_corr_features = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            feature_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
            high_corr_features.append(feature_pair)

# Print features having high correlation
print("\nHighly Correlated Features (Correlation > 0.7):")
for feature1, feature2, corr_value in high_corr_features:
    print(f"{feature1} ↔ {feature2} : Correlation = {corr_value:.2f}")
