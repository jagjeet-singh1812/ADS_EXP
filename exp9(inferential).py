import numpy as np
import pandas as pd
import math as m

# 📥 Load the dataset
df = pd.read_csv('supermarket_sales - Sheet1.csv')

print("✅ Dataset Loaded Successfully!")
print(f"Total Rows and Columns: {df.shape}")
print("\n🔍 First 5 rows of the dataset:\n", df.head())
print("\n📊 Summary of dataset info:\n")
print(df.info())

# ------------------------------
# 🧪 Z-Test (Large Sample n > 30)
# ------------------------------
print("\n" + "=" * 40)
print("🔬 Performing One-Sample Z-Test (n > 30)")
print("=" * 40)

df1 = df.sample(n=100, random_state=1)
population_mean = df['Total'].mean()
sample_mean = df1['Total'].mean()
sigma = df1['Total'].std()
nz = len(df1)

print(f"📌 Population Mean (µ): {population_mean:.2f}")
print(f"📌 Sample Mean (x̄): {sample_mean:.2f}")
print(f"📌 Sample Size (n): {nz}")
print(f"📌 Sample Standard Deviation (σ): {sigma:.2f}")

# Z-Score Calculation
z_score = (sample_mean - population_mean) / (sigma / m.sqrt(nz))
print(f"🧮 Calculated Z-Score: {z_score:.4f}")

# One-tail test critical z-value at 5% significance (right tail)
# for 2 tail 95 % confidence value is 1.96
critical_val_z = 1.65
print(f"🎯 Critical Value (Right Tail, α=0.05): {critical_val_z}")

# Hypothesis Decision
if z_score > critical_val_z:
    print("📢 Result: Z-Score > Critical Value → ❌ Reject Null Hypothesis")
else:
    print("📢 Result: Z-Score ≤ Critical Value → ✅ Fail to Reject Null Hypothesis")

# ------------------------------
# 🧪 T-Test (Small Sample n < 30)
# ------------------------------
print("\n" + "=" * 40)
print("🔬 Performing One-Sample T-Test (n < 30)")
print("=" * 40)

df2 = df.sample(n=28, random_state=2)
mue = df['Total'].mean()
x_bar = df2['Total'].mean()
s_t = df2['Total'].std()
nt = len(df2)

print(f"📌 Population Mean (µ): {mue:.2f}")
print(f"📌 Sample Mean (x̄): {x_bar:.2f}")
print(f"📌 Sample Size (n): {nt}")
print(f"📌 Sample Standard Deviation (s): {s_t:.2f}")
print(f"📌 Degrees of Freedom (df): {nt - 1}")

# T-Score Calculation
t_score = (x_bar - mue) / (s_t / m.sqrt(nt))
print(f"🧮 Calculated T-Score: {t_score:.4f}")

# One-tail test critical t-value at 5% significance for df=27
critical_val_t = 1.703  # This varies with df, use table or scipy for exact
print(f"🎯 Critical Value (Right Tail, α=0.05, df=27): {critical_val_t}")

# Hypothesis Decision
if t_score > critical_val_t:
    print("📢 Result: T-Score > Critical Value → ❌ Reject Null Hypothesis")
else:
    print("📢 Result: T-Score ≤ Critical Value → ✅ Fail to Reject Null Hypothesis")

# independent_sample

dfx = df.sample(28)
male = dfx[dfx['Gender'] == 'Male']['Quantity']
women = dfx[dfx['Gender'] == 'Female']['Quantity']

male_mean = male.mean()
women_mean = women.mean()

male_s = male.std()
women_s = women.std()

dof = len(male) + len(women) - 1
n1 = len(male)
n2 = len(women)
print('\n' + '=' * 40)
print("Doing Independent 1 tail test : ")
print('\n' + '=' * 40)

print(f'Dof: {dof}')

lower = m.sqrt((1 / n1) + (1 / n2))
sp = np.sqrt(((n1 - 1) * male_s * male_s) + ((n2 - 1) * women_s * women_s) / (dof - 1))

t_score_ind = (abs(male_mean - women_mean) )/ (sp * lower)
print(t_score_ind)
cv = 1.703  # for dof=27 , aplha =0.05 i.e one tail for

if t_score_ind > cv:
    print("We reject Null Hypothesis")
else:
    print("We Accept")
