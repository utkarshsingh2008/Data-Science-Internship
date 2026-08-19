import pandas as pd

df = pd.read_csv("titanic.csv")
# Outlier Detection
import matplotlib.pyplot as plt

# Age Boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df["Age"].dropna())
plt.title("Boxplot of Age")
plt.ylabel("Age")
plt.show()

# Fare Boxplot
plt.figure(figsize=(6, 4))
plt.boxplot(df["Fare"].dropna())
plt.title("Boxplot of Fare")
plt.ylabel("Fare")
plt.show()

# IQR Outlier Detection


Q1_age = df["Age"].quantile(0.25)
Q3_age = df["Age"].quantile(0.75)

IQR_age = Q3_age - Q1_age

lower_age = Q1_age - 1.5 * IQR_age
upper_age = Q3_age + 1.5 * IQR_age

age_outliers = df[
    (df["Age"] < lower_age) |
    (df["Age"] > upper_age)
]

print("\nAge Outlier Analysis:")
print("Q1:", Q1_age)
print("Q3:", Q3_age)
print("IQR:", IQR_age)
print("Lower Bound:", lower_age)
print("Upper Bound:", upper_age)
print("Number of Age Outliers:", len(age_outliers))

Q1_fare = df["Fare"].quantile(0.25)
Q3_fare = df["Fare"].quantile(0.75)

IQR_fare = Q3_fare - Q1_fare

lower_fare = Q1_fare - 1.5 * IQR_fare
upper_fare = Q3_fare + 1.5 * IQR_fare

fare_outliers = df[
    (df["Fare"] < lower_fare) |
    (df["Fare"] > upper_fare)
]

print("\nFare Outlier Analysis:")
print("Q1:", Q1_fare)
print("Q3:", Q3_fare)
print("IQR:", IQR_fare)
print("Lower Bound:", lower_fare)
print("Upper Bound:", upper_fare)
print("Number of Fare Outliers:", len(fare_outliers))



# 7. Missing Value Handling


print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Age: Fill missing values with median

age_median = df["Age"].median()
df["Age"] = df["Age"].fillna(age_median)

print("\nMissing Age values after cleaning:")
print(df["Age"].isnull().sum())

# Embarked: Fill missing values with mode

embarked_mode = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(embarked_mode)

print("\nMissing Embarked values after cleaning:")
print(df["Embarked"].isnull().sum())

# Cabin: Too many missing values, so remove the column

df = df.drop(columns=["Cabin"])

print("\nCabin column removed.")
print("Current shape:", df.shape)

# 8. Duplicate Records

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())

df = df.drop_duplicates()

print("\nDuplicate rows after cleaning:")
print(df.duplicated().sum())

print("\nShape after duplicate removal:")
print(df.shape)


print("\nData types before preprocessing:")
print(df.dtypes)

# 9(a). Categorical Encoding

df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

print("\nData after categorical encoding:")
print(df.head())

print("\nData types after encoding:")
print(df.dtypes)

# 9(b). Feature Scaling


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df[["Age", "Fare"]] = scaler.fit_transform(
    df[["Age", "Fare"]]
)

print("\nData after feature scaling:")
print(df[["Age", "Fare"]].head())

print("\nMean after scaling:")
print(df[["Age", "Fare"]].mean())

print("\nStandard deviation after scaling:")
print(df[["Age", "Fare"]].std())

# 10. Final Data Quality Check


print("\nFinal missing values:")
print(df.isnull().sum())

print("\nFinal duplicate rows:")
print(df.duplicated().sum())

print("\nFinal dataset shape:")
print(df.shape)
#11
df.to_csv("cleaned_titanic.csv", index=False)

# 11. Before vs After Comparison

print("\n===== BEFORE vs AFTER CLEANING =====")

print("\nOriginal Dataset:")
print("Rows: 891")
print("Columns: 12")
print("Missing Age values: 177")
print("Missing Cabin values: 687")
print("Missing Embarked values: 2")
print("Duplicate rows: 0")

print("\nCleaned Dataset:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())

# 12. Final Visualization
# ==============================

import matplotlib.pyplot as plt

# Age Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Age"], bins=20)
plt.title("Age Distribution After Preprocessing")
plt.xlabel("Standardized Age")
plt.ylabel("Number of Passengers")
plt.show()

# Fare Distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Fare"], bins=20)
plt.title("Fare Distribution After Preprocessing")
plt.xlabel("Standardized Fare")
plt.ylabel("Number of Passengers")
plt.show()

# 13. Export Final Dataset


df.to_csv("cleaned_titanic.csv", index=False)

print("\nFinal cleaned dataset saved successfully.")
print("File: cleaned_titanic.csv")