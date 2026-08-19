import pandas as pd;
import matplotlib.pyplot as plt
import  seaborn as sns
#1
df = pd.read_csv("cleaned_titanic.csv")

print("Dataset Loaded Successfully")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nBasic Statistical Summary:")
print(df.describe())
#2

print("\nMedian Values:")
print(df.median(numeric_only=True))

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nSurvival Percentage:")
print(df["Survived"].value_counts(normalize=True) * 100)

# 3. Univariate Analysis - Age Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Age"], bins=20, kde=True)

plt.title("Distribution of Passenger Age")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()

# 4. Univariate Analysis - Gender Distribution

plt.figure(figsize=(7, 5))

sns.countplot(x="Sex", data=df)

plt.title("Passenger Distribution by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")

plt.show()

# 5. Univariate Analysis - Passenger Class Distribution

plt.figure(figsize=(7, 5))

sns.countplot(x="Pclass", data=df)

plt.title("Passenger Distribution by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")

plt.show()

# 6. Univariate Analysis - Fare Distribution

plt.figure(figsize=(8, 5))

sns.histplot(df["Fare"], bins=20, kde=True)

plt.title("Distribution of Passenger Fare")
plt.xlabel("Fare")
plt.ylabel("Number of Passengers")

plt.show()

# 7. Bivariate Analysis - Gender vs Survival

plt.figure(figsize=(7, 5))

sns.countplot(x="Sex", hue="Survived", data=df)

plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["Did Not Survive", "Survived"])

plt.show()

# 8. Bivariate Analysis - Passenger Class vs Survival

plt.figure(figsize=(7, 5))

sns.countplot(x="Pclass", hue="Survived", data=df)

plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["Did Not Survive", "Survived"])

plt.show()

# 9. Bivariate Analysis - Age vs Survival

plt.figure(figsize=(8, 5))

sns.boxplot(x="Survived", y="Age", data=df)

plt.title("Age Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Age")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])

plt.show()

# 10. Bivariate Analysis - Fare vs Survival

plt.figure(figsize=(8, 5))

sns.boxplot(x="Survived", y="Fare", data=df)

plt.title("Fare Distribution by Survival Status")
plt.xlabel("Survival Status")
plt.ylabel("Fare")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])

plt.show()

# 11. Bivariate Analysis - Embarked vs Survival

plt.figure(figsize=(7, 5))

sns.countplot(x="Embarked", hue="Survived", data=df)

plt.title("Survival by Port of Embarkation")
plt.xlabel("Port of Embarkation")
plt.ylabel("Number of Passengers")
plt.legend(
    title="Survived",
    labels=["Did Not Survive", "Survived"]
)

plt.show()

# 12. Multivariate Analysis - Class, Gender and Survival

sns.catplot(
    data=df,
    x="Pclass",
    hue="Sex",
    col="Survived",
    kind="count",
    height=5,
    aspect=1
)


plt.show()

# 13. Correlation Analysis

correlation = df.corr(numeric_only=True)

print("\nCorrelation Matrix:")
print(correlation)

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap of Titanic Dataset")

plt.show()

# 14. Data Aggregation - Survival Rate by Gender

survival_gender = df.groupby("Sex")["Survived"].mean() * 100

print("\nSurvival Rate by Gender:")
print(survival_gender)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=survival_gender.index,
    y=survival_gender.values
)

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate (%)")

plt.show()

# 15. Data Aggregation - Survival Rate by Passenger Class

survival_class = df.groupby("Pclass")["Survived"].mean() * 100

print("\nSurvival Rate by Passenger Class:")
print(survival_class)

plt.figure(figsize=(7, 5))

sns.barplot(
    x=survival_class.index,
    y=survival_class.values
)

plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")

plt.show()

# 16. Data Aggregation - Survival Rate by Gender and Passenger Class

survival_gender_class = df.groupby(
    ["Sex", "Pclass"]
)["Survived"].mean() * 100

print("\nSurvival Rate by Gender and Passenger Class:")
print(survival_gender_class)

# Convert result to a DataFrame for visualization
survival_gender_class = survival_gender_class.reset_index()

plt.figure(figsize=(9, 5))

sns.barplot(
    data=survival_gender_class,
    x="Pclass",
    y="Survived",
    hue="Sex"
)

plt.title("Survival Rate by Gender and Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate (%)")
plt.legend(title="Gender")

plt.show()

# 17. Key Findings and Anomalies

print("\n========== KEY FINDINGS ==========")

# Overall survival rate
overall_survival = df["Survived"].mean() * 100
print(f"Overall Survival Rate: {overall_survival:.2f}%")

# Survival rate by gender
gender_survival = df.groupby("Sex")["Survived"].mean() * 100
print("\nSurvival Rate by Gender:")
print(gender_survival)

# Survival rate by class
class_survival = df.groupby("Pclass")["Survived"].mean() * 100
print("\nSurvival Rate by Passenger Class:")
print(class_survival)

# Survival rate by gender and class
gender_class_survival = df.groupby(
    ["Sex", "Pclass"]
)["Survived"].mean() * 100

print("\nSurvival Rate by Gender and Passenger Class:")
print(gender_class_survival)

# Highest and lowest survival groups
highest_group = gender_class_survival.idxmax()
highest_rate = gender_class_survival.max()

lowest_group = gender_class_survival.idxmin()
lowest_rate = gender_class_survival.min()

print("\nHighest Survival Group:")
print(highest_group, f"{highest_rate:.2f}%")

print("\nLowest Survival Group:")
print(lowest_group, f"{lowest_rate:.2f}%")

# 18. Final EDA Summary

print("\n========== FINAL EDA SUMMARY ==========")

print("Dataset Shape:", df.shape)

print("\nTotal Passengers:", len(df))

print("\nSurvival Count:")
print(df["Survived"].value_counts())

print("\nSurvival Percentage:")
print((df["Survived"].value_counts(normalize=True) * 100).round(2))

print("\nAverage Age:")
print(df["Age"].mean())

print("\nAverage Fare:")
print(df["Fare"].mean())

print("\nPassenger Class Distribution:")
print(df["Pclass"].value_counts().sort_index())

print("\nGender Distribution:")
print(df["Sex"].value_counts())

print("\nEmbarkation Distribution:")
print(df["Embarked"].value_counts())