# 1. Dataset Loading and Initial Exploration

import pandas as pd
from sklearn.datasets import load_breast_cancer


data = load_breast_cancer()


df = pd.DataFrame(data.data, columns=data.feature_names)


df["target"] = data.target

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
df.info()

print("\nTarget Distribution:")
print(df["target"].value_counts())

print("\nStatistical Summary:")
print(df.describe())

# 2. Data Quality and Class Distribution

print("\nMissing Values in Each Column:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nTarget Class Distribution:")
print(df["target"].value_counts())

print("\nTarget Class Percentage:")
print((df["target"].value_counts(normalize=True) * 100).round(2))

# 3. Feature and Target Preparation

# Separate features and target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nNumber of Features:")
print(X.shape[1])

print("\nTarget Classes:")
print(sorted(y.unique()))

print("\nFeature Names:")
print(list(X.columns))

# 4. Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Distribution:")
print(y_train.value_counts())

print("\nTesting Target Distribution:")
print(y_test.value_counts())

# 5. Feature Scaling

from sklearn.preprocessing import StandardScaler

# Check feature ranges before scaling
print("\nBefore Scaling:")
print(X_train[["mean radius", "mean texture", "mean area"]].describe().loc[["mean", "min", "max"]].round(2))

# Scale the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled Training Data - First 5 Rows:")
print(pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).head().round(2))

print("\nScaled Feature Means (First 5 Features):")
print(pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).mean().head().round(2))

print("\nScaled Feature Standard Deviations (First 5 Features):")
print(pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
).std().head().round(2))

# 6. Logistic Regression with Cross-Validation

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Create Logistic Regression model
model = LogisticRegression(max_iter=5000, random_state=42)

# 5-fold cross-validation
cv_scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=5,
    scoring="accuracy"
)

print("\nCross-Validation Accuracy Scores:")
print(cv_scores.round(4))

print("\nMean Cross-Validation Accuracy:")
print(round(cv_scores.mean(), 4))

print("\nStandard Deviation of CV Accuracy:")
print(round(cv_scores.std(), 4))

# Train final model
model.fit(X_train_scaled, y_train)

print("\nLogistic Regression model trained successfully!")

# 7. Model Evaluation on Test Data

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Make predictions on test data
y_pred = model.predict(X_test_scaled)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nTest Set Evaluation:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix Visualization

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# Create confusion matrix display
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Malignant", "Benign"]
)

disp.plot(cmap="Blues", values_format="d")

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

# 9. Detailed Error Analysis

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

print("\nDetailed Error Analysis:")
print(f"True Negatives  : {tn}")
print(f"False Positives : {fp}")
print(f"False Negatives : {fn}")
print(f"True Positives  : {tp}")

print("\nTotal Test Samples:")
print(len(y_test))

print("\nCorrect Predictions:")
print(tn + tp)

print("\nIncorrect Predictions:")
print(fp + fn)

print("\nError Rate:")
print(round((fp + fn) / len(y_test) * 100, 2), "%")

# 10. Model Performance Analysis

print("\nPerformance Analysis:")

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1-Score: {f1 * 100:.2f}%")

print(f"\nCorrectly Classified: {tn + tp} out of {len(y_test)}")
print(f"Incorrectly Classified: {fp + fn} out of {len(y_test)}")

print("\nCross-Validation vs Test Accuracy:")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Difference: {abs(cv_scores.mean() - accuracy) * 100:.2f} percentage points")

# 11. Feature Coefficient Analysis

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

feature_importance["Absolute_Coefficient"] = (
    feature_importance["Coefficient"].abs()
)

feature_importance = feature_importance.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop 10 Features by Absolute Coefficient:")
print(feature_importance.head(10).round(4))

print("\nMost Positive Feature:")
print(feature_importance.loc[
    feature_importance["Coefficient"].idxmax(),
    ["Feature", "Coefficient"]
])

print("\nMost Negative Feature:")
print(feature_importance.loc[
    feature_importance["Coefficient"].idxmin(),
    ["Feature", "Coefficient"]
])

# 12. Model Comparison - Logistic Regression vs Random Forest

from sklearn.ensemble import RandomForestClassifier

# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train Random Forest
rf_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)

# Calculate metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print("\nModel Comparison:")
print("-" * 55)
print(f"{'Metric':<15}{'Logistic Regression':<20}{'Random Forest'}")
print("-" * 55)
print(f"{'Accuracy':<15}{accuracy:.4f}{'':<10}{rf_accuracy:.4f}")
print(f"{'Precision':<15}{precision:.4f}{'':<10}{rf_precision:.4f}")
print(f"{'Recall':<15}{recall:.4f}{'':<10}{rf_recall:.4f}")
print(f"{'F1-Score':<15}{f1:.4f}{'':<10}{rf_f1:.4f}")
print("-" * 55)