# 1. Dataset Load & Initial Exploration

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nBasic Statistics:")
print(df.describe(include="all"))

# 2. Data Cleaning & Data Quality Analysis

# Check missing values
print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check duplicate records
print("\nDuplicate Records:", df.duplicated().sum())

# Check missing values after conversion
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Remove rows with missing values
df = df.dropna()

print("\nDataset Shape After Cleaning:", df.shape)

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 3. Exploratory Data Analysis (EDA)

# Churn distribution
print("\nChurn Distribution:")
print(df["Churn"].value_counts())

# Churn percentage
print("\nChurn Percentage:")
print(df["Churn"].value_counts(normalize=True) * 100)

# Numerical feature statistics
print("\nNumerical Feature Statistics:")
print(df[["tenure", "MonthlyCharges", "TotalCharges"]].describe())

# Churn distribution visualization
plt.figure(figsize=(7, 5))
sns.countplot(x="Churn", data=df)
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.show()

# Monthly Charges distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.show()

# Tenure distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure (Months)")
plt.ylabel("Number of Customers")
plt.show()

# 4. Feature Analysis & Relationship Analysis

# Churn by Contract Type
print("\nChurn by Contract Type:")
print(pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100)

# Churn by Internet Service
print("\nChurn by Internet Service:")
print(pd.crosstab(df["InternetService"], df["Churn"], normalize="index") * 100)

# Churn vs Monthly Charges
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="MonthlyCharges", data=df)
plt.title("Monthly Charges vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges")
plt.show()

# Churn vs Tenure
plt.figure(figsize=(8, 5))
sns.boxplot(x="Churn", y="tenure", data=df)
plt.title("Tenure vs Customer Churn")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.show()

# Correlation heatmap for numerical features
plt.figure(figsize=(8, 6))
correlation = df[["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]].corr()
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.show()

# 5. Feature Engineering & Encoding

from sklearn.preprocessing import LabelEncoder

# Create a copy for modeling
model_df = df.copy()

# Remove customer ID because it does not provide useful predictive information
model_df = model_df.drop("customerID", axis=1)

# Encode binary categorical columns
binary_columns = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling", "Churn"]

label_encoder = LabelEncoder()

for column in binary_columns:
    model_df[column] = label_encoder.fit_transform(model_df[column])

# Convert categorical features into numerical features
model_df = pd.get_dummies(
    model_df,
    columns=[
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod"
    ],
    drop_first=True
)

# Convert boolean columns to integers
model_df = model_df.astype(int)

print("\nFeature Engineering Completed")
print("Original Dataset Shape:", df.shape)
print("Model Dataset Shape:", model_df.shape)

print("\nEncoded Dataset Preview:")
print(model_df.head())

print("\nFinal Data Types:")
print(model_df.dtypes)

# 6. Train-Test Split & Feature Scaling

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Separate features and target
X = model_df.drop("Churn", axis=1)
y = model_df["Churn"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTrain-Test Split:")
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

print("\nFeature Scaling:")
print("Original Number of Features:", X_train.shape[1])
print("Scaled Training Shape:", X_train_scaled.shape)
print("Scaled Testing Shape:", X_test_scaled.shape)

# 7. Supervised Learning Model Training

from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
logistic_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logistic_model.fit(X_train_scaled, y_train)

print("\nLogistic Regression Model Training Completed")

# Generate predictions
y_pred = logistic_model.predict(X_test_scaled)

print("Predictions Generated Successfully")
print("Number of Predictions:", len(y_pred))

# 8. Model Evaluation & Performance Metrics

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nLogistic Regression Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# 9. Confusion Matrix & Error Analysis

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Display confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix - Logistic Regression")

plt.tight_layout()
plt.show()


# Extract confusion matrix values
tn, fp, fn, tp = cm.ravel()

print("\nError Analysis:")
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print("True Positives:", tp)

print("\nIncorrect Predictions:", fp + fn)
print("Correct Predictions:", tn + tp)

# 10. Unsupervised Learning — K-Means Customer Segmentation

from sklearn.cluster import KMeans

# Select features for customer segmentation
clustering_features = df[["tenure", "MonthlyCharges", "TotalCharges"]].copy()

# Scale clustering features
clustering_scaler = StandardScaler()
clustering_scaled = clustering_scaler.fit_transform(clustering_features)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(clustering_scaled)

# Add cluster labels to dataset
df["Cluster"] = clusters

print("\nK-Means Clustering Completed")
print("Number of Clusters:", 3)

print("\nCustomer Count in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Visualize customer segments
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="tenure",
    y="MonthlyCharges",
    hue="Cluster",
    data=df,
    palette="viridis",
    s=60
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Tenure (Months)")
plt.ylabel("Monthly Charges")
plt.show()

# 11. Clustering Evaluation & Analysis

from sklearn.metrics import silhouette_score

# Calculate Silhouette Score
silhouette = silhouette_score(clustering_scaled, clusters)

print("\nClustering Evaluation:")
print(f"Silhouette Score: {silhouette:.4f}")

# Analyze average values for each cluster
cluster_summary = df.groupby("Cluster")[[
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]].mean()

print("\nCluster-wise Average Values:")
print(cluster_summary)

# Visualize cluster-wise average values
cluster_summary.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Average Customer Characteristics by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Value")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()



# 12. Deep Learning Model using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim

# Convert data into PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Define Neural Network
class ChurnNeuralNetwork(nn.Module):
    def __init__(self, input_features):
        super(ChurnNeuralNetwork, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


# Create model
input_features = X_train_scaled.shape[1]

neural_model = ChurnNeuralNetwork(input_features)

print("\nNeural Network Architecture:")
print(neural_model)

# Loss function and optimizer
criterion_nn = nn.BCEWithLogitsLoss()
optimizer_nn = optim.Adam(neural_model.parameters(), lr=0.001)

# Count trainable parameters
total_parameters_nn = sum(
    parameter.numel()
    for parameter in neural_model.parameters()
    if parameter.requires_grad
)

print("\nDeep Learning Model Information:")
print("Input Features:", input_features)
print("Trainable Parameters:", total_parameters_nn)
print("Loss Function: BCEWithLogitsLoss")
print("Optimizer: Adam")
print("Learning Rate: 0.001")

# 13. Deep Learning Model Training

from torch.utils.data import TensorDataset, DataLoader

# Create validation split from training data
validation_split = int(0.8 * len(X_train_tensor))

X_nn_train = X_train_tensor[:validation_split]
y_nn_train = y_train_tensor[:validation_split]

X_nn_validation = X_train_tensor[validation_split:]
y_nn_validation = y_train_tensor[validation_split:]

# Create datasets
train_dataset_nn = TensorDataset(X_nn_train, y_nn_train)
validation_dataset_nn = TensorDataset(X_nn_validation, y_nn_validation)

# Create DataLoaders
batch_size_nn = 64

train_loader_nn = DataLoader(
    train_dataset_nn,
    batch_size=batch_size_nn,
    shuffle=True
)

validation_loader_nn = DataLoader(
    validation_dataset_nn,
    batch_size=batch_size_nn,
    shuffle=False
)

# Training configuration
num_epochs_nn = 20

train_losses_nn = []
train_accuracies_nn = []
validation_losses_nn = []
validation_accuracies_nn = []

# Train Neural Network
for epoch in range(num_epochs_nn):

    neural_model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader_nn:

        optimizer_nn.zero_grad()

        outputs = neural_model(features)

        loss = criterion_nn(outputs, labels)

        loss.backward()
        optimizer_nn.step()

        running_loss += loss.item()

        predictions = (torch.sigmoid(outputs) >= 0.5).float()

        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    train_loss = running_loss / len(train_loader_nn)
    train_accuracy = 100 * correct / total

    # Validation
    neural_model.eval()

    validation_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for features, labels in validation_loader_nn:

            outputs = neural_model(features)

            loss = criterion_nn(outputs, labels)

            validation_loss += loss.item()

            predictions = (torch.sigmoid(outputs) >= 0.5).float()

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    validation_loss = validation_loss / len(validation_loader_nn)
    validation_accuracy = 100 * correct / total

    train_losses_nn.append(train_loss)
    train_accuracies_nn.append(train_accuracy)

    validation_losses_nn.append(validation_loss)
    validation_accuracies_nn.append(validation_accuracy)

    print(
        f"Epoch [{epoch + 1}/{num_epochs_nn}] "
        f"Train Loss: {train_loss:.4f}, "
        f"Train Accuracy: {train_accuracy:.2f}%, "
        f"Validation Loss: {validation_loss:.4f}, "
        f"Validation Accuracy: {validation_accuracy:.2f}%"
    )

# Save trained neural network
torch.save(
    neural_model.state_dict(),
    "telco_churn_neural_network.pth"
)

print("\nNeural Network Training Completed")
print("Trained model saved successfully.")

# 14. Deep Learning Model Evaluation

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Set model to evaluation mode
neural_model.eval()

# Generate predictions on test data
with torch.no_grad():
    test_outputs = neural_model(X_test_tensor)

    test_probabilities = torch.sigmoid(test_outputs)

    nn_predictions = (test_probabilities >= 0.5).int().numpy().flatten()

# Convert actual labels to NumPy
actual_labels = y_test_tensor.numpy().flatten().astype(int)

# Calculate evaluation metrics
nn_accuracy = accuracy_score(actual_labels, nn_predictions)
nn_precision = precision_score(actual_labels, nn_predictions)
nn_recall = recall_score(actual_labels, nn_predictions)
nn_f1 = f1_score(actual_labels, nn_predictions)

print("\nDeep Learning Model Evaluation:")
print(f"Accuracy: {nn_accuracy:.4f}")
print(f"Precision: {nn_precision:.4f}")
print(f"Recall: {nn_recall:.4f}")
print(f"F1-Score: {nn_f1:.4f}")

# Classification report
print("\nClassification Report:")
print(
    classification_report(
        actual_labels,
        nn_predictions,
        target_names=["No Churn", "Churn"]
    )
)

# Confusion Matrix
nn_cm = confusion_matrix(actual_labels, nn_predictions)

print("\nConfusion Matrix:")
print(nn_cm)

# Display confusion matrix
fig, ax = plt.subplots(figsize=(7, 5))

nn_disp = ConfusionMatrixDisplay(
    confusion_matrix=nn_cm,
    display_labels=["No Churn", "Churn"]
)

nn_disp.plot(cmap="Blues", ax=ax)
ax.set_title("Confusion Matrix - Neural Network")

plt.tight_layout()
plt.show()

# Error analysis
nn_tn, nn_fp, nn_fn, nn_tp = nn_cm.ravel()

print("\nNeural Network Error Analysis:")
print("True Negatives:", nn_tn)
print("False Positives:", nn_fp)
print("False Negatives:", nn_fn)
print("True Positives:", nn_tp)

print("\nIncorrect Predictions:", nn_fp + nn_fn)
print("Correct Predictions:", nn_tn + nn_tp)

# 15. Logistic Regression vs Neural Network Comparison

# Create model comparison
comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Neural Network"
    ],
    "Accuracy": [
        accuracy,
        nn_accuracy
    ],
    "Precision": [
        precision,
        nn_precision
    ],
    "Recall": [
        recall,
        nn_recall
    ],
    "F1-Score": [
        f1,
        nn_f1
    ]
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))

# Visualize model accuracy
plt.figure(figsize=(8, 5))

sns.barplot(
    x="Model",
    y="Accuracy",
    data=comparison
)

plt.title("Model Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.ylim(0, 1)

plt.show()

# 16. Final Insights, Recommendations & Conclusion

print("\n" + "=" * 60)
print("FINAL CAPSTONE PROJECT SUMMARY")
print("=" * 60)

print("\n1. Logistic Regression Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\n2. Neural Network Results:")
print(f"Accuracy: {nn_accuracy:.4f}")
print(f"Precision: {nn_precision:.4f}")
print(f"Recall: {nn_recall:.4f}")
print(f"F1-Score: {nn_f1:.4f}")

print("\n3. K-Means Clustering:")
print(f"Number of Clusters: {kmeans.n_clusters}")
print(f"Silhouette Score: {silhouette:.4f}")

print("\n4. Key Insights:")
print("- Customer churn was predicted using supervised learning.")
print("- Customer groups were identified using K-Means clustering.")
print("- Neural Network performance was compared with Logistic Regression.")
print("- Tenure and billing-related features were analyzed to understand customer behavior.")
print("- Confusion matrices were used to analyze classification errors.")

print("\n5. Recommendations:")
print("- Identify customers with a high probability of churn.")
print("- Develop targeted retention offers for high-risk customers.")
print("- Use customer segments to create personalized marketing strategies.")
print("- Consider additional features and models for further improvement.")

print("\n6. Conclusion:")
print("This project demonstrated a complete data science pipeline")
print("from data collection and cleaning to EDA, feature engineering,")
print("supervised learning, unsupervised learning, deep learning,")
print("model evaluation, and business-oriented recommendations.")

print("\n" + "=" * 60)
print("CAPSTONE PROJECT COMPLETED")
print("=" * 60)