# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Suppress FutureWarnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load the dataset
dataset_path = "C:/Users/durga/.cache/kagglehub/datasets/mlg-ulb/creditcardfraud/versions/3/creditcard.csv"
df = pd.read_csv(dataset_path)

# Display dataset overview
print("Dataset Loaded Successfully\n")
print("First five rows of dataset")
print(df.head())

print("\nDataset Information")
print(df.info())

# Fraud vs Non-Fraud Distribution
print("\nFraud and Non-Fraud Distribution")
print(df["Class"].value_counts())

# Visualizing class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Class"], palette="coolwarm")  # Fix for Seaborn FutureWarning
plt.title("Fraud vs Non-Fraud Transactions")
plt.show()

# Separate features and target
X = df.drop(columns=["Class"])
y = df["Class"]

# Apply SMOTE to handle class imbalance
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Display balanced class distribution
print("\nData after SMOTE balancing")
print(pd.Series(y_resampled).value_counts())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation metrics
print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
