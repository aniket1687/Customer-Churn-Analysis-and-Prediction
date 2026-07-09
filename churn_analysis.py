import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv("Telco_Customer_Churn_Dataset(3).csv")

# Dataset Shape
print("Dataset Shape:", df.shape)

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Column Names
print("\nColumns:")
print(df.columns)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Dataset Information
print("\nDataset Information:")
print(df.info())

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe(include="all"))

# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Check Missing Values Again
print("\nMissing Values After Conversion:")
print(df.isnull().sum())

# Remove Missing Values
df = df.dropna()

print("\nDataset Shape After Cleaning:", df.shape)

# Encoding Categorical Data
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

print("\nEncoded Dataset:")
print(df.head())