import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ===========================
# Load Dataset
# ===========================
df = pd.read_csv("Telco_Customer_Churn_Dataset(3).csv")

# ===========================
# Data Cleaning
# ===========================
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

print("Dataset Shape:", df.shape)
print(df.head())

# ===========================
# Churn Count
# ===========================
print("\nChurn Count:")
print(df["Churn"].value_counts())

plt.figure(figsize=(6,4))
sns.countplot(x="Churn", data=df)
plt.title("Overall Customer Churn")
plt.show()

# ===========================
# Gender vs Churn
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(x="gender", hue="Churn", data=df)
plt.title("Gender vs Churn")
plt.show()

# ===========================
# Partner vs Churn
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(x="Partner", hue="Churn", data=df)
plt.title("Partner vs Churn")
plt.show()

# ===========================
# Dependents vs Churn
# ===========================
plt.figure(figsize=(6,4))
sns.countplot(x="Dependents", hue="Churn", data=df)
plt.title("Dependents vs Churn")
plt.show()

# ===========================
# Tenure Distribution
# ===========================
plt.figure(figsize=(8,5))
sns.histplot(df["tenure"], bins=30, kde=True)
plt.title("Customer Tenure Distribution")
plt.xlabel("Tenure")
plt.show()

# ===========================
# Contract Type
# ===========================
plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=20)
plt.show()

# ===========================
# Payment Method
# ===========================
plt.figure(figsize=(10,5))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=45)
plt.show()

# ===========================
# Monthly Charges
# ===========================
plt.figure(figsize=(8,5))
sns.histplot(df["MonthlyCharges"], bins=30, kde=True)
plt.title("Monthly Charges Distribution")
plt.xlabel("Monthly Charges")
plt.show()

# ===========================
# Correlation Heatmap
# ===========================
numeric_df = df.copy()

le = LabelEncoder()

for col in numeric_df.columns:
    if numeric_df[col].dtype == "object":
        numeric_df[col] = le.fit_transform(numeric_df[col].astype(str))

corr = numeric_df.corr(numeric_only=True)

plt.figure(figsize=(15,10))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

print("\n==============================")
print("Task 2 (EDA) Completed Successfully!")
print("==============================")