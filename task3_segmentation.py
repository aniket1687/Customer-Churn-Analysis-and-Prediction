import os
print("Current Directory:", os.getcwd())
import pandas as pd
df = pd.read_csv("Telco_Customer_Churn_Dataset.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)

print("Dataset Shape:", df.shape)

# ======================================
# Customer Segmentation by Tenure
# ======================================
df["Tenure_Group"] = pd.cut(
    df["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12 Months", "13-24 Months", "25-48 Months", "49-72 Months"]
)

plt.figure(figsize=(8,5))
sns.countplot(x="Tenure_Group", hue="Churn", data=df)
plt.title("Customer Churn by Tenure Group")
plt.xticks(rotation=15)
plt.show()

# ======================================
# Customer Segmentation by Monthly Charges
# ======================================
df["MonthlyCharges_Group"] = pd.cut(
    df["MonthlyCharges"],
    bins=[0, 35, 70, 120],
    labels=["Low", "Medium", "High"]
)

plt.figure(figsize=(8,5))
sns.countplot(x="MonthlyCharges_Group", hue="Churn", data=df)
plt.title("Customer Churn by Monthly Charges")
plt.show()

# ======================================
# Contract Type Analysis
# ======================================
plt.figure(figsize=(8,5))
sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Contract Type vs Churn")
plt.xticks(rotation=15)
plt.show()

# ======================================
# Payment Method Analysis
# ======================================
plt.figure(figsize=(10,5))
sns.countplot(x="PaymentMethod", hue="Churn", data=df)
plt.title("Payment Method vs Churn")
plt.xticks(rotation=30)
plt.show()

# ======================================
# Internet Service Analysis
# ======================================
plt.figure(figsize=(8,5))
sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Internet Service vs Churn")
plt.show()

# ======================================
# High Value Customers
# ======================================
high_value = df[
    (df["MonthlyCharges"] > 70) &
    (df["Churn"] == "Yes")
]

print("\nHigh Value Customers at Risk")
print(high_value[["customerID", "MonthlyCharges", "Contract", "tenure"]].head(10))

print("\nTask 3 (Customer Segmentation) Completed Successfully!")