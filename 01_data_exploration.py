import pandas as pd

# Load dataset
df = pd.read_csv('creditcard.csv')

# Basic info
print(f"Shape: {df.shape}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Check for missing values
print(f"\nMissing values: {df.isnull().sum().sum()}")

# Class distribution
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud percentage: {(df['Class'].sum() / len(df)) * 100:.2f}%")

# Sample rows
print(f"\nFirst few rows:")
print(df.head(3))

