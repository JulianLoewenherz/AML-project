import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('creditcard.csv')

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split with stratification to preserve fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Save splits
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print(f"Train set: {len(X_train)} samples ({y_train.sum()} fraud)")
print(f"Test set: {len(X_test)} samples ({y_test.sum()} fraud)")
print(f"Fraud ratio train: {y_train.mean():.4f}")
print(f"Fraud ratio test: {y_test.mean():.4f}")

