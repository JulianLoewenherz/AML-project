import pandas as pd

# Load train data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()

print(f"Loaded training data: {len(X_train)} samples")
print(f"Features: {X_train.shape[1]}")
print(f"Fraud cases: {y_train.sum()} ({y_train.mean()*100:.2f}%)")

