import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import mlflow
import mlflow.xgboost
import joblib

# Load train and test data
X_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
X_test = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv').squeeze()

print(f"Loaded training data: {len(X_train)} samples ({y_train.sum()} fraud)")
print(f"Loaded test data: {len(X_test)} samples ({y_test.sum()} fraud)")

# Scale Time and Amount using RobustScaler
scaler = RobustScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

# Save scaler for later use (production API)
joblib.dump(scaler, 'models/scaler.pkl')
print(f"\nScaled features and saved scaler")

# Set up MLflow
mlflow.set_experiment("fraud_detection")

def train_and_evaluate(experiment_name, use_smote=False, use_cost_sensitive=False):
    """Train model with specified configuration and log to MLflow"""
    
    with mlflow.start_run(run_name=experiment_name):
        print(f"\n{'='*60}")
        print(f"Running: {experiment_name}")
        print(f"{'='*60}")
        
        # Prepare data based on experiment configuration
        X_train_exp = X_train.copy()
        y_train_exp = y_train.copy()
        
        # Apply SMOTE if specified
        if use_smote:
            print("Applying SMOTE...")
            smote = SMOTE(random_state=42)
            X_train_exp, y_train_exp = smote.fit_resample(X_train_exp, y_train_exp)
            print(f"After SMOTE: {len(X_train_exp)} samples ({y_train_exp.sum()} fraud)")
            mlflow.log_param("smote", "yes")
        else:
            mlflow.log_param("smote", "no")
        
        # Calculate scale_pos_weight if cost-sensitive
        if use_cost_sensitive:
            scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
            print(f"Using cost-sensitive learning: scale_pos_weight={scale_pos_weight:.1f}")
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
        else:
            scale_pos_weight = 1
            mlflow.log_param("scale_pos_weight", 1)
        
        # Train XGBoost
        print("Training XGBoost...")
        model = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train_exp, y_train_exp)
        
        # Log hyperparameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        auprc = auc(recall, precision)
        
        # Log metrics to MLflow
        mlflow.log_metric("auprc", auprc)
        
        # Print results
        print(f"\nResults:")
        print(f"AUPRC: {auprc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        print(f"\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # Save model
        model_path = f"models/{experiment_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        
        print(f"Model saved: {model_path}")
        
        return auprc

# Run the 4 experiments
print("\n" + "="*60)
print("STARTING EXPERIMENTS")
print("="*60)

results = {}

# Experiment 1: Baseline
results['baseline'] = train_and_evaluate(
    "Experiment 1: Baseline",
    use_smote=False,
    use_cost_sensitive=False
)

# Experiment 2: SMOTE only
results['smote'] = train_and_evaluate(
    "Experiment 2: SMOTE Only",
    use_smote=True,
    use_cost_sensitive=False
)

# Experiment 3: Cost-sensitive only
results['cost_sensitive'] = train_and_evaluate(
    "Experiment 3: Cost-Sensitive Only",
    use_smote=False,
    use_cost_sensitive=True
)

# Experiment 4: Both
results['both'] = train_and_evaluate(
    "Experiment 4: SMOTE + Cost-Sensitive",
    use_smote=True,
    use_cost_sensitive=True
)

# Summary
print("\n" + "="*60)
print("SUMMARY OF ALL EXPERIMENTS")
print("="*60)
for name, auprc in results.items():
    print(f"{name:20s}: AUPRC = {auprc:.4f}")

print("\n" + "="*60)
print("View results in MLflow UI:")
print("Run: mlflow ui")
print("Then open: http://localhost:5000")
print("="*60)

