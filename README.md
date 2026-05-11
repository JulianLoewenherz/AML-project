# Credit Card Fraud Detection Pipeline and API

This project is an end-to-end applied machine learning workflow for detecting fraudulent credit card transactions. It demonstrates how to move from exploratory data analysis to reproducible model experiments and then expose the selected model through a FastAPI prediction service.

The repository is designed to show hiring managers the full lifecycle of a practical fraud-detection project: understanding an imbalanced dataset, creating reliable train/test splits, comparing modeling strategies, tracking experiments, saving production artifacts, validating API inputs, and serving real-time predictions.

## What This Project Does

The project uses a credit card transaction dataset with anonymized PCA features (`V1` through `V28`), transaction `Time`, transaction `Amount`, and a binary `Class` label where fraud is the positive class.

The workflow includes:

1. **Exploratory data analysis** to inspect dataset shape, columns, data types, missing values, sample records, class balance, and fraud rate.
2. **Stratified train/test splitting** to preserve the rare fraud ratio in both training and testing data.
3. **Feature preprocessing** with `RobustScaler` for `Time` and `Amount`, which are the non-PCA numeric fields most sensitive to outliers.
4. **Model experimentation** with XGBoost across four strategies for imbalanced classification:
   - Baseline XGBoost
   - SMOTE oversampling only
   - Cost-sensitive learning only
   - SMOTE plus cost-sensitive learning
5. **Experiment tracking** with MLflow, including logged parameters, AUPRC metrics, and model artifacts.
6. **Production artifact creation** using `joblib` to save the trained models and scaler.
7. **Real-time inference API** using FastAPI with Pydantic validation and a fraud probability response.

## Why This Project Matters

Credit card fraud detection is a highly imbalanced classification problem: fraudulent transactions are rare, but missing them can be expensive. Accuracy alone is not useful in this setting because a model can appear accurate by predicting nearly every transaction as legitimate.

This project focuses on a more realistic evaluation approach by using **AUPRC** (Area Under the Precision-Recall Curve), which is better suited for rare-event detection. It also compares imbalance-handling strategies to show the tradeoff between catching fraud and limiting false alarms.

## Repository Structure

| File | Purpose |
| --- | --- |
| `01_data_exploration.py` | Loads `creditcard.csv` and prints dataset shape, columns, data types, missing values, class distribution, fraud percentage, and sample rows. |
| `02_train_test_split.py` | Creates an 80/20 stratified train/test split and writes processed CSV files into `data/`. |
| `train.py` | Scales features, trains four XGBoost experiments, tracks results in MLflow, evaluates AUPRC, prints classification reports/confusion matrices, and saves model artifacts. |
| `schemas.py` | Defines the Pydantic request schema for validating incoming transaction payloads. |
| `main.py` | Defines the FastAPI application with health, root, and fraud prediction endpoints. |
| `requirements.txt` | Lists Python dependencies needed for data processing, modeling, experiment tracking, and serving. |
| `mlflow.db` | Local MLflow backend database for experiment metadata. |

Generated files are intentionally ignored by Git where appropriate:

- `creditcard.csv` because the raw dataset is too large for normal repository storage.
- `data/` because train/test splits can be regenerated.
- `models/` because trained model artifacts can be recreated by running the training pipeline.
- `mlruns/` because MLflow run artifacts are local experiment outputs.

## Technical Approach

### 1. Data Exploration

The exploration script reads the raw dataset and reports core quality and distribution checks:

- Number of rows and columns
- Column names and data types
- Missing-value count
- Legitimate vs. fraudulent transaction counts
- Fraud percentage
- Example records

This gives a quick understanding of dataset readiness and confirms the class imbalance that drives the modeling choices.

### 2. Train/Test Split

The split script separates features from the target label and uses `train_test_split` with `stratify=y`. Stratification is important because fraud is rare; without it, the test set could have a different fraud ratio from the training set, making evaluation less trustworthy.

The resulting files are saved as:

- `data/X_train.csv`
- `data/X_test.csv`
- `data/y_train.csv`
- `data/y_test.csv`

### 3. Preprocessing

The training pipeline applies `RobustScaler` to `Time` and `Amount`. These fields are scaled separately because the `V1`-`V28` features are already PCA-transformed. The fitted scaler is saved to `models/scaler.pkl` so the API can apply the exact same transformation at prediction time.

### 4. Model Training and Experimentation

The model training script uses `XGBClassifier` and evaluates four experiment configurations:

| Experiment | SMOTE | Cost-Sensitive Weighting | Goal |
| --- | --- | --- | --- |
| Baseline | No | No | Establish a simple benchmark. |
| SMOTE Only | Yes | No | Test whether synthetic minority oversampling improves fraud detection. |
| Cost-Sensitive Only | No | Yes | Penalize missed fraud more heavily by weighting the positive class. |
| SMOTE + Cost-Sensitive | Yes | Yes | Combine data-level and algorithm-level imbalance handling. |

For each experiment, the script logs parameters and AUPRC to MLflow, prints classification metrics, prints a confusion matrix, and saves the trained model as a `.pkl` file.

### 5. Model Serving API

The FastAPI service loads the selected model and scaler when the application starts. The `/predict` endpoint accepts a single transaction, validates it with Pydantic, applies preprocessing, orders columns to match training, and returns:

- `fraud_probability`: model-estimated probability of fraud
- `is_fraud`: boolean decision using a 0.30 threshold
- `threshold`: the decision threshold used by the API

The API also includes:

- `GET /` for service information
- `GET /health` for model/scaler readiness checks
- `POST /predict` for fraud predictions

## Key Skills Demonstrated

- Applied machine learning workflow design
- Imbalanced classification strategy comparison
- Fraud detection domain awareness
- XGBoost model training
- SMOTE oversampling
- Cost-sensitive learning with `scale_pos_weight`
- Precision-recall based evaluation
- MLflow experiment tracking
- Model and preprocessing artifact serialization
- FastAPI service development
- Pydantic request validation
- Reproducible preprocessing between training and inference

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

Place the raw Kaggle credit card fraud dataset in the project root as:

```text
creditcard.csv
```

The expected target column is `Class`, where `1` indicates fraud and `0` indicates a legitimate transaction.

## How to Run the Project

### Explore the data

```bash
python 01_data_exploration.py
```

### Create train/test splits

```bash
mkdir -p data
python 02_train_test_split.py
```

### Train and compare models

```bash
mkdir -p models
python train.py
```

### View MLflow experiments

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

### Start the API

```bash
uvicorn main:app --reload
```

Then open the interactive API documentation at:

```text
http://localhost:8000/docs
```

## Example Prediction Request

Send a `POST` request to `/predict` with all required transaction fields:

```json
{
  "Time": 12345.0,
  "Amount": 99.99,
  "V1": -1.0,
  "V2": 0.5,
  "V3": 1.2,
  "V4": -0.3,
  "V5": 0.1,
  "V6": -0.8,
  "V7": 0.4,
  "V8": 0.2,
  "V9": -0.1,
  "V10": 0.6,
  "V11": -0.4,
  "V12": 0.7,
  "V13": -0.2,
  "V14": 0.3,
  "V15": -0.5,
  "V16": 0.9,
  "V17": -0.7,
  "V18": 0.8,
  "V19": -0.6,
  "V20": 0.2,
  "V21": -0.1,
  "V22": 0.4,
  "V23": -0.3,
  "V24": 0.5,
  "V25": -0.2,
  "V26": 0.1,
  "V27": -0.4,
  "V28": 0.3
}
```

Example response:

```json
{
  "fraud_probability": 0.42,
  "is_fraud": true,
  "threshold": 0.3
}
```

## Notes for Reviewers

This repository emphasizes the engineering and modeling workflow rather than committing large data/model files. To keep the repository lightweight, the raw dataset, generated train/test splits, trained model files, and MLflow run artifacts are excluded from version control. Running the pipeline locally recreates those artifacts.

The current API is structured for local demonstration. In a production setting, the next improvements would include model registry integration, automated tests, Docker packaging, CI/CD, authentication, monitoring for data drift, and threshold tuning based on business cost assumptions.
