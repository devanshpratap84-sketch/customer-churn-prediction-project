# Customer Churn Prediction – Minor Project

## Objective
Predict whether a telecom customer is likely to churn using customer demographics,
services, account information and billing information.

## Dataset
Telco Customer Churn dataset by BlastChar / IBM Sample Data Sets.
Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The Python script can load a local CSV named:
`WA_Fn-UseC_-Telco-Customer-Churn.csv`

If the local file is absent, the script uses a public raw CSV mirror so that the
project can be demonstrated without manual Kaggle authentication.

## Technologies
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Machine Learning
- Logistic Regression
- Random Forest
- One-Hot Encoding for categorical variables
- Standard Scaling for numerical variables
- Stratified 80/20 train-test split

## Evaluation
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Run
```bash
pip install -r requirements.txt
python customer_churn_prediction.py
```

Outputs are generated inside `outputs/`.
