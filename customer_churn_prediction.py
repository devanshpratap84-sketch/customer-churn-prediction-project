"""
Customer Churn Prediction - Minor Project
Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn

How to run:
1. Install: pip install -r requirements.txt
2. Either place WA_Fn-UseC_-Telco-Customer-Churn.csv in this folder,
   or keep USE_REMOTE_DATA=True to load a public raw CSV mirror.
3. Run: python customer_churn_prediction.py

Outputs are saved inside ./outputs/
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

LOCAL_FILE = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# The original project dataset is the Kaggle Telco Customer Churn dataset.
# This public raw mirror is used only so the code can run without manual Kaggle login.
REMOTE_URL = (
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/"
    "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
USE_REMOTE_DATA = True
RANDOM_STATE = 42


def load_data():
    """Load the CSV from the local folder, otherwise from the public raw URL."""
    if LOCAL_FILE.exists():
        print(f"Loading local dataset: {LOCAL_FILE}")
        return pd.read_csv(LOCAL_FILE)

    if USE_REMOTE_DATA:
        print("Local CSV not found. Loading dataset from public raw URL...")
        return pd.read_csv(REMOTE_URL)

    raise FileNotFoundError(
        "Dataset not found. Place WA_Fn-UseC_-Telco-Customer-Churn.csv "
        "in the project folder."
    )


def clean_data(df):
    """Basic cleaning and target encoding."""
    df = df.copy()

    # TotalCharges contains some blank strings in the original dataset.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Customer ID is an identifier, not a predictive business feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Remove rows where the target or essential numeric field is missing.
    df = df.dropna(subset=["TotalCharges", "Churn"])

    # Convert Yes/No target to 1/0.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def build_pipeline(X):
    """Create preprocessing + classification pipelines."""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }

    pipelines = {
        name: Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        for name, model in models.items()
    }

    return pipelines


def evaluate_model(name, model, X_test, y_test):
    """Calculate the requested classification metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }
    return result, y_pred, y_prob


def save_confusion_matrix(y_test, y_pred, filename):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.close()


def save_roc_curve(results, y_test, model_outputs):
    plt.figure(figsize=(7, 5))
    for name, y_prob in model_outputs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close()


def save_churn_distribution(df):
    counts = df["Churn"].value_counts().sort_index()
    labels = ["No Churn", "Churn"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Customer Churn Distribution")
    plt.ylabel("Number of Customers")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "churn_distribution.png", dpi=200)
    plt.close()


def save_feature_importance(best_model, filename="feature_importance.csv"):
    """Save model feature importance/coefficient information."""
    preprocessor = best_model.named_steps["preprocessor"]
    model = best_model.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        table = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importance
        }).sort_values("Importance", ascending=False)
    elif hasattr(model, "coef_"):
        coefficients = model.coef_[0]
        table = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Absolute_Importance": np.abs(coefficients)
        }).sort_values("Absolute_Importance", ascending=False)
    else:
        return

    table.to_csv(OUTPUT_DIR / filename, index=False)

    top = table.head(15).copy()
    importance_col = (
        "Importance" if "Importance" in top.columns else "Absolute_Importance"
    )

    plt.figure(figsize=(9, 6))
    top.sort_values(importance_col).plot(
        x="Feature", y=importance_col, kind="barh", legend=False
    )
    plt.title("Top 15 Predictive Features")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()


def main():
    print("\n=== CUSTOMER CHURN PREDICTION ===\n")

    # 1. Load
    raw_df = load_data()
    print("Raw shape:", raw_df.shape)

    # 2. Explore
    print("\nFirst 5 rows:")
    print(raw_df.head())
    print("\nMissing values:")
    print(raw_df.isnull().sum())

    # 3. Clean
    df = clean_data(raw_df)
    print("\nCleaned shape:", df.shape)
    print("Churn rate:", round(df["Churn"].mean() * 100, 2), "%")
    save_churn_distribution(df)

    # 4. Split features and target
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nTraining rows:", len(X_train))
    print("Testing rows:", len(X_test))

    # 5. Build and train models
    pipelines = build_pipeline(X)

    all_results = []
    model_outputs = {}
    trained_models = {}

    for name, pipeline in pipelines.items():
        print(f"\nTraining {name}...")
        pipeline.fit(X_train, y_train)

        result, y_pred, y_prob = evaluate_model(
            name, pipeline, X_test, y_test
        )
        all_results.append(result)
        model_outputs[name] = y_prob
        trained_models[name] = pipeline

        print(classification_report(
            y_test, y_pred,
            target_names=["No Churn", "Churn"],
            zero_division=0
        ))

        safe_name = name.lower().replace(" ", "_")
        save_confusion_matrix(
            y_test, y_pred, f"{safe_name}_confusion_matrix.png"
        )

    # 6. Compare models
    results_df = pd.DataFrame(all_results).sort_values(
        "F1-Score", ascending=False
    )
    results_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    # 7. Save ROC comparison
    save_roc_curve(results_df, y_test, model_outputs)

    # 8. Select best model by F1-score
    best_name = results_df.iloc[0]["Model"]
    best_model = trained_models[best_name]
    joblib.dump(best_model, OUTPUT_DIR / "best_churn_model.joblib")
    save_feature_importance(best_model)

    # 9. Save a small project summary
    summary = {
        "dataset_rows_after_cleaning": int(len(df)),
        "features_used": int(X.shape[1]),
        "churn_rate_percent": round(float(y.mean() * 100), 2),
        "best_model_by_f1": best_name,
        "best_f1": round(float(results_df.iloc[0]["F1-Score"]), 4),
    }
    with open(OUTPUT_DIR / "project_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\nBest model:", best_name)
    print("Files saved in:", OUTPUT_DIR.resolve())
    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
