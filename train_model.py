
import pandas as pd
import os
import glob
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# 1. Load dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError("Dataset CSV file not found!")

df = pd.read_csv(csv_files[0])

# 2. Clean dataset
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

df = df.dropna()
df = df.drop("customerID", axis=1)

df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# 3. Separate features and target
X = df.drop("Churn", axis=1)
y = df["Churn"]

# 4. Identify numerical and categorical columns
numerical_cols = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_cols = X.select_dtypes(
    include=["object", "str", "category"]
).columns

# 5. Create preprocessing pipelines
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_cols),
    ("cat", categorical_pipeline, categorical_cols)
])

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# 7. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

# 8. Train and evaluate models
results = {}
trained_models = {}
model_metrics = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print(f"\nTraining {name}...")

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    cm = confusion_matrix(y_test, predictions)

    results[name] = f1
    trained_models[name] = pipeline

    model_metrics[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist()
    }

    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC Score:", round(roc_auc, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    print("\nConfusion Matrix:")
    print(cm)

# 9. Select best model based on F1 score
best_model_name = max(results, key=results.get)
best_model = trained_models[best_model_name]
best_metrics = model_metrics[best_model_name]

print("\nBest Model:", best_model_name)
print("Best F1 Score:", round(results[best_model_name], 4))

# 10. Save model
MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "churn_model.pkl")
joblib.dump(best_model, model_path)

print("\nModel saved successfully!")
print("Saved location:", model_path)

# 11. Save best model evaluation results
metrics_path = os.path.join(MODEL_DIR, "model_metrics.pkl")
joblib.dump(best_metrics, metrics_path)

print("Model metrics saved successfully!")
print("Metrics location:", metrics_path)
