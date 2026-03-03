import pandas as pd
import numpy as np
import joblib
import re

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

print("Loading dataset...")

df = pd.read_csv(
    "data/startup-investments-crunchbase.csv",
    encoding="latin1",
    low_memory=False
)

df.columns = df.columns.str.strip()

print("Loaded", len(df), "rows")

# ==========================
# CLEAN FUNDING COLUMN PROPERLY
# ==========================

def clean_money(x):
    if pd.isna(x):
        return 0
    x = str(x)
    x = re.sub(r"[^\d.]", "", x)  # remove $, commas, etc.
    return float(x) if x != "" else 0

df["funding_total_usd"] = df["funding_total_usd"].apply(clean_money)

print("Funding stats:")
print(df["funding_total_usd"].describe())

# ==========================
# CREATE HIGH-GROWTH TARGET
# ==========================

threshold = df["funding_total_usd"].quantile(0.80)
df["high_growth"] = (df["funding_total_usd"] >= threshold).astype(int)

print("\nTarget Distribution:")
print(df["high_growth"].value_counts())

# ==========================
# REMOVE LEAKAGE
# ==========================

df_model = df.drop(columns=["funding_total_usd", "high_growth"])

# ==========================
# SELECT FEATURES
# ==========================

categorical_features = [
    "category_list",
    "country_code",
    "state_code",
    "market"
]

numeric_features = [
    "funding_rounds",
    "founded_year"
]

# Keep only existing columns
categorical_features = [c for c in categorical_features if c in df_model.columns]
numeric_features = [c for c in numeric_features if c in df_model.columns]

X = df_model[categorical_features + numeric_features].copy()
y = df["high_growth"]

# Clean numeric columns
for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
    X[col] = X[col].fillna(0)

# Clean categorical columns
for col in categorical_features:
    X[col] = X[col].astype(str).fillna("Unknown")

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# PREPROCESSING
# ==========================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# ==========================
# PIPELINE
# ==========================

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [200, 300],
    "classifier__max_depth": [None, 10],
    "classifier__min_samples_split": [2, 5]
}

search = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=6,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    random_state=42
)

print("\nTraining model...")
search.fit(X_train, y_train)

print("\nBest Parameters:")
print(search.best_params_)

y_pred = search.predict(X_test)

print("\n===== HIGH-GROWTH CLASSIFICATION REPORT =====")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(search.best_estimator_, "high_growth_model.pkl")

print("\nModel saved as high_growth_model.pkl")
print("\nDONE ✅")