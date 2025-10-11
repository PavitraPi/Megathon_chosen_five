import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# --- 1. Data Loading ---
print("--- 1. Data Loading ---")

# ✅ Change this path to your local CSV file
DATA_FILENAME = './data/autoinsurance_churn.csv'

if not os.path.exists(DATA_FILENAME):
    raise FileNotFoundError(f"Dataset not found at {DATA_FILENAME}. Please check your file path.")

df = pd.read_csv(DATA_FILENAME)
print(f"Dataset '{DATA_FILENAME}' loaded successfully.")
print("Dataset shape:", df.shape)
print("\nDataset head:\n", df.head())
print("\nDataset info:\n")
df.info()
print("\nMissing values before handling:\n", df.isnull().sum())

# Define target variable
TARGET = 'Churn'
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset.")

# Drop ID columns
id_cols = [col for col in df.columns if 'ID' in col or 'id' in col]
features_to_drop = [col for col in id_cols if col != TARGET]
if TARGET in features_to_drop:
    features_to_drop.remove(TARGET)

X = df.drop(features_to_drop + [TARGET], axis=1, errors='ignore')
y = df[TARGET]

# --- 2. Data Cleaning ---
print("\n--- 2. Data Cleaning ---")

# Handle Missing Values
numerical_cols = X.select_dtypes(include=np.number).columns
for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"Filled missing values in '{col}' with median: {median_val}")

categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)
        print(f"Filled missing values in '{col}' with mode: {mode_val}")

print("\nMissing values after handling:", X.isnull().sum().sum())

# --- 3. Feature Engineering ---
print("\n--- 3. Feature Engineering ---")

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"Encoded categorical column: {col}")

# Scale numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
print("Scaled numerical columns using StandardScaler.")

# --- 4. Model Training ---
print("\n--- 4. Model Training ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Handle class imbalance
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}
print(f"Class weights: {class_weights}")

# Compute scale_pos_weight for XGBoost
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count
print(f"scale_pos_weight for XGBoost: {scale_pos_weight:.2f}")

# Initialize models
models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weights),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'XGBoostClassifier': XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight
    )
}

trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")

# --- 5. Model Evaluation ---
print("\n--- 5. Model Evaluation ---")

results = {}
best_model_name = None
best_roc_auc = -1

for name, model in trained_models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC-AUC': roc_auc
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-score: {f1:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model_name = name

print("\n--- Model Comparison ---")
results_df = pd.DataFrame(results).T
print(results_df)

print(f"\nBest performing model (ROC-AUC): {best_model_name}")
best_model = trained_models[best_model_name]

# --- 6. Save Best Model ---
print("\n--- 6. Saving Model ---")
model_filename = f'{best_model_name}_churn_model.pkl'
joblib.dump(best_model, model_filename)
print(f"✅ Best model saved as '{model_filename}' in current directory.")
