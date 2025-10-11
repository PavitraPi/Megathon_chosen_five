# ===============================
# 1. Importing Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.utils import class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# ===============================
# 2. Data Loading
# ===============================
print("\n--- 1. Data Loading ---")

def load_data_from_kaggle():   
    try:
        if os.path.exists('/kaggle/input'):
            base_path = '/kaggle/input'
            dataset_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
            if dataset_folders:
                data_path = os.path.join(base_path, dataset_folders[0])
        else:
            data_path = '.'
        
        main_file = os.path.join(data_path, 'autoinsurance_churn.csv')
        if not os.path.exists(main_file):
            raise FileNotFoundError(f"Main dataset not found at {main_file}")
        
        df = pd.read_csv(main_file)
        print(f"✅ Main dataset loaded: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None

# Load dataset
df = load_data_from_kaggle()

# --- Parse Date Columns ---
date_cols = ['cust_orig_date', 'date_of_birth', 'acct_suspd_date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

print("\nParsed date columns:", date_cols)
print("Dataset shape:", df.shape)
print("\nMissing values before handling:\n", df.isnull().sum())

# ===============================
# 3. Data Cleaning
# ===============================
print("\n--- 2. Data Cleaning ---")

# Drop ID columns
id_cols = ['individual_id', 'address_id']
df.drop(columns=id_cols, inplace=True, errors='ignore')

# Drop irrelevant or fully null columns (like acct_suspd_date if all NaN)
null_frac = df.isnull().mean()
drop_cols = [col for col in df.columns if null_frac[col] > 0.9]
if drop_cols:
    df.drop(columns=drop_cols, inplace=True)
    print(f"Dropped mostly null columns: {drop_cols}")

# Drop 'acct_suspd_date' to prevent data leakage
if 'acct_suspd_date' in df.columns:
    df.drop(columns=['acct_suspd_date'], inplace=True)
    print("Dropped column 'acct_suspd_date' to prevent target leakage.")

df = preprocess_insurance_data(df)
# Define target and features
TARGET = 'Churn'
X = df.drop(columns=[TARGET], errors='ignore')
y = df[TARGET]

# Handle missing values
# Numeric: median imputation, Categorical: mode imputation
numerical_cols = X.select_dtypes(include=np.number).columns
categorical_cols = X.select_dtypes(include='object').columns

for col in numerical_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)

print("Missing values after cleaning:", X.isnull().sum().sum())

# ===============================
# 4. Feature Engineering
# ===============================
print("\n--- 3. Feature Engineering ---")

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Scale numerical columns
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print(f"Encoded {len(categorical_cols)} categorical columns.")
print(f"Scaled {len(numerical_cols)} numerical columns.")

# ===============================
# 5. Model Training
# ===============================
print("\n--- 4. Model Training ---")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Handle class imbalance
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights_dict[i] for i in range(len(class_weights_dict))}
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

models = {
    'RandomForestClassifier': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weights),
    'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42),
    'XGBoostClassifier': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1,
                                       scale_pos_weight=scale_pos_weight)
}

# --- Handle datetime columns ---
print("\n--- Converting datetime columns ---")
datetime_cols = X.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()

# If dates were loaded as object strings, detect and convert them
for col in X.columns:
    if X[col].dtype == 'object':
        # Try converting columns that look like dates
        try:
            X[col] = pd.to_datetime(X[col], errors='raise')
            datetime_cols.append(col)
        except Exception:
            pass  # ignore non-date text columns

# Remove duplicates
datetime_cols = list(set(datetime_cols))

print(f"Detected datetime columns: {datetime_cols}")

# Convert datetime columns to numeric (days since earliest date)
for col in datetime_cols:
    # Fill missing values with the median date to avoid NaT
    median_date = X[col].dropna().median()
    X[col] = X[col].fillna(median_date)
    # Convert to days since median
    X[f'{col}_days_since'] = (X[col] - median_date).dt.days
    X.drop(columns=[col], inplace=True)

for col in X.columns:
    if np.issubdtype(X[col].dtype, np.datetime64):
        X[col] = X[col].view('int64')  # convert datetime to integer timestamp
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
print("✅ Ensured all feature columns are numeric before model training.")
    

print("Datetime columns successfully converted to numeric features.")


trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} trained successfully.")

# ===============================
# 6. Model Evaluation
# ===============================
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

    # Plot ROC curve
    plt.figure(figsize=(6, 4))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
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

print(f"\n✅ Best performing model based on ROC-AUC: {best_model_name}")
best_model = trained_models[best_model_name]

# ===============================
# 7. Model Saving
# ===============================

print("\n--- 6. Model Saving ---")

saved_model_paths = {}

# Save all trained models
for name, model in trained_models.items():
    model_filename = f'/kaggle/working/{name}_churn_prediction_model.pkl'
    joblib.dump(model, model_filename)
    saved_model_paths[name] = model_filename
    print(f"✅ Saved {name} model as '{model_filename}'")

print("\nAll models have been saved successfully:")
for name, path in saved_model_paths.items():
    print(f"   - {name}: {path}")

# Optionally, also save the best model separately for quick access
best_model_filename = f'/kaggle/working/{best_model_name}_best_churn_model.pkl'
joblib.dump(best_model, best_model_filename)
print(f"\n🏆 Best model ('{best_model_name}') also saved separately as '{best_model_filename}'")
