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
        return df
    
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

# Load dataset
df = load_data_from_kaggle()

# ===============================
# 3. Data Cleaning & Preprocessing
# ===============================
print("\n--- 2. Data Cleaning & Preprocessing ---")

# Make a copy to avoid modifying original
df_clean = df.copy()

# Drop ID columns completely
id_cols = ['individual_id', 'address_id']
df_clean.drop(columns=id_cols, inplace=True, errors='ignore')

# Parse date columns
date_cols = ['cust_orig_date', 'date_of_birth', 'acct_suspd_date']
for col in date_cols:
    if col in df_clean.columns:
        df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')

# CRITICAL: Create date-based features BEFORE splitting
# Use fixed reference dates to avoid data leakage
reference_date = pd.Timestamp('2023-12-31')  # Fixed reference date

if 'cust_orig_date' in df_clean.columns:
    df_clean['cust_tenure_days'] = (reference_date - df_clean['cust_orig_date']).dt.days
    df_clean.drop(columns=['cust_orig_date'], inplace=True)

if 'date_of_birth' in df_clean.columns:
    df_clean['age'] = (reference_date - df_clean['date_of_birth']).dt.days // 365
    df_clean.drop(columns=['date_of_birth'], inplace=True)

if 'acct_suspd_date' in df_clean.columns:
    df_clean['was_suspended'] = df_clean['acct_suspd_date'].notna().astype(int)
    df_clean['days_since_suspension'] = (reference_date - df_clean['acct_suspd_date']).dt.days
    df_clean['days_since_suspension'].fillna(-1, inplace=True)
    df_clean.drop(columns=['acct_suspd_date'], inplace=True)

# Drop columns with too many missing values
null_frac = df_clean.isnull().mean()
drop_cols = [col for col in df_clean.columns if null_frac[col] > 0.9]
if drop_cols:
    df_clean.drop(columns=drop_cols, inplace=True)
    print(f"Dropped mostly null columns: {drop_cols}")

# Check for problematic columns that might cause leakage
problematic_cols = []
for col in df_clean.columns:
    if df_clean[col].nunique() == len(df_clean):
        problematic_cols.append(col)

if problematic_cols:
    print(f"⚠️  Dropping columns with unique values for each row: {problematic_cols}")
    df_clean.drop(columns=problematic_cols, inplace=True)

print(f"Dataset shape after cleaning: {df_clean.shape}")
print(f"Missing values:\n{df_clean.isnull().sum()}")

# Define target and features
TARGET = 'Churn'
if TARGET not in df_clean.columns:
    print("❌ Target column 'Churn' not found!")
    print(f"Available columns: {list(df_clean.columns)}")
    exit()

X = df_clean.drop(columns=[TARGET])
y = df_clean[TARGET]

print(f"Features: {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts()}")

# ===============================
# 4. Train-Test Split (CRITICAL)
# ===============================
print("\n--- 3. Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts()}")
print(f"Test target distribution:\n{y_test.value_counts()}")

# ===============================
# 5. Feature Engineering (Separately for train and test)
# ===============================
print("\n--- 4. Feature Engineering ---")

# Identify column types
numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Handle missing values
print("\nHandling missing values...")

# Numerical imputation
if numerical_cols:
    num_imputer = SimpleImputer(strategy='median')
    X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Categorical imputation
if categorical_cols:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
    X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Encode categorical variables
label_encoders = {}
if categorical_cols:
    for col in categorical_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        
        # Handle unseen categories in test set
        test_categories = set(X_test[col].astype(str).unique())
        train_categories = set(le.classes_)
        unseen_categories = test_categories - train_categories
        
        if unseen_categories:
            print(f"⚠️  Unseen categories in {col}: {unseen_categories}")
            # Map unseen categories to a special value
            X_test[col] = X_test[col].astype(str).apply(
                lambda x: x if x in train_categories else 'UNSEEN'
            )
            # If we still have UNSEEN, need to handle it
            if 'UNSEEN' not in le.classes_:
                # Add UNSEEN to the encoder
                le.classes_ = np.append(le.classes_, 'UNSEEN')
        
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le

# Scale numerical features
if numerical_cols:
    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

print("✅ Preprocessing completed")

# ===============================
# 6. Data Leakage Check
# ===============================
print("\n--- 5. Data Leakage Check ---")

# Check if there are any perfect predictors
from sklearn.feature_selection import mutual_info_classif

# Calculate mutual information
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)

print("Top features by Mutual Information:")
print(mi_series.head(10))

# Check for features with very high correlation with target
high_mi_features = mi_series[mi_series > 0.5]
if len(high_mi_features) > 0:
    print(f"⚠️  High MI features detected: {list(high_mi_features.index)}")

# ===============================
# 7. Model Training
# ===============================
print("\n--- 6. Model Training ---")

# Handle class imbalance
neg_count = y_train.value_counts()[0]
pos_count = y_train.value_counts()[1]
scale_pos_weight = neg_count / pos_count

print(f"Class distribution - Negative: {neg_count}, Positive: {pos_count}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# Use simpler models with regularization
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=50,  # Reduced
        random_state=42, 
        n_jobs=-1, 
        max_depth=5,  # Strict regularization
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=50,  # Reduced
        random_state=42,
        max_depth=3,  # Strict regularization
        min_samples_split=20,
        min_samples_leaf=10,
        subsample=0.7
    ),
    'XGBoost': XGBClassifier(
        n_estimators=50,  # Reduced
        use_label_encoder=False, 
        eval_metric='logloss', 
        random_state=42, 
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        max_depth=3,  # Strict regularization
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1,  # L1 regularization
        reg_lambda=1   # L2 regularization
    )
}

trained_models = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    trained_models[name] = model
    
    # Check training performance (should not be perfect)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f"  Training Accuracy: {train_accuracy:.4f}")

# ===============================
# 8. Model Evaluation
# ===============================
print("\n--- 7. Model Evaluation ---")

results = {}
best_model_name = None
best_roc_auc = -1

plt.figure(figsize=(10, 8))

for name, model in trained_models.items():
    print(f"\nEvaluating {name}...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'ROC-AUC': roc_auc
    }

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        best_model_name = name

# Finalize ROC plot
plt.plot([0, 1], [0, 1], 'r--', alpha=0.5, label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Model Comparison')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n--- Model Comparison ---")
results_df = pd.DataFrame(results).T
print(results_df.round(4))

print(f"\n✅ Best performing model: {best_model_name} (ROC-AUC: {best_roc_auc:.4f})")

# ===============================
# 9. Debugging: Check for Perfect Separation
# ===============================
print("\n--- 8. Debugging Info ---")

# Check if any single feature perfectly predicts the target
print("Checking for perfect predictors...")
perfect_predictors = []
for col in X_train.columns:
    unique_combinations = X_train[col].value_counts()
    if len(unique_combinations) == 2:  # Binary feature
        temp_df = pd.DataFrame({'feature': X_train[col], 'target': y_train})
        grouped = temp_df.groupby('feature')['target'].mean()
        if any(grouped == 0) and any(grouped == 1):
            perfect_predictors.append(col)
            print(f"⚠️  Perfect predictor found: {col}")

if perfect_predictors:
    print(f"🚨 Dropping perfect predictors: {perfect_predictors}")
    X_train = X_train.drop(columns=perfect_predictors)
    X_test = X_test.drop(columns=perfect_predictors)
    
    # Retrain models without perfect predictors
    print("Retraining models without perfect predictors...")
    trained_models = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

# Check dataset characteristics
print(f"\nFinal dataset info:")
print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print(f"Feature names: {list(X_train.columns)}")

print("\n🎯 Model training and evaluation completed!")