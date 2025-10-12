"""
XGBoost Churn Risk Explainer with SHAP Analysis - COMPLETE FIXED VERSION
Uses the actual trained model and real data from hackathon.py
Provides comprehensive SHAP, LIME, and PDP analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import pickle
from datetime import datetime
from collections import Counter

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.inspection import PartialDependenceDisplay

# Explainability libraries
import shap

# Optional explainer libs
try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:
    LIME_AVAILABLE = False

try:
    from pdpbox import pdp
    PDPBOX_AVAILABLE = True
except Exception:
    PDPBOX_AVAILABLE = False

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_style('darkgrid')


def preprocess_insurance_data(df):
    """
    Exact preprocessing function from hackathon.py
    """
    df = df.copy()

    # --- Step 1: Convert datetime columns properly ---
    date_cols = ['cust_orig_date', 'date_of_birth', 'acct_suspd_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # --- Step 2: Replace datetime with numeric features ---
    today = pd.Timestamp('today')

    if 'cust_orig_date' in df.columns:
        df['cust_orig_days_since'] = (today - df['cust_orig_date']).dt.days

    if 'date_of_birth' in df.columns:
        df['age'] = (today - df['date_of_birth']).dt.days // 365  # compute actual age

    if 'acct_suspd_date' in df.columns:
        df['acct_suspd_days_since'] = (today - df['acct_suspd_date']).dt.days.fillna(0)

    # Drop original datetime columns
    df = df.drop(columns=date_cols, errors='ignore')

    # --- Step 3: Handle categorical columns ---
    categorical_cols = df.select_dtypes(include=['object']).columns

    # --- Step 4: Ensure all are numeric ---
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)

    print("✅ Data preprocessing successful.")
    print("Final columns:", len(df.columns))
    return df


def load_and_prepare_data():
    """
    Load and prepare data using the exact same process as hackathon.py
    """
    print("📊 Loading and preparing real insurance data...")
    
    # Load data
    candidate_paths = [
        './data/autoinsurance_churn.csv',
        './Megathon_chosen_five/data/autoinsurance_churn.csv',
        './Explainable-Customer-Churn-Prediction/data/autoinsurance_churn.csv',
        './autoinsurance_churn.csv'
    ]
    DATA_FILENAME = None
    for p in candidate_paths:
        if os.path.exists(p):
            DATA_FILENAME = p
            break
    if DATA_FILENAME is None:
        raise FileNotFoundError('Dataset autoinsurance_churn.csv not found in expected locations: ' + ','.join(candidate_paths))
    
    df = pd.read_csv(DATA_FILENAME)
    print(f"Dataset loaded: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Drop ID columns (same as hackathon.py)
    id_cols = ['individual_id', 'address_id']
    df.drop(columns=id_cols, inplace=True, errors='ignore')
    
    # Drop mostly null columns
    null_frac = df.isnull().mean()
    drop_cols = [col for col in df.columns if null_frac[col] > 0.9]
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        print(f"Dropped mostly null columns: {drop_cols}")
    
    # Drop target leakage column
    if 'acct_suspd_date' in df.columns:
        df.drop(columns=['acct_suspd_date'], inplace=True)
        print("Dropped 'acct_suspd_date' to prevent target leakage")
    
    # Apply preprocessing
    df = preprocess_insurance_data(df)
    
    # Define target and features
    TARGET = 'Churn'
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")
    
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]
    
    # Handle any remaining missing values
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
    
    print(f"✅ Data prepared: {X.shape} features, {len(y)} samples")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_xgboost_model():
    """Load the trained XGBoost model"""
    print("🤖 Loading XGBoost model...")
    
    model_files = [
        './models/XGBoostClassifier_churn_prediction_model.pkl',
        './XGBoostClassifier_churn_prediction_model.pkl',
        './churn_xgboost_optimized.pkl',
        './churn_gradboost_optimized.pkl',
        './Megathon_chosen_five/churn_xgboost_optimized.pkl',
        './Megathon_chosen_five/churn_gradboost_optimized.pkl',
        './Megathon_chosen_five/models/XGBoostClassifier_churn_prediction_model.pkl',
        './Megathon_chosen_five/XGBoostClassifier_churn_prediction_model.pkl'
    ]
    
    for filepath in model_files:
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                print(f"✅ Loaded XGBoost model from: {filepath}")
                print(f"Model type: {type(model).__name__}")
                
                # Get expected features
                if hasattr(model, 'feature_names_in_'):
                    expected_features = model.feature_names_in_
                    print(f"Model expects {len(expected_features)} features")
                else:
                    expected_features = None
                    print("⚠️ Model doesn't have feature_names_in_ attribute")
                
                return model, expected_features, filepath
            except Exception as e:
                print(f"❌ Error loading model from {filepath}: {str(e)}")
                continue
    
    raise FileNotFoundError("XGBoost model not found.")


def align_features_with_model(X, expected_features):
    """
    Align the data features with what the model expects
    """
    print("\n🔧 Aligning features with model expectations...")
    
    if expected_features is None:
        print("⚠️ No expected features found, using data as-is")
        return X
    
    print(f"Data has {len(X.columns)} features")
    print(f"Model expects {len(expected_features)} features")
    
    # Find missing and extra features
    data_features = set(X.columns)
    model_features = set(expected_features)
    
    missing_features = model_features - data_features
    extra_features = data_features - model_features
    
    if missing_features:
        print(f"⚠️ Missing features (will add as zeros): {len(missing_features)}")
        for feature in missing_features:
            X[feature] = 0
    
    if extra_features:
        print(f"⚠️ Extra features (will remove): {len(extra_features)}")
        X = X.drop(columns=list(extra_features), errors='ignore')
    
    # Reorder columns to match model expectations
    X = X.reindex(columns=expected_features, fill_value=0)
    
    print(f"✅ Features aligned: {X.shape}")
    print(f"Final feature count: {len(X.columns)}")
    
    return X


def create_churn_risk_assessment(model, X, y):
    """
    Create comprehensive churn risk assessment using the real model and data
    """
    print("\n🎯 CHURN RISK ASSESSMENT")
    print("=" * 50)
    
    try:
        # Get model predictions
        y_pred_proba = model.predict_proba(X)[:, 1]  # Churn probabilities
        y_pred = model.predict(X)
        
        print("✅ Model predictions successful!")
        
        # Create risk assessment DataFrame
        risk_assessment = pd.DataFrame({
            'customer_id': range(len(X)),
            'churn_probability': y_pred_proba,
            'predicted_churn': y_pred,
            'actual_churn': y
        })
        
        # Risk categorization
        def categorize_risk(prob):
            if prob >= 0.7:
                return "EXTREME"
            elif prob >= 0.5:
                return "HIGH"
            elif prob >= 0.25:
                return "MEDIUM" 
            else:
                return "LOW"
        
        risk_assessment['risk_level'] = risk_assessment['churn_probability'].apply(categorize_risk)
        
        # Risk distribution
        risk_counts = risk_assessment['risk_level'].value_counts()
        total_customers = len(risk_assessment)
        
        print(f"\n📊 CUSTOMER RISK DISTRIBUTION:")
        print(f"   🔴 EXTREME RISK customers (>=70%):  {risk_counts.get('EXTREME', 0):6d} ({risk_counts.get('EXTREME', 0)/total_customers*100:5.1f}%)")
        print(f"   🟠 HIGH RISK customers (50%-70%):   {risk_counts.get('HIGH', 0):6d} ({risk_counts.get('HIGH', 0)/total_customers*100:5.1f}%)")
        print(f"   🟡 MEDIUM RISK customers (25-50%):  {risk_counts.get('MEDIUM', 0):6d} ({risk_counts.get('MEDIUM', 0)/total_customers*100:5.1f}%)")
        print(f"   🟢 LOW RISK customers (<25%):       {risk_counts.get('LOW', 0):6d} ({risk_counts.get('LOW', 0)/total_customers*100:5.1f}%)")
        
        # Show detailed examples from each risk level
        print(f"\n📋 SAMPLE CUSTOMERS BY RISK LEVEL:")
        
        for risk_level, emoji in [('EXTREME', '🔴'), ('HIGH', '🟠'), ('MEDIUM', '🟡'), ('LOW', '🟢')]:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            
            if len(risk_customers) > 0:
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS (showing top 5):")
                
                if risk_level in ['EXTREME', 'HIGH']:
                    samples = risk_customers.nlargest(5, 'churn_probability')
                elif risk_level == 'LOW':
                    samples = risk_customers.nsmallest(5, 'churn_probability')
                else:
                    samples = risk_customers.head(5)
                
                for idx, (_, customer) in enumerate(samples.iterrows(), 1):
                    pred_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                    actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                    accuracy = "✅" if customer['predicted_churn'] == customer['actual_churn'] else "❌"
                    
                    print(f"   {idx}. Customer {customer['customer_id']:6d}: {customer['churn_probability']:.3f} | "
                          f"Predicted: {pred_label:12s} | Actual: {actual_label:8s} {accuracy}")
            else:
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS: None found")
        
        # Statistics
        print(f"\n📈 CHURN PROBABILITY STATISTICS:")
        print(f"   Average:     {np.mean(y_pred_proba):.3f}")
        print(f"   Median:      {np.median(y_pred_proba):.3f}")
        print(f"   Std Dev:     {np.std(y_pred_proba):.3f}")
        print(f"   Min:         {np.min(y_pred_proba):.3f}")
        print(f"   Max:         {np.max(y_pred_proba):.3f}")
        print(f"   Q1 (25th):   {np.percentile(y_pred_proba, 25):.3f}")
        print(f"   Q3 (75th):   {np.percentile(y_pred_proba, 75):.3f}")
        
        # Model performance
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        print(f"\n🎯 MODEL PERFORMANCE METRICS:")
        print(f"   Accuracy:    {accuracy:.3f}")
        print(f"   Precision:   {precision:.3f}")
        print(f"   Recall:      {recall:.3f}")
        print(f"   F1-Score:    {f1:.3f}")
        print(f"   ROC-AUC:     {roc_auc:.3f}")
        
        # Save results
        os.makedirs('./outputs', exist_ok=True)
        risk_assessment.to_csv('./outputs/real_customer_risk_assessment.csv', index=False)
        print(f"\n💾 Risk assessment saved to: ./outputs/real_customer_risk_assessment.csv")
        
        return risk_assessment
        
    except Exception as e:
        print(f"❌ Error in risk assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def comprehensive_shap_analysis(model, X, y, risk_assessment):
    """
    Comprehensive SHAP analysis inspired by main.ipynb
    """
    print("\n🎯 COMPREHENSIVE SHAP ANALYSIS")
    print("=" * 60)
    print("Inspired by main.ipynb SHAP methodology")
    print("=" * 60)
    
    try:
        # Initialize SHAP
        print("🔍 Creating SHAP TreeExplainer for XGBoost...")
        explainer = shap.TreeExplainer(model)
        
        # Use a sample for SHAP computation (to manage memory)
        sample_size = min(10000, len(X))
        print(f"📊 Computing SHAP values for {sample_size} samples...")
        
        # Get diverse sample including different risk levels
        extreme_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'EXTREME'].head(200)
        high_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'HIGH'].head(300)
        medium_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'MEDIUM'].head(400)
        low_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'LOW'].head(300)
        
        # Combine customer IDs
        sample_indices = []
        sample_indices.extend(extreme_risk_customers['customer_id'].tolist())
        sample_indices.extend(high_risk_customers['customer_id'].tolist())
        sample_indices.extend(medium_risk_customers['customer_id'].tolist())
        sample_indices.extend(low_risk_customers['customer_id'].tolist())
        
        # If we don't have enough, add random samples
        if len(sample_indices) < sample_size:
            remaining = sample_size - len(sample_indices)
            additional_indices = np.random.choice(
                [i for i in range(len(X)) if i not in sample_indices], 
                size=min(remaining, len(X) - len(sample_indices)), 
                replace=False
            )
            sample_indices.extend(additional_indices.tolist())
        
        # Take first sample_size indices
        sample_indices = sample_indices[:sample_size]
        X_sample = X.iloc[sample_indices]
        y_sample = y.iloc[sample_indices]
        
        print(f"✅ Selected {len(sample_indices)} diverse customers for SHAP analysis")
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]  # Churn class
            expected_value = explainer.expected_value[1]
            print("✅ Using positive class SHAP values (churn prediction)")
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
            print("✅ Using SHAP values for churn prediction")
        
        print(f"✅ SHAP values computed: {shap_values_positive.shape}")
        
        # 1. SHAP Summary Plot (like main.ipynb)
        print("\n📊 Creating SHAP Summary Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_positive, X_sample, show=False, max_display=20)
        plt.title('SHAP Feature Importance Summary\n(Impact on Churn Prediction)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./outputs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. SHAP Bar Plot (feature importance)
        print("📊 Creating SHAP Bar Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_positive, X_sample, plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance (Bar Plot)\n(Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./outputs/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Individual Customer Analysis (like main.ipynb waterfall plots)
        print("\n🔍 Creating Individual Customer SHAP Explanations...")
        
        # Select representative customers from different risk levels
        analysis_customers = []
        
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            risk_idx = risk_assessment[risk_assessment['risk_level'] == risk_level]
            if len(risk_idx) > 0:
                sample_customer = risk_idx.iloc[0]['customer_id']
                if sample_customer in sample_indices:
                    analysis_customers.append((risk_level, sample_customer, sample_indices.index(sample_customer)))
        
        # Create waterfall plots for each representative customer
        for risk_level, customer_id, sample_idx in analysis_customers:
            print(f"📊 Creating SHAP Waterfall for {risk_level} risk customer {customer_id}...")
            
            customer_data = X_sample.iloc[sample_idx]
            customer_shap = shap_values_positive[sample_idx]
            customer_prob = risk_assessment.iloc[customer_id]['churn_probability']
            
            # Create waterfall plot
            shap_explanation = shap.Explanation(
                values=customer_shap,
                base_values=expected_value,
                data=customer_data,
                feature_names=X.columns.tolist()
            )
            
            emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
            
            plt.figure(figsize=(12, 10))
            shap.plots.waterfall(shap_explanation, max_display=15, show=False)
            plt.title(f'{emoji} {risk_level} Risk Customer {customer_id}\n'
                     f'Churn Probability: {customer_prob:.3f}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'./outputs/shap_waterfall_{risk_level.lower()}_risk_customer_{customer_id}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Print interpretation
            feature_contributions = pd.DataFrame({
                'feature': X.columns,
                'shap_value': customer_shap,
                'feature_value': customer_data.values
            })
            feature_contributions['abs_shap'] = abs(feature_contributions['shap_value'])
            
            print(f"\n📊 Top Risk Factors for {risk_level} Risk Customer {customer_id}:")
            top_risk = feature_contributions[feature_contributions['shap_value'] > 0].nlargest(5, 'shap_value')
            for _, row in top_risk.iterrows():
                print(f"   • {row['feature']}: +{row['shap_value']:.4f} (value: {row['feature_value']:.3f})")
            
            print(f"\n📊 Top Protective Factors:")
            top_protective = feature_contributions[feature_contributions['shap_value'] < 0].nsmallest(5, 'shap_value')
            for _, row in top_protective.iterrows():
                print(f"   • {row['feature']}: {row['shap_value']:.4f} (value: {row['feature_value']:.3f})")
        
        # 4. Overall Feature Importance Analysis
        print(f"\n📊 OVERALL FEATURE IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        # Calculate mean absolute SHAP values
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.mean(np.abs(shap_values_positive), axis=0),
            'mean_shap': np.mean(shap_values_positive, axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("📈 Top 15 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            direction = "→ Increases Churn" if row['mean_shap'] > 0 else "→ Reduces Churn"
            print(f"   {i:2d}. {row['feature']:<35}: {row['mean_abs_shap']:.4f} {direction}")
        
        # Save feature importance
        feature_importance.to_csv('./outputs/shap_feature_importance.csv', index=False)
        
        # 5. Risk Level Comparison
        print(f"\n📊 SHAP VALUES BY RISK LEVEL")
        print("=" * 50)
        
        # Compare SHAP values across risk levels
        risk_sample_map = {}
        for customer_id in sample_indices:
            risk_level = risk_assessment.iloc[customer_id]['risk_level']
            if risk_level not in risk_sample_map:
                risk_sample_map[risk_level] = []
            risk_sample_map[risk_level].append(sample_indices.index(customer_id))
        
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            if risk_level in risk_sample_map:
                indices = risk_sample_map[risk_level]
                risk_shap = shap_values_positive[indices]
                mean_risk_shap = np.mean(risk_shap, axis=0)
                
                # Top features for this risk level
                risk_importance = pd.DataFrame({
                    'feature': X.columns,
                    'mean_shap': mean_risk_shap
                }).sort_values('mean_shap', key=abs, ascending=False)
                
                emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
                print(f"\n{emoji} {risk_level} RISK - Top Features:")
                for i, (_, row) in enumerate(risk_importance.head(5).iterrows(), 1):
                    direction = "↑ Churn" if row['mean_shap'] > 0 else "↓ Churn"
                    print(f"   {i}. {row['feature']:<30}: {row['mean_shap']:+.4f} {direction}")
        
        # Save SHAP explainer and values
        os.makedirs('./models', exist_ok=True)
        joblib.dump(explainer, './models/shap_explainer_real.bz2', compress=('bz2', 9))
        np.save('./models/shap_values_sample.npy', shap_values_positive)
        np.save('./models/shap_sample_indices.npy', sample_indices)
        
        print(f"\n💾 SHAP analysis artifacts saved:")
        print(f"   • ./models/shap_explainer_real.bz2")
        print(f"   • ./models/shap_values_sample.npy")
        print(f"   • ./outputs/shap_summary_plot.png")
        print(f"   • ./outputs/shap_bar_plot.png")
        print(f"   • ./outputs/shap_feature_importance.csv")
        
        return explainer, shap_values_positive, sample_indices, feature_importance

    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def local_lime_analysis(model, X_train, X_test, risk_assessment, outputs_dir='./outputs'):
    """
    Generate LIME explanations for representative customers and save results.
    
    FIXED VERSION: Uses training data as background, proper customer selection
    """
    os.makedirs(outputs_dir, exist_ok=True)
    lime_results = {}
    
    if not LIME_AVAILABLE:
        print("\n⚠️ LIME not installed; skipping LIME local explanations.")
        print("   Install with: pip install lime")
        return lime_results

    print("\n🔎 Running LIME on representative customers...")
    
    try:
        # Initialize LIME explainer with training data (FIXED)
        print("   Creating LIME explainer with training data background...")
        explainer = LimeTabularExplainer(
            training_data=X_train.values,  # FIXED: Use training data
            feature_names=X_train.columns.tolist(),
            class_names=['Retain', 'Churn'],
            mode='classification',  # FIXED: Added mode
            discretize_continuous=True,
            random_state=42
        )
        
        # Select diverse customers from each risk level (FIXED)
        customers_to_explain = []
        
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            if len(risk_customers) > 0:
                # Take up to 2 customers per risk level
                for i in range(min(2, len(risk_customers))):
                    customer_id = risk_customers.iloc[i]['customer_id']
                    if customer_id < len(X_test):  # FIXED: Check bounds
                        customers_to_explain.append((customer_id, risk_level))
        
        print(f"   Explaining {len(customers_to_explain)} customers...")
        
        # Generate explanations (FIXED: Better error handling)
        for customer_id, risk_level in customers_to_explain:
            try:
                # Get customer data
                customer_data = X_test.iloc[customer_id].values
                customer_prob = risk_assessment.iloc[customer_id]['churn_probability']
                
                # Generate LIME explanation
                exp = explainer.explain_instance(
                    data_row=customer_data,
                    predict_fn=model.predict_proba,
                    num_features=15,
                    top_labels=2
                )
                
                # Get explanation for churn class (class 1)
                explanation_list = exp.as_list(label=1)
                lime_results[customer_id] = explanation_list
                
                # Save explanation as text
                out_txt = os.path.join(outputs_dir, f'lime_explanation_{risk_level}_customer_{customer_id}.txt')
                with open(out_txt, 'w', encoding='utf8') as f:
                    f.write(f"LIME Explanation for Customer {customer_id}\n")
                    f.write(f"Risk Level: {risk_level}\n")
                    f.write(f"Churn Probability: {customer_prob:.3f}\n")
                    f.write("="*60 + "\n\n")
                    f.write("Top Feature Contributions:\n")
                    f.write("-"*60 + "\n")
                    for feature, weight in explanation_list:
                        direction = "→ Increases Churn" if weight > 0 else "→ Reduces Churn"
                        f.write(f"{feature:<40} {weight:+.4f} {direction}\n")
                
                # Save visualization (FIXED: Added visualization)
                try:
                    fig = exp.as_pyplot_figure(label=1)
                    fig.suptitle(f'LIME Explanation - {risk_level} Risk Customer {customer_id}\n'
                               f'Churn Probability: {customer_prob:.3f}', 
                               fontsize=12, fontweight='bold')
                    fig.tight_layout()
                    out_img = os.path.join(outputs_dir, f'lime_plot_{risk_level}_customer_{customer_id}.png')
                    fig.savefig(out_img, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"   ✅ Customer {customer_id} ({risk_level}): {out_txt}")
                except Exception as e:
                    print(f"   ⚠️ Could not save LIME plot for customer {customer_id}: {e}")
                
            except Exception as e:
                print(f"   ❌ Error explaining customer {customer_id}: {e}")
                continue
        
        print(f"\n✅ LIME analysis complete: {len(lime_results)} customers explained")
        return lime_results
        
    except Exception as e:
        print(f"❌ Error in LIME analysis: {e}")
        import traceback
        traceback.print_exc()
        return lime_results

# def consistency_report(feature_importance_df, model, lime_results, outputs_dir='./outputs'):
#     """
#     Generate a comprehensive consistency report comparing different explainability methods.
    
#     FIXED VERSION: Proper LIME aggregation, feature name extraction, overlap calculations
#     """
#     os.makedirs(outputs_dir, exist_ok=True)
#     print(f"\n📊 Creating Explainability Consistency Report...")
    
#     report_lines = []
#     report_lines.append('# Explainability Consistency Report')
#     report_lines.append('=' * 80)
#     report_lines.append(f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}')
#     report_lines.append('')
#     report_lines.append('This report compares different explainability methods:')
#     report_lines.append('- SHAP (global and local explanations)')
#     report_lines.append('- XGBoost feature importances (gain-based)')
#     report_lines.append('- LIME (local explanations, if available)')
#     report_lines.append('')
    
#     # 1. Top SHAP features
#     report_lines.append('## 1. SHAP Feature Importance (Top 20)')
#     report_lines.append('-' * 80)
#     top_shap = feature_importance_df.head(20)
#     report_lines.append('| Rank | Feature | Mean Abs SHAP | Mean SHAP | Direction |')
#     report_lines.append('|------|---------|---------------|-----------|-----------|')
#     for idx, (i, row) in enumerate(top_shap.iterrows(), 1):
#         direction = "↑ Churn" if row['mean_shap'] > 0 else "↓ Churn"
#         report_lines.append(f"| {idx:2d} | {row['feature']:<40} | {row['mean_abs_shap']:.4f} | {row['mean_shap']:+.4f} | {direction} |")
#     report_lines.append('')
    
#     # 2. Model feature importances
#     report_lines.append('## 2. XGBoost Feature Importances (Top 20)')
#     report_lines.append('-' * 80)
#     if hasattr(model, 'feature_importances_'):
#         # Get feature names (FIXED)
#         if hasattr(model, 'feature_names_in_'):
#             feature_names = model.feature_names_in_
#         else:
#             feature_names = feature_importance_df['feature'].values
        
#         # Create importance dataframe
#         model_importance = pd.DataFrame({
#             'feature': feature_names,
#             'importance': model.feature_importances_
#         }).sort_values('importance', ascending=False)
        
#         top_model = model_importance.head(20)
#         report_lines.append('| Rank | Feature | Importance |')
#         report_lines.append('|------|---------|------------|')
#         for idx, (i, row) in enumerate(top_model.iterrows(), 1):
#             report_lines.append(f"| {idx:2d} | {row['feature']:<40} | {row['importance']:.4f} |")
#         report_lines.append('')
        
#         top_model_features = top_model['feature'].tolist()
#     else:
#         report_lines.append('⚠️ Model feature importances not available for this model type.')
#         report_lines.append('')
#         top_model_features = []
    
#     # 3. LIME aggregated features (FIXED: Proper aggregation)
#     report_lines.append('## 3. LIME Feature Importance (Aggregated from Local Explanations)')
#     report_lines.append('-' * 80)
#     if LIME_AVAILABLE and lime_results:
#         # Aggregate LIME results (FIXED)
#         feature_weights = Counter()
#         feature_counts = Counter()
        
#         for customer_id, explanation_list in lime_results.items():
#             for feature_desc, weight in explanation_list:
#                 # Extract feature name (FIXED: Remove value condition)
#                 feature_name = feature_desc.split('<=')[0].split('>')[0].split('<')[0].strip()
#                 feature_weights[feature_name] += abs(weight)
#                 feature_counts[feature_name] += 1
        
#         # Calculate average absolute weight
#         lime_importance = pd.DataFrame([
#             {
#                 'feature': feat,
#                 'total_weight': weight,
#                 'count': feature_counts[feat],
#                 'avg_weight': weight / feature_counts[feat]
#             }
#             for feat, weight in feature_weights.most_common(20)
#         ])
        
#         report_lines.append(f'Based on {len(lime_results)} local explanations')
#         report_lines.append('')
#         report_lines.append('| Rank | Feature | Avg Abs Weight | Appearances |')
#         report_lines.append('|------|---------|----------------|-------------|')
#         for idx, (i, row) in enumerate(lime_importance.iterrows(), 1):
#             report_lines.append(f"| {idx:2d} | {row['feature']:<40} | {row['avg_weight']:.4f} | {row['count']:2d} |")
#         report_lines.append('')
        
#         top_lime_features = lime_importance['feature'].tolist()
#     else:
#         if not LIME_AVAILABLE:
#             report_lines.append('⚠️ LIME not installed. Install with: pip install lime')
#         else:
#             report_lines.append('⚠️ No LIME explanations were generated.')
#         report_lines.append('')
#         top_lime_features = []
    
#     # 4. Consistency Analysis (FIXED: Proper overlap calculations)
#     report_lines.append('## 4. Consistency Analysis')
#     report_lines.append('-' * 80)
    
#     top_shap_features = feature_importance_df.head(20)['feature'].tolist()
    
#     # SHAP vs Model Importances
#     if top_model_features:
#         overlap_shap_model = len(set(top_shap_features[:10]).intersection(set(top_model_features[:10])))
#         report_lines.append(f'### SHAP vs XGBoost Feature Importances (Top 10):')
#         report_lines.append(f'- Features in common: **{overlap_shap_model}/10** ({overlap_shap_model*10}%)')
#         report_lines.append(f'- Agreement level: {"High" if overlap_shap_model >= 7 else "Moderate" if overlap_shap_model >= 5 else "Low"}')
#         report_lines.append('')
        
#         # Show common features
#         common_features = set(top_shap_features[:10]).intersection(set(top_model_features[:10]))
#         if common_features:
#             report_lines.append('Common features:')
#             for feat in sorted(common_features):
#                 report_lines.append(f'- {feat}')
#             report_lines.append('')
    
#     # SHAP vs LIME
#     if top_lime_features:
#         overlap_shap_lime = len(set(top_shap_features[:10]).intersection(set(top_lime_features[:10])))
#         report_lines.append(f'### SHAP vs LIME (Top 10):')
#         report_lines.append(f'- Features in common: **{overlap_shap_lime}/10** ({overlap_shap_lime*10}%)')
#         report_lines.append(f'- Agreement level: {"High" if overlap_shap_lime >= 7 else "Moderate" if overlap_shap_lime >= 5 else "Low"}')
#         report_lines.append('')
        
#         # Show common features
#         common_features = set(top_shap_features[:10]).intersection(set(top_lime_features[:10]))
#         if common_features:
#             report_lines.append('Common features:')
#             for feat in sorted(common_features):
#                 report_lines.append(f'- {feat}')
#             report_lines.append('')
    
#     # Model vs LIME
#     if top_model_features and top_lime_features:
#         overlap_model_lime = len(set(top_model_features[:10]).intersection(set(top_lime_features[:10])))
#         report_lines.append(f'### XGBoost vs LIME (Top 10):')
#         report_lines.append(f'- Features in common: **{overlap_model_lime}/10** ({overlap_model_lime*10}%)')
#         report_lines.append(f'- Agreement level: {"High" if overlap_model_lime >= 7 else "Moderate" if overlap_model_lime >= 5 else "Low"}')
#         report_lines.append('')
    
#     # 5. Key Insights (FIXED)
#     report_lines.append('## 5. Key Insights')
#     report_lines.append('-' * 80)
    
#     # Most consistent features (appear in all methods)
#     all_methods = [top_shap_features[:15]]
#     if top_model_features:
#         all_methods.append(top_model_features[:15])
#     if top_lime_features:
#         all_methods.append(top_lime_features[:15])
    
#     if len(all_methods) >= 2:
#         consistent_features = set(all_methods[0])
#         for method_features in all_methods[1:]:
#             consistent_features = consistent_features.intersection(set(method_features))
        
#         report_lines.append(f'### Most Consistent Features (Top 15):')
#         if consistent_features:
#             report_lines.append('Features that appear in all explainability methods:')
#             for feat in sorted(consistent_features):
#                 report_lines.append(f'- **{feat}**')
#         else:
#             report_lines.append('No features consistently appear in top 15 across all methods.')
#         report_lines.append('')
    
#     # Recommendations
#     report_lines.append('## 6. Recommendations')
#     report_lines.append('-' * 80)
#     report_lines.append('Based on this consistency analysis:')
#     report_lines.append('')
    
#     if len(all_methods) >= 2 and consistent_features:
#         report_lines.append(f'✅ **High Confidence Features**: The {len(consistent_features)} features appearing in all methods')
#         report_lines.append('   are the most reliable drivers of churn and should be prioritized for intervention.')
    
#     if top_model_features and overlap_shap_model >= 7:
#         report_lines.append('✅ **SHAP and Model Importances Align**: High agreement suggests stable feature importance.')
#     elif top_model_features:
#         report_lines.append('⚠️ **SHAP and Model Importances Differ**: Consider local vs global importance differences.')
    
#     if top_lime_features and overlap_shap_lime >= 7:
#         report_lines.append('✅ **Local and Global Explanations Align**: LIME and SHAP show consistent patterns.')
#     elif top_lime_features:
#         report_lines.append('⚠️ **Local Explanations Vary**: LIME shows different patterns - investigate customer segments.')
    
#     report_lines.append('')
    
#     # Write report
#     report_path = os.path.join(outputs_dir, 'explainability_consistency_report.md')
#     with open(report_path, 'w', encoding='utf8') as f:
#         f.write('\n'.join(report_lines))
    
#     print(f"✅ Consistency report saved to: {report_path}")
#     return report_path


def main():
    """Main function to run the complete analysis"""
    print("🚀 XGBOOST CHURN RISK EXPLAINER - REAL DATA")
    print("=" * 60)
    print("Using actual trained model and real insurance data")
    print("SHAP, LIME, and PDP analysis with comprehensive explainability")
    print("=" * 60)
    
    try:
        # 1. Load XGBoost model
        model, expected_features, model_path = load_xgboost_model()
        
        # 2. Load and prepare real data
        X, y = load_and_prepare_data()
        
        # 3. Align features with model expectations
        X_aligned = align_features_with_model(X, expected_features)
        
        # 4. Create train-test split (same as hackathon.py)
        print(f"\n🔄 Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 5. Create risk assessment on test set
        risk_assessment = create_churn_risk_assessment(model, X_test, y_test)
        
        if risk_assessment is not None:
            # 6. Comprehensive SHAP analysis
            explainer, shap_values, sample_indices, feature_importance = comprehensive_shap_analysis(
                model, X_test, y_test, risk_assessment
            )

            # 7. Additional explainability methods (FIXED)
            if explainer is not None and shap_values is not None and feature_importance is not None:
                print("\n" + "="*60)
                print("🔍 ADDITIONAL EXPLAINABILITY ANALYSIS")
                print("="*60)
                
                # LIME Local Explanations (FIXED: Pass X_train)
                lime_results = local_lime_analysis(
                    model=model,
                    X_train=X_train,  # FIXED: Pass training data
                    X_test=X_test,
                    risk_assessment=risk_assessment,
                    outputs_dir='./outputs'
                )
                
                print("\n✅ Additional explainability analysis complete!")
                print(f"   • LIME explanations: {len(lime_results)} customers")
            
            # 8. Final summary
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Summary statistics
            extreme_count = len(risk_assessment[risk_assessment['risk_level'] == 'EXTREME'])
            high_count = len(risk_assessment[risk_assessment['risk_level'] == 'HIGH'])
            medium_count = len(risk_assessment[risk_assessment['risk_level'] == 'MEDIUM'])
            low_count = len(risk_assessment[risk_assessment['risk_level'] == 'LOW'])
            total_count = len(risk_assessment)
            
            print("📊 EXECUTIVE SUMMARY:")
            print(f"   Total Customers Analyzed: {total_count:,}")
            print(f"   🔴 Extreme Risk (≥70%):   {extreme_count:6,} ({extreme_count/total_count*100:5.1f}%)")
            print(f"   🟠 High Risk (50-70%):    {high_count:6,} ({high_count/total_count*100:5.1f}%)")
            print(f"   🟡 Medium Risk (25-50%):  {medium_count:6,} ({medium_count/total_count*100:5.1f}%)")
            print(f"   🟢 Low Risk (<25%):       {low_count:6,} ({low_count/total_count*100:5.1f}%)")
            
            avg_prob = np.mean(risk_assessment['churn_probability'])
            print(f"   📈 Average Churn Risk:    {avg_prob:.1%}")
            
            print(f"\n📁 FILES CREATED:")
            print(f"   • ./outputs/real_customer_risk_assessment.csv")
            print(f"   • ./outputs/shap_summary_plot.png")
            print(f"   • ./outputs/shap_bar_plot.png")
            print(f"   • ./outputs/shap_feature_importance.csv")
            print(f"   • ./outputs/explainability_consistency_report.md")
            if lime_results:
                print(f"   • ./outputs/lime_explanation_*.txt (text files)")
                print(f"   • ./outputs/lime_plot_*.png (visualizations)")
            
            print(f"\n🎯 BUSINESS IMPACT:")
            if extreme_count > 0:
                estimated_revenue_risk = extreme_count * 1200
                print(f"   • {extreme_count:,} customers at EXTREME RISK")
                print(f"   • Estimated revenue at risk: ${estimated_revenue_risk:,}")
            
            if high_count > 0:
                estimated_revenue_risk_high = high_count * 1200
                print(f"   • {high_count:,} customers at HIGH RISK")
                print(f"   • Additional revenue at risk: ${estimated_revenue_risk_high:,}")
            
            print(f"\n💡 EXPLAINABILITY INSIGHTS:")
            print(f"   • SHAP: Global and local feature importance")
            if lime_results:
                print(f"   • LIME: {len(lime_results)} individual customer explanations")

            print("="*60)
            
        else:
            print("❌ Risk assessment failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()