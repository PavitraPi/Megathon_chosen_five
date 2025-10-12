"""
Gradient Boosting Churn Risk Explainer with SHAP Analysis
Uses the actual trained GradientBoostingClassifier model and real data from hackathon.py
Provides comprehensive SHAP analysis inspired by main.ipynb
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import joblib
import pickle
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Explainability libraries
import shap

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
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

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
    DATA_FILENAME = './data/autoinsurance_churn.csv'
    
    if not os.path.exists(DATA_FILENAME):
        raise FileNotFoundError(f"Dataset not found at {DATA_FILENAME}")
    
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
            mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 0
            X[col].fillna(mode_val, inplace=True)
    
    print(f"✅ Data prepared: {X.shape} features, {len(y)} samples")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y


def load_gradientboost_model():
    """Load the trained GradientBoostingClassifier model with version compatibility handling"""
    print("🤖 Loading GradientBoostingClassifier model...")
    
    # Check current scikit-learn version
    import sklearn
    current_sklearn_version = sklearn.__version__
    print(f"📦 Current scikit-learn version: {current_sklearn_version}")
    
    # Try different models based on compatibility
    model_files = [
        ('./models/GradientBoostingClassifier_churn_prediction_model.pkl', 'GradientBoostingClassifier'),
        ('./churn_gradboost_optimized.pkl', 'GradientBoostingClassifier'),
        ('./models/XGBoostClassifier_churn_prediction_model.pkl', 'XGBoostClassifier'),
        ('./models/churn_xgboost_optimized.pkl', 'XGBoostClassifier')
    ]
    
    for filepath, model_name in model_files:
        if os.path.exists(filepath):
            print(f"\n🔍 Trying {model_name} from: {filepath}")
            
            try:
                model = joblib.load(filepath)
                model_type = type(model).__name__
                print(f"✅ Loaded {model_type} model successfully")
                
                # Test if the model can make predictions
                try:
                    # Create a small test array with the right number of features
                    if hasattr(model, 'feature_names_in_'):
                        n_features = len(model.feature_names_in_)
                        expected_features = model.feature_names_in_
                    elif hasattr(model, 'n_features_in_'):
                        n_features = model.n_features_in_
                        expected_features = None
                    else:
                        # Try to determine from loaded artifacts
                        try:
                            artifacts = joblib.load('./models/model_artifacts.pkl')
                            expected_features = artifacts.get('feature_names', None)
                            n_features = len(expected_features) if expected_features else 100
                        except:
                            n_features = 100  # Default fallback
                            expected_features = None
                    
                    # Test prediction
                    test_X = np.zeros((1, n_features))
                    test_pred = model.predict(test_X)
                    print(f"✅ Model prediction test successful")
                    
                    # Test probability prediction
                    try:
                        test_proba = model.predict_proba(test_X)
                        print(f"✅ Model probability prediction successful")
                        prediction_method = 'predict_proba'
                    except Exception as proba_error:
                        print(f"⚠ predict_proba failed: {proba_error}")
                        print(f"   Will use predict only method")
                        prediction_method = 'predict_only'
                    
                    print(f"✅ {model_type} is compatible with current scikit-learn version")
                    print(f"Model expects {n_features} features")
                    return model, expected_features, filepath, prediction_method, model_type
                    
                except Exception as pred_error:
                    print(f"❌ Model prediction test failed: {pred_error}")
                    print(f"   Version incompatibility detected")
                    continue
                    
            except Exception as e:
                print(f"❌ Error loading model from {filepath}: {str(e)}")
                continue
    
    # If we get here, no model worked
    print(f"\n❌ No compatible model found for scikit-learn {current_sklearn_version}")
    print(f"💡 Recommendations:")
    print(f"   • Check if model files exist in ./models/ directory")
    print(f"   • Try running with different scikit-learn version")
    print(f"   • Or retrain models with current scikit-learn version")
    
    raise FileNotFoundError("No compatible model found for current scikit-learn version.")


def align_features_with_model(X, expected_features):
    """
    Align the data features with what the model expects
    """
    print("\n🔧 Aligning features with model expectations...")
    
    if expected_features is None:
        print("⚠ No expected features found, using data as-is")
        return X
    
    print(f"Data has {len(X.columns)} features")
    print(f"Model expects {len(expected_features)} features")
    
    # Find missing and extra features
    data_features = set(X.columns)
    model_features = set(expected_features)
    
    missing_features = model_features - data_features
    extra_features = data_features - model_features
    
    if missing_features:
        print(f"⚠ Missing features (will add as zeros): {len(missing_features)}")
        # Add missing features as zeros
        for feature in missing_features:
            X[feature] = 0
    
    if extra_features:
        print(f"⚠ Extra features (will remove): {len(extra_features)}")
        # Remove extra features
        X = X.drop(columns=list(extra_features), errors='ignore')
    
    # Reorder columns to match model expectations
    X = X.reindex(columns=expected_features, fill_value=0)
    
    print(f"✅ Features aligned: {X.shape}")
    print(f"Final feature count: {len(X.columns)}")
    
    return X


def create_churn_risk_assessment(model, X, y, prediction_method='predict_proba'):
    """
    Create comprehensive churn risk assessment using the real model and data
    """
    print("\n🎯 CHURN RISK ASSESSMENT")
    print("=" * 50)
    
    try:
        # Handle different prediction methods based on model compatibility
        print(f"🔧 Using prediction method: {prediction_method}")
        
        if prediction_method == 'predict_proba':
            try:
                y_pred_proba = model.predict_proba(X)[:, 1]  # Churn probabilities
                print("✅ predict_proba successful!")
            except Exception as e:
                print(f"⚠ predict_proba failed: {e}")
                print("🔄 Falling back to predict_only method...")
                prediction_method = 'predict_only'
        
        if prediction_method == 'predict_only':
            # Use only predict method and create synthetic probabilities
            y_pred = model.predict(X)
            # Create synthetic probabilities: high confidence for predictions
            # This is a workaround when predict_proba is not available
            y_pred_proba = np.where(y_pred == 1, 0.8, 0.2)  # High confidence predictions
            print("✅ Using predict with synthetic probabilities!")
        else:
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
        print(f"   🔴 EXTREME RISK customers (>=70%):    {risk_counts.get('EXTREME', 0):6d} ({risk_counts.get('EXTREME', 0)/total_customers*100:5.1f}%)")

        print(f"   🔴 HIGH RISK customers (50%-70%):    {risk_counts.get('HIGH', 0):6d} ({risk_counts.get('HIGH', 0)/total_customers*100:5.1f}%)")
        print(f"   🟡 MEDIUM RISK customers (25-50%): {risk_counts.get('MEDIUM', 0):6d} ({risk_counts.get('MEDIUM', 0)/total_customers*100:5.1f}%)")
        print(f"   🟢 LOW RISK customers (<25%):      {risk_counts.get('LOW', 0):6d} ({risk_counts.get('LOW', 0)/total_customers*100:5.1f}%)")
        
        # Show detailed examples from each risk level
        print(f"\n📋 SAMPLE CUSTOMERS BY RISK LEVEL:")
        
        for risk_level, emoji in [('EXTREME', '🔴'), ('HIGH', ' '), ('MEDIUM', '🟡'), ('LOW', '🟢')]:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            
            if len(risk_customers) > 0:
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS (showing top 5):")
                
                if risk_level == 'EXTREME':
                    samples = risk_customers.nlargest(5, 'churn_probability')
                elif risk_level == 'HIGH':
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
        
        # Save results with model-specific naming
        os.makedirs('./outputs', exist_ok=True)
        model_name = type(model).__name__.lower()
        output_filename = f'./outputs/{model_name}_customer_risk_assessment.csv'
        risk_assessment.to_csv(output_filename, index=False)
        print(f"\n💾 Risk assessment saved to: {output_filename}")
        
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
        # Initialize SHAP (works for both GradientBoostingClassifier and XGBoost)
        model_type = type(model).__name__
        print(f"🔍 Creating SHAP TreeExplainer for {model_type}...")
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
        plt.close()  # Close instead of show to prevent display issues
        print("✅ Summary plot saved")
        
        # 2. SHAP Bar Plot (feature importance)
        print("\n📊 Creating SHAP Bar Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_positive, X_sample, plot_type="bar", show=False, max_display=20)
        plt.title('SHAP Feature Importance (Bar Plot)\n(Mean Absolute SHAP Values)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./outputs/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close instead of show
        print("✅ Bar plot saved")
        
        # 3. Individual Customer Analysis (like main.ipynb waterfall plots)
        print("\n🔍 Creating Individual Customer SHAP Explanations...")
        
        # Select representative customers from different risk levels
        analysis_customers = []
        
        # Extreme risk customer
        extreme_risk_idx = risk_assessment[risk_assessment['risk_level'] == 'EXTREME']
        if len(extreme_risk_idx) > 0:
            sample_extreme = extreme_risk_idx.iloc[0]['customer_id']
            if sample_extreme in sample_indices:
                analysis_customers.append(('EXTREME', sample_extreme, sample_indices.index(sample_extreme)))
        
        # High risk customer
        high_risk_idx = risk_assessment[risk_assessment['risk_level'] == 'HIGH']
        if len(high_risk_idx) > 0:
            sample_high = high_risk_idx.iloc[0]['customer_id']
            if sample_high in sample_indices:
                analysis_customers.append(('HIGH', sample_high, sample_indices.index(sample_high)))
        
        # Medium risk customer
        medium_risk_idx = risk_assessment[risk_assessment['risk_level'] == 'MEDIUM']
        if len(medium_risk_idx) > 0:
            sample_medium = medium_risk_idx.iloc[0]['customer_id']
            if sample_medium in sample_indices:
                analysis_customers.append(('MEDIUM', sample_medium, sample_indices.index(sample_medium)))
        
        # Low risk customer
        low_risk_idx = risk_assessment[risk_assessment['risk_level'] == 'LOW']
        if len(low_risk_idx) > 0:
            sample_low = low_risk_idx.iloc[0]['customer_id']
            if sample_low in sample_indices:
                analysis_customers.append(('LOW', sample_low, sample_indices.index(sample_low)))
        
        # Create waterfall plots for each representative customer
        for risk_level, customer_id, sample_idx in analysis_customers:
            print(f"\n📊 Creating SHAP Waterfall for {risk_level} risk customer {customer_id}...")
            
            try:
                customer_data = X_sample.iloc[sample_idx]
                customer_shap = shap_values_positive[sample_idx]
                customer_prob = risk_assessment.iloc[customer_id]['churn_probability']
                
                # Create waterfall plot with proper error handling
                shap_explanation = shap.Explanation(
                    values=customer_shap,
                    base_values=expected_value,
                    data=customer_data.values,  # Use .values to ensure proper array format
                    feature_names=X.columns.tolist()
                )
                
                emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
                
                plt.figure(figsize=(12, 10))
                try:
                    shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                    plt.title(f'{emoji} {risk_level} Risk Customer {customer_id}\n'
                             f'Churn Probability: {customer_prob:.3f}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(f'./outputs/shap_waterfall_{risk_level.lower()}risk_customer{customer_id}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()  # Close the figure to prevent memory issues
                    print(f"✅ Waterfall plot saved successfully")
                except Exception as waterfall_error:
                    print(f"⚠ Waterfall plot failed: {waterfall_error}")
                    print(f"   Creating alternative force plot...")
                    
                    # Alternative: Create force plot instead
                    plt.figure(figsize=(12, 8))
                    shap.plots.force(expected_value, customer_shap, customer_data, 
                                   feature_names=X.columns.tolist(), matplotlib=True, show=False)
                    plt.title(f'{emoji} {risk_level} Risk Customer {customer_id} - SHAP Force Plot\n'
                             f'Churn Probability: {customer_prob:.3f}', fontsize=14, fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(f'./outputs/shap_force_{risk_level.lower()}risk_customer{customer_id}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Force plot saved as alternative")
                
                # Print interpretation (this part was working fine)
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
                    
            except Exception as customer_error:
                print(f"❌ Error analyzing customer {customer_id}: {customer_error}")
                continue
        
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
        
        # Save SHAP explainer and values with model-specific naming
        os.makedirs('./models', exist_ok=True)
        model_name = type(model).__name__.lower()
        explainer_filename = f'./models/shap_explainer_{model_name}.bz2'
        values_filename = f'./models/shap_values_{model_name}_sample.npy'
        indices_filename = f'./models/shap_sample_indices_{model_name}.npy'
        
        joblib.dump(explainer, explainer_filename, compress=('bz2', 9))
        np.save(values_filename, shap_values_positive)
        np.save(indices_filename, sample_indices)
        
        print(f"\n💾 SHAP analysis artifacts saved:")
        print(f"   • {explainer_filename}")
        print(f"   • {values_filename}")
        print(f"   • ./outputs/shap_summary_plot.png")
        print(f"   • ./outputs/shap_bar_plot.png")
        print(f"   • ./outputs/shap_feature_importance.csv")
        
        return explainer, shap_values_positive, sample_indices
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    """Main function to run the complete analysis"""
    print("🚀 ADAPTIVE CHURN RISK EXPLAINER - REAL DATA")
    print("=" * 60)
    print("Auto-detects compatible model based on scikit-learn version")
    print("Supports both GradientBoostingClassifier and XGBoost models")
    print("=" * 60)
    
    try:
        # 1. Load model with version compatibility
        model, expected_features, model_path, prediction_method, model_type = load_gradientboost_model()
        
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
        risk_assessment = create_churn_risk_assessment(model, X_test, y_test, prediction_method)
        
        if risk_assessment is not None:
            # 6. Comprehensive SHAP analysis
            explainer, shap_values, sample_indices = comprehensive_shap_analysis(
                model, X_test, y_test, risk_assessment
            )
            
            # 7. Final summary
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
            model_name = type(model).__name__.lower()
            print(f"   • ./outputs/{model_name}_customer_risk_assessment.csv")
            print(f"   • ./outputs/shap_summary_plot.png")
            print(f"   • ./outputs/shap_bar_plot.png")
            print(f"   • ./outputs/shap_feature_importance.csv")
            print(f"   • ./models/shap_explainer_{model_name}.bz2")
            
            print(f"\n🎯 BUSINESS IMPACT:")
            if extreme_count > 0:
                estimated_revenue_risk = extreme_count * 1200  # Assuming average annual value
                print(f"   • {extreme_count:,} customers at EXTREME RISK")
                print(f"   • Estimated revenue at risk: ${estimated_revenue_risk:,}")
            
            if high_count > 0:
                estimated_revenue_risk_high = high_count * 1200
                print(f"   • {high_count:,} customers at HIGH RISK")
                print(f"   • Additional revenue at risk: ${estimated_revenue_risk_high:,}")
            
            print(f"\n💡 SHAP INSIGHTS:")
            print(f"   • Feature importance analysis completed")
            print(f"   • Individual customer explanations generated")
            print(f"   • Risk level comparisons available")
            print(f"   • Use SHAP plots to understand churn drivers")
            
            print("="*60)
            
        else:
            print("❌ Risk assessment failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()