"""
Churn Risk Explainer for XGBoost Model
Provides high risk and low risk churn probability assessments for data points
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

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Explainability libraries
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')
plt.style.use('default')


def preprocess_insurance_data(df):
    """
    Preprocess auto insurance data for XGBoost model
    Based on the structure from your hackathon.py
    """
    print("🔄 Preprocessing insurance data...")
    
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Drop ID columns if they exist
    id_cols = ['individual_id', 'address_id']
    for col in id_cols:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)
            print(f"   Dropped ID column: {col}")
    
    # Drop target leakage columns
    if 'acct_suspd_date' in data.columns:
        data.drop(columns=['acct_suspd_date'], inplace=True)
        print("   Dropped 'acct_suspd_date' to prevent target leakage")
    
    # Handle missing values
    # Numeric: median imputation
    numerical_cols = data.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"   Filled missing values in '{col}' with median: {median_val}")
    
    # Categorical: mode imputation
    categorical_cols = data.select_dtypes(include='object').columns
    categorical_cols = [col for col in categorical_cols if col != 'Churn']  # Exclude target
    
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode_val = data[col].mode()[0]
            data[col].fillna(mode_val, inplace=True)
            print(f"   Filled missing values in '{col}' with mode: {mode_val}")
    
    print(f"✅ Preprocessing completed. Dataset shape: {data.shape}")
    return data


def load_xgboost_model():
    """Load the trained XGBoost model"""
    print("🤖 Loading XGBoost model...")
    
    # Look for XGBoost model files in different locations
    model_files = [
        './models/XGBoostClassifier_churn_prediction_model.pkl',
        './XGBoostClassifier_churn_prediction_model.pkl',
        './models/churn_xgboost_optimized.pkl',
        './churn_xgboost_optimized.pkl',
        './models/model_xgb.pkl'
    ]
    
    model = None
    model_path = None
    
    for filepath in model_files:
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                model_path = filepath
                print(f"✅ Loaded XGBoost model from: {filepath}")
                break
            except Exception as e:
                print(f"❌ Error loading model from {filepath}: {str(e)}")
                continue
    
    if model is None:
        raise FileNotFoundError("XGBoost model not found. Please ensure the .pkl file is available.")
    
    return model, model_path


def prepare_data_for_model(data_path='./data/autoinsurance_churn.csv'):
    """Load and prepare data for the XGBoost model using the exact same preprocessing as training"""
    print("📊 Loading and preparing data...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape}")
    
    # Apply the exact same preprocessing as in hackathon.py
    print("🔄 Applying same preprocessing as training...")
    
    # Drop ID columns
    id_cols = ['individual_id', 'address_id']
    for col in id_cols:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Drop target leakage columns
    if 'acct_suspd_date' in df.columns:
        df.drop(columns=['acct_suspd_date'], inplace=True)
    
    # Handle datetime columns the same way as training
    date_cols = ['cust_orig_date', 'date_of_birth']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Define target and features
    TARGET = 'Churn'
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset.")
    
    X = df.drop(columns=[TARGET], errors='ignore')
    y = df[TARGET]
    
    # Handle missing values exactly as in training
    numerical_cols = X.select_dtypes(include=np.number).columns
    for col in numerical_cols:
        if X[col].isnull().sum() > 0:
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
    
    categorical_cols = X.select_dtypes(include='object').columns
    for col in categorical_cols:
        if X[col].isnull().sum() > 0:
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)
    
    # Handle datetime columns the same way as training
    datetime_cols = X.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
    
    # Convert datetime columns to numeric (days since earliest date)
    for col in datetime_cols:
        median_date = X[col].dropna().median()
        X[col] = X[col].fillna(median_date)
        X[f'{col}_days_since'] = (X[col] - median_date).dt.days
        X.drop(columns=[col], inplace=True)
    
    # Encode categorical columns
    categorical_cols = X.select_dtypes(include='object').columns
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Ensure all columns are numeric
    for col in X.columns:
        if np.issubdtype(X[col].dtype, np.datetime64):
            X[col] = X[col].view('int64')
    
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Scale numerical columns
    numerical_cols = X.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    
    print(f"✅ Data prepared: {X.shape} features, {len(y)} samples")
    print(f"   Feature columns: {list(X.columns)}")
    print(f"   Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, label_encoders, scaler


def create_churn_risk_assessment(model, X, y=None):
    """
    Create comprehensive churn risk assessment with high/medium/low risk categorization
    """
    print("\n🎯 CHURN RISK ASSESSMENT")
    print("=" * 50)
    
    # Get model predictions
    try:
        y_pred_proba = model.predict_proba(X)[:, 1]  # Churn probabilities
        y_pred = model.predict(X)
    except Exception as e:
        print(f"❌ Error making predictions: {str(e)}")
        return None
    
    # Create risk assessment DataFrame
    risk_assessment = pd.DataFrame({
        'customer_id': range(len(X)),
        'churn_probability': y_pred_proba,
        'predicted_churn': y_pred
    })
    
    # Add actual churn if available
    if y is not None:
        risk_assessment['actual_churn'] = y.values
    
    # Define risk categories based on churn probability
    def categorize_risk(prob):
        if prob >= 0.7:
            return "HIGH"
        elif prob >= 0.4:
            return "MEDIUM" 
        else:
            return "LOW"
    
    risk_assessment['risk_level'] = risk_assessment['churn_probability'].apply(categorize_risk)
    
    # Risk distribution analysis
    risk_counts = risk_assessment['risk_level'].value_counts()
    total_customers = len(risk_assessment)
    
    print(f"\n📊 CUSTOMER RISK DISTRIBUTION:")
    print(f"   🔴 HIGH RISK customers (≥70%):    {risk_counts.get('HIGH', 0):4d} ({risk_counts.get('HIGH', 0)/total_customers*100:5.1f}%)")
    print(f"   🟡 MEDIUM RISK customers (40-70%): {risk_counts.get('MEDIUM', 0):4d} ({risk_counts.get('MEDIUM', 0)/total_customers*100:5.1f}%)")
    print(f"   🟢 LOW RISK customers (<40%):      {risk_counts.get('LOW', 0):4d} ({risk_counts.get('LOW', 0)/total_customers*100:5.1f}%)")
    
    # Show detailed examples from each risk category
    print(f"\n📋 DETAILED RISK ANALYSIS:")
    
    for risk_level, emoji in [('HIGH', '🔴'), ('MEDIUM', '🟡'), ('LOW', '🟢')]:
        risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
        
        if len(risk_customers) > 0:
            print(f"\n{emoji} {risk_level} RISK CUSTOMERS:")
            
            # Sort by probability (highest first for HIGH, lowest first for LOW)
            if risk_level == 'HIGH':
                sample_customers = risk_customers.nlargest(5, 'churn_probability')
            elif risk_level == 'LOW':
                sample_customers = risk_customers.nsmallest(5, 'churn_probability')
            else:
                sample_customers = risk_customers.head(5)
            
            for idx, (_, customer) in enumerate(sample_customers.iterrows(), 1):
                pred_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                
                if y is not None:
                    actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                    accuracy_indicator = "✅" if customer['predicted_churn'] == customer['actual_churn'] else "❌"
                    
                    print(f"   {idx}. Customer {customer['customer_id']:4d}: {customer['churn_probability']:.3f} probability "
                          f"| Predicted: {pred_label} | Actual: {actual_label} {accuracy_indicator}")
                else:
                    print(f"   {idx}. Customer {customer['customer_id']:4d}: {customer['churn_probability']:.3f} probability "
                          f"| Predicted: {pred_label}")
    
    # Statistical summary
    print(f"\n📈 STATISTICAL SUMMARY:")
    print(f"   Average churn probability: {np.mean(y_pred_proba):.3f}")
    print(f"   Median churn probability:  {np.median(y_pred_proba):.3f}")
    print(f"   Std deviation:             {np.std(y_pred_proba):.3f}")
    print(f"   Min probability:           {np.min(y_pred_proba):.3f}")
    print(f"   Max probability:           {np.max(y_pred_proba):.3f}")
    
    # Model performance if actual labels available
    if y is not None:
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_pred_proba)
        
        print(f"\n🎯 MODEL PERFORMANCE:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   F1-Score:  {f1:.4f}")
        print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    # Save risk assessment
    os.makedirs('./outputs', exist_ok=True)
    risk_assessment.to_csv('./outputs/customer_risk_assessment.csv', index=False)
    print(f"\n💾 Risk assessment saved to: ./outputs/customer_risk_assessment.csv")
    
    # Create visualizations
    create_risk_visualizations(risk_assessment, y_pred_proba)
    
    return risk_assessment


def create_risk_visualizations(risk_assessment, y_pred_proba):
    """Create visualizations for risk assessment"""
    print("\n📊 Creating risk visualizations...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Risk level distribution (pie chart)
    risk_counts = risk_assessment['risk_level'].value_counts()
    colors = ['red', 'green', 'orange']  # HIGH, LOW, MEDIUM
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Customer Risk Distribution')
    
    # 2. Probability distribution histogram
    axes[0, 1].hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Medium Risk Threshold')
    axes[0, 1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
    axes[0, 1].set_xlabel('Churn Probability')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Churn Probability Distribution')
    axes[0, 1].legend()
    
    # 3. Risk level bar chart
    risk_counts_ordered = risk_assessment['risk_level'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'])
    colors_ordered = ['green', 'orange', 'red']
    bars = axes[1, 0].bar(risk_counts_ordered.index, risk_counts_ordered.values, color=colors_ordered)
    axes[1, 0].set_title('Customer Count by Risk Level')
    axes[1, 0].set_ylabel('Number of Customers')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
    
    # 4. Box plot of probabilities by risk level
    risk_levels = ['LOW', 'MEDIUM', 'HIGH']
    prob_by_risk = [risk_assessment[risk_assessment['risk_level'] == level]['churn_probability'] 
                   for level in risk_levels]
    
    box_plot = axes[1, 1].boxplot(prob_by_risk, labels=risk_levels, patch_artist=True)
    colors_box = ['lightgreen', 'orange', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes[1, 1].set_title('Churn Probability by Risk Level')
    axes[1, 1].set_ylabel('Churn Probability')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('./outputs', exist_ok=True)
    plt.savefig('./outputs/churn_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved to: ./outputs/churn_risk_analysis.png")


def explain_individual_predictions(model, X, y=None, sample_indices=[0, 1, 2]):
    """
    Provide detailed explanations for individual customer predictions using SHAP
    """
    print("\n🔍 INDIVIDUAL CUSTOMER EXPLANATIONS")
    print("=" * 50)
    
    try:
        # Create SHAP explainer
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for selected samples
        X_sample = X.iloc[sample_indices]
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification SHAP values
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]  # Churn class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        
        print(f"✅ SHAP values calculated for {len(sample_indices)} customers")
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_sample)[:, 1]
        y_pred = model.predict(X_sample)
        
        # Analyze each customer
        for i, idx in enumerate(sample_indices):
            if idx >= len(X):
                continue
                
            customer_data = X.iloc[idx]
            churn_prob = y_pred_proba[i]
            prediction = y_pred[i]
            
            # Determine risk level
            if churn_prob >= 0.7:
                risk_level = "🔴 HIGH RISK"
                risk_color = 'red'
            elif churn_prob >= 0.4:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = 'orange'
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = 'green'
            
            print(f"\n{'='*30} CUSTOMER {idx} {'='*30}")
            print(f"Churn Probability: {churn_prob:.3f}")
            print(f"Prediction: {'Will Churn' if prediction == 1 else 'Will Retain'}")
            print(f"Risk Level: {risk_level}")
            
            if y is not None:
                actual = y.iloc[idx]
                print(f"Actual Outcome: {'Churned' if actual == 1 else 'Retained'}")
                accuracy = "✅ Correct" if prediction == actual else "❌ Incorrect"
                print(f"Prediction Accuracy: {accuracy}")
            
            # Get SHAP values for this customer
            customer_shap = shap_values_positive[i]
            
            # Create feature contribution analysis
            feature_contributions = pd.DataFrame({
                'feature': X.columns,
                'shap_value': customer_shap,
                'feature_value': customer_data.values
            })
            feature_contributions['abs_shap'] = abs(feature_contributions['shap_value'])
            
            # Top features contributing to churn risk
            top_risk_features = feature_contributions[feature_contributions['shap_value'] > 0].nlargest(5, 'shap_value')
            top_protective_features = feature_contributions[feature_contributions['shap_value'] < 0].nsmallest(5, 'shap_value')
            
            print(f"\n📈 TOP FACTORS INCREASING CHURN RISK:")
            if len(top_risk_features) > 0:
                for _, row in top_risk_features.iterrows():
                    impact = "HIGH" if abs(row['shap_value']) > 0.2 else "MEDIUM" if abs(row['shap_value']) > 0.1 else "LOW"
                    print(f"   • {row['feature']}: +{row['shap_value']:.4f} [{impact} impact] (value: {row['feature_value']:.3f})")
            else:
                print("   • No significant factors increasing churn risk")
            
            print(f"\n📉 TOP FACTORS REDUCING CHURN RISK:")
            if len(top_protective_features) > 0:
                for _, row in top_protective_features.iterrows():
                    impact = "HIGH" if abs(row['shap_value']) > 0.2 else "MEDIUM" if abs(row['shap_value']) > 0.1 else "LOW"
                    print(f"   • {row['feature']}: {row['shap_value']:.4f} [{impact} impact] (value: {row['feature_value']:.3f})")
            else:
                print("   • No significant protective factors found")
            
            # Create waterfall plot
            try:
                shap_explanation = shap.Explanation(
                    values=customer_shap,
                    base_values=expected_value,
                    data=customer_data,
                    feature_names=X.columns.tolist()
                )
                
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                plt.title(f'SHAP Explanation - Customer {idx}\n'
                         f'Risk: {risk_level}, Churn Probability: {churn_prob:.3f}', 
                         fontsize=14)
                plt.tight_layout()
                
                # Save plot
                os.makedirs('./outputs', exist_ok=True)
                plt.savefig(f'./outputs/shap_explanation_customer_{idx}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"   ⚠️ Could not create SHAP plot for customer {idx}: {str(e)}")
        
        # Save SHAP explainer for future use
        os.makedirs('./models', exist_ok=True)
        joblib.dump(explainer, './models/explainer_xgb.bz2', compress=('bz2', 9))
        print(f"\n💾 SHAP explainer saved to: ./models/explainer_xgb.bz2")
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main function to run XGBoost churn risk assessment and explanation
    """
    print("🚀 XGBOOST CHURN RISK EXPLAINER")
    print("=" * 60)
    print("Analyzing customer churn risk with high/medium/low categorization")
    print("=" * 60)
    
    try:
        # Step 1: Load XGBoost model
        model, model_path = load_xgboost_model()
        print(f"Model type: {type(model).__name__}")
        
        # Step 2: Load and prepare data
        X, y, label_encoders, scaler = prepare_data_for_model()
        
        # Step 3: Create comprehensive risk assessment
        risk_assessment = create_churn_risk_assessment(model, X, y)
        
        # Step 4: Explain individual predictions for sample customers
        print("\n" + "="*60)
        print("INDIVIDUAL CUSTOMER ANALYSIS")
        print("="*60)
        
        # Select diverse samples for explanation
        high_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'HIGH'].head(2)['customer_id'].tolist()
        medium_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'MEDIUM'].head(1)['customer_id'].tolist()
        low_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'LOW'].head(2)['customer_id'].tolist()
        
        sample_customers = high_risk_customers + medium_risk_customers + low_risk_customers
        sample_customers = sample_customers[:5]  # Limit to 5 for detailed analysis
        
        if sample_customers:
            explain_individual_predictions(model, X, y, sample_customers)
        
        # Final summary
        print("\n" + "="*60)
        print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("📁 Files created:")
        print("   • ./outputs/customer_risk_assessment.csv - Complete risk assessment")
        print("   • ./outputs/churn_risk_analysis.png - Risk visualization")
        print("   • ./outputs/shap_explanation_customer_*.png - Individual explanations")
        print("   • ./models/explainer_xgb.bz2 - SHAP explainer for future use")
        print("\n🎯 Key insights:")
        print("   • High risk customers (≥70% churn probability) need immediate attention")
        print("   • Medium risk customers (40-70%) are good candidates for retention campaigns")
        print("   • Low risk customers (<40%) are likely to stay but monitor for changes")
        print("   • Use SHAP explanations to understand what drives each customer's risk")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
