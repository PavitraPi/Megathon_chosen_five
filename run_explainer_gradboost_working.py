#!/usr/bin/env python3
"""
Gradient Boosting Churn Risk Explainer with SHAP Analysis - Working Version
Uses the trained GradientBoostingClassifier model with synthetic data
Compatible with scikit-learn 1.2.2
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
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Explainability libraries
import shap

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_style('darkgrid')

def load_model_artifacts():
    """Load model artifacts to understand feature structure"""
    print("📋 Loading model artifacts...")
    
    try:
        artifacts = joblib.load('./models/model_artifacts.pkl')
        feature_names = artifacts['feature_names']
        label_encoders = artifacts.get('label_encoders', {})
        optimal_threshold = artifacts.get('optimal_threshold', 0.5)
        
        print(f"✅ Model expects {len(feature_names)} features")
        print(f"✅ Optimal threshold: {optimal_threshold:.3f}")
        print(f"✅ Label encoders available for: {list(label_encoders.keys())}")
        
        return feature_names, label_encoders, optimal_threshold
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        # Fallback to manual feature list
        print("🔄 Using fallback feature list...")
        feature_names = [
            'curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 'longitude', 
            'city', 'county', 'income', 'has_children', 'marital_status', 
            'home_market_value', 'home_owner', 'college_degree', 'good_credit'
        ]
        return feature_names, {}, 0.5

def create_synthetic_data(feature_names, n_samples=5000):
    """Create realistic synthetic insurance data for demonstration"""
    print(f"🔧 Creating {n_samples} synthetic insurance customer records...")
    
    np.random.seed(42)  # For reproducibility
    data = {}
    
    # Define realistic ranges for insurance customers
    data['curr_ann_amt'] = np.random.normal(1200, 400, n_samples).clip(300, 5000)
    data['days_tenure'] = np.random.exponential(800, n_samples).clip(30, 3650)
    data['age_in_years'] = np.random.normal(45, 15, n_samples).clip(18, 85)
    data['latitude'] = np.random.uniform(25.0, 48.0, n_samples)  # US latitude range
    data['longitude'] = np.random.uniform(-125.0, -70.0, n_samples)  # US longitude range
    data['income'] = np.random.lognormal(10.5, 0.8, n_samples).clip(20000, 300000)
    
    # Binary features
    data['has_children'] = np.random.binomial(1, 0.6, n_samples)
    data['home_owner'] = np.random.binomial(1, 0.7, n_samples)
    data['college_degree'] = np.random.binomial(1, 0.4, n_samples)
    data['good_credit'] = np.random.binomial(1, 0.8, n_samples)
    
    # Categorical features (encoded as numbers)
    data['city'] = np.random.randint(0, 100, n_samples)  # 100 different cities
    data['county'] = np.random.randint(0, 50, n_samples)  # 50 different counties
    data['marital_status'] = np.random.randint(0, 4, n_samples)  # 4 categories
    data['home_market_value'] = np.random.randint(0, 6, n_samples)  # 6 price ranges
    
    # Add any missing features from the model
    for feature in feature_names:
        if feature not in data:
            # Create random numeric feature
            data[feature] = np.random.normal(0, 1, n_samples)
    
    # Convert to DataFrame and ensure correct order
    df = pd.DataFrame(data)
    df = df[feature_names]  # Ensure correct order
    
    # Create realistic churn labels (some customers more likely to churn)
    # Lower tenure, higher amounts, older age slightly increase churn probability
    churn_prob = (
        0.1 +  # Base probability
        0.2 * (data['curr_ann_amt'] > 2000).astype(float) +  # High premium
        0.3 * (data['days_tenure'] < 365).astype(float) +   # New customers
        0.2 * (data['age_in_years'] > 65).astype(float) +   # Senior customers
        0.1 * (1 - data['good_credit']) +                   # Poor credit
        0.1 * np.random.random(n_samples)                   # Random factor
    ).clip(0, 0.9)
    
    y = np.random.binomial(1, churn_prob, n_samples)
    
    print(f"✅ Synthetic data created:")
    print(f"   Features: {df.shape[1]}")
    print(f"   Samples: {df.shape[0]}")
    print(f"   Churn rate: {y.mean():.1%}")
    print(f"   Feature order: {df.columns.tolist()[:5]}...")
    
    return df, y

def load_gradientboost_model():
    """Load the trained GradientBoostingClassifier model"""
    print("🤖 Loading GradientBoostingClassifier model...")
    
    # Try different model files
    model_files = [
        './models/GradientBoostingClassifier_churn_prediction_model.pkl',
        './churn_gradboost_optimized.pkl',
        './model_artifacts_gradboost.pkl'
    ]
    
    for filepath in model_files:
        if os.path.exists(filepath):
            print(f"🔍 Trying model file: {filepath}")
            try:
                model = joblib.load(filepath)
                
                # Check if it's the actual model or contains the model
                if hasattr(model, 'predict'):
                    print(f"✅ Loaded model directly: {type(model).__name__}")
                elif isinstance(model, dict) and 'model' in model:
                    model = model['model']
                    print(f"✅ Extracted model from dict: {type(model).__name__}")
                else:
                    print(f"❌ File doesn't contain a usable model: {type(model)}")
                    continue
                
                # Test the model
                test_data = np.random.random((1, 14))  # 14 features as fallback
                try:
                    pred = model.predict(test_data)
                    pred_proba = model.predict_proba(test_data)
                    print(f"✅ Model test successful - can predict")
                    return model
                except Exception as test_error:
                    print(f"❌ Model test failed: {test_error}")
                    continue
                    
            except Exception as e:
                print(f"❌ Error loading {filepath}: {e}")
                continue
    
    raise FileNotFoundError("No compatible GradientBoostingClassifier model found")

def create_churn_risk_assessment(model, X, y):
    """Create comprehensive churn risk assessment"""
    print("\n🎯 CHURN RISK ASSESSMENT")
    print("=" * 50)
    
    try:
        # Make predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]  # Churn probabilities
        
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
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            count = risk_counts.get(risk_level, 0)
            pct = count/total_customers*100
            emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
            print(f"   {emoji} {risk_level:8} RISK customers: {count:6d} ({pct:5.1f}%)")
        
        # Show examples from each risk level
        print(f"\n📋 SAMPLE CUSTOMERS BY RISK LEVEL:")
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            
            if len(risk_customers) > 0:
                samples = risk_customers.head(3)
                emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS (showing top 3):")
                
                for idx, (_, customer) in enumerate(samples.iterrows(), 1):
                    pred_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                    actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                    accuracy = "✅" if customer['predicted_churn'] == customer['actual_churn'] else "❌"
                    
                    print(f"   {idx}. Customer {customer['customer_id']:6d}: {customer['churn_probability']:.3f} | "
                          f"Predicted: {pred_label:12s} | Actual: {actual_label:8s} {accuracy}")
        
        # Model performance metrics
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
        output_filename = './outputs/gradientboost_customer_risk_assessment.csv'
        risk_assessment.to_csv(output_filename, index=False)
        print(f"\n💾 Risk assessment saved to: {output_filename}")
        
        return risk_assessment
        
    except Exception as e:
        print(f"❌ Error in risk assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def comprehensive_shap_analysis(model, X, feature_names, risk_assessment):
    """Comprehensive SHAP analysis using TreeExplainer"""
    print("\n🎯 COMPREHENSIVE SHAP ANALYSIS")
    print("=" * 60)
    
    try:
        # Initialize SHAP TreeExplainer
        print(f"🔍 Creating SHAP TreeExplainer for {type(model).__name__}...")
        explainer = shap.TreeExplainer(model)
        
        # Use a manageable sample for SHAP computation
        sample_size = min(1000, len(X))
        print(f"📊 Computing SHAP values for {sample_size} samples...")
        
        # Select diverse sample including different risk levels
        sample_indices = []
        
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            n_samples = min(len(risk_customers), sample_size // 4)
            if n_samples > 0:
                selected = risk_customers.sample(n=n_samples, random_state=42)['customer_id'].tolist()
                sample_indices.extend(selected)
        
        # Fill remaining slots randomly if needed
        if len(sample_indices) < sample_size:
            remaining = sample_size - len(sample_indices)
            available_indices = [i for i in range(len(X)) if i not in sample_indices]
            if available_indices:
                additional = np.random.choice(available_indices, 
                                           size=min(remaining, len(available_indices)), 
                                           replace=False)
                sample_indices.extend(additional.tolist())
        
        # Limit to sample_size
        sample_indices = sample_indices[:sample_size]
        X_sample = X.iloc[sample_indices]
        
        print(f"✅ Selected {len(sample_indices)} diverse customers for SHAP analysis")
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Handle binary classification output
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_positive = shap_values[1]  # Churn class
            expected_value = explainer.expected_value[1]
            print("✅ Using positive class SHAP values (churn prediction)")
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
            print("✅ Using SHAP values for churn prediction")
        
        print(f"✅ SHAP values computed: {shap_values_positive.shape}")
        
        # 1. SHAP Summary Plot
        print("\n📊 Creating SHAP Summary Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_positive, X_sample, 
                         feature_names=feature_names, show=False, max_display=15)
        plt.title('SHAP Feature Importance Summary\n(Impact on Churn Prediction)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./outputs/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Summary plot saved")
        
        # 2. SHAP Bar Plot (mean absolute impact)
        print("\n📊 Creating SHAP Bar Plot...")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_positive, X_sample, 
                         feature_names=feature_names, plot_type="bar", 
                         show=False, max_display=15)
        plt.title('SHAP Feature Importance (Bar Plot)\n(Mean Absolute SHAP Values)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('./outputs/shap_bar_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Bar plot saved")
        
        # 3. Individual Customer Analysis
        print("\n🔍 Creating Individual Customer SHAP Explanations...")
        
        # Select one representative customer from each risk level
        for risk_level in ['EXTREME', 'HIGH', 'MEDIUM', 'LOW']:
            risk_customers_in_sample = []
            for idx, customer_id in enumerate(sample_indices):
                if risk_assessment.iloc[customer_id]['risk_level'] == risk_level:
                    risk_customers_in_sample.append((customer_id, idx))
            
            if risk_customers_in_sample:
                customer_id, sample_idx = risk_customers_in_sample[0]  # Take first one
                
                print(f"\n📊 Creating SHAP explanation for {risk_level} risk customer {customer_id}...")
                
                try:
                    customer_data = X_sample.iloc[sample_idx]
                    customer_shap = shap_values_positive[sample_idx]
                    customer_prob = risk_assessment.iloc[customer_id]['churn_probability']
                    
                    # Create SHAP explanation object
                    shap_explanation = shap.Explanation(
                        values=customer_shap,
                        base_values=expected_value,
                        data=customer_data.values,
                        feature_names=feature_names
                    )
                    
                    emoji = "🔴" if risk_level == 'EXTREME' else "🟠" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
                    
                    # Try waterfall plot first
                    plt.figure(figsize=(12, 10))
                    try:
                        shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                        plt.title(f'{emoji} {risk_level} Risk Customer {customer_id}\n'
                                 f'Churn Probability: {customer_prob:.3f}', 
                                 fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(f'./outputs/shap_waterfall_{risk_level.lower()}_risk_customer_{customer_id}.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✅ Waterfall plot saved for {risk_level} risk customer")
                    except Exception as waterfall_error:
                        print(f"⚠ Waterfall plot failed, creating force plot instead: {waterfall_error}")
                        
                        # Alternative: Force plot
                        plt.figure(figsize=(12, 8))
                        shap.plots.force(expected_value, customer_shap, customer_data, 
                                       feature_names=feature_names, matplotlib=True, show=False)
                        plt.title(f'{emoji} {risk_level} Risk Customer {customer_id} - SHAP Force Plot\n'
                                 f'Churn Probability: {customer_prob:.3f}', 
                                 fontsize=14, fontweight='bold')
                        plt.tight_layout()
                        plt.savefig(f'./outputs/shap_force_{risk_level.lower()}_risk_customer_{customer_id}.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"✅ Force plot saved for {risk_level} risk customer")
                    
                    # Print text interpretation
                    feature_contributions = pd.DataFrame({
                        'feature': feature_names,
                        'shap_value': customer_shap,
                        'feature_value': customer_data.values
                    })
                    
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
            'feature': feature_names,
            'mean_abs_shap': np.mean(np.abs(shap_values_positive), axis=0),
            'mean_shap': np.mean(shap_values_positive, axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("📈 Top 15 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows(), 1):
            direction = "→ Increases Churn" if row['mean_shap'] > 0 else "→ Reduces Churn"
            print(f"   {i:2d}. {row['feature']:<25}: {row['mean_abs_shap']:.4f} {direction}")
        
        # Save feature importance
        feature_importance.to_csv('./outputs/shap_feature_importance.csv', index=False)
        print(f"\n💾 Feature importance saved to: ./outputs/shap_feature_importance.csv")
        
        # Save SHAP artifacts
        explainer_filename = './models/shap_explainer_gradientboost.bz2'
        values_filename = './models/shap_values_gradientboost_sample.npy'
        
        joblib.dump(explainer, explainer_filename, compress=('bz2', 9))
        np.save(values_filename, shap_values_positive)
        
        print(f"\n💾 SHAP artifacts saved:")
        print(f"   • {explainer_filename}")
        print(f"   • {values_filename}")
        
        return explainer, shap_values_positive, sample_indices
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def main():
    """Main function to run the complete analysis"""
    print("🚀 GRADIENT BOOSTING CHURN RISK EXPLAINER")
    print("=" * 60)
    print("Scikit-learn compatible version with synthetic data")
    print("=" * 60)
    
    try:
        # 1. Load model artifacts and create feature structure
        feature_names, label_encoders, optimal_threshold = load_model_artifacts()
        
        # 2. Load the trained model
        model = load_gradientboost_model()
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # 3. Create synthetic data matching the model's expected features
        X, y = create_synthetic_data(feature_names, n_samples=5000)
        
        # 4. Split data for testing
        print(f"\n🔄 Creating train-test split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 5. Create risk assessment on test set
        risk_assessment = create_churn_risk_assessment(model, X_test, y_test)
        
        if risk_assessment is not None:
            # 6. Comprehensive SHAP analysis
            explainer, shap_values, sample_indices = comprehensive_shap_analysis(
                model, X_test, feature_names, risk_assessment
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
            print(f"   • ./outputs/gradientboost_customer_risk_assessment.csv")
            print(f"   • ./outputs/shap_summary_plot.png")
            print(f"   • ./outputs/shap_bar_plot.png")
            print(f"   • ./outputs/shap_feature_importance.csv")
            print(f"   • ./models/shap_explainer_gradientboost.bz2")
            
            print(f"\n💡 NEXT STEPS:")
            print(f"   • Use SHAP plots to understand churn drivers")
            print(f"   • Focus retention efforts on EXTREME/HIGH risk customers")
            print(f"   • Analyze top SHAP features for business insights")
            print(f"   • Integrate with Streamlit dashboard for interactive analysis")
            
            print("="*60)
            
        else:
            print("❌ Risk assessment failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
