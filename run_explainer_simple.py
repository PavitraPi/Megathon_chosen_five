"""
Simple XGBoost Churn Risk Explainer
Works with existing XGBoostClassifier_churn_prediction_model.pkl
Provides high risk and low risk churn probability assessments
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

warnings.filterwarnings('ignore')
plt.style.use('default')


def load_xgboost_model():
    """Load the trained XGBoost model"""
    print("🤖 Loading XGBoost model...")
    
    model_files = [
        './models/XGBoostClassifier_churn_prediction_model.pkl',
        './XGBoostClassifier_churn_prediction_model.pkl',
        './models/churn_xgboost_optimized.pkl',
        './churn_xgboost_optimized.pkl'
    ]
    
    for filepath in model_files:
        if os.path.exists(filepath):
            try:
                model = joblib.load(filepath)
                print(f"✅ Loaded XGBoost model from: {filepath}")
                return model, filepath
            except Exception as e:
                print(f"❌ Error loading model from {filepath}: {str(e)}")
                continue
    
    raise FileNotFoundError("XGBoost model not found.")


def create_synthetic_data_for_demo():
    """Create synthetic data matching the model's expected features for demonstration"""
    print("📊 Creating synthetic data for demonstration...")
    
    # Get model expected features (from the error message)
    expected_features = [
        'curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 'longitude', 
        'income', 'has_children', 'length_of_residence', 'home_owner', 
        'college_degree', 'good_credit', 'cust_orig_days_since', 'age'
    ]
    
    # City features (sample)
    city_features = [
        'city_Dallas', 'city_Houston', 'city_Austin', 'city_Fort Worth', 'city_Plano',
        'city_Irving', 'city_Garland', 'city_Arlington', 'city_Frisco', 'city_Mckinney'
    ]
    
    # County features (sample)  
    county_features = [
        'county_Dallas', 'county_Tarrant', 'county_Harris', 'county_Travis', 'county_Collin'
    ]
    
    # Home market value features
    home_value_features = [
        'home_market_value_75000 - 99999', 'home_market_value_100000 - 124999',
        'home_market_value_125000 - 149999', 'home_market_value_150000 - 174999',
        'home_market_value_200000 - 224999', 'home_market_value_250000 - 274999'
    ]
    
    # Marital status features
    marital_features = ['marital_status_Single']
    
    # Combine all features
    all_features = (expected_features + city_features + county_features + 
                   home_value_features + marital_features)
    
    # Create synthetic data (100 samples for demo)
    np.random.seed(42)
    n_samples = 100
    
    data = {}
    
    # Numerical features
    data['curr_ann_amt'] = np.random.normal(1200, 300, n_samples)
    data['days_tenure'] = np.random.randint(30, 2000, n_samples)
    data['age_in_years'] = np.random.randint(25, 75, n_samples)
    data['latitude'] = np.random.uniform(32.5, 33.0, n_samples)
    data['longitude'] = np.random.uniform(-97.5, -96.5, n_samples)
    data['income'] = np.random.normal(65000, 25000, n_samples)
    data['has_children'] = np.random.binomial(1, 0.4, n_samples)
    data['length_of_residence'] = np.random.randint(1, 20, n_samples)
    data['home_owner'] = np.random.binomial(1, 0.7, n_samples)
    data['college_degree'] = np.random.binomial(1, 0.6, n_samples)
    data['good_credit'] = np.random.binomial(1, 0.8, n_samples)
    data['cust_orig_days_since'] = np.random.randint(-1000, 0, n_samples)
    data['age'] = data['age_in_years']  # Duplicate for compatibility
    
    # One-hot encoded categorical features (binary)
    for feature in city_features + county_features + home_value_features + marital_features:
        data[feature] = np.random.binomial(1, 0.1, n_samples)  # 10% chance of being 1
    
    # Ensure at least one city and county is selected for each sample
    for i in range(n_samples):
        # Select one random city
        city_idx = np.random.randint(0, len(city_features))
        data[city_features[city_idx]][i] = 1
        
        # Select one random county
        county_idx = np.random.randint(0, len(county_features))
        data[county_features[county_idx]][i] = 1
        
        # Select one random home value
        home_idx = np.random.randint(0, len(home_value_features))
        data[home_value_features[home_idx]][i] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Higher income, education -> lower churn risk
    # Longer tenure -> lower churn risk
    churn_logits = (
        -2.0 +  # Base low churn probability
        -0.0001 * df['income'] +  # Higher income -> lower churn
        -0.0005 * df['days_tenure'] +  # Longer tenure -> lower churn
        -0.5 * df['college_degree'] +  # Education -> lower churn
        -0.3 * df['home_owner'] +  # Home ownership -> lower churn
        0.3 * df['marital_status_Single'] +  # Single -> higher churn
        np.random.normal(0, 0.5, n_samples)  # Random noise
    )
    
    # Convert to probabilities
    churn_probabilities = 1 / (1 + np.exp(-churn_logits))
    actual_churn = np.random.binomial(1, churn_probabilities, n_samples)
    
    print(f"✅ Created synthetic data: {df.shape}")
    print(f"   Churn rate: {np.mean(actual_churn):.1%}")
    print(f"   Features: {len(df.columns)}")
    
    return df, actual_churn


def create_churn_risk_assessment(model, X, y=None):
    """Create comprehensive churn risk assessment"""
    print("\n🎯 CHURN RISK ASSESSMENT")
    print("=" * 50)
    
    try:
        # Get model predictions
        y_pred_proba = model.predict_proba(X)[:, 1]  # Churn probabilities
        y_pred = model.predict(X)
        
        # Create risk assessment DataFrame
        risk_assessment = pd.DataFrame({
            'customer_id': range(len(X)),
            'churn_probability': y_pred_proba,
            'predicted_churn': y_pred
        })
        
        if y is not None:
            risk_assessment['actual_churn'] = y
        
        # Risk categorization
        def categorize_risk(prob):
            if prob >= 0.7:
                return "HIGH"
            elif prob >= 0.4:
                return "MEDIUM" 
            else:
                return "LOW"
        
        risk_assessment['risk_level'] = risk_assessment['churn_probability'].apply(categorize_risk)
        
        # Risk distribution
        risk_counts = risk_assessment['risk_level'].value_counts()
        total_customers = len(risk_assessment)
        
        print(f"\n📊 CUSTOMER RISK DISTRIBUTION:")
        print(f"   🔴 HIGH RISK customers (≥70%):    {risk_counts.get('HIGH', 0):3d} ({risk_counts.get('HIGH', 0)/total_customers*100:5.1f}%)")
        print(f"   🟡 MEDIUM RISK customers (40-70%): {risk_counts.get('MEDIUM', 0):3d} ({risk_counts.get('MEDIUM', 0)/total_customers*100:5.1f}%)")
        print(f"   🟢 LOW RISK customers (<40%):      {risk_counts.get('LOW', 0):3d} ({risk_counts.get('LOW', 0)/total_customers*100:5.1f}%)")
        
        # Show detailed examples
        print(f"\n📋 DETAILED CUSTOMER EXAMPLES:")
        
        for risk_level, emoji in [('HIGH', '🔴'), ('MEDIUM', '🟡'), ('LOW', '🟢')]:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            
            if len(risk_customers) > 0:
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS (showing top 3):")
                
                if risk_level == 'HIGH':
                    samples = risk_customers.nlargest(3, 'churn_probability')
                elif risk_level == 'LOW':
                    samples = risk_customers.nsmallest(3, 'churn_probability')
                else:
                    samples = risk_customers.head(3)
                
                for idx, (_, customer) in enumerate(samples.iterrows(), 1):
                    pred_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                    
                    if y is not None:
                        actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                        accuracy = "✅" if customer['predicted_churn'] == customer['actual_churn'] else "❌"
                        print(f"   {idx}. Customer {customer['customer_id']:2d}: {customer['churn_probability']:.3f} | "
                              f"Predicted: {pred_label} | Actual: {actual_label} {accuracy}")
                    else:
                        print(f"   {idx}. Customer {customer['customer_id']:2d}: {customer['churn_probability']:.3f} | "
                              f"Predicted: {pred_label}")
        
        # Statistics
        print(f"\n📈 PROBABILITY STATISTICS:")
        print(f"   Average: {np.mean(y_pred_proba):.3f}")
        print(f"   Median:  {np.median(y_pred_proba):.3f}")
        print(f"   Std Dev: {np.std(y_pred_proba):.3f}")
        print(f"   Range:   [{np.min(y_pred_proba):.3f}, {np.max(y_pred_proba):.3f}]")
        
        # Model performance if ground truth available
        if y is not None:
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y, y_pred_proba)
            
            print(f"\n🎯 MODEL PERFORMANCE:")
            print(f"   Accuracy:  {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall:    {recall:.3f}")
            print(f"   F1-Score:  {f1:.3f}")
            print(f"   ROC-AUC:   {roc_auc:.3f}")
        
        # Save results
        os.makedirs('./outputs', exist_ok=True)
        risk_assessment.to_csv('./outputs/customer_risk_assessment.csv', index=False)
        print(f"\n💾 Risk assessment saved to: ./outputs/customer_risk_assessment.csv")
        
        # Create visualization
        create_risk_visualization(risk_assessment, y_pred_proba)
        
        return risk_assessment
        
    except Exception as e:
        print(f"❌ Error in risk assessment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_risk_visualization(risk_assessment, y_pred_proba):
    """Create risk visualization"""
    print("\n📊 Creating risk visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Risk distribution pie chart
    risk_counts = risk_assessment['risk_level'].value_counts()
    colors = ['red', 'green', 'orange']  # HIGH, LOW, MEDIUM
    axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Customer Risk Distribution')
    
    # 2. Probability histogram
    axes[0, 1].hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0.4, color='orange', linestyle='--', linewidth=2, label='Medium Risk')
    axes[0, 1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='High Risk')
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
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
    
    # 4. Box plot
    risk_levels = ['LOW', 'MEDIUM', 'HIGH']
    prob_by_risk = []
    for level in risk_levels:
        level_data = risk_assessment[risk_assessment['risk_level'] == level]['churn_probability']
        if len(level_data) > 0:
            prob_by_risk.append(level_data)
        else:
            prob_by_risk.append([0])  # Empty placeholder
    
    box_plot = axes[1, 1].boxplot(prob_by_risk, labels=risk_levels, patch_artist=True)
    colors_box = ['lightgreen', 'orange', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
    
    axes[1, 1].set_title('Churn Probability by Risk Level')
    axes[1, 1].set_ylabel('Churn Probability')
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('./outputs/churn_risk_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization saved to: ./outputs/churn_risk_visualization.png")


def explain_high_risk_customers(model, X, risk_assessment):
    """Provide SHAP explanations for high-risk customers"""
    print("\n🔍 EXPLAINING HIGH-RISK CUSTOMERS")
    print("=" * 50)
    
    try:
        # Get high-risk customers
        high_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'HIGH']
        
        if len(high_risk_customers) == 0:
            print("ℹ️ No high-risk customers found for explanation")
            return
        
        # Select top 3 highest risk customers
        top_high_risk = high_risk_customers.nlargest(3, 'churn_probability')
        sample_indices = top_high_risk['customer_id'].tolist()
        
        print(f"Analyzing top {len(sample_indices)} high-risk customers...")
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Get SHAP values for high-risk customers
        X_samples = X.iloc[sample_indices]
        shap_values = explainer.shap_values(X_samples)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]  # Churn class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        
        # Analyze each high-risk customer
        for i, customer_id in enumerate(sample_indices):
            customer_data = X.iloc[customer_id]
            customer_risk = top_high_risk.iloc[i]
            churn_prob = customer_risk['churn_probability']
            
            print(f"\n{'='*25} HIGH-RISK CUSTOMER {customer_id} {'='*25}")
            print(f"🔴 Churn Probability: {churn_prob:.3f} (HIGH RISK)")
            print(f"📊 Predicted Outcome: {'Will Churn' if customer_risk['predicted_churn'] == 1 else 'Will Retain'}")
            
            # Get SHAP values for this customer
            customer_shap = shap_values_positive[i]
            
            # Feature importance analysis
            feature_contrib = pd.DataFrame({
                'feature': X.columns,
                'shap_value': customer_shap,
                'feature_value': customer_data.values
            })
            feature_contrib['abs_shap'] = abs(feature_contrib['shap_value'])
            
            # Top factors increasing churn risk
            top_risk_factors = feature_contrib[feature_contrib['shap_value'] > 0.01].nlargest(5, 'shap_value')
            top_protective_factors = feature_contrib[feature_contrib['shap_value'] < -0.01].nsmallest(5, 'shap_value')
            
            print(f"\n📈 TOP FACTORS INCREASING CHURN RISK:")
            if len(top_risk_factors) > 0:
                for _, row in top_risk_factors.iterrows():
                    impact = "🔥 CRITICAL" if abs(row['shap_value']) > 0.1 else "⚠️ HIGH" if abs(row['shap_value']) > 0.05 else "📊 MEDIUM"
                    print(f"   • {row['feature']}: +{row['shap_value']:.4f} {impact}")
                    print(f"     └─ Feature value: {row['feature_value']:.3f}")
            else:
                print("   • No significant risk factors identified")
            
            print(f"\n📉 PROTECTIVE FACTORS (reducing churn risk):")
            if len(top_protective_factors) > 0:
                for _, row in top_protective_factors.iterrows():
                    impact = "🛡️ STRONG" if abs(row['shap_value']) > 0.1 else "✅ MODERATE" if abs(row['shap_value']) > 0.05 else "📊 WEAK"
                    print(f"   • {row['feature']}: {row['shap_value']:.4f} {impact}")
                    print(f"     └─ Feature value: {row['feature_value']:.3f}")
            else:
                print("   • No significant protective factors found")
            
            # Actionable insights
            print(f"\n💡 RECOMMENDED ACTIONS FOR CUSTOMER {customer_id}:")
            print("   🎯 PRIORITY: Immediate retention intervention needed")
            print("   📞 Contact customer within 24-48 hours")
            print("   💰 Consider personalized offers or discounts")
            print("   🤝 Assign dedicated customer success manager")
            
            # Create waterfall plot for this customer
            try:
                shap_explanation = shap.Explanation(
                    values=customer_shap,
                    base_values=expected_value,
                    data=customer_data,
                    feature_names=X.columns.tolist()
                )
                
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                plt.title(f'🔴 High-Risk Customer {customer_id} - SHAP Analysis\n'
                         f'Churn Probability: {churn_prob:.3f}', fontsize=14, color='red')
                plt.tight_layout()
                
                # Save plot
                plt.savefig(f'./outputs/high_risk_customer_{customer_id}_explanation.png', 
                           dpi=300, bbox_inches='tight')
                plt.show()
                
            except Exception as e:
                print(f"   ⚠️ Could not create SHAP plot for customer {customer_id}: {str(e)}")
        
        # Save SHAP explainer
        joblib.dump(explainer, './models/explainer_xgb.bz2', compress=('bz2', 9))
        print(f"\n💾 SHAP explainer saved to: ./models/explainer_xgb.bz2")
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("🚀 XGBOOST CHURN RISK EXPLAINER")
    print("=" * 60)
    print("Analyzing customer churn risk with explainable AI")
    print("=" * 60)
    
    try:
        # Load model
        model, model_path = load_xgboost_model()
        print(f"Model type: {type(model).__name__}")
        
        # Create synthetic data for demonstration
        # (In production, replace this with your actual data preparation)
        X, y = create_synthetic_data_for_demo()
        
        print(f"\n🔄 Using synthetic demo data with {len(X)} customers")
        print("ℹ️  In production, replace this with your actual preprocessed data")
        
        # Create risk assessment
        risk_assessment = create_churn_risk_assessment(model, X, y)
        
        if risk_assessment is not None:
            # Explain high-risk customers
            explain_high_risk_customers(model, X, risk_assessment)
            
            # Success summary
            print("\n" + "="*60)
            print("🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            print("📁 Files created:")
            print("   • ./outputs/customer_risk_assessment.csv")
            print("   • ./outputs/churn_risk_visualization.png")
            print("   • ./outputs/high_risk_customer_*_explanation.png")
            print("   • ./models/explainer_xgb.bz2")
            
            print("\n🎯 Key Insights:")
            high_count = len(risk_assessment[risk_assessment['risk_level'] == 'HIGH'])
            medium_count = len(risk_assessment[risk_assessment['risk_level'] == 'MEDIUM'])
            low_count = len(risk_assessment[risk_assessment['risk_level'] == 'LOW'])
            
            print(f"   🔴 {high_count} HIGH-RISK customers need immediate attention")
            print(f"   🟡 {medium_count} MEDIUM-RISK customers for proactive campaigns")
            print(f"   🟢 {low_count} LOW-RISK customers are likely to stay")
            
            print("\n📋 Next Steps:")
            print("   1. Contact high-risk customers immediately")
            print("   2. Design retention campaigns for medium-risk customers")
            print("   3. Use SHAP explanations to understand risk drivers")
            print("   4. Implement real-time risk monitoring")
            print("="*60)
            
        else:
            print("❌ Risk assessment failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
