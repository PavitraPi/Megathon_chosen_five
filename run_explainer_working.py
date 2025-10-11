"""
Working XGBoost Churn Risk Explainer
Creates data with the exact feature names expected by the trained model
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


def create_data_with_exact_features():
    """Create data with the exact feature names the model expects"""
    print("📊 Creating data with exact model features...")
    
    # These are the exact feature names the model expects (from the error message)
    model_features = [
        'curr_ann_amt', 'days_tenure', 'age_in_years', 'latitude', 'longitude', 
        'income', 'has_children', 'length_of_residence', 'home_owner', 
        'college_degree', 'good_credit', 'cust_orig_days_since', 'age',
        
        # City features (all the cities the model was trained on)
        'city_Aledo', 'city_Allen', 'city_Anna', 'city_Argyle', 'city_Arlington', 
        'city_Aubrey', 'city_Azle', 'city_Balch Springs', 'city_Bedford', 
        'city_Blue Ridge', 'city_Burleson', 'city_Caddo Mills', 'city_Carrollton', 
        'city_Cedar Hill', 'city_Celina', 'city_Chatfield', 'city_Colleyville', 
        'city_Coppell', 'city_Crandall', 'city_Crowley', 'city_Dallas', 
        'city_Denton', 'city_Desoto', 'city_Duncanville', 'city_Ennis', 
        'city_Era', 'city_Euless', 'city_Farmersville', 'city_Ferris', 
        'city_Flower Mound', 'city_Forney', 'city_Forreston', 'city_Fort Worth', 
        'city_Frisco', 'city_Garland', 'city_Grand Prairie', 'city_Grapevine', 
        'city_Haltom City', 'city_Haslet', 'city_Hurst', 'city_Hutchins', 
        'city_Irving', 'city_Italy', 'city_Joshua', 'city_Justin', 'city_Kaufman', 
        'city_Keller', 'city_Kemp', 'city_Kennedale', 'city_Krum', 
        'city_Lake Dallas', 'city_Lancaster', 'city_Lavon', 'city_Lewisville', 
        'city_Little Elm', 'city_Mansfield', 'city_Maypearl', 'city_Mckinney', 
        'city_Melissa', 'city_Mertens', 'city_Mesquite', 'city_Midlothian', 
        'city_Milford', 'city_Naval Air Station Jrb', 'city_Nevada', 
        'city_North Richland Hills', 'city_Palmer', 'city_Pilot Point', 
        'city_Plano', 'city_Ponder', 'city_Princeton', 'city_Prosper', 
        'city_Red Oak', 'city_Rice', 'city_Richardson', 'city_Roanoke', 
        'city_Rockwall', 'city_Rowlett', 'city_Royse City', 'city_Sachse', 
        'city_Sanger', 'city_Scurry', 'city_Seagoville', 'city_Southlake', 
        'city_Springtown', 'city_Sunnyvale', 'city_Terrell', 'city_The Colony', 
        'city_Tioga', 'city_Valley View', 'city_Waxahachie', 'city_Weatherford', 
        'city_Wilmer', 'city_Wylie',
        
        # County features
        'county_Cooke', 'county_Dallas', 'county_Denton', 'county_Ellis', 
        'county_Grayson', 'county_Hill', 'county_Hunt', 'county_Johnson', 
        'county_Kaufman', 'county_Navarro', 'county_Parker', 'county_Rockwall', 
        'county_Tarrant',
        
        # Marital status
        'marital_status_Single',
        
        # Home market value features
        'home_market_value_100000 - 124999', 'home_market_value_1000000 Plus',
        'home_market_value_125000 - 149999', 'home_market_value_150000 - 174999',
        'home_market_value_175000 - 199999', 'home_market_value_200000 - 224999',
        'home_market_value_225000 - 249999', 'home_market_value_25000 - 49999',
        'home_market_value_250000 - 274999', 'home_market_value_275000 - 299999',
        'home_market_value_300000 - 349999', 'home_market_value_350000 - 399999',
        'home_market_value_400000 - 449999', 'home_market_value_450000 - 499999',
        'home_market_value_50000 - 74999', 'home_market_value_500000 - 749999',
        'home_market_value_75000 - 99999', 'home_market_value_750000 - 999999'
    ]
    
    print(f"Total expected features: {len(model_features)}")
    
    # Create synthetic data for 50 customers
    np.random.seed(42)
    n_samples = 50
    
    # Initialize all features with zeros
    data = {feature: np.zeros(n_samples) for feature in model_features}
    
    # Fill numerical features with realistic values
    data['curr_ann_amt'] = np.random.normal(1200, 300, n_samples)
    data['days_tenure'] = np.random.randint(30, 2000, n_samples)
    data['age_in_years'] = np.random.randint(25, 75, n_samples)
    data['age'] = data['age_in_years']  # Duplicate
    data['latitude'] = np.random.uniform(32.5, 33.0, n_samples)
    data['longitude'] = np.random.uniform(-97.5, -96.5, n_samples)
    data['income'] = np.random.normal(65000, 25000, n_samples)
    data['has_children'] = np.random.binomial(1, 0.4, n_samples)
    data['length_of_residence'] = np.random.randint(1, 20, n_samples)
    data['home_owner'] = np.random.binomial(1, 0.7, n_samples)
    data['college_degree'] = np.random.binomial(1, 0.6, n_samples)
    data['good_credit'] = np.random.binomial(1, 0.8, n_samples)
    data['cust_orig_days_since'] = np.random.randint(-1000, 0, n_samples)
    
    # Most popular cities to activate
    popular_cities = ['city_Dallas', 'city_Fort Worth', 'city_Plano', 'city_Irving', 
                      'city_Garland', 'city_Arlington', 'city_Frisco', 'city_Mckinney']
    
    # Most popular counties
    popular_counties = ['county_Dallas', 'county_Tarrant', 'county_Denton']
    
    # Popular home values
    popular_home_values = ['home_market_value_100000 - 124999', 
                          'home_market_value_150000 - 174999',
                          'home_market_value_200000 - 224999']
    
    # For each customer, activate one city, county, and home value
    for i in range(n_samples):
        # Activate one city
        city = np.random.choice(popular_cities)
        data[city][i] = 1
        
        # Activate one county  
        county = np.random.choice(popular_counties)
        data[county][i] = 1
        
        # Activate one home value
        home_value = np.random.choice(popular_home_values)
        data[home_value][i] = 1
        
        # Sometimes single
        if np.random.random() < 0.3:
            data['marital_status_Single'][i] = 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create synthetic churn labels for evaluation
    # Make some customers more likely to churn by adjusting the logic
    churn_logits = (
        -0.5 +  # Higher base probability for more churners
        -0.00002 * df['income'] +  # Income effect (reduced)
        -0.0001 * df['days_tenure'] +  # Tenure effect (reduced)
        -0.2 * df['college_degree'] +  # Education effect (reduced)
        -0.1 * df['home_owner'] +  # Home ownership effect (reduced)
        0.5 * df['marital_status_Single'] +  # Being single increases churn
        0.3 * (df['age_in_years'] < 30).astype(int) +  # Young people churn more
        -0.2 * (df['age_in_years'] > 60).astype(int) +  # Older people churn less
        np.random.normal(0, 0.8, n_samples)  # More noise for variety
    )
    
    churn_probabilities = 1 / (1 + np.exp(-churn_logits))
    actual_churn = np.random.binomial(1, churn_probabilities, n_samples)
    
    print(f"✅ Created data with exact model features: {df.shape}")
    print(f"   Synthetic churn rate: {np.mean(actual_churn):.1%}")
    print(f"   All {len(model_features)} expected features present")
    
    return df, actual_churn


def create_churn_risk_assessment(model, X, y=None):
    """Create comprehensive churn risk assessment"""
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
        print(f"\n📋 DETAILED RISK BREAKDOWN:")
        
        for risk_level, emoji in [('HIGH', '🔴'), ('MEDIUM', '🟡'), ('LOW', '🟢')]:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            
            if len(risk_customers) > 0:
                print(f"\n{emoji} {risk_level} RISK CUSTOMERS:")
                
                if risk_level == 'HIGH':
                    samples = risk_customers.nlargest(min(5, len(risk_customers)), 'churn_probability')
                elif risk_level == 'LOW':
                    samples = risk_customers.nsmallest(min(5, len(risk_customers)), 'churn_probability')
                else:
                    samples = risk_customers.head(min(5, len(risk_customers)))
                
                for idx, (_, customer) in enumerate(samples.iterrows(), 1):
                    pred_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                    
                    if y is not None:
                        actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                        accuracy = "✅" if customer['predicted_churn'] == customer['actual_churn'] else "❌"
                        print(f"   {idx}. Customer {customer['customer_id']:2d}: {customer['churn_probability']:.3f} | "
                              f"Predicted: {pred_label:12s} | Actual: {actual_label:8s} {accuracy}")
                    else:
                        print(f"   {idx}. Customer {customer['customer_id']:2d}: {customer['churn_probability']:.3f} | "
                              f"Predicted: {pred_label}")
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
        
        # Model performance if ground truth available
        if y is not None:
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
            
            # Confusion matrix breakdown
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred)
            
            # Handle case where confusion matrix might not be 2x2
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                print(f"\n📊 CONFUSION MATRIX:")
                print(f"   True Negatives:  {tn:3d} (correctly predicted no churn)")
                print(f"   False Positives: {fp:3d} (incorrectly predicted churn)")
                print(f"   False Negatives: {fn:3d} (missed actual churners)")
                print(f"   True Positives:  {tp:3d} (correctly predicted churn)")
            else:
                print(f"\n📊 CONFUSION MATRIX:")
                print(f"   Matrix shape: {cm.shape}")
                print(f"   Only one class present in predictions")
        
        # Risk-based recommendations
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
        high_risk_count = risk_counts.get('HIGH', 0)
        medium_risk_count = risk_counts.get('MEDIUM', 0)
        low_risk_count = risk_counts.get('LOW', 0)
        
        if high_risk_count > 0:
            print(f"   🚨 IMMEDIATE ACTION: {high_risk_count} high-risk customers need urgent intervention")
            print(f"      → Contact within 24 hours")
            print(f"      → Offer personalized retention packages")
            print(f"      → Assign dedicated account managers")
        
        if medium_risk_count > 0:
            print(f"   ⚠️  PROACTIVE CARE: {medium_risk_count} medium-risk customers for targeted campaigns")
            print(f"      → Implement loyalty programs")
            print(f"      → Regular check-ins and satisfaction surveys")
            print(f"      → Preventive offers and upgrades")
        
        if low_risk_count > 0:
            print(f"   ✅ MAINTAIN: {low_risk_count} low-risk customers are stable")
            print(f"      → Focus on upselling opportunities")
            print(f"      → Monitor for any changes in behavior")
            print(f"      → Use as brand advocates and referral sources")
        
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
    """Create comprehensive risk visualization"""
    print("\n📊 Creating risk visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Risk distribution pie chart
    risk_counts = risk_assessment['risk_level'].value_counts()
    colors = ['red', 'green', 'orange']  # HIGH, LOW, MEDIUM
    wedges, texts, autotexts = axes[0, 0].pie(risk_counts.values, labels=risk_counts.index, 
                                              autopct='%1.1f%%', colors=colors, startangle=90)
    axes[0, 0].set_title('Customer Risk Distribution', fontsize=14, fontweight='bold')
    
    # 2. Probability histogram with risk thresholds
    axes[0, 1].hist(y_pred_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].axvline(x=0.4, color='orange', linestyle='--', linewidth=2, 
                      label='Medium Risk Threshold (40%)')
    axes[0, 1].axvline(x=0.7, color='red', linestyle='--', linewidth=2, 
                      label='High Risk Threshold (70%)')
    axes[0, 1].set_xlabel('Churn Probability')
    axes[0, 1].set_ylabel('Number of Customers')
    axes[0, 1].set_title('Churn Probability Distribution', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Risk level bar chart with values
    risk_counts_ordered = risk_assessment['risk_level'].value_counts().reindex(['LOW', 'MEDIUM', 'HIGH'])
    colors_ordered = ['green', 'orange', 'red']
    bars = axes[1, 0].bar(risk_counts_ordered.index, risk_counts_ordered.values, 
                         color=colors_ordered, alpha=0.8)
    axes[1, 0].set_title('Customer Count by Risk Level', fontweight='bold')
    axes[1, 0].set_ylabel('Number of Customers')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Box plot of probabilities by risk level
    risk_levels = ['LOW', 'MEDIUM', 'HIGH']
    prob_by_risk = []
    labels_with_counts = []
    
    for level in risk_levels:
        level_data = risk_assessment[risk_assessment['risk_level'] == level]['churn_probability']
        if len(level_data) > 0:
            prob_by_risk.append(level_data)
            labels_with_counts.append(f'{level}\n(n={len(level_data)})')
        else:
            prob_by_risk.append([0])  # Empty placeholder
            labels_with_counts.append(f'{level}\n(n=0)')
    
    box_plot = axes[1, 1].boxplot(prob_by_risk, labels=labels_with_counts, patch_artist=True)
    colors_box = ['lightgreen', 'orange', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    
    axes[1, 1].set_title('Churn Probability Distribution by Risk Level', fontweight='bold')
    axes[1, 1].set_ylabel('Churn Probability')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add risk threshold lines
    axes[1, 1].axhline(y=0.4, color='orange', linestyle='--', alpha=0.7)
    axes[1, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('./outputs/comprehensive_risk_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Comprehensive visualization saved to: ./outputs/comprehensive_risk_analysis.png")


def explain_predictions_with_shap(model, X, risk_assessment):
    """Provide SHAP explanations for different risk levels"""
    print("\n🔍 SHAP EXPLANATIONS FOR DIFFERENT RISK LEVELS")
    print("=" * 60)
    
    try:
        # Create SHAP explainer
        print("Creating SHAP explainer for XGBoost model...")
        explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values for all customers
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X)
        
        # Handle binary classification
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]  # Churn class
            expected_value = explainer.expected_value[1]
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        
        print(f"✅ SHAP values calculated for {len(X)} customers")
        
        # Select representative customers from each risk level
        high_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'HIGH']
        medium_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'MEDIUM']
        low_risk_customers = risk_assessment[risk_assessment['risk_level'] == 'LOW']
        
        # Analyze one customer from each risk level
        analysis_customers = []
        
        if len(high_risk_customers) > 0:
            analysis_customers.append(('HIGH', high_risk_customers.iloc[0]['customer_id']))
        
        if len(medium_risk_customers) > 0:
            analysis_customers.append(('MEDIUM', medium_risk_customers.iloc[0]['customer_id']))
        
        if len(low_risk_customers) > 0:
            analysis_customers.append(('LOW', low_risk_customers.iloc[0]['customer_id']))
        
        print(f"\nAnalyzing {len(analysis_customers)} representative customers...")
        
        for risk_level, customer_id in analysis_customers:
            customer_data = X.iloc[customer_id]
            customer_risk = risk_assessment.iloc[customer_id]
            churn_prob = customer_risk['churn_probability']
            
            # Risk level emoji
            emoji = "🔴" if risk_level == 'HIGH' else "🟡" if risk_level == 'MEDIUM' else "🟢"
            
            print(f"\n{emoji} {risk_level} RISK CUSTOMER {customer_id}")
            print("=" * 40)
            print(f"Churn Probability: {churn_prob:.3f}")
            print(f"Prediction: {'Will Churn' if customer_risk['predicted_churn'] == 1 else 'Will Retain'}")
            
            # Get SHAP values for this customer
            customer_shap = shap_values_positive[customer_id]
            
            # Feature importance analysis
            feature_contrib = pd.DataFrame({
                'feature': X.columns,
                'shap_value': customer_shap,
                'feature_value': customer_data.values
            })
            feature_contrib['abs_shap'] = abs(feature_contrib['shap_value'])
            
            # Top risk-increasing factors
            top_risk_factors = feature_contrib[feature_contrib['shap_value'] > 0.001].nlargest(5, 'shap_value')
            
            # Top protective factors
            top_protective_factors = feature_contrib[feature_contrib['shap_value'] < -0.001].nsmallest(5, 'shap_value')
            
            print(f"\n📈 FACTORS INCREASING CHURN RISK:")
            if len(top_risk_factors) > 0:
                for _, row in top_risk_factors.iterrows():
                    impact = "🔥 CRITICAL" if abs(row['shap_value']) > 0.1 else "⚠️ HIGH" if abs(row['shap_value']) > 0.05 else "📊 MEDIUM"
                    feature_name = row['feature'].replace('_', ' ').title()
                    print(f"   • {feature_name}: +{row['shap_value']:.4f} {impact}")
                    
                    # Add interpretation for binary features
                    if row['feature_value'] in [0, 1]:
                        status = "YES" if row['feature_value'] == 1 else "NO"
                        print(f"     └─ {status}")
                    else:
                        print(f"     └─ Value: {row['feature_value']:.2f}")
            else:
                print("   • No significant risk-increasing factors")
            
            print(f"\n📉 FACTORS REDUCING CHURN RISK:")
            if len(top_protective_factors) > 0:
                for _, row in top_protective_factors.iterrows():
                    impact = "🛡️ STRONG" if abs(row['shap_value']) > 0.1 else "✅ MODERATE" if abs(row['shap_value']) > 0.05 else "📊 WEAK"
                    feature_name = row['feature'].replace('_', ' ').title()
                    print(f"   • {feature_name}: {row['shap_value']:.4f} {impact}")
                    
                    if row['feature_value'] in [0, 1]:
                        status = "YES" if row['feature_value'] == 1 else "NO"
                        print(f"     └─ {status}")
                    else:
                        print(f"     └─ Value: {row['feature_value']:.2f}")
            else:
                print("   • No significant protective factors")
            
            # Risk-specific recommendations
            print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
            if risk_level == 'HIGH':
                print("   🚨 URGENT: Immediate intervention required")
                print("   📞 Call customer within 24 hours")
                print("   💰 Offer retention package with discounts")
                print("   🤝 Assign senior account manager")
                print("   📊 Monitor daily for engagement changes")
            elif risk_level == 'MEDIUM':
                print("   ⚠️ PROACTIVE: Schedule retention call this week")
                print("   🎁 Consider loyalty rewards or upgrades")
                print("   📧 Send personalized satisfaction survey")
                print("   📈 Monitor weekly for risk changes")
            else:
                print("   ✅ MAINTAIN: Customer is stable")
                print("   🚀 Explore upselling opportunities")
                print("   🌟 Consider for referral program")
                print("   📅 Regular quarterly check-ins")
        
        # Create overall SHAP summary
        print(f"\n📊 OVERALL FEATURE IMPORTANCE (All Customers)")
        print("=" * 50)
        
        # Calculate mean absolute SHAP values for feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'mean_abs_shap': np.mean(np.abs(shap_values_positive), axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        print("Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
            feature_name = row['feature'].replace('_', ' ').title()
            print(f"   {i:2d}. {feature_name}: {row['mean_abs_shap']:.4f}")
        
        # Save SHAP explainer
        os.makedirs('./models', exist_ok=True)
        joblib.dump(explainer, './models/explainer_xgb.bz2', compress=('bz2', 9))
        print(f"\n💾 SHAP explainer saved to: ./models/explainer_xgb.bz2")
        
    except Exception as e:
        print(f"❌ Error in SHAP analysis: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main function"""
    print("🚀 WORKING XGBOOST CHURN RISK EXPLAINER")
    print("=" * 60)
    print("High-Risk & Low-Risk Customer Analysis with Explainable AI")
    print("=" * 60)
    
    try:
        # Load model
        model, model_path = load_xgboost_model()
        print(f"Model type: {type(model).__name__}")
        
        # Create data with exact features the model expects
        X, y = create_data_with_exact_features()
        
        # Create comprehensive risk assessment
        risk_assessment = create_churn_risk_assessment(model, X, y)
        
        if risk_assessment is not None:
            # Provide SHAP explanations
            explain_predictions_with_shap(model, X, risk_assessment)
            
            # Final success summary
            print("\n" + "="*60)
            print("🎉 CHURN RISK ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*60)
            
            # Summary statistics
            high_count = len(risk_assessment[risk_assessment['risk_level'] == 'HIGH'])
            medium_count = len(risk_assessment[risk_assessment['risk_level'] == 'MEDIUM'])
            low_count = len(risk_assessment[risk_assessment['risk_level'] == 'LOW'])
            total_count = len(risk_assessment)
            
            print("📊 EXECUTIVE SUMMARY:")
            print(f"   Total Customers Analyzed: {total_count}")
            print(f"   🔴 High Risk (≥70%):     {high_count:2d} ({high_count/total_count*100:5.1f}%)")
            print(f"   🟡 Medium Risk (40-70%):  {medium_count:2d} ({medium_count/total_count*100:5.1f}%)")
            print(f"   🟢 Low Risk (<40%):       {low_count:2d} ({low_count/total_count*100:5.1f}%)")
            
            avg_prob = np.mean(risk_assessment['churn_probability'])
            print(f"   📈 Average Churn Risk:    {avg_prob:.1%}")
            
            print(f"\n📁 FILES CREATED:")
            print(f"   • ./outputs/customer_risk_assessment.csv")
            print(f"   • ./outputs/comprehensive_risk_analysis.png")
            print(f"   • ./models/explainer_xgb.bz2")
            
            print(f"\n🎯 BUSINESS IMPACT:")
            if high_count > 0:
                print(f"   • {high_count} customers at CRITICAL RISK - immediate action needed")
                estimated_revenue_risk = high_count * 1200  # Assuming $1200 annual value
                print(f"   • Estimated revenue at risk: ${estimated_revenue_risk:,}")
            
            if medium_count > 0:
                print(f"   • {medium_count} customers for PROACTIVE retention campaigns")
            
            if low_count > 0:
                print(f"   • {low_count} stable customers for upselling and referrals")
            
            print(f"\n📋 NEXT STEPS:")
            print(f"   1. Review customer_risk_assessment.csv for detailed customer list")
            print(f"   2. Implement immediate outreach for high-risk customers")
            print(f"   3. Design targeted campaigns for medium-risk customers")
            print(f"   4. Use SHAP explanations to understand risk drivers")
            print(f"   5. Set up automated monitoring for risk level changes")
            
            print("="*60)
            
        else:
            print("❌ Risk assessment failed")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
