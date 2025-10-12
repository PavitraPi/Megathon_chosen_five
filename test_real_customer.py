#!/usr/bin/env python3

import pandas as pd
import joblib
import numpy as np

print("=== TESTING WITH REAL HIGH-RISK CUSTOMER ===")

# Load real data and get a high-risk customer
df = pd.read_csv('data/autoinsurance_churn.csv')

# Find highest-risk customers (churned ones)
churned_customers = df[df['Churn'] == 1]
print(f"Total churned customers: {len(churned_customers)}")

# Get a sample with high premium (more likely to be detected as high risk by model)
high_premium_churned = churned_customers[churned_customers['curr_ann_amt'] > 1000]
print(f"High premium churned customers: {len(high_premium_churned)}")

if len(high_premium_churned) > 0:
    # Pick the first one
    real_customer = high_premium_churned.iloc[0]
    
    print(f"\n🎯 REAL CHURNED CUSTOMER DATA:")
    print(f"Annual Premium: {real_customer['curr_ann_amt']:.2f}")
    print(f"Income: {real_customer['income']:.2f}")
    print(f"Days Tenure: {real_customer['days_tenure']:.0f}")
    print(f"Age: {real_customer['age_in_years']:.0f}")
    print(f"City: {real_customer['city']}")
    print(f"State: {real_customer['state']}")
    print(f"County: {real_customer['county']}")
    print(f"Latitude: {real_customer['latitude']:.6f}")
    print(f"Longitude: {real_customer['longitude']:.6f}")
    print(f"Marital Status: {real_customer['marital_status']}")
    print(f"Has Children: {real_customer['has_children']}")
    print(f"Home Owner: {real_customer['home_owner']}")
    print(f"College Degree: {real_customer['college_degree']}")
    print(f"Good Credit: {real_customer['good_credit']}")
    print(f"Home Market Value: {real_customer['home_market_value']}")
    print(f"Length of Residence: {real_customer['length_of_residence']}")
    
    # Test with models
    print(f"\n=== TESTING WITH MODELS ===")
    
    # Prepare data exactly like the models expect
    features = real_customer.drop(['Churn', 'individual_id', 'address_id', 'cust_orig_date', 'date_of_birth', 'acct_suspd_date'])
    features_df = pd.DataFrame([features])
    
    print(f"Feature shape: {features_df.shape}")
    print(f"Features: {list(features_df.columns)}")
    
    # Load models and test
    try:
        gb_model = joblib.load('models/GradientBoostingClassifier_churn_prediction_model.pkl')
        gb_prob = gb_model.predict_proba(features_df)[0][1]
        print(f"✅ GradientBoost prediction: {gb_prob:.3f} ({gb_prob*100:.1f}%)")
        
        xgb_model = joblib.load('models/XGBoostClassifier_churn_prediction_model.pkl')
        xgb_prob = xgb_model.predict_proba(features_df)[0][1]
        print(f"✅ XGBoost prediction: {xgb_prob:.3f} ({xgb_prob*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Model error: {e}")
    
    print(f"\n📋 STREAMLIT DASHBOARD INPUT FORMAT:")
    print(f"curr_ann_amt: {real_customer['curr_ann_amt']}")
    print(f"days_tenure: {real_customer['days_tenure']:.0f}")
    print(f"age_in_years: {real_customer['age_in_years']:.0f}")
    print(f"latitude: {real_customer['latitude']:.6f}")
    print(f"longitude: {real_customer['longitude']:.6f}")
    print(f"city: {real_customer['city']}")
    print(f"state: {real_customer['state']}")
    print(f"county: {real_customer['county']}")
    print(f"income: {real_customer['income']}")
    print(f"has_children: {'Yes' if real_customer['has_children'] == 1 else 'No'}")
    print(f"length_of_residence: {real_customer['length_of_residence']:.0f}")
    print(f"marital_status: {real_customer['marital_status']}")
    print(f"home_market_value: {real_customer['home_market_value']}")
    print(f"home_owner: {'Yes' if real_customer['home_owner'] == 1 else 'No'}")
    print(f"college_degree: {'Yes' if real_customer['college_degree'] == 1 else 'No'}")
    print(f"good_credit: {'Yes' if real_customer['good_credit'] == 1 else 'No'}")
