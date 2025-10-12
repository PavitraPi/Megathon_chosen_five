#!/usr/bin/env python3

import pandas as pd
import joblib
import numpy as np

print("=== DEBUGGING FRONTEND-BACKEND COMMUNICATION ===")

# Load real data
df = pd.read_csv('data/autoinsurance_churn.csv')
print(f"Dataset shape: {df.shape}")

# Check if Terrell exists
terrell_customers = df[df['city'] == 'Terrell']
print(f"\nTerrell customers found: {len(terrell_customers)}")

if len(terrell_customers) == 0:
    print("❌ PROBLEM FOUND: Terrell doesn't exist in dataset!")
    print("\nFinding ACTUAL high-risk cities:")
    
    # Find actual high-risk cities
    city_stats = df.groupby('city').agg({
        'churn': ['count', 'mean'],
        'annual_premium': 'mean'
    }).round(3)
    
    city_stats.columns = ['customer_count', 'churn_rate', 'avg_premium']
    high_risk_cities = city_stats[
        (city_stats['customer_count'] >= 20) & 
        (city_stats['churn_rate'] >= 0.7)
    ].sort_values('churn_rate', ascending=False)
    
    print("\n🚨 ACTUAL HIGH-RISK CITIES (70%+ churn rate):")
    print(high_risk_cities.head(10))
    
    if len(high_risk_cities) > 0:
        # Get the highest risk city
        top_city = high_risk_cities.index[0]
        top_city_data = df[df['city'] == top_city]
        
        print(f"\n✅ GUARANTEED HIGH-RISK CITY: {top_city}")
        print(f"   Churn Rate: {high_risk_cities.loc[top_city, 'churn_rate']:.1%}")
        print(f"   Customers: {high_risk_cities.loc[top_city, 'customer_count']}")
        
        # Get a sample high-risk customer
        high_risk_sample = top_city_data[top_city_data['churn'] == 1].iloc[0]
        
        print(f"\n🎯 GUARANTEED TEST CASE (Real customer who churned):")
        test_fields = [
            'city', 'state', 'county', 'latitude', 'longitude',
            'annual_premium', 'annual_income', 'age_in_years',
            'marital_status', 'has_children', 'home_owner',
            'college_degree', 'good_credit', 'days_tenure'
        ]
        
        for field in test_fields:
            if field in high_risk_sample.index:
                print(f"   {field}: {high_risk_sample[field]}")
else:
    print("✅ Terrell exists, checking churn rate:")
    print(f"   Churn rate: {terrell_customers['churn'].mean():.1%}")

# Also check what the models expect
print(f"\n=== MODEL INFO ===")
try:
    gb_model = joblib.load('models/GradientBoostingClassifier_churn_prediction_model.pkl')
    print(f"GradientBoost model loaded: {type(gb_model)}")
    
    xgb_model = joblib.load('models/XGBoostClassifier_churn_prediction_model.pkl')
    print(f"XGBoost model loaded: {type(xgb_model)}")
    
    # Check feature count
    if hasattr(gb_model, 'n_features_'):
        print(f"Expected features: {gb_model.n_features_}")
    
except Exception as e:
    print(f"Model loading error: {e}")
