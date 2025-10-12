#!/usr/bin/env python3
"""
Quick test to find guaranteed high-risk customer data from the real dataset
"""
import pandas as pd
import joblib
import numpy as np

def preprocess_insurance_data(df):
    """Exact preprocessing function from training"""
    df = df.copy()
    
    # Convert datetime columns
    date_cols = ['cust_orig_date', 'date_of_birth', 'acct_suspd_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Replace datetime with numeric features
    today = pd.Timestamp('today')
    
    if 'cust_orig_date' in df.columns:
        df['cust_orig_days_since'] = (today - df['cust_orig_date']).dt.days
    
    if 'date_of_birth' in df.columns:
        df['age'] = (today - df['date_of_birth']).dt.days // 365
    
    if 'acct_suspd_date' in df.columns:
        df['acct_suspd_days_since'] = (today - df['acct_suspd_date']).dt.days.fillna(0)
    
    # Drop original datetime columns
    df = df.drop(columns=date_cols, errors='ignore')
    
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all are numeric
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    return df

def main():
    print("🔍 Finding actual high-risk customers from real data...")
    
    # Load real data
    df = pd.read_csv('./data/autoinsurance_churn.csv')
    print(f"Loaded {len(df)} customers")
    
    # Drop ID columns
    df.drop(columns=['individual_id', 'address_id'], inplace=True, errors='ignore')
    
    # Drop target leakage
    if 'acct_suspd_date' in df.columns:
        df.drop(columns=['acct_suspd_date'], inplace=True)
    
    # Get target
    y = df['Churn']
    X_raw = df.drop(columns=['Churn'])
    
    # Preprocess
    X = preprocess_insurance_data(X_raw)
    
    print(f"Processed features: {X.shape}")
    print(f"Churn rate: {y.mean():.1%}")
    
    # Load model
    model = joblib.load('./models/GradientBoostingClassifier_churn_prediction_model.pkl')
    
    # Get predictions for all customers
    print("🧮 Computing predictions for all customers...")
    probabilities = model.predict_proba(X)[:, 1]
    
    # Find highest risk customers
    high_risk_indices = np.argsort(probabilities)[-10:][::-1]
    
    print(f"\n🔴 TOP 10 HIGHEST RISK CUSTOMERS:")
    for i, idx in enumerate(high_risk_indices, 1):
        prob = probabilities[idx]
        actual_churn = y.iloc[idx]
        
        print(f"\n{i}. Customer {idx}: {prob:.1%} churn risk")
        print(f"   Actual outcome: {'CHURNED' if actual_churn == 1 else 'RETAINED'}")
        
        # Get original data for this customer
        customer_raw = df.iloc[idx]
        customer_processed = X.iloc[idx]
        
        print(f"   Raw data:")
        print(f"     curr_ann_amt: ${customer_raw.get('curr_ann_amt', 'N/A')}")
        print(f"     days_tenure: {customer_raw.get('days_tenure', 'N/A')}")
        print(f"     age_in_years: {customer_raw.get('age_in_years', 'N/A')}")
        print(f"     income: ${customer_raw.get('income', 'N/A')}")
        print(f"     marital_status: {customer_raw.get('marital_status', 'N/A')}")
        print(f"     city: {customer_raw.get('city', 'N/A')}")
        print(f"     state: {customer_raw.get('state', 'N/A')}")
        
        if i <= 3:  # Show detailed settings for top 3
            print(f"\n   📋 DASHBOARD TEST SETTINGS FOR CUSTOMER {idx}:")
            print(f"     Current Annual Amount: {customer_raw.get('curr_ann_amt', 0)}")
            print(f"     Days Tenure: {customer_raw.get('days_tenure', 0)}")
            print(f"     Customer Orig Date: {customer_raw.get('cust_orig_date', '2020-01-01')}")
            print(f"     Age in Years: {customer_raw.get('age_in_years', 30)}")
            print(f"     Date of Birth: {customer_raw.get('date_of_birth', '1990-01-01')}")
            print(f"     Latitude: {customer_raw.get('latitude', 30.0)}")
            print(f"     Longitude: {customer_raw.get('longitude', -90.0)}")
            print(f"     City: {customer_raw.get('city', 'Unknown')}")
            print(f"     State: {customer_raw.get('state', 'TX')}")
            print(f"     County: {customer_raw.get('county', 'Unknown')}")
            print(f"     Annual Premium: {customer_raw.get('curr_ann_amt', 0)}")
            print(f"     Annual Income: {customer_raw.get('income', 50000)}")
            print(f"     Has Children: {customer_raw.get('has_children', 0)}")
            print(f"     Length of Residence: {customer_raw.get('length_of_residence', 1)}")
            print(f"     Marital Status: {customer_raw.get('marital_status', 'Single')}")
            print(f"     Home Market Value: {customer_raw.get('home_market_value', '50000 - 74999')}")
            print(f"     Home Owner: {customer_raw.get('home_owner', 0)}")
            print(f"     College Degree: {customer_raw.get('college_degree', 0)}")
            print(f"     Good Credit: {customer_raw.get('good_credit', 0)}")

if __name__ == "__main__":
    main()
