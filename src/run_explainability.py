"""
Explainability Analysis Script for Auto Insurance Churn Models
Loads pre-trained models and performs comprehensive explainability analysis
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
import eli5
from eli5.sklearn import PermutationImportance
from pdpbox import pdp, info_plots

warnings.filterwarnings('ignore')
plt.style.use('default')

class ExplainabilityAnalyzer:
    def __init__(self, data_path='./data/autoinsurance_churn.csv', model_dir='./models'):
        self.data_path = data_path
        self.model_dir = model_dir
        self.models = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.shap_explainers = {}
        self.lime_explainer = None
        
        # Create output directories
        os.makedirs('./models', exist_ok=True)
        os.makedirs('./plots', exist_ok=True)
        
    def load_data(self):
        """Load and preprocess the dataset using saved preprocessing objects"""
        print("📊 Loading and preprocessing data...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        df = pd.read_csv(self.data_path)
        print(f"Dataset shape: {df.shape}")
        
        # Define target and features
        TARGET = 'Churn'
        if TARGET not in df.columns:
            raise ValueError(f"Target column '{TARGET}' not found in dataset.")
        
        # Try to load saved preprocessing objects first
        try:
            if os.path.exists('./models/label_encoders.pkl'):
                label_encoders = joblib.load('./models/label_encoders.pkl')
                print("✅ Loaded saved label encoders")
            else:
                label_encoders = None
                
            if os.path.exists('./models/scaler.pkl'):
                scaler = joblib.load('./models/scaler.pkl')
                print("✅ Loaded saved scaler")
            else:
                scaler = None
                
            if os.path.exists('./models/feature_columns.pkl'):
                feature_columns = joblib.load('./models/feature_columns.pkl')
                print("✅ Loaded saved feature columns")
            else:
                feature_columns = None
        except Exception as e:
            print(f"⚠️ Could not load preprocessing objects: {str(e)}")
            label_encoders = None
            scaler = None
            feature_columns = None
        
        # Drop ID columns
        id_cols = [col for col in df.columns if 'ID' in col or 'id' in col]
        features_to_drop = [col for col in id_cols if col != TARGET]
        
        X = df.drop(features_to_drop + [TARGET], axis=1, errors='ignore')
        y = df[TARGET]
        
        print(f"Features: {len(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Handle missing values
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
        
        # Use saved encoders or create new ones
        if label_encoders:
            print("Using saved label encoders...")
            for col in categorical_cols:
                if col in label_encoders:
                    try:
                        X[col] = label_encoders[col].transform(X[col])
                    except ValueError as e:
                        print(f"⚠️ New categories in {col}, fitting new encoder")
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col])
                else:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
        else:
            print("Creating new label encoders...")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
        
        # Use saved scaler or create new one
        if scaler:
            print("Using saved scaler...")
            try:
                X[numerical_cols] = scaler.transform(X[numerical_cols])
            except Exception as e:
                print(f"⚠️ Error with saved scaler: {str(e)}, creating new one")
                scaler_new = StandardScaler()
                X[numerical_cols] = scaler_new.fit_transform(X[numerical_cols])
        else:
            print("Creating new scaler...")
            scaler_new = StandardScaler()
            X[numerical_cols] = scaler_new.fit_transform(X[numerical_cols])
        
        # Ensure column order matches saved feature columns if available
        if feature_columns:
            missing_cols = set(feature_columns) - set(X.columns)
            extra_cols = set(X.columns) - set(feature_columns)
            
            if missing_cols:
                print(f"⚠️ Missing columns from saved features: {missing_cols}")
            if extra_cols:
                print(f"⚠️ Extra columns not in saved features: {extra_cols}")
                
            # Reorder columns to match saved order
            common_cols = [col for col in feature_columns if col in X.columns]
            X = X[common_cols]
            print(f"✅ Reordered columns to match saved feature order")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"✅ Data preprocessed successfully!")
        print(f"Train shape: {self.X_train.shape}, Test shape: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def load_models(self):
        """Load pre-trained models from various possible locations"""
        print("\n🤖 Loading pre-trained models...")
        
        # Model file patterns to search for
        model_patterns = [
            # From your training script
            ('RandomForest', ['RandomForestClassifier_churn_model.pkl', 'model_rfc.pkl', 'rf_model.pkl']),
            ('GradientBoosting', ['GradientBoostingClassifier_churn_model.pkl', 'model_gb.pkl', 'gb_model.pkl']),
            ('XGBoost', ['XGBoostClassifier_churn_model.pkl', 'model_xgb.pkl', 'xgb_model.pkl']),
            ('MLP', ['mlp_model.pkl', 'model_mlp.pkl']),
            ('Best', ['best_model_RandomForestClassifier_churn_model.pkl', 'best_model_GradientBoostingClassifier_churn_model.pkl', 'best_model_XGBoostClassifier_churn_model.pkl'])
        ]
        
        # Search in current directory and models directory
        search_paths = ['.', './models']
        
        models_loaded = 0
        for model_name, filenames in model_patterns:
            model_found = False
            
            for search_path in search_paths:
                if model_found:
                    break
                    
                for filename in filenames:
                    filepath = os.path.join(search_path, filename)
                    if os.path.exists(filepath):
                        try:
                            model = joblib.load(filepath)
                            self.models[model_name] = model
                            models_loaded += 1
                            print(f"✅ Loaded {model_name} from {filepath}")
                            model_found = True
                            break
                        except Exception as e:
                            print(f"❌ Error loading {model_name} from {filepath}: {str(e)}")
        
        # Also check for any other .pkl files in current directory
        print("\n🔍 Searching for additional model files...")
        for file in os.listdir('.'):
            if file.endswith('.pkl') and 'model' in file.lower():
                if not any(file in filenames for _, filenames in model_patterns for filenames in [filenames]):
                    try:
                        model = joblib.load(file)
                        # Try to determine if it's a sklearn model
                        if hasattr(model, 'predict') and hasattr(model, 'predict_proba'):
                            model_key = file.replace('.pkl', '').replace('_churn_model', '')
                            if model_key not in self.models:
                                self.models[model_key] = model
                                models_loaded += 1
                                print(f"✅ Found additional model: {model_key} from {file}")
                    except Exception as e:
                        print(f"⚠️ Could not load {file}: {str(e)}")
        
        if models_loaded == 0:
            raise ValueError("No pre-trained models found! Please ensure your .pkl model files are in the current directory or ./models/ directory.")
        
        print(f"\n✅ Successfully loaded {models_loaded} models!")
        print(f"Available models: {list(self.models.keys())}")
        return self.models
    
    def evaluate_models(self):
        """Evaluate loaded models and assess churn risk probabilities"""
        print("\n📈 Evaluating model performance and churn risk assessment...")
        
        results = {}
        for name, model in self.models.items():
            try:
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]  # Churn probabilities
                
                # Calculate standard metrics
                results[name] = {
                    'Accuracy': accuracy_score(self.y_test, y_pred),
                    'Precision': precision_score(self.y_test, y_pred, zero_division=0),
                    'Recall': recall_score(self.y_test, y_pred, zero_division=0),
                    'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
                    'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba)
                }
                
                # Churn risk assessment
                high_risk_threshold = 0.7
                medium_risk_threshold = 0.4
                
                high_risk_count = np.sum(y_pred_proba >= high_risk_threshold)
                medium_risk_count = np.sum((y_pred_proba >= medium_risk_threshold) & (y_pred_proba < high_risk_threshold))
                low_risk_count = np.sum(y_pred_proba < medium_risk_threshold)
                
                results[name]['High_Risk_Customers'] = high_risk_count
                results[name]['Medium_Risk_Customers'] = medium_risk_count
                results[name]['Low_Risk_Customers'] = low_risk_count
                results[name]['Avg_Churn_Probability'] = np.mean(y_pred_proba)
                
                print(f"\n{name} Performance:")
                for metric, value in results[name].items():
                    if 'Customers' in metric or 'Probability' in metric:
                        if 'Probability' in metric:
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
                    else:
                        print(f"  {metric}: {value:.4f}")
                
                print(f"\n📊 {name} Churn Risk Distribution:")
                print(f"  🔴 High Risk (≥70%): {high_risk_count} customers ({high_risk_count/len(y_pred_proba)*100:.1f}%)")
                print(f"  🟡 Medium Risk (40-70%): {medium_risk_count} customers ({medium_risk_count/len(y_pred_proba)*100:.1f}%)")
                print(f"  🟢 Low Risk (<40%): {low_risk_count} customers ({low_risk_count/len(y_pred_proba)*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ Error evaluating {name}: {str(e)}")
        
        # Create comparison DataFrame
        if results:
            results_df = pd.DataFrame(results).T
            print(f"\n📊 Model Comparison:")
            print(results_df.round(4))
            
            # Save results
            results_df.to_csv('./models/explainability_model_comparison.csv')
            print("✅ Model comparison saved")
            
        return results
    
    def permutation_importance_analysis(self):
        """Perform permutation importance analysis"""
        print("\n🔍 PERMUTATION IMPORTANCE ANALYSIS")
        print("=" * 50)
        
        permutation_results = {}
        
        for name, model in self.models.items():
            try:
                print(f"\n📊 Analyzing {name}...")
                
                perm = PermutationImportance(model, random_state=42).fit(self.X_test, self.y_test)
                
                feature_importance = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': perm.feature_importances_,
                    'std': perm.feature_importances_std_
                }).sort_values('importance', ascending=False)
                
                permutation_results[name] = feature_importance
                
                # Plot top 10 features
                top_features = feature_importance.head(10)
                
                plt.figure(figsize=(10, 6))
                plt.barh(range(len(top_features)), top_features['importance'], 
                         xerr=top_features['std'], color='skyblue', alpha=0.7)
                plt.yticks(range(len(top_features)), top_features['feature'])
                plt.xlabel('Permutation Importance')
                plt.title(f'{name} - Top 10 Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.savefig(f'./plots/permutation_importance_{name.lower()}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"Top 5 features for {name}:")
                print(top_features.head()[['feature', 'importance']].to_string(index=False))
                
            except Exception as e:
                print(f"❌ Error computing permutation importance for {name}: {str(e)}")
        
        return permutation_results
    
    def shap_analysis(self):
        """Perform SHAP analysis for all models with full test set"""
        print("\n🎯 SHAP ANALYSIS")
        print("=" * 50)
        
        # Initialize SHAP JavaScript for force plots (if in notebook)
        try:
            shap.initjs()
        except:
            print("⚠️ SHAP JavaScript not initialized (not in notebook environment)")
        
        print(f"Calculating SHAP values for full test set: {len(self.X_test)} samples")
        
        shap_values_dict = {}
        
        for name, model in self.models.items():
            try:
                print(f"\n🔍 Creating SHAP explainer for {name}...")
                
                # Create appropriate explainer based on model type
                if any(keyword in name.lower() for keyword in ['xgboost', 'xgb', 'randomforest', 'rf', 'gradientboosting', 'gb']):
                    print(f"   Using TreeExplainer for {name}")
                    explainer = shap.TreeExplainer(model)
                    
                    # Calculate SHAP values for entire test set
                    print(f"   Calculating SHAP values for {len(self.X_test)} samples...")
                    shap_values = explainer.shap_values(self.X_test)
                    
                    # Handle binary classification - get positive class values
                    if isinstance(shap_values, list):
                        shap_values_positive = shap_values[1]  # Positive class (churn)
                        shap_values_negative = shap_values[0]  # Negative class (no churn)
                        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    else:
                        shap_values_positive = shap_values
                        shap_values_negative = None
                        expected_value = explainer.expected_value
                        
                else:
                    print(f"   Using KernelExplainer for {name}")
                    # Kernel explainer for other models with larger background
                    background = shap.sample(self.X_train, min(2000, len(self.X_train)))
                    explainer = shap.KernelExplainer(model.predict_proba, background)
                    
                    # Calculate SHAP values for entire test set
                    print(f"   Calculating SHAP values for {len(self.X_test)} samples...")
                    shap_values = explainer.shap_values(self.X_test)
                    
                    if isinstance(shap_values, list):
                        shap_values_positive = shap_values[1]
                        shap_values_negative = shap_values[0]
                        expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                    else:
                        shap_values_positive = shap_values
                        shap_values_negative = None
                        expected_value = explainer.expected_value
                
                # Store explainer and values
                self.shap_explainers[name] = explainer
                shap_values_dict[name] = {
                    'positive': shap_values_positive,
                    'negative': shap_values_negative,
                    'expected_value': expected_value
                }
                
                print(f"✅ SHAP values calculated: {shap_values_positive.shape}")
                
                # Create individual prediction examples for first few samples
                self._create_individual_shap_plots(name, model, explainer, shap_values_positive, expected_value)
                
                # Summary plots using full dataset or sample if too large
                sample_size = min(500, len(self.X_test))  # Use up to 500 samples for summary plots
                X_test_sample = self.X_test.iloc[:sample_size]
                shap_values_sample = shap_values_positive[:sample_size]
                
                # Summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values_sample, X_test_sample, show=False, max_display=15)
                plt.title(f'{name} - SHAP Feature Importance Summary')
                plt.tight_layout()
                plt.savefig(f'./plots/shap_summary_{name.lower()}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                # Bar plot
                plt.figure(figsize=(12, 6))
                shap.summary_plot(shap_values_sample, X_test_sample, plot_type="bar", show=False, max_display=15)
                plt.title(f'{name} - SHAP Feature Importance (Bar Plot)')
                plt.tight_layout()
                plt.savefig(f'./plots/shap_bar_{name.lower()}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"✅ SHAP analysis completed for {name}")
                
            except Exception as e:
                print(f"❌ Error creating SHAP explainer for {name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return shap_values_dict
    
    def create_churn_risk_assessment(self):
        """Create comprehensive churn risk assessment for all customers"""
        print("\n🎯 COMPREHENSIVE CHURN RISK ASSESSMENT")
        print("=" * 60)
        
        if not self.models:
            print("❌ No models available for risk assessment")
            return
        
        # Use the first (primary) model for risk assessment
        primary_model_name = list(self.models.keys())[0]
        primary_model = self.models[primary_model_name]
        
        print(f"Using {primary_model_name} for risk assessment...")
        
        # Get predictions for all test customers
        y_pred_proba = primary_model.predict_proba(self.X_test)[:, 1]  # Churn probabilities
        y_pred = primary_model.predict(self.X_test)
        
        # Create risk assessment DataFrame
        risk_assessment = pd.DataFrame({
            'customer_id': range(len(self.X_test)),
            'actual_churn': self.y_test.values,
            'churn_probability': y_pred_proba,
            'predicted_churn': y_pred
        })
        
        # Add risk categories
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
        print(f"   🔴 HIGH RISK customers: {risk_counts.get('HIGH', 0):3d} ({risk_counts.get('HIGH', 0)/total_customers*100:5.1f}%)")
        print(f"   🟡 MEDIUM RISK customers: {risk_counts.get('MEDIUM', 0):3d} ({risk_counts.get('MEDIUM', 0)/total_customers*100:5.1f}%)")
        print(f"   🟢 LOW RISK customers: {risk_counts.get('LOW', 0):3d} ({risk_counts.get('LOW', 0)/total_customers*100:5.1f}%)")
        
        # Show examples from each risk category
        print(f"\n📋 SAMPLE CUSTOMERS BY RISK LEVEL:")
        
        for risk_level in ['HIGH', 'MEDIUM', 'LOW']:
            risk_customers = risk_assessment[risk_assessment['risk_level'] == risk_level]
            if len(risk_customers) > 0:
                print(f"\n{risk_level} RISK Examples (showing top 3):")
                sample_customers = risk_customers.nlargest(3, 'churn_probability') if risk_level == 'HIGH' else risk_customers.head(3)
                
                for _, customer in sample_customers.iterrows():
                    actual_label = "Churned" if customer['actual_churn'] == 1 else "Retained"
                    predicted_label = "Will Churn" if customer['predicted_churn'] == 1 else "Will Retain"
                    
                    print(f"   Customer {customer['customer_id']:3d}: {customer['churn_probability']:.3f} probability "
                          f"(Actual: {actual_label}, Predicted: {predicted_label})")
        
        # Save risk assessment
        risk_assessment.to_csv('./models/customer_risk_assessment.csv', index=False)
        print(f"\n✅ Risk assessment saved to ./models/customer_risk_assessment.csv")
        
        # Create risk distribution visualization
        plt.figure(figsize=(12, 5))
        
        # Risk level distribution
        plt.subplot(1, 2, 1)
        risk_counts.plot(kind='bar', color=['red', 'green', 'orange'])
        plt.title('Customer Risk Distribution')
        plt.xlabel('Risk Level')
        plt.ylabel('Number of Customers')
        plt.xticks(rotation=0)
        
        # Probability distribution
        plt.subplot(1, 2, 2)
        plt.hist(y_pred_proba, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0.4, color='orange', linestyle='--', label='Medium Risk Threshold')
        plt.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
        plt.xlabel('Churn Probability')
        plt.ylabel('Number of Customers')
        plt.title('Churn Probability Distribution')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('./plots/churn_risk_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return risk_assessment
    
    def _create_individual_shap_plots(self, model_name, model, explainer, shap_values, expected_value):
        """Create individual SHAP plots for specific instances with risk assessment"""
        print(f"\n💧 Creating individual SHAP plots for {model_name}...")
        
        # Analyze first 3 samples
        sample_indices = [0, 1, 2]
        
        for idx in sample_indices:
            if idx >= len(self.X_test):
                continue
                
            try:
                # Get sample data
                sample_data = self.X_test.iloc[idx]
                actual_churn = self.y_test.iloc[idx]
                
                # Get model prediction
                pred_proba = model.predict_proba([sample_data])[0]
                predicted_churn = model.predict([sample_data])[0]
                churn_probability = pred_proba[1]
                
                # Determine risk level
                if churn_probability >= 0.7:
                    risk_level = "🔴 HIGH RISK"
                elif churn_probability >= 0.4:
                    risk_level = "🟡 MEDIUM RISK"
                else:
                    risk_level = "🟢 LOW RISK"
                
                print(f"   Sample {idx}: Actual={'Churn' if actual_churn == 1 else 'No Churn'}, "
                      f"Predicted={'Churn' if predicted_churn == 1 else 'No Churn'}, "
                      f"Churn Prob={churn_probability:.3f} ({risk_level})")
                
                # Get SHAP values for this sample
                sample_shap_values = shap_values[idx]
                
                # Create force plot
                try:
                    force_plot = shap.force_plot(
                        expected_value, 
                        sample_shap_values, 
                        sample_data,
                        matplotlib=True,
                        show=False
                    )
                    plt.title(f'{model_name} - SHAP Force Plot (Sample {idx}) - {risk_level}')
                    plt.tight_layout()
                    plt.savefig(f'./plots/shap_force_{model_name.lower()}_sample_{idx}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    
                except Exception as e:
                    print(f"   ⚠️ Could not create force plot for sample {idx}: {str(e)}")
                
                # Create waterfall plot
                try:
                    shap_explanation = shap.Explanation(
                        values=sample_shap_values,
                        base_values=expected_value,
                        data=sample_data,
                        feature_names=self.X_test.columns.tolist()
                    )
                    
                    plt.figure(figsize=(10, 8))
                    shap.plots.waterfall(shap_explanation, max_display=15, show=False)
                    plt.title(f'{model_name} - SHAP Waterfall (Sample {idx})\n'
                             f'Actual: {"Churn" if actual_churn == 1 else "No Churn"}, '
                             f'Risk: {risk_level}, Churn Prob: {churn_probability:.3f}')
                    plt.tight_layout()
                    plt.savefig(f'./plots/shap_waterfall_{model_name.lower()}_sample_{idx}.png', 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    # Print top contributing features with risk interpretation
                    feature_contributions = pd.DataFrame({
                        'feature': self.X_test.columns,
                        'shap_value': sample_shap_values,
                        'feature_value': sample_data.values
                    })
                    feature_contributions['abs_shap'] = abs(feature_contributions['shap_value'])
                    top_features = feature_contributions.nlargest(5, 'abs_shap')
                    
                    print(f"   Top 5 features influencing churn risk for sample {idx}:")
                    for _, row in top_features.iterrows():
                        direction = "→ INCREASES CHURN RISK" if row['shap_value'] > 0 else "→ REDUCES CHURN RISK"
                        impact = "HIGH" if abs(row['shap_value']) > 0.2 else "MEDIUM" if abs(row['shap_value']) > 0.1 else "LOW"
                        print(f"     {row['feature']}: {row['shap_value']:.4f} {direction} [{impact} impact] (value: {row['feature_value']:.3f})")
                    
                except Exception as e:
                    print(f"   ❌ Error creating waterfall plot for sample {idx}: {str(e)}")
                    
            except Exception as e:
                print(f"   ❌ Error processing sample {idx}: {str(e)}")
    
    def lime_analysis(self):
        """Perform LIME analysis"""
        print("\n🍋 LIME ANALYSIS")
        print("=" * 50)
        
        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.X_train.columns.tolist(),
            class_names=['No Churn', 'Churn'],
            mode='classification',
            discretize_continuous=True
        )
        
        # Use first model for LIME analysis
        main_model = list(self.models.values())[0]
        main_model_name = list(self.models.keys())[0]
        
        sample_indices = [0, 1, 2]  # Analyze first 3 samples
        
        for sample_idx in sample_indices:
            if sample_idx >= len(self.X_test):
                continue
                
            sample_instance = self.X_test.iloc[sample_idx].values
            actual_class = self.y_test.iloc[sample_idx]
            
            try:
                lime_exp = self.lime_explainer.explain_instance(
                    sample_instance,
                    main_model.predict_proba,
                    num_features=10,
                    top_labels=1
                )
                
                exp_data = lime_exp.as_list()
                if exp_data:
                    features, importances = zip(*exp_data)
                    
                    plt.figure(figsize=(10, 6))
                    colors = ['red' if imp < 0 else 'green' for imp in importances]
                    plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
                    plt.yticks(range(len(features)), features)
                    plt.xlabel('LIME Importance')
                    plt.title(f'LIME Explanation - Sample {sample_idx} ({main_model_name})')
                    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f'./plots/lime_sample_{sample_idx}.png', dpi=300, bbox_inches='tight')
                    plt.show()
                    
                    pred_proba = main_model.predict_proba([sample_instance])[0]
                    print(f"Sample {sample_idx}: {pred_proba[1]:.3f} probability of churn (Actual: {'Churn' if actual_class == 1 else 'No Churn'})")
                    
            except Exception as e:
                print(f"❌ Error creating LIME explanation for sample {sample_idx}: {str(e)}")
    
    def pdp_analysis(self):
        """Perform Partial Dependence Plot analysis"""
        print("\n📈 PARTIAL DEPENDENCE PLOTS")
        print("=" * 50)
        
        # Use first model for PDP analysis
        main_model = list(self.models.values())[0]
        main_model_name = list(self.models.keys())[0]
        
        # Get top features
        if hasattr(main_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': main_model.feature_importances_
            }).sort_values('importance', ascending=False)
            top_features = feature_importance.head(6)['feature'].tolist()
        else:
            top_features = self.X_train.columns[:6].tolist()
        
        print(f"Analyzing PDP for top features: {top_features}")
        
        # Create subplot grid
        n_features = len(top_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            try:
                pdp_result = pdp.pdp_isolate(
                    model=main_model,
                    dataset=self.X_test,
                    model_features=self.X_test.columns,
                    feature=feature
                )
                
                pdp.pdp_plot(pdp_result, feature, ax=axes[i])
                axes[i].set_title(f'PDP: {feature}')
                
            except Exception as e:
                print(f"❌ Error creating PDP for {feature}: {str(e)}")
                axes[i].text(0.5, 0.5, f'Error: {feature}', 
                            ha='center', va='center', transform=axes[i].transAxes)
        
        # Hide unused subplots
        for i in range(len(top_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'./plots/pdp_analysis_{main_model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_explainers(self):
        """Save all explainers with proper naming for XGBoost models"""
        print("\n💾 Saving explainers and models...")
        
        # Enhanced model mapping to handle your specific model names
        model_mapping = {
            'RandomForest': 'model_rfc.pkl',
            'GradientBoosting': 'model_gb.pkl', 
            'XGBoost': 'model_xgb.pkl',
            'XGBoostClassifier': 'model_xgb.pkl',
            'xgboost_optimized': 'model_xgb.pkl'
        }
        
        # Save models with appropriate names
        for model_name, model in self.models.items():
            # Check if model name matches any pattern
            saved_model = False
            for pattern, filename in model_mapping.items():
                if pattern.lower() in model_name.lower():
                    joblib.dump(model, f'./models/{filename}')
                    print(f"✅ Saved {model_name} as {filename}")
                    saved_model = True
                    break
            
            if not saved_model:
                # Save with original name if no pattern matches
                safe_name = model_name.replace(' ', '_').lower()
                joblib.dump(model, f'./models/model_{safe_name}.pkl')
                print(f"✅ Saved {model_name} as model_{safe_name}.pkl")
        
        # Save SHAP explainers with enhanced mapping
        explainer_mapping = {
            'RandomForest': 'explainer_rfc.bz2',
            'GradientBoosting': 'explainer_gb.bz2',
            'XGBoost': 'explainer_xgb.bz2',
            'XGBoostClassifier': 'explainer_xgb.bz2', 
            'xgboost_optimized': 'explainer_xgb.bz2'
        }
        
        # Save explainers with appropriate names
        for name, explainer in self.shap_explainers.items():
            saved_explainer = False
            for pattern, filename in explainer_mapping.items():
                if pattern.lower() in name.lower():
                    joblib.dump(explainer, f'./models/{filename}', compress=('bz2', 9))
                    print(f"✅ Saved SHAP explainer for {name} as {filename}")
                    saved_explainer = True
                    break
            
            if not saved_explainer:
                # Save with original name if no pattern matches
                safe_name = name.replace(' ', '_').lower()
                joblib.dump(explainer, f'./models/explainer_{safe_name}.bz2', compress=('bz2', 9))
                print(f"✅ Saved SHAP explainer for {name} as explainer_{safe_name}.bz2")
        
        # Save main explainer as explainer_model.bz2 (as requested)
        if self.shap_explainers:
            main_explainer = list(self.shap_explainers.values())[0]
            main_model_name = list(self.shap_explainers.keys())[0]
            
            joblib.dump(main_explainer, './models/explainer_model.bz2', compress=('bz2', 9))
            print("✅ Saved main SHAP explainer as explainer_model.bz2")
            
            # Save backup copies for compatibility
            joblib.dump(main_explainer, './models/explainer_rfc.bz2', compress=('bz2', 9))
            print("✅ Saved backup SHAP explainer as explainer_rfc.bz2")
            
            print(f"🎯 Main explainer is from: {main_model_name}")
        
        # Save LIME explainer
        if self.lime_explainer:
            joblib.dump(self.lime_explainer, './models/lime_explainer.bz2', compress=('bz2', 9))
            print("✅ Saved LIME explainer")
        
        # Save comprehensive feature info
        feature_info = {
            'feature_names': self.X_train.columns.tolist(),
            'n_features': len(self.X_train.columns),
            'target_names': ['No Churn', 'Churn'],
            'models_available': list(self.models.keys()),
            'explainers_available': list(self.shap_explainers.keys()),
            'test_set_size': len(self.X_test),
            'train_set_size': len(self.X_train)
        }
        joblib.dump(feature_info, './models/feature_info.pkl')
        print("✅ Saved comprehensive feature information")
        
        # Save test predictions for verification
        self._save_test_predictions()
    
    def _save_test_predictions(self):
        """Save test set predictions and SHAP values for later analysis"""
        print("\n📊 Saving test predictions and SHAP values...")
        
        predictions_data = {}
        
        for model_name, model in self.models.items():
            try:
                # Get predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)
                
                predictions_data[model_name] = {
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'actual': self.y_test.values
                }
                
                print(f"✅ Saved predictions for {model_name}")
                
            except Exception as e:
                print(f"❌ Error saving predictions for {model_name}: {str(e)}")
        
        # Save predictions
        if predictions_data:
            joblib.dump(predictions_data, './models/test_predictions.pkl')
            print("✅ Saved all test predictions")
        
        print(f"📈 Prediction Summary:")
        for model_name in predictions_data:
            accuracy = accuracy_score(self.y_test, predictions_data[model_name]['predictions'])
            print(f"   {model_name}: {accuracy:.4f} accuracy")
    
    def run_complete_analysis(self):
        """Run the complete explainability analysis pipeline"""
        print("🚀 STARTING EXPLAINABILITY ANALYSIS FOR EXISTING MODELS")
        print("=" * 60)
        
        try:
            # Step 1: Load models first to see what we have
            print("Step 1: Loading existing models...")
            self.load_models()
            
            # Step 2: Load and preprocess data
            print("\nStep 2: Loading and preprocessing data...")
            self.load_data()
            
            # Step 3: Evaluate models
            print("\nStep 3: Evaluating model performance...")
            self.evaluate_models()
            
            # Step 4: Permutation importance
            print("\nStep 4: Running permutation importance analysis...")
            self.permutation_importance_analysis()
            
            # Step 5: SHAP analysis
            print("\nStep 5: Running SHAP analysis...")
            self.shap_analysis()
            
            # Step 5.5: Comprehensive churn risk assessment
            print("\nStep 5.5: Creating comprehensive churn risk assessment...")
            self.create_churn_risk_assessment()
            
            # Step 6: LIME analysis
            print("\nStep 6: Running LIME analysis...")
            self.lime_analysis()
            
            # Step 7: PDP analysis
            print("\nStep 7: Running PDP analysis...")
            self.pdp_analysis()
            
            # Step 8: Save explainers
            print("\nStep 8: Saving explainers for Streamlit...")
            self.save_explainers()
            
            # Final summary
            self._print_final_summary()
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _print_final_summary(self):
        """Print final summary of the analysis"""
        print("\n" + "=" * 60)
        print("🎉 EXPLAINABILITY ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"📊 Models analyzed: {len(self.models)}")
        print(f"📁 Models available: {list(self.models.keys())}")
        
        # Check what files were created
        model_files = []
        explainer_files = []
        plot_files = []
        
        if os.path.exists('./models'):
            for file in os.listdir('./models'):
                if file.endswith('.pkl'):
                    model_files.append(file)
                elif file.endswith('.bz2'):
                    explainer_files.append(file)
        
        if os.path.exists('./plots'):
            plot_files = [f for f in os.listdir('./plots') if f.endswith('.png')]
        
        print(f"\n📁 Files created:")
        print(f"   • Model files ({len(model_files)}): {model_files[:3]}{'...' if len(model_files) > 3 else ''}")
        print(f"   • Explainer files ({len(explainer_files)}): {explainer_files[:3]}{'...' if len(explainer_files) > 3 else ''}")
        print(f"   • Plot files ({len(plot_files)}): {plot_files[:3]}{'...' if len(plot_files) > 3 else ''}")
        
        print("\n🚀 Next steps:")
        print("1. All explainers are ready for churn risk assessment!")
        print("2. Use SHAP to explain why customers are high/medium/low risk")
        print("3. Review customer_risk_assessment.csv for actionable insights")
        print("4. Target high-risk customers for retention campaigns")
        print("5. Use explainer_model.bz2 for real-time risk assessment")
        print("=" * 60)


def main():
    """Main function to run explainability analysis"""
    analyzer = ExplainabilityAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
