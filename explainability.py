# ===============================
# 1. Importing Libraries
# ===============================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, classification_report
from sklearn.utils import class_weight
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import time
import shap
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ===============================
# 2. Configuration
# ===============================
class Config:
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    SHAP_SAMPLE_SIZE = 1000  # For large datasets
    MODEL_TYPE = 'xgboost'  # Options: 'xgboost', 'randomforest'
    
config = Config()

# ===============================
# 3. Data Processing Component
# ===============================
class DataProcessor:
    def __init__(self):
        self.num_imputer = None
        self.cat_imputer = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = None
        self.numerical_cols = None
        self.categorical_cols = None
        
    def load_data(self, file_path):
        """Load and validate dataset"""
        print("📊 Loading data...")
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded: {df.shape}")
        return df
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess data with leak-proof methodology"""
        df_clean = df.copy()
        
        # Drop ID columns
        id_cols = ['individual_id', 'address_id']
        df_clean.drop(columns=id_cols, inplace=True, errors='ignore')
        
        # Handle date columns with fixed reference date
        reference_date = pd.Timestamp('2023-12-31')
        
        if 'cust_orig_date' in df_clean.columns:
            df_clean['cust_orig_date'] = pd.to_datetime(df_clean['cust_orig_date'], errors='coerce')
            df_clean['cust_tenure_days'] = (reference_date - df_clean['cust_orig_date']).dt.days
            df_clean.drop(columns=['cust_orig_date'], inplace=True)
            
        if 'date_of_birth' in df_clean.columns:
            df_clean['date_of_birth'] = pd.to_datetime(df_clean['date_of_birth'], errors='coerce')
            df_clean['age'] = (reference_date - df_clean['date_of_birth']).dt.days // 365
            df_clean.drop(columns=['date_of_birth'], inplace=True)
            
        if 'acct_suspd_date' in df_clean.columns:
            df_clean['acct_suspd_date'] = pd.to_datetime(df_clean['acct_suspd_date'], errors='coerce')
            df_clean['was_suspended'] = df_clean['acct_suspd_date'].notna().astype(int)
            df_clean['days_since_suspension'] = (reference_date - df_clean['acct_suspd_date']).dt.days
            df_clean['days_since_suspension'].fillna(-1, inplace=True)
            df_clean.drop(columns=['acct_suspd_date'], inplace=True)
        
        # Drop high null columns
        null_frac = df_clean.isnull().mean()
        drop_cols = [col for col in df_clean.columns if null_frac[col] > 0.9]
        if drop_cols:
            df_clean.drop(columns=drop_cols, inplace=True)
            print(f"📤 Dropped high-null columns: {drop_cols}")
        
        return df_clean
    
    def prepare_features(self, X_train, X_test, y_train, y_test):
        """Prepare features with proper train-test separation"""
        
        # Identify column types from training data only
        self.numerical_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
        
        print(f"🔢 Numerical columns: {len(self.numerical_cols)}")
        print(f"📝 Categorical columns: {len(self.categorical_cols)}")
        
        # Handle missing values
        if self.numerical_cols:
            self.num_imputer = SimpleImputer(strategy='median')
            X_train[self.numerical_cols] = self.num_imputer.fit_transform(X_train[self.numerical_cols])
            X_test[self.numerical_cols] = self.num_imputer.transform(X_test[self.numerical_cols])
        
        if self.categorical_cols:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            X_train[self.categorical_cols] = self.cat_imputer.fit_transform(X_train[self.categorical_cols])
            X_test[self.categorical_cols] = self.cat_imputer.transform(X_test[self.categorical_cols])
            
            # Encode categorical variables
            for col in self.categorical_cols:
                le = LabelEncoder()
                X_train[col] = le.fit_transform(X_train[col].astype(str))
                
                # Handle unseen categories in test set
                test_categories = set(X_test[col].astype(str).unique())
                train_categories = set(le.classes_)
                unseen_categories = test_categories - train_categories
                
                if unseen_categories:
                    print(f"⚠️  Unseen categories in {col}: {len(unseen_categories)}")
                    X_test[col] = X_test[col].astype(str).apply(
                        lambda x: x if x in train_categories else 'UNSEEN'
                    )
                    if 'UNSEEN' not in le.classes_:
                        le.classes_ = np.append(le.classes_, 'UNSEEN')
                
                X_test[col] = le.transform(X_test[col].astype(str))
                self.label_encoders[col] = le
        
        # Scale numerical features
        if self.numerical_cols:
            self.scaler = StandardScaler()
            X_train[self.numerical_cols] = self.scaler.fit_transform(X_train[self.numerical_cols])
            X_test[self.numerical_cols] = self.scaler.transform(X_test[self.numerical_cols])
        
        self.feature_names = X_train.columns.tolist()
        print(f"✅ Feature preparation completed. Total features: {len(self.feature_names)}")
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_data(self, X_new):
        """Transform new data using fitted preprocessors"""
        X_transformed = X_new.copy()
        
        if self.numerical_cols:
            X_transformed[self.numerical_cols] = self.num_imputer.transform(X_transformed[self.numerical_cols])
            X_transformed[self.numerical_cols] = self.scaler.transform(X_transformed[self.numerical_cols])
        
        if self.categorical_cols:
            X_transformed[self.categorical_cols] = self.cat_imputer.transform(X_transformed[self.categorical_cols])
            for col in self.categorical_cols:
                le = self.label_encoders[col]
                X_transformed[col] = X_transformed[col].astype(str).apply(
                    lambda x: x if x in le.classes_ else 'UNSEEN'
                )
                X_transformed[col] = le.transform(X_transformed[col])
        
        return X_transformed

# ===============================
# 4. Model Training Component
# ===============================
class ChurnModel:
    def __init__(self, model_type='xgboost'):
        self.model = None
        self.model_type = model_type
        self.training_time = None
        self.feature_importance = None
        
    def train(self, X_train, y_train):
        """Train the churn prediction model"""
        print(f"🤖 Training {self.model_type} model...")
        start_time = time.time()
        
        # Handle class imbalance
        neg_count = y_train.value_counts()[0]
        pos_count = y_train.value_counts()[1]
        scale_pos_weight = neg_count / pos_count
        
        print(f"📊 Class balance - Negative: {neg_count}, Positive: {pos_count}")
        print(f"⚖️  Scale pos weight: {scale_pos_weight:.2f}")
        
        if self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            )
        elif self.model_type == 'randomforest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"✅ Model trained in {self.training_time:.2f} seconds")
        
    def predict(self, X):
        """Predict churn probabilities"""
        return self.model.predict_proba(X)[:, 1]
    
    def predict_churn(self, X, threshold=0.5):
        """Predict churn classification"""
        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int), probabilities
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred, y_prob = self.predict_churn(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
        
        print("\n📈 Model Performance:")
        for metric, value in metrics.items():
            print(f"   {metric.capitalize()}: {value:.4f}")
        
        return metrics

# ===============================
# 5. SHAP Explainability Component
# ===============================
class SHAPExplainer:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self, X_background=None, sample_size=1000):
        """Initialize SHAP explainer with background data"""
        print("🔍 Initializing SHAP explainer...")
        
        # Sample background data for efficiency
        if X_background is not None and len(X_background) > sample_size:
            X_background = X_background.sample(sample_size, random_state=config.RANDOM_STATE)
        
        # Use TreeExplainer for tree-based models
        self.explainer = shap.TreeExplainer(self.model, X_background)
        print("✅ SHAP explainer initialized")
        
    def compute_shap_values(self, X):
        """Compute SHAP values for given data"""
        print("🧮 Computing SHAP values...")
        start_time = time.time()
        
        self.shap_values = self.explainer.shap_values(X)
        
        # Handle XGBoost which returns a list of arrays
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use class 1 (churn) values
        
        computation_time = time.time() - start_time
        print(f"✅ SHAP values computed in {computation_time:.2f} seconds")
        
        return self.shap_values
    
    def global_explanations(self, X):
        """Generate global feature importance plots"""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        print("\n🌍 Generating global explanations...")
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, show=False)
        plt.title("SHAP Feature Importance Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Bar plot for mean absolute SHAP values
        plt.figure(figsize=(10, 6))
        shap.summary_plot(self.shap_values, X, feature_names=self.feature_names, plot_type="bar", show=False)
        plt.title("Mean |SHAP Value| Feature Importance", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def individual_explanations(self, X, customer_index=0, top_features=5):
        """Generate individual customer explanations"""
        if self.shap_values is None:
            self.compute_shap_values(X)
        
        print(f"\n👤 Individual explanation for customer {customer_index}:")
        
        # Force plot for individual prediction
        plt.figure(figsize=(12, 4))
        shap.force_plot(
            self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
            self.shap_values[customer_index],
            X.iloc[customer_index],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f"SHAP Force Plot - Customer {customer_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Waterfall plot
        plt.figure(figsize=(12, 8))
        shap.waterfall_plot(
            shap.Explanation(
                values=self.shap_values[customer_index],
                base_values=self.explainer.expected_value[1] if isinstance(self.explainer.expected_value, list) else self.explainer.expected_value,
                data=X.iloc[customer_index],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f"SHAP Waterfall Plot - Customer {customer_index}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Generate reason codes
        reason_codes = self._generate_reason_codes(customer_index, top_features)
        return reason_codes
    
    def _generate_reason_codes(self, customer_index, top_features=5):
        """Generate human-readable reason codes for predictions"""
        feature_contributions = []
        
        for i, feature in enumerate(self.feature_names):
            contribution = self.shap_values[customer_index][i]
            feature_contributions.append({
                'feature': feature,
                'contribution': contribution,
                'abs_contribution': abs(contribution)
            })
        
        # Sort by absolute contribution
        feature_contributions.sort(key=lambda x: x['abs_contribution'], reverse=True)
        
        reason_codes = []
        for i, fc in enumerate(feature_contributions[:top_features]):
            direction = "increases" if fc['contribution'] > 0 else "decreases"
            reason_codes.append({
                'rank': i + 1,
                'feature': fc['feature'],
                'impact': fc['contribution'],
                'reason': f"{fc['feature']} {direction} churn risk"
            })
        
        return reason_codes
    
    def batch_explanations(self, X, customer_indices, output_file=None):
        """Generate explanations for multiple customers"""
        print(f"📊 Generating batch explanations for {len(customer_indices)} customers...")
        
        all_reason_codes = {}
        for idx in customer_indices:
            reason_codes = self.individual_explanations(X, idx, top_features=3)
            all_reason_codes[f"customer_{idx}"] = {
                'prediction_probability': self.model.predict_proba(X.iloc[idx:idx+1])[0, 1],
                'reason_codes': reason_codes
            }
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(all_reason_codes, f, indent=2)
            print(f"✅ Batch explanations saved to {output_file}")
        
        return all_reason_codes

# ===============================
# 6. Churn Prediction Pipeline
# ===============================
class ChurnPredictionPipeline:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.model = None
        self.shap_explainer = None
        self.is_trained = False
        
    def run_pipeline(self, data_path):
        """Run complete churn prediction pipeline"""
        print("🚀 Starting Churn Prediction Pipeline...")
        
        # 1. Load and preprocess data
        df = self.data_processor.load_data(data_path)
        df_clean = self.data_processor.preprocess_data(df, is_training=True)
        
        # 2. Prepare features and target
        TARGET = 'Churn'
        X = df_clean.drop(columns=[TARGET])
        y = df_clean[TARGET]
        
        # 3. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
        )
        
        print(f"📊 Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # 4. Feature engineering
        X_train_processed, X_test_processed, y_train, y_test = self.data_processor.prepare_features(
            X_train, X_test, y_train, y_test
        )
        
        # 5. Train model
        self.model = ChurnModel(model_type=config.MODEL_TYPE)
        self.model.train(X_train_processed, y_train)
        
        # 6. Evaluate model
        metrics = self.model.evaluate(X_test_processed, y_test)
        
        # 7. Initialize SHAP explainer
        self.shap_explainer = SHAPExplainer(self.model.model, self.data_processor.feature_names)
        self.shap_explainer.initialize_explainer(X_train_processed)
        
        self.is_trained = True
        print("✅ Pipeline completed successfully!")
        
        return metrics
    
    def predict_customer_churn(self, customer_data, customer_id=None, top_reasons=5):
        """Predict churn risk for individual customer with explanations"""
        if not self.is_trained:
            raise ValueError("Pipeline must be trained first!")
        
        print(f"🎯 Predicting churn risk for customer {customer_id if customer_id else 'unknown'}...")
        
        # Preprocess customer data
        customer_processed = self.data_processor.transform_new_data(customer_data)
        
        # Predict churn probability
        churn_prob = self.model.predict(customer_processed)[0]
        churn_risk = "High" if churn_prob >= 0.5 else "Low"
        
        # Generate SHAP explanations
        shap_values = self.shap_explainer.compute_shap_values(customer_processed)
        reason_codes = self.shap_explainer._generate_reason_codes(0, top_reasons)
        
        # Create comprehensive result
        result = {
            'customer_id': customer_id,
            'churn_probability': round(churn_prob, 4),
            'churn_risk': churn_risk,
            'prediction_timestamp': datetime.now().isoformat(),
            'reason_codes': reason_codes,
            'feature_contributions': {
                rc['feature']: round(rc['impact'], 4) for rc in reason_codes
            }
        }
        
        print(f"\n📋 Prediction Results:")
        print(f"   Churn Probability: {churn_prob:.4f}")
        print(f"   Churn Risk: {churn_risk}")
        print(f"\n🔍 Top Reasons:")
        for rc in reason_codes:
            print(f"   {rc['rank']}. {rc['reason']} (impact: {rc['impact']:.4f})")
        
        return result
    
    def batch_predict(self, customer_batch, output_file=None):
        """Batch prediction for multiple customers"""
        print(f"📦 Processing batch prediction for {len(customer_batch)} customers...")
        start_time = time.time()
        
        results = []
        for idx, (_, customer_data) in enumerate(customer_batch.iterrows()):
            result = self.predict_customer_churn(
                customer_data.to_frame().T, 
                customer_id=f"batch_{idx}",
                top_reasons=3
            )
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1} customers...")
        
        batch_time = time.time() - start_time
        print(f"✅ Batch prediction completed in {batch_time:.2f} seconds")
        print(f"   Average time per customer: {batch_time/len(customer_batch):.3f} seconds")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"📁 Results saved to {output_file}")
        
        return results
    
    def save_pipeline(self, output_dir='churn_model'):
        """Save complete pipeline"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save model
        joblib.dump(self.model.model, f'{output_dir}/model.pkl')
        
        # Save data processor
        joblib.dump(self.data_processor, f'{output_dir}/data_processor.pkl')
        
        # Save configuration
        config_data = {
            'feature_names': self.data_processor.feature_names,
            'model_type': config.MODEL_TYPE,
            'training_date': datetime.now().isoformat()
        }
        
        with open(f'{output_dir}/config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"💾 Pipeline saved to {output_dir}/")
    
    @classmethod
    def load_pipeline(cls, input_dir='churn_model'):
        """Load saved pipeline"""
        pipeline = cls()
        
        # Load model
        pipeline.model = ChurnModel()
        pipeline.model.model = joblib.load(f'{input_dir}/model.pkl')
        
        # Load data processor
        pipeline.data_processor = joblib.load(f'{input_dir}/data_processor.pkl')
        
        # Initialize SHAP explainer
        with open(f'{input_dir}/config.json', 'r') as f:
            config_data = json.load(f)
        
        pipeline.shap_explainer = SHAPExplainer(
            pipeline.model.model, 
            pipeline.data_processor.feature_names
        )
        pipeline.is_trained = True
        
        print(f"📂 Pipeline loaded from {input_dir}/")
        return pipeline

# ===============================
# 7. Usage Example
# ===============================
def main():
    """Main execution function"""
    
    # Initialize pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Run complete pipeline
    metrics = pipeline.run_pipeline('autoinsurance_churn.csv')
    
    # Generate global explanations
    df = pipeline.data_processor.load_data('autoinsurance_churn.csv')
    df_clean = pipeline.data_processor.preprocess_data(df)
    X_all = df_clean.drop(columns=['Churn'])
    X_processed = pipeline.data_processor.transform_new_data(X_all)
    
    # Sample for SHAP (for efficiency)
    X_sample = X_processed.sample(min(1000, len(X_processed)), random_state=config.RANDOM_STATE)
    
    pipeline.shap_explainer.global_explanations(X_sample)
    
    # Individual prediction example
    sample_customer = X_processed.iloc[0:1]  # First customer
    individual_result = pipeline.predict_customer_churn(
        sample_customer, 
        customer_id="sample_customer_001",
        top_reasons=5
    )
    
    # Generate individual SHAP plots
    pipeline.shap_explainer.individual_explanations(X_sample, customer_index=0)
    
    # Save pipeline for future use
    pipeline.save_pipeline('churn_prediction_pipeline')
    
    return pipeline, individual_result

# ===============================
# 8. Fast Inference Component
# ===============================
class ChurnPredictor:
    """Lightweight predictor for fast inference"""
    
    def __init__(self, pipeline_dir='churn_prediction_pipeline'):
        self.pipeline = ChurnPredictionPipeline.load_pipeline(pipeline_dir)
        
    def predict_fast(self, customer_data, return_reasons=True):
        """Fast prediction with minimal overhead"""
        start_time = time.time()
        
        # Preprocess
        customer_processed = self.pipeline.data_processor.transform_new_data(customer_data)
        
        # Predict
        churn_prob = self.pipeline.model.predict(customer_processed)[0]
        
        # Generate reasons if needed
        reasons = []
        if return_reasons:
            shap_values = self.pipeline.shap_explainer.compute_shap_values(customer_processed)
            reasons = self.pipeline.shap_explainer._generate_reason_codes(0, 3)
        
        inference_time = time.time() - start_time
        
        result = {
            'churn_probability': round(churn_prob, 4),
            'churn_risk': "High" if churn_prob >= 0.5 else "Low",
            'inference_time_seconds': round(inference_time, 4),
            'top_reasons': reasons
        }
        
        return result

# Run the pipeline
if __name__ == "__main__":
    pipeline, results = main()
    
    # Demonstrate fast inference
    print("\n⚡ Fast Inference Demo:")
    fast_predictor = ChurnPredictor()
    
    # Load sample customer
    df = pipeline.data_processor.load_data('autoinsurance_churn.csv')
    df_clean = pipeline.data_processor.preprocess_data(df)
    sample_customer = df_clean.drop(columns=['Churn']).iloc[0:1]
    
    # Fast prediction
    fast_result = fast_predictor.predict_fast(sample_customer)
    print(f"Fast prediction: {fast_result}")