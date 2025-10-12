import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date
import warnings
import time
warnings.filterwarnings('ignore')

# Configure Streamlit page
st.set_page_config(
    page_title="Chubb Churn Detector",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'GradientBoostingClassifier'

if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

# Theme configuration
def get_theme_config():
    if st.session_state.dark_mode:
        return {
            'bg_color': '#000000',
            'secondary_bg': '#1a1a1a',
            'text_color': '#ffffff',
            'accent_color': '#ffffff',
            'primary_color': '#ffffff',
            'success_color': '#4ade80',
            'warning_color': '#fbbf24',
            'danger_color': '#ef4444',
            'info_color': '#60a5fa',
            'card_bg': '#1f1f1f',
            'sidebar_bg': '#0a0a0a'
        }
    else:
        return {
            'bg_color': '#ffffff',
            'secondary_bg': '#f8f9fa',
            'text_color': '#000000',
            'accent_color': '#000000',
            'primary_color': '#000000',
            'success_color': '#22c55e',
            'warning_color': '#f59e0b',
            'danger_color': '#dc2626',
            'info_color': '#3b82f6',
            'card_bg': '#ffffff',
            'sidebar_bg': '#f1f3f4'
        }

def apply_custom_css():
    theme = get_theme_config()
    
    # Dynamic shadow and border opacity based on theme
    shadow_opacity = "0.3" if st.session_state.dark_mode else "0.1"
    border_opacity = "0.3" if st.session_state.dark_mode else "0.1"
    
    st.markdown(f"""
    <style>
        /* Main app styling */
        .stApp {{
            background-color: {theme['bg_color']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Sidebar styling */
        .css-1d391kg, .css-1f3w014, .stSidebar {{
            background-color: {theme['sidebar_bg']} !important;
        }}
        
        /* Force sidebar text color */
        .stSidebar .stMarkdown, .stSidebar label {{
            color: {theme['text_color']} !important;
        }}
        
        /* Custom card styling */
        .custom-card {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
            padding: 1.5rem;
            border-radius: 15px;
            box-shadow: {('0 8px 16px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '0 8px 16px rgba(0,0,0,0.1)')};
            margin: 1rem 0;
            border: {('1px solid #333333' if st.session_state.dark_mode else '1px solid #e5e7eb')};
            transition: all 0.3s ease;
        }}
        
        .custom-card:hover {{
            transform: translateY(-2px);
            box-shadow: {('0 12px 24px rgba(255,255,255,0.15)' if st.session_state.dark_mode else '0 12px 24px rgba(0,0,0,0.15)')};
        }}
        
        /* Profile card with dynamic theming */
        .profile-card {{
            background: {('linear-gradient(135deg, #1a1a1a, #2a2a2a)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa, #ffffff)')} !important;
            color: {theme['text_color']} !important;
            padding: 2rem;
            border-radius: 20px;
            box-shadow: {('0 12px 24px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '0 12px 24px rgba(0,0,0,0.1)')};
            margin: 1rem 0;
            border: {('2px solid #333333' if st.session_state.dark_mode else '2px solid #e5e7eb')};
        }}
        
        /* Fix profile card content */
        .profile-card * {{
            color: inherit !important;
        }}
        
        .profile-card h2 {{
            color: {theme['accent_color']} !important;
        }}
        
        .profile-card p {{
            color: {theme['text_color']} !important;
        }}
        
        /* Metric cards with theme colors */
        .metric-card {{
            background: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
            padding: 1rem !important;
            border-radius: 10px !important;
            text-align: center !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, {shadow_opacity}) !important;
            margin: 0.5rem 0 !important;
            border: 1px solid rgba(0, 0, 0, {border_opacity}) !important;
        }}
        
        .metric-card h4 {{
            color: {theme['accent_color']} !important;
            margin: 0 !important;
        }}
        
        .metric-card p {{
            color: {theme['text_color']} !important;
            margin: 0 !important;
            font-size: 1.2rem !important;
            font-weight: bold !important;
        }}
        
        /* Risk level cards with proper theming */
        .risk-extreme {{
            background: {('linear-gradient(135deg, #2d1b1b, #3d2626)' if st.session_state.dark_mode else 'linear-gradient(135deg, #fef2f2, #fee2e2)')} !important;
            border-left: 5px solid {theme['danger_color']};
            color: {theme['danger_color']} !important;
            border: {('1px solid #4d2626' if st.session_state.dark_mode else '1px solid #fecaca')};
        }}
        
        .risk-high {{
            background: {('linear-gradient(135deg, #2d251b, #3d3226)' if st.session_state.dark_mode else 'linear-gradient(135deg, #fffbeb, #fef3c7)')} !important;
            border-left: 5px solid {theme['warning_color']};
            color: {theme['warning_color']} !important;
            border: {('1px solid #4d3d26' if st.session_state.dark_mode else '1px solid #fed7aa')};
        }}
        
        .risk-medium {{
            background: {('linear-gradient(135deg, #1f1f2e, #2f2f3e)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8fafc, #f1f5f9)')} !important;
            border-left: 5px solid #8b5cf6;
            color: {'#a78bfa' if st.session_state.dark_mode else '#7c3aed'} !important;
            border: {('1px solid #3f3f4e' if st.session_state.dark_mode else '1px solid #e2e8f0')};
        }}
        
        .risk-low {{
            background: {('linear-gradient(135deg, #1b2d1b, #263d26)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f0fdf4, #dcfce7)')} !important;
            border-left: 5px solid {theme['success_color']};
            color: {theme['success_color']} !important;
            border: {('1px solid #2d4d2d' if st.session_state.dark_mode else '1px solid #bbf7d0')};
        }}
        
        /* Header styling */
        .main-header {{
            font-size: 2.5rem;
            color: {theme['accent_color']} !important;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, {shadow_opacity});
        }}
        
        .section-header {{
            font-size: 1.5rem;
            color: {theme['accent_color']} !important;
            margin: 1.5rem 0 1rem 0;
            font-weight: 600;
            border-bottom: 2px solid {theme['accent_color']};
            padding-bottom: 0.5rem;
        }}
        
        /* Button styling */
        .stButton > button {{
            background: {('linear-gradient(135deg, #1f1f1f, #333333)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa, #e9ecef)')} !important;
            color: {theme['text_color']} !important;
            border: {('2px solid #ffffff' if st.session_state.dark_mode else '2px solid #000000')} !important;
            border-radius: 25px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: {('0 4px 8px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '0 4px 8px rgba(0,0,0,0.1)')} !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: {('0 8px 16px rgba(255,255,255,0.2)' if st.session_state.dark_mode else '0 8px 16px rgba(0,0,0,0.2)')} !important;
            background: {('linear-gradient(135deg, #333333, #4d4d4d)' if st.session_state.dark_mode else 'linear-gradient(135deg, #e9ecef, #dee2e6)')} !important;
        }}
        
        /* Override Streamlit primary button styling */
        .stButton > button[kind="primary"] {{
            background: {('linear-gradient(135deg, #2a2a2a, #404040)' if st.session_state.dark_mode else 'linear-gradient(135deg, #ffffff, #f0f0f0)')} !important;
            color: {theme['text_color']} !important;
            border: {('3px solid #ffffff' if st.session_state.dark_mode else '3px solid #000000')} !important;
            border-radius: 30px !important;
            padding: 1rem 2.5rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
            box-shadow: {('0 6px 12px rgba(255,255,255,0.15)' if st.session_state.dark_mode else '0 6px 12px rgba(0,0,0,0.15)')} !important;
            text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
        }}
        
        .stButton > button[kind="primary"]:hover {{
            transform: translateY(-3px) !important;
            box-shadow: {('0 12px 24px rgba(255,255,255,0.25)' if st.session_state.dark_mode else '0 12px 24px rgba(0,0,0,0.25)')} !important;
            background: {('linear-gradient(135deg, #404040, #5a5a5a)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f0f0f0, #e0e0e0)')} !important;
            border-color: {('#cccccc' if st.session_state.dark_mode else '#333333')} !important;
        }}
        
        /* Enhanced expandable sections */
        .stExpander {{
            background: {('linear-gradient(135deg, #1a1a1a, #2a2a2a)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa, #ffffff)')} !important;
            border: {('2px solid #333333' if st.session_state.dark_mode else '2px solid #e5e7eb')} !important;
            border-radius: 15px !important;
            margin: 1rem 0 !important;
            box-shadow: {('0 8px 16px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '0 8px 16px rgba(0,0,0,0.1)')} !important;
            overflow: hidden !important;
        }}
        
        .streamlit-expanderHeader {{
            background: {('linear-gradient(135deg, #2a2a2a, #3a3a3a)' if st.session_state.dark_mode else 'linear-gradient(135deg, #ffffff, #f8f9fa)')} !important;
            color: {theme['text_color']} !important;
            border: none !important;
            padding: 1.5rem !important;
            font-weight: 700 !important;
            font-size: 1.1rem !important;
            transition: all 0.3s ease !important;
        }}
        
        .streamlit-expanderHeader:hover {{
            background: {('linear-gradient(135deg, #3a3a3a, #4a4a4a)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa, #e9ecef)')} !important;
        }}
        
        .streamlit-expanderContent {{
            background: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
            padding: 2rem !important;
            border-top: {('1px solid #333333' if st.session_state.dark_mode else '1px solid #e5e7eb')} !important;
        }}
        
        /* Force all expander text to follow theme */
        .streamlit-expanderContent * {{
            color: {theme['text_color']} !important;
        }}
        
        .streamlit-expanderContent h1, .streamlit-expanderContent h2, 
        .streamlit-expanderContent h3, .streamlit-expanderContent h4 {{
            color: {theme['text_color']} !important;
            font-weight: 600 !important;
        }}
        
        .streamlit-expanderContent strong {{
            color: {theme['text_color']} !important;
        }}
        }}
        
        .stExpander > div > div {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Sidebar input styling with better contrast */
        .stSelectbox label, .stNumberInput label, .stDateInput label {{
            color: {theme['text_color']} !important;
            font-weight: 500;
        }}
        
        /* Input fields with proper contrast */
        .stSelectbox > div > div, .stNumberInput > div > div, .stDateInput > div > div {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
            border: 1px solid {theme['accent_color']} !important;
        }}
        
        /* Input field text */
        .stSelectbox input, .stNumberInput input, .stDateInput input {{
            color: {theme['text_color']} !important;
            background-color: {theme['card_bg']} !important;
        }}
        
        /* Selectbox options */
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Dropdown menu */
        .stSelectbox div[role="listbox"] {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Dropdown options */
        .stSelectbox div[role="option"] {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
        }}
        
        .stSelectbox div[role="option"]:hover {{
            background-color: {theme['accent_color']} !important;
            color: white !important;
        }}
        
        /* Expander content with proper contrast */
        .streamlit-expanderContent {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
        }}
        
        /* Expander content text */
        .streamlit-expanderContent p, .streamlit-expanderContent div, 
        .streamlit-expanderContent li, .streamlit-expanderContent span {{
            color: {theme['text_color']} !important;
        }}
        
        /* Expander headers */
        .streamlit-expanderHeader {{
            background-color: {theme['card_bg']} !important;
            color: {theme['accent_color']} !important;
            border: 1px solid {theme['accent_color']} !important;
        }}
        
        /* Risk factor boxes */
        .risk-factor-box {{
            background-color: {theme['card_bg']} !important;
            color: {theme['text_color']} !important;
            padding: 0.8rem !important;
            margin: 0.5rem 0 !important;
            border-radius: 8px !important;
            border: 1px solid {theme['accent_color']} !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        }}
        
        .risk-factor-positive {{
            border-left: 4px solid #dc2626 !important;
            background: {'#2d1b1b' if st.session_state.dark_mode else '#fef2f2'} !important;
        }}
        
        .risk-factor-negative {{
            border-left: 4px solid #059669 !important;
            background: {'#1b2d1b' if st.session_state.dark_mode else '#f0fdf4'} !important;
        }}
        
        /* Professional header styling */
        .chubb-header {{
            background: {('linear-gradient(135deg, #1a1a1a, #333333)' if st.session_state.dark_mode else 'linear-gradient(135deg, #f8f9fa, #ffffff)')};
            color: {theme['text_color']};
            padding: 3rem 2rem;
            border-radius: 20px;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: {('0 20px 40px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '0 20px 40px rgba(0,0,0,0.15)')};
            border: {('2px solid #333333' if st.session_state.dark_mode else '2px solid #e5e7eb')};
            position: relative;
            overflow: hidden;
        }}
        
        .chubb-header::before {{
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: {('radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%)' if st.session_state.dark_mode else 'radial-gradient(circle, rgba(0,0,0,0.05) 0%, transparent 70%)')};
            animation: rotate 20s linear infinite;
            z-index: 0;
        }}
        
        @keyframes rotate {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .chubb-header h1 {{
            font-size: 3rem;
            font-weight: 900;
            margin: 0;
            color: {theme['text_color']};
            text-shadow: {('2px 2px 8px rgba(255,255,255,0.1)' if st.session_state.dark_mode else '2px 2px 8px rgba(0,0,0,0.1)')};
            position: relative;
            z-index: 1;
            letter-spacing: 2px;
        }}
        
        .chubb-header p {{
            font-size: 1.2rem;
            margin: 1rem 0 0 0;
            color: {theme['text_color']};
            opacity: 0.8;
            position: relative;
            z-index: 1;
            font-weight: 500;
        }}
        
        /* Add elevation effect */
        .chubb-header:hover {{
            transform: translateY(-5px);
            box-shadow: {('0 25px 50px rgba(255,255,255,0.15)' if st.session_state.dark_mode else '0 25px 50px rgba(0,0,0,0.2)')};
            transition: all 0.3s ease;
        }}
        
        /* Force text colors in all elements */
        div[data-testid="stMarkdownContainer"] p, 
        div[data-testid="stMarkdownContainer"] div,
        div[data-testid="stMarkdownContainer"] span {{
            color: {theme['text_color']} !important;
        }}
        
        /* Profile card text override */
        .profile-card div, .profile-card p, .profile-card strong {{
            color: {theme['text_color']} !important;
        }}
        
        .profile-card h2 {{
            color: {theme['accent_color']} !important;
        }}
        
        /* Animation for loading */
        @keyframes pulse {{
            0% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
            100% {{ opacity: 1; }}
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        /* Theme toggle positioning */
        .theme-toggle {{
            position: fixed !important;
            top: 1rem !important;
            right: 1rem !important;
            z-index: 999 !important;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load both models and return them"""
    models = {}
    
    # Try to load GradientBoosting model
    try:
        gb_model = joblib.load('./models/GradientBoostingClassifier_churn_prediction_model.pkl')
        models['GradientBoostingClassifier'] = gb_model
    except:
        st.warning("GradientBoostingClassifier model not found")
    
    # Try to load XGBoost model
    try:
        xgb_model = joblib.load('./models/XGBoostClassifier_churn_prediction_model.pkl')
        models['XGBoostClassifier'] = xgb_model
    except:
        try:
            xgb_model = joblib.load('./models/churn_xgboost_optimized.pkl')
            models['XGBoostClassifier'] = xgb_model
        except:
            st.warning("XGBoostClassifier model not found")
    
    if not models:
        st.error("No models could be loaded!")
        return {}
    
    return models

def preprocess_input(input_data):
    """Preprocess the input data to match model expectations"""
    today = pd.Timestamp('today')
    
    # Remove any ID fields
    id_fields = ['individual_id', 'address_id', 'customer_id', 'id']
    for field in id_fields:
        input_data.pop(field, None)
    
    # Calculate age from date_of_birth
    if 'date_of_birth' in input_data:
        birth_date = pd.to_datetime(input_data['date_of_birth'])
        input_data['age'] = (today - birth_date).days // 365
    
    # Calculate days since customer origination
    if 'cust_orig_date' in input_data:
        orig_date = pd.to_datetime(input_data['cust_orig_date'])
        input_data['cust_orig_days_since'] = (today - orig_date).days
    
    # NEW METRICS CALCULATION
    # 1. Premium to Income Ratio
    if 'curr_ann_amt' in input_data and 'income' in input_data:
        if input_data['income'] > 0:
            input_data['premium_to_income_ratio'] = input_data['curr_ann_amt'] / input_data['income']
        else:
            input_data['premium_to_income_ratio'] = 0
    
    # 2. Income Adequacy Score (based on income percentiles and local cost of living)
    if 'income' in input_data:
        # Define income adequacy thresholds (can be adjusted based on business rules)
        income = input_data['income']
        if income >= 75000:
            input_data['income_adequacy_score'] = 5  # Excellent
        elif income >= 50000:
            input_data['income_adequacy_score'] = 4  # Good
        elif income >= 35000:
            input_data['income_adequacy_score'] = 3  # Fair
        elif income >= 25000:
            input_data['income_adequacy_score'] = 2  # Poor
        else:
            input_data['income_adequacy_score'] = 1  # Very Poor
    
    # 3. Customer Tenure (Reference Date)
    if 'cust_orig_date' in input_data:
        orig_date = pd.to_datetime(input_data['cust_orig_date'])
        reference_date = pd.Timestamp('2024-01-01')  # Reference date
        input_data['customer_tenure_reference'] = (reference_date - orig_date).days
    
    # Remove the original date columns
    input_data.pop('date_of_birth', None)
    input_data.pop('cust_orig_date', None)
    
    # Handle categorical variables
    categorical_columns = ['city', 'state', 'county', 'marital_status', 'home_market_value']
    
    for col in categorical_columns:
        if col in input_data:
            if col == 'marital_status':
                input_data[f'{col}_Single'] = 1 if input_data[col] == 'Single' else 0
            elif col == 'city':
                cities = ['Kaufman', 'Grand Prairie', 'Dallas', 'Terrell', 'Royse City']
                for city in cities:
                    input_data[f'{col}_{city}'] = 1 if input_data[col] == city else 0
            elif col == 'home_market_value':
                ranges = ['50000 - 74999', '75000 - 99999', '100000 - 124999']
                for range_val in ranges:
                    input_data[f'{col}_{range_val}'] = 1 if input_data[col] == range_val else 0
    
    # Remove original categorical columns
    for col in categorical_columns:
        input_data.pop(col, None)
    
    return input_data

def create_gauge_chart(probability):
    """Create a gauge chart for churn probability"""
    theme = get_theme_config()
    
    if probability >= 0.7:
        color = theme['danger_color']
        risk_level = "EXTREME RISK"
    elif probability >= 0.5:
        color = theme['warning_color'] 
        risk_level = "HIGH RISK"
    elif probability >= 0.25:
        color = "#8b5cf6"
        risk_level = "MEDIUM RISK"
    else:
        color = theme['success_color']
        risk_level = "LOW RISK"
    
    # Dynamic background colors based on theme
    bg_colors = {
        'step1': '#f0fdf4' if not st.session_state.dark_mode else '#1b2d1b',
        'step2': '#fffbeb' if not st.session_state.dark_mode else '#2d251b', 
        'step3': '#fef2f2' if not st.session_state.dark_mode else '#2d1b1b',
        'step4': '#fee2e2' if not st.session_state.dark_mode else '#3d2626'
    }
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"<b>Churn Probability</b><br><span style='font-size:0.8em;color:{color}'>{risk_level}</span>"},
        delta = {'reference': 50, 'increasing': {'color': color}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': theme['text_color']},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': theme['card_bg'],
            'borderwidth': 2,
            'bordercolor': theme['accent_color'],
            'steps': [
                {'range': [0, 25], 'color': bg_colors['step1']},
                {'range': [25, 50], 'color': bg_colors['step2']},
                {'range': [50, 70], 'color': bg_colors['step3']},
                {'range': [70, 100], 'color': bg_colors['step4']}
            ],
            'threshold': {
                'line': {'color': theme['danger_color'], 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': theme['text_color']}
    )
    return fig

def create_shap_explanation(model, input_df, model_name):
    """Create SHAP explanation for the prediction"""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        
        if isinstance(shap_values, list):
            shap_values_positive = shap_values[1]
            expected_value = explainer.expected_value[1]
        else:
            shap_values_positive = shap_values
            expected_value = explainer.expected_value
        
        feature_importance = pd.DataFrame({
            'feature': input_df.columns,
            'shap_value': shap_values_positive[0],
            'abs_shap_value': np.abs(shap_values_positive[0])
        }).sort_values('abs_shap_value', ascending=False).head(15)
        
        return feature_importance
        
    except Exception as e:
        st.error(f"Error creating SHAP explanation: {e}")
        return None

def display_user_profile(input_data):
    """Display user profile in a professional layout"""
    theme = get_theme_config()
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
    
    # Calculate age
    today = pd.Timestamp('today')
    birth_date = pd.to_datetime(input_data['date_of_birth'])
    age = (today - birth_date).days // 365
    
    # Calculate tenure
    orig_date = pd.to_datetime(input_data['cust_orig_date'])
    tenure_years = (today - orig_date).days // 365
    
    # Use streamlit columns for better layout
    with st.container():
        # Profile header
        st.markdown(f"""
        <div style="background: {theme['secondary_bg']}; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem; color: {theme['text_color']};">
            <h3 style="margin: 0; color: {theme['accent_color']};">Customer Overview</h3>
            <p style="margin: 0; color: {theme['text_color']};">Risk Assessment Profile</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate new metrics for display
        premium_to_income_ratio = input_data['curr_ann_amt'] / input_data['income'] if input_data['income'] > 0 else 0
        income_adequacy_score = 5 if input_data['income'] >= 75000 else (4 if input_data['income'] >= 50000 else (3 if input_data['income'] >= 35000 else (2 if input_data['income'] >= 25000 else 1)))
        reference_date = pd.Timestamp('2024-01-01')
        customer_tenure_reference = (reference_date - pd.to_datetime(input_data['cust_orig_date'])).days
        
        # Metrics in columns (expanded to 6 columns for new metrics)
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Age</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {theme['text_color']};">{age} years</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Tenure</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {theme['text_color']};">{tenure_years} years</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Annual Premium</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {theme['text_color']};">${input_data['curr_ann_amt']:,.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Income</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {theme['text_color']};">${input_data['income']:,.0f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col5:
            # Premium to Income Ratio with color coding
            ratio_color = theme['danger_color'] if premium_to_income_ratio > 0.05 else (theme['warning_color'] if premium_to_income_ratio > 0.03 else theme['success_color'])
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Premium/Income</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {ratio_color};">{premium_to_income_ratio:.1%}</p>
                <small style="color: {theme['text_color']};">Ratio</small>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            # Income Adequacy Score with color coding
            adequacy_color = theme['success_color'] if income_adequacy_score >= 4 else (theme['warning_color'] if income_adequacy_score >= 3 else theme['danger_color'])
            adequacy_labels = {5: "Excellent", 4: "Good", 3: "Fair", 2: "Poor", 1: "Very Poor"}
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; text-align: center; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); border: 1px solid {theme['accent_color']};">
                <h4 style="margin: 0; color: {theme['accent_color']};">Income Score</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold; color: {adequacy_color};">{income_adequacy_score}/5</p>
                <small style="color: {theme['text_color']};">{adequacy_labels[income_adequacy_score]}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Customer Tenure Reference metric
        st.markdown(f"""
        <div style="background: {theme['secondary_bg']}; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; color: {theme['text_color']};">
            <h4 style="margin: 0 0 1rem 0; color: {theme['accent_color']};">Customer Tenure Reference (Jan 1, 2024)</h4>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <p style="margin: 0; font-size: 1.5rem; font-weight: bold; color: {theme['text_color']};">{customer_tenure_reference:,} days</p>
                    <p style="margin: 0; color: {theme['text_color']};">({customer_tenure_reference/365:.1f} years from reference date)</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Personal details in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div style="background: {theme['secondary_bg']}; padding: 1rem; border-radius: 8px; color: {theme['text_color']};">
                <strong>Location:</strong> {input_data['city']}, {input_data['state']}<br>
                <strong>Marital Status:</strong> {input_data['marital_status']}
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background: {theme['secondary_bg']}; padding: 1rem; border-radius: 8px; color: {theme['text_color']};">
                <strong>Home Owner:</strong> {'Yes' if input_data['home_owner'] else 'No'}<br>
                <strong>College Degree:</strong> {'Yes' if input_data['college_degree'] else 'No'}
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div style="background: {theme['secondary_bg']}; padding: 1rem; border-radius: 8px; color: {theme['text_color']};">
                <strong>Good Credit:</strong> {'Yes' if input_data['good_credit'] else 'No'}<br>
                <strong>Has Children:</strong> {'Yes' if input_data['has_children'] else 'No'}
            </div>
            """, unsafe_allow_html=True)
        
        # Risk Metrics Summary
        with st.expander("Risk Metrics Summary", expanded=False):
            st.markdown(f"""
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; text-align: center; margin: 1rem 0;">
                <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 8px; border: 1px solid {theme['accent_color']};">
                    <h5 style="margin: 0; color: {theme['text_color']};">Premium/Income</h5>
                    <p style="margin: 0; color: {ratio_color}; font-weight: bold; font-size: 1.2rem;">{premium_to_income_ratio:.1%}</p>
                    <small style="color: {theme['text_color']}; opacity: 0.8;">Higher ratio increases churn risk</small>
                </div>
                <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 8px; border: 1px solid {theme['accent_color']};">
                    <h5 style="margin: 0; color: {theme['text_color']};">Income Score</h5>
                    <p style="margin: 0; color: {adequacy_color}; font-weight: bold; font-size: 1.2rem;">{income_adequacy_score}/5</p>
                    <small style="color: {theme['text_color']}; opacity: 0.8;">Lower scores indicate higher churn risk</small>
                </div>
                <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 8px; border: 1px solid {theme['accent_color']};">
                    <h5 style="margin: 0; color: {theme['text_color']};">Tenure</h5>
                    <p style="margin: 0; color: {theme['info_color']}; font-weight: bold; font-size: 1.2rem;">{(customer_tenure_reference/365):.1f}y</p>
                    <small style="color: {theme['text_color']}; opacity: 0.8;">Longer tenure reduces churn risk</small>
                </div>
            </div>
            """, unsafe_allow_html=True)

def display_prediction_results(prediction_proba, model_name, feature_importance):
    """Display prediction results with professional analysis"""
    theme = get_theme_config()
    
    # Risk level determination
    if prediction_proba >= 0.7:
        risk_level = "EXTREME RISK"
        risk_color = "#dc2626"
        css_class = "risk-extreme"
        risk_description = "Immediate intervention required - customer likely to churn within 30 days"
    elif prediction_proba >= 0.5:
        risk_level = "HIGH RISK"
        risk_color = "#d97706"
        css_class = "risk-high"
        risk_description = "High probability of churn - proactive retention needed within 60 days"
    elif prediction_proba >= 0.25:
        risk_level = "MEDIUM RISK"
        risk_color = "#9333ea"
        css_class = "risk-medium"
        risk_description = "Moderate risk - monitor closely and engage with value-added services"
    else:
        risk_level = "LOW RISK"
        risk_color = "#059669"
        css_class = "risk-low"
        risk_description = "Low churn probability - maintain relationship and explore upsell opportunities"
    
    st.markdown('<div class="section-header">Churn Risk Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Risk summary card
        risk_html = f"""
        <div class="custom-card {css_class}">
            <div style="text-align: center;">
                <h2 style="margin: 0; color: {risk_color};">{risk_level}</h2>
                <h1 style="margin: 0.5rem 0; font-size: 3rem; color: {risk_color};">{prediction_proba:.1%}</h1>
                <p style="margin: 0; font-weight: 500; color: {theme['text_color']};">Churn Probability</p>
                <hr style="margin: 1rem 0;">
                <p style="margin: 0; font-style: italic; color: {theme['text_color']};">{risk_description}</p>
                <div style="margin-top: 1rem; padding: 0.5rem; background: rgba(0,0,0,0.1); border-radius: 5px;">
                    <strong style="color: {theme['text_color']};">Model:</strong> <span style="color: {theme['text_color']};">{model_name}</span>
                </div>
            </div>
        </div>
        """
        st.markdown(risk_html, unsafe_allow_html=True)
    
    with col2:
        # Gauge chart
        gauge_fig = create_gauge_chart(prediction_proba)
        st.plotly_chart(gauge_fig, width='stretch')

def display_risk_factors(feature_importance):
    """Display risk factors analysis with enhanced styling"""
    theme = get_theme_config()
    
    with st.expander("Risk Factors Analysis", expanded=False):
        if feature_importance is not None:
            # Prepare data for plotting
            plot_data = feature_importance.head(10).copy()
            plot_data['Impact Type'] = plot_data['shap_value'].apply(lambda x: 'Increases Risk' if x > 0 else 'Decreases Risk')
            
            fig = px.bar(
                plot_data,
                x='shap_value',
                y='feature',
                color='Impact Type',
                color_discrete_map={'Increases Risk': theme['danger_color'], 'Decreases Risk': theme['success_color']},
                title='Top 10 Risk Factors Impact',
                labels={'shap_value': 'SHAP Impact Value', 'feature': 'Features'}
            )
            
            fig.update_layout(
                height=400,
                yaxis={'categoryorder': 'total ascending', 'tickfont': {'color': theme['text_color']}, 'titlefont': {'color': theme['text_color']}},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font={'color': theme['text_color']},
                title_font_color=theme['text_color'],
                xaxis={'tickfont': {'color': theme['text_color']}, 'titlefont': {'color': theme['text_color']}},
                legend={'font': {'color': theme['text_color']}}
            )
            
            st.plotly_chart(fig, width='stretch')
        else:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 2rem; border-radius: 10px; text-align: center;
                        border: {('1px solid #333333' if st.session_state.dark_mode else '1px solid #e5e7eb')};">
                <h4 style="color: {theme['warning_color']};">Analysis Unavailable</h4>
                <p style="color: {theme['text_color']};">Unable to generate risk factor analysis for this prediction.</p>
            </div>
            """, unsafe_allow_html=True)

def display_retention_strategies(prediction_proba, input_data=None):
    """Display retention strategies and recommendations with enhanced styling"""
    theme = get_theme_config()
    
    # Calculate new metrics for strategy recommendations
    if input_data:
        premium_to_income_ratio = input_data['curr_ann_amt'] / input_data['income'] if input_data['income'] > 0 else 0
        income_adequacy_score = 5 if input_data['income'] >= 75000 else (4 if input_data['income'] >= 50000 else (3 if input_data['income'] >= 35000 else (2 if input_data['income'] >= 25000 else 1)))
        reference_date = pd.Timestamp('2024-01-01')
        customer_tenure_reference = (reference_date - pd.to_datetime(input_data['cust_orig_date'])).days
        tenure_years = customer_tenure_reference / 365
    
    with st.expander("Retention Strategies", expanded=False):
        # Risk level assessment
        if prediction_proba >= 0.7:
            risk_level = "EXTREME"
            risk_color = theme['danger_color']
            urgency = "Immediate Action (24-48 hours)"
        elif prediction_proba >= 0.5:
            risk_level = "HIGH"
            risk_color = theme['warning_color']
            urgency = "Proactive Measures (3-7 days)"
        elif prediction_proba >= 0.25:
            risk_level = "MEDIUM"
            risk_color = theme['info_color']
            urgency = "Scheduled Monitoring (2-4 weeks)"
        else:
            risk_level = "LOW"
            risk_color = theme['success_color']
            urgency = "Regular Maintenance (Quarterly)"
        
        # Risk level indicator
        st.markdown(f"""
        <div style="background: {theme['card_bg']}; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
                    border-left: 5px solid {risk_color};">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h4 style="margin: 0; color: {theme['text_color']};">Risk Level: 
                        <span style="color: {risk_color};">{risk_level}</span>
                    </h4>
                    <p style="margin: 0.5rem 0 0 0; color: {theme['text_color']}; opacity: 0.8;">{urgency}</p>
                </div>
                <div style="font-size: 1.5rem;">
                    {'!' if risk_level == 'EXTREME' else '!' if risk_level == 'HIGH' else '!' if risk_level == 'MEDIUM' else '✓'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if prediction_proba >= 0.7:
            strategies = [
                "Personal call from senior relationship manager",
                "Offer 20-30% premium discount for next renewal",
                "Expedite any pending claims or service requests",
                "Premium reduction or payment plan options"
            ]
        elif prediction_proba >= 0.5:
            strategies = [
                "Outbound satisfaction survey call",
                "Policy benefits and usage review",
                "10-15% renewal discount",
                "Monthly engagement tracking"
            ]
        elif prediction_proba >= 0.25:
            strategies = [
                "Quarterly check-in calls",
                "Annual policy review meeting",
                "New product announcements",
                "Coverage gap analysis"
            ]
        else:
            strategies = [
                "Reference customer program",
                "Additional coverage consultation",
                "VIP customer recognition",
                "Exclusive events and experiences"
            ]
        
        # Display strategies in simple list
        for action in strategies:
            st.markdown(f"""
            <div style="background: {theme['card_bg']}; padding: 0.8rem; margin: 0.5rem 0; 
                       border-radius: 8px; border-left: 4px solid {theme['primary_color']};">
                <span style="color: {theme['text_color']};">{action}</span>
            </div>
            """, unsafe_allow_html=True)

def main():
    # Apply custom CSS
    apply_custom_css()
    
    # Theme toggle button in a cleaner position
    col1, col2, col3 = st.columns([6, 1, 1])
    with col3:
        if st.button("🌓 Toggle Theme" if st.session_state.dark_mode else "🌙 Dark Mode", help="Switch between Light/Dark mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
    
    # Header with professional Chubb branding
    st.markdown("""
    <div class="chubb-header">
        <h1>Chubb Churn Detector</h1>
        <p>Advanced AI-Powered Customer Risk Assessment Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if not models:
        st.error(" No models available. Please check the models directory.")
        return
    
    # Sidebar - 30% width for inputs
    with st.sidebar:
        st.markdown('<div class="section-header">Customer Assessment</div>', unsafe_allow_html=True)
        
        # Model selection
        st.markdown("### AI Model Selection")
        selected_model = st.selectbox(
            "Choose Prediction Model:",
            list(models.keys()),
            index=0,
            help="Select the machine learning model for prediction"
        )
        st.session_state.selected_model = selected_model
        
        st.markdown("---")
        
        # Customer input form
        st.markdown("### Financial Information")
        curr_ann_amt = st.number_input("Annual Premium ($)", min_value=0.0, value=818.88, step=10.0)
        income = st.number_input("Annual Income ($)", min_value=0.0, value=22500.0, step=1000.0)
        
        st.markdown("### Important Dates")
        cust_orig_date = st.date_input("Customer Since", value=date(2018, 12, 9))
        date_of_birth = st.date_input("Date of Birth", value=date(1978, 6, 23))
        
        st.markdown("### Demographics")
        days_tenure = st.number_input("Tenure (days)", min_value=0, value=1454, step=1)
        age_in_years = st.number_input("Age (years)", min_value=18, max_value=100, value=44, step=1)
        has_children = st.selectbox("Has Children", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        length_of_residence = st.number_input("Residence Length (years)", min_value=0, value=15, step=1)
        
        st.markdown("### Location")
        latitude = st.number_input("Latitude", value=32.578829, format="%.6f")
        longitude = st.number_input("Longitude", value=-96.305006, format="%.6f")
        city = st.selectbox("City", ["Kaufman", "Grand Prairie", "Dallas", "Terrell", "Royse City", "Other"], index=0)
        state = st.selectbox("State", ["TX", "Other"], index=0)
        county = st.selectbox("County", ["Kaufman", "Dallas", "Other"], index=0)
        
        st.markdown("### Personal Details")
        marital_status = st.selectbox("Marital Status", ["Married", "Single"], index=0)
        home_market_value = st.selectbox("Home Value Range", ["50000 - 74999", "75000 - 99999", "100000 - 124999", "Other"], index=0)
        home_owner = st.selectbox("Home Owner", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        college_degree = st.selectbox("College Degree", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        good_credit = st.selectbox("Good Credit", [0, 1], index=1, format_func=lambda x: "Yes" if x == 1 else "No")
        
        st.markdown("---")
        
        # Predict button
        if st.button("Generate Risk Assessment", type="primary"):
            st.session_state.prediction_made = True
            
            # Store input data in session state
            st.session_state.input_data = {
                'curr_ann_amt': curr_ann_amt,
                'days_tenure': days_tenure,
                'cust_orig_date': cust_orig_date,
                'age_in_years': age_in_years,
                'date_of_birth': date_of_birth,
                'latitude': latitude,
                'longitude': longitude,
                'city': city,
                'state': state,
                'county': county,
                'income': income,
                'has_children': has_children,
                'length_of_residence': length_of_residence,
                'marital_status': marital_status,
                'home_market_value': home_market_value,
                'home_owner': home_owner,
                'college_degree': college_degree,
                'good_credit': good_credit
            }
            
            st.rerun()
    
    # Main content area - 70% width
    if st.session_state.prediction_made and 'input_data' in st.session_state:
        try:
            # Get model
            model = models[st.session_state.selected_model]
            
            # Preprocess input
            processed_data = preprocess_input(st.session_state.input_data.copy())
            
            # Create feature vector
            if hasattr(model, 'feature_names_in_'):
                expected_features = model.feature_names_in_
                feature_vector = pd.DataFrame(0, index=[0], columns=expected_features)
                
                for feature, value in processed_data.items():
                    if feature in feature_vector.columns:
                        feature_vector[feature] = value
            else:
                feature_vector = pd.DataFrame([processed_data])
            
            # Make prediction
            prediction_proba = model.predict_proba(feature_vector)[:, 1][0]
            
            # Show user profile
            display_user_profile(st.session_state.input_data)
            
            # Create SHAP explanation
            feature_importance = create_shap_explanation(model, feature_vector, st.session_state.selected_model)
            
            # Show prediction results
            display_prediction_results(prediction_proba, st.session_state.selected_model, feature_importance)
            
            # Show risk factors analysis
            display_risk_factors(feature_importance)
            
            # Show retention strategies
            display_retention_strategies(prediction_proba, st.session_state.input_data)
            
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
            st.session_state.prediction_made = False
    
    else:
        # Welcome screen
        st.markdown("### Chubb AI Risk Assessment Platform")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: {get_theme_config()['accent_color']};">Features</h4>
                <ul style="color: {get_theme_config()['text_color']};">
                    <li>Customer profile analysis</li>
                    <li>AI-powered churn prediction</li>
                    <li>Risk factor identification</li>
                    <li>Retention strategies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="custom-card">
                <h4 style="color: {get_theme_config()['accent_color']};">Available Models</h4>
                <ul style="color: {get_theme_config()['text_color']};">
                    <li>GradientBoostingClassifier</li>
                    <li>XGBoostClassifier</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()