import io
import base64
from flask import Flask, request, jsonify, render_template
import time
import numpy as np
import pickle
import joblib
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle
import shap
shap.initjs()

app = Flask(__name__)
model = pickle.load(open('./models/model_rfc.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    gender = 0
    if request.form["gender"] == 1:
        gender = 1
    SeniorCitizen = 0
    if 'SeniorCitizen' in request.form:
        SeniorCitizen = 1
    Partner = 0
    if 'Partner' in request.form:
        Partner = 1
    Dependents = 0
    if 'Dependents' in request.form:
        Dependents = 1
    PaperlessBilling = 0
    if 'PaperlessBilling' in request.form:
        PaperlessBilling = 1

    MonthlyCharges = int(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])
    TotalCharges = MonthlyCharges*Tenure

    PhoneService = 0
    if 'PhoneService' in request.form:
        PhoneService = 1

    MultipleLines = 0
    if 'MultipleLines' in request.form and PhoneService == 1:
        MultipleLines = 1

    InternetService_Fiberoptic = 0
    InternetService_No = 0
    if request.form["InternetService"] == 0:
        InternetService_No = 1
    elif request.form["InternetService"] == 2:
        InternetService_Fiberoptic = 1

    OnlineSecurity = 0
    if 'OnlineSecurity' in request.form and InternetService_No == 0:
        OnlineSecurity = 1

    OnlineBackup = 0
    if 'OnlineBackup' in request.form and InternetService_No == 0:
        OnlineBackup = 1

    DeviceProtection = 0
    if 'DeviceProtection' in request.form and InternetService_No == 0:
        DeviceProtection = 1

    TechSupport = 0
    if 'TechSupport' in request.form and InternetService_No == 0:
        TechSupport = 1

    StreamingTV = 0
    if 'StreamingTV' in request.form and InternetService_No == 0:
        StreamingTV = 1

    StreamingMovies = 0
    if 'StreamingMovies' in request.form and InternetService_No == 0:
        StreamingMovies = 1

    Contract_Oneyear = 0
    Contract_Twoyear = 0
    if request.form["Contract"] == 1:
        Contract_Oneyear = 1
    elif request.form["Contract"] == 2:
        Contract_Twoyear = 1

    PaymentMethod_CreditCard = 0
    PaymentMethod_ElectronicCheck = 0
    PaymentMethod_MailedCheck = 0
    if request.form["PaymentMethod"] == 1:
        PaymentMethod_CreditCard = 1
    elif request.form["PaymentMethod"] == 2:
        PaymentMethod_ElectronicCheck = 1
    elif request.form["PaymentMethod"] == 3:
        PaymentMethod_MailedCheck = 1

    features = [gender, SeniorCitizen, Partner, Dependents, Tenure, PhoneService, MultipleLines, OnlineSecurity, OnlineBackup,
                DeviceProtection, TechSupport, StreamingTV, StreamingMovies, PaperlessBilling, MonthlyCharges, TotalCharges,
                InternetService_Fiberoptic, InternetService_No, Contract_Oneyear, Contract_Twoyear,
                PaymentMethod_CreditCard, PaymentMethod_ElectronicCheck, PaymentMethod_MailedCheck]

    columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
               'InternetService_Fiber optic', 'InternetService_No', 'Contract_One year', 'Contract_Two year',
               'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']

    final_features = [np.array(features)]
    # print(final_features)
    # print(model)
    
    # Create DataFrame with proper feature names to avoid sklearn warning
    import pandas as pd
    feature_df = pd.DataFrame(final_features, columns=columns)
    prediction = model.predict_proba(feature_df)

    output = prediction[0, 1]
    print(f"Churn probability: {output}")  # Debug output

    # Shap Values - Create new explainer to avoid compatibility issues
    try:
        import pandas as pd
        # Create a DataFrame with proper feature names for the model
        feature_df = pd.DataFrame(final_features, columns=columns)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_df)
        
        # Create a simple SHAP bar plot instead of force plot
        plt.figure(figsize=(12, 8))
        
        # Get the SHAP values for the positive class (churn = 1)
        feature_importance = shap_values[1][0]  # First (and only) sample, positive class
        
        # Create a simple bar plot of SHAP values
        sorted_idx = np.argsort(np.abs(feature_importance))[-10:]  # Top 10 features
        
        plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx])
        plt.yticks(range(len(sorted_idx)), [columns[i] for i in sorted_idx])
        plt.xlabel('SHAP Value (Impact on Churn Prediction)')
        plt.title('Feature Importance for Churn Prediction')
        plt.tight_layout()
        
        shap_img = io.BytesIO()
        plt.savefig(shap_img, bbox_inches="tight", format='png', dpi=100)
        plt.close()  # Close the figure to free memory
        shap_img.seek(0)
        shap_url = base64.b64encode(shap_img.getvalue()).decode()
        
        print(f"SHAP plot created successfully")
    except Exception as e:
        print(f"SHAP error: {e}")
        # If SHAP fails, create a placeholder image
        shap_url = ""

    def degree_range(n):
        start = np.linspace(0, 180, n+1, endpoint=True)[0:-1]
        end = np.linspace(0, 180, n+1, endpoint=True)[1::]
        mid_points = start + ((end-start)/2.)
        return np.c_[start, end], mid_points

    def rot_text(ang):
        rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
        return rotation

    def gauge(labels=['LOW', 'MEDIUM', 'HIGH', 'EXTREME'],
              colors=['#007A00', '#0063BF', '#FFCC00', '#ED1C24'], Probability=1, model_name='rf'):

        N = len(labels)
        colors = colors[::-1]

        gauge_img = io.BytesIO()
        fig, ax = plt.subplots()

        ang_range, mid_points = degree_range(4)

        labels = labels[::-1]
        patches = []
        for ang, c in zip(ang_range, colors):
            # sectors
            patches.append(Wedge((0., 0.), .4, *ang, facecolor='w', lw=2))
            # arcs
            patches.append(Wedge((0., 0.), .4, *ang, width=0.10,
                           facecolor=c, lw=2, alpha=0.5))

        [ax.add_patch(p) for p in patches]

        for mid, lab in zip(mid_points, labels):

            ax.text(0.35 * np.cos(np.radians(mid)), 0.35 * np.sin(np.radians(mid)), lab,
                    horizontalalignment='center', verticalalignment='center', fontsize=14,
                    fontweight='bold', rotation=rot_text(mid))

        r = Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2)
        ax.add_patch(r)

        ax.text(0, -0.05, 'Churn Probability ' + np.round(Probability, 2).astype(str), horizontalalignment='center',
                verticalalignment='center', fontsize=22, fontweight='bold')

        pos = (1-Probability)*180
        ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos)), 0.225 * np.sin(np.radians(pos)),
                 width=0.04, head_width=0.09, head_length=0.1, fc='k', ec='k')

        ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
        ax.add_patch(Circle((0, 0), radius=0.01, facecolor='w', zorder=11))

        ax.set_frame_on(False)
        ax.axes.set_xticks([])
        ax.axes.set_yticks([])
        ax.axis('equal')
        plt.tight_layout()

        plt.savefig(gauge_img, format='png')
        plt.close()  # Close the figure to free memory
        gauge_img.seek(0)
        url = base64.b64encode(gauge_img.getvalue()).decode()
        return url

    gauge_url = gauge(Probability=output)
    t = time.time()
    return render_template('index.html', 
                         prediction_text='Churn Probability: {}'.format(np.round(output, 2)), 
                         url_1=gauge_url, 
                         url_2=shap_url,
                         url_3="",  # Placeholder for additional charts
                         url_4=""   # Placeholder for additional charts
                         )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
