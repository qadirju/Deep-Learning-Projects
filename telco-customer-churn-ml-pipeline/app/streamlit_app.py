"""
Streamlit Web Application for Telco Churn Prediction

This interactive web app allows users to input customer information and get
real-time churn predictions using the trained ML pipeline.

Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #bdc3c7;
        padding-bottom: 0.5rem;
    }
    .prediction-high {
        background-color: #ffcdd2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #c62828;
        color: #000;
    }
    .prediction-high h3 {
        color: #c62828 !important;
        margin: 0.5rem 0;
    }
    .prediction-high p {
        color: #000 !important;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .prediction-low {
        background-color: #c8e6c9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2e7d32;
        color: #000;
    }
    .prediction-low h3 {
        color: #2e7d32 !important;
        margin: 0.5rem 0;
    }
    .prediction-low p {
        color: #000 !important;
        margin: 0.3rem 0;
        font-weight: 500;
    }
    .metric-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    """Load the trained pipeline model."""
    try:
        # Try multiple possible model paths
        possible_paths = [
            'models/churn_pipeline.joblib',
            './models/churn_pipeline.joblib',
            os.path.join(os.path.dirname(__file__), 'models', 'churn_pipeline.joblib'),
            os.path.join(os.path.dirname(__file__), '..', 'models', 'churn_pipeline.joblib'),
        ]
        
        for model_path in possible_paths:
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                st.info(f"✅ Model loaded successfully from: {model_path}")
                return model
        
        # If no model found, show error with helpful info
        st.warning("""
        ⚠️ **Model file not found**
        
        The machine learning model file is not available. This could be because:
        1. The model training notebook hasn't been executed yet
        2. The GridSearchCV training is still in progress (15-60 minutes)
        3. The model file path is incorrect
        
        **To fix this:**
        - Run the notebook cells to train the model
        - Or wait for GridSearchCV to complete
        - The model should be saved at: `models/churn_pipeline.joblib`
        """)
        return None
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("Please check that the model file exists and is not corrupted.")
        return None


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📊 Telco Customer Churn Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    This application predicts whether a customer is likely to churn (cancel their service)
    based on their profile and usage patterns. Input customer information below to get a prediction.
    """)
    
    # Load model
    model = load_model()
    
    if model is None:
        st.warning("""
        ⚠️ **Cannot proceed without a model**
        
        Please:
        1. Run the training notebook to train and save the model
        2. Ensure the model file exists at: `models/churn_pipeline.joblib`
        3. Then restart this Streamlit app
        """)
        st.stop()  # Stop execution here
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch Prediction", "About"])
    
    # ==================== TAB 1: SINGLE PREDICTION ====================
    with tab1:
        st.markdown('<h2 class="section-header">Single Customer Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            customer_id = st.text_input("Customer ID", value="CUST001")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        
        with col2:
            st.subheader("Service Usage")
            tenure = st.slider("Tenure (months)", 0, 72, value=12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Additional Services")
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
        
        with col4:
            st.subheader("Contract & Payment")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, value=65.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, value=1500.0)
        
        # Multiple lines
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        
        # Prepare prediction
        if st.button("🔮 Predict Churn", use_container_width=True):
            try:
                # Create customer dataframe
                customer_data = pd.DataFrame({
                    'customerID': [customer_id],
                    'gender': [gender],
                    'SeniorCitizen': [1 if senior_citizen == "Yes" else 0],
                    'Partner': [partner],
                    'Dependents': [dependents],
                    'tenure': [tenure],
                    'PhoneService': [phone_service],
                    'MultipleLines': [multiple_lines],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [online_security],
                    'OnlineBackup': [online_backup],
                    'DeviceProtection': [device_protection],
                    'TechSupport': [tech_support],
                    'StreamingTV': [streaming_tv],
                    'StreamingMovies': [streaming_movies],
                    'Contract': [contract],
                    'PaperlessBilling': [paperless_billing],
                    'PaymentMethod': [payment_method],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges]
                })
                
                # Make prediction
                prediction = model.predict(customer_data)[0]
                probabilities = model.predict_proba(customer_data)[0]
                classes = model.classes_
                
                # Get probabilities
                prob_no = probabilities[list(classes).index('No')]
                prob_yes = probabilities[list(classes).index('Yes')]
                
                # Display results
                st.divider()
                
                # Main prediction result - uses better spacing
                with st.container():
                    if prediction == 'Yes':
                        st.markdown(f"""
                        <div class="prediction-high">
                        <h3>⚠️ CHURN PREDICTION: YES</h3>
                        <p>Risk Level: <strong>HIGH</strong></p>
                        <p>This customer is likely to churn.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="prediction-low">
                        <h3>✅ CHURN PREDICTION: NO</h3>
                        <p>Risk Level: <strong>LOW</strong></p>
                        <p>This customer is likely to stay.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Add space
                st.write("")
                
                # Display probabilities in a clear format
                col_prob1, col_prob2, col_prob3 = st.columns(3)
                
                with col_prob1:
                    st.metric("No Churn", f"{prob_no:.2%}")
                
                with col_prob2:
                    st.metric("Churn", f"{prob_yes:.2%}")
                
                with col_prob3:
                    risk_level = "HIGH 🔴" if prediction == 'Yes' else "LOW 🟢"
                    st.metric("Risk Level", risk_level)
                
                # Recommendations
                st.divider()
                st.subheader("📋 Recommendations")
                
                if prediction == 'Yes':
                    st.warning("""
                    **This customer is at risk of churning. Consider:**
                    - Offering a retention discount or special promotion
                    - Upgrading their service package
                    - Direct outreach from customer success team
                    - Addressing any service quality issues
                    """)
                else:
                    st.success("""
                    **This customer is satisfied. Consider:**
                    - Maintaining current service quality
                    - Offering upsell opportunities
                    - Regular check-ins to ensure satisfaction
                    - Loyalty programs or referral incentives
                    """)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    # ==================== TAB 2: BATCH PREDICTION ====================
    with tab2:
        st.markdown('<h2 class="section-header">Batch Prediction</h2>', unsafe_allow_html=True)
        
        st.info("Upload a CSV file with customer data to make predictions for multiple customers.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(df)} customers")
                st.dataframe(df.head())
                
                if st.button("🔮 Predict for All Customers"):
                    predictions = model.predict(df)
                    probabilities = model.predict_proba(df)
                    
                    classes = model.classes_
                    prob_churn = probabilities[:, list(classes).index('Yes')]
                    
                    results_df = pd.DataFrame({
                        'Prediction': predictions,
                        'Churn_Probability': prob_churn,
                        'Risk_Level': ['HIGH' if p > 0.7 else 'MEDIUM' if p > 0.3 else 'LOW' for p in prob_churn]
                    })
                    
                    st.success("Predictions completed!")
                    st.dataframe(results_df)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results (CSV)",
                        data=csv,
                        file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    # ==================== TAB 3: ABOUT ====================
    with tab3:
        st.markdown('<h2 class="section-header">About This Application</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        ### 🎯 Purpose
        This application predicts whether a telecommunications customer is likely to churn
        (discontinue their service) using a machine learning model trained on historical
        customer data.
        
        ### 🤖 Model Details
        - **Algorithm**: Random Forest / Logistic Regression
        - **Training Data**: Telco customer dataset
        - **Features**: 20 customer attributes (demographics, services, charges)
        - **Cross-Validation**: 5-fold stratified cross-validation
        - **Performance**: Optimized for F1-score
        
        ### 📊 How It Works
        1. You input customer information
        2. The model preprocesses the data (imputation, scaling, encoding)
        3. The trained classifier makes a prediction
        4. Results show churn probability and recommendations
        
        ### 📈 Key Metrics
        - **Accuracy**: High predictive accuracy on test set
        - **Precision**: Reliable positive predictions
        - **Recall**: Captures most churning customers
        - **F1-Score**: Balanced performance metric
        
        ### 🔧 Technologies
        - **Framework**: Scikit-learn
        - **Interface**: Streamlit
        - **Language**: Python
        
        ### 📝 Notes
        - This tool provides predictions based on historical patterns
        - Always combine ML predictions with domain expertise
        - Regularly retrain the model with new data
        """)
        
        st.divider()
        st.markdown("""
        **Created with ❤️ by the ML Engineering Team**
        
        For questions or issues, please contact the data science team.
        """)


if __name__ == "__main__":
    main()
