"""
Prediction Module

This module handles making predictions on new customer data using trained models.
"""

import pandas as pd
import numpy as np


def create_sample_customer():
    """
    Create a sample customer for testing predictions.
    
    Returns:
        pd.DataFrame: Sample customer data
    """
    customer = pd.DataFrame({
        'customerID': ['CUST12345'],
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['No'],
        'Dependents': ['No'],
        'tenure': [12],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['No'],
        'DeviceProtection': ['Yes'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [89.45],
        'TotalCharges': [1185.50]
    })
    return customer


def make_prediction(pipeline, customer_data):
    """
    Make churn prediction for a customer.
    
    Args:
        pipeline: Trained pipeline model
        customer_data (pd.DataFrame): Customer features
        
    Returns:
        dict: Prediction results with probabilities
    """
    # Make prediction
    prediction = pipeline.predict(customer_data)[0]
    probabilities = pipeline.predict_proba(customer_data)[0]
    
    # Get class indices
    classes = pipeline.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    result = {
        'prediction': prediction,
        'probability_no': probabilities[class_to_idx['No']],
        'probability_yes': probabilities[class_to_idx['Yes']],
        'churn_risk': probabilities[class_to_idx['Yes']]
    }
    
    return result


def predict_customer(pipeline, customer_data):
    """
    Predict churn for a customer and print human-readable result.
    
    Args:
        pipeline: Trained pipeline model
        customer_data (pd.DataFrame): Customer features
        
    Returns:
        dict: Prediction results
    """
    result = make_prediction(pipeline, customer_data)
    
    print("="*80)
    print("CHURN PREDICTION RESULT")
    print("="*80)
    print(f"\nCustomer ID: {customer_data['customerID'].values[0]}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability of NOT churning: {result['probability_no']:.2%}")
    print(f"Probability of CHURNING: {result['probability_yes']:.2%}")
    
    # Human readable interpretation
    print(f"\n{'='*80}")
    if result['prediction'] == 'Yes':
        print(f"⚠️  WARNING: This customer is LIKELY TO CHURN")
        print(f"   Churn Risk: {result['churn_risk']:.2%}")
        print(f"   Recommended Action: Provide immediate retention offer")
    else:
        print(f"✅ GOOD NEWS: This customer is LIKELY TO STAY")
        print(f"   Churn Risk: {result['churn_risk']:.2%}")
        print(f"   Recommended Action: Continue standard service")
    print(f"{'='*80}\n")
    
    return result


def batch_predict(pipeline, customers_df):
    """
    Make predictions for multiple customers.
    
    Args:
        pipeline: Trained pipeline model
        customers_df (pd.DataFrame): Multiple customer records
        
    Returns:
        pd.DataFrame: Predictions for all customers
    """
    predictions = pipeline.predict(customers_df)
    probabilities = pipeline.predict_proba(customers_df)
    
    classes = pipeline.classes_
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    results_df = pd.DataFrame({
        'customerID': customers_df['customerID'],
        'prediction': predictions,
        'probability_stay': probabilities[:, class_to_idx['No']],
        'probability_churn': probabilities[:, class_to_idx['Yes']],
        'churn_risk': probabilities[:, class_to_idx['Yes']]
    })
    
    return results_df


def format_prediction_output(result):
    """
    Format prediction result for display/export.
    
    Args:
        result (dict): Prediction result from make_prediction()
        
    Returns:
        str: Formatted output string
    """
    output = f"""
    ╔════════════════════════════════════════════════════════════════════════════╗
    ║                      CHURN PREDICTION RESULT                              ║
    ╠════════════════════════════════════════════════════════════════════════════╣
    ║                                                                            ║
    ║  Prediction:             {("CHURN" if result['prediction'] == "Yes" else "NO CHURN"):40s} ║
    ║  Probability (No Churn): {f"{result['probability_no']:.2%}":40s} ║
    ║  Probability (Churn):    {f"{result['probability_yes']:.2%}":40s} ║
    ║  Churn Risk Level:       {("HIGH" if result['churn_risk'] > 0.7 else "MEDIUM" if result['churn_risk'] > 0.3 else "LOW"):40s} ║
    ║                                                                            ║
    ╚════════════════════════════════════════════════════════════════════════════╝
    """
    return output
