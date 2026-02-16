"""
Data Preprocessing Module

This module handles data loading, cleaning, and preprocessing for the Telco churn dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_dataset(filepath):
    """
    Load the Telco churn dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def handle_missing_values(df):
    """
    Replace blank strings with NaN values.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with NaN values
    """
    print("Handling missing values...")
    df_clean = df.replace(" ", np.nan)
    print(f"Missing values found: {df_clean.isnull().sum().sum()}")
    return df_clean


def convert_data_types(df):
    """
    Convert TotalCharges column to numeric datatype.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with corrected data types
    """
    print("Converting data types...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    print(f"TotalCharges datatype: {df['TotalCharges'].dtype}")
    return df


def separate_features_target(df, target_column='Churn'):
    """
    Separate features (X) and target (y) from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        
    Returns:
        tuple: (X, y) where X is features and y is target
    """
    print(f"Separating features and target ({target_column})...")
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    return X, y


def split_train_test(X, y, test_size=0.2, random_state=42):
    """
    Split dataset into training and testing sets with stratification.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"Splitting dataset (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
    return X_train, X_test, y_train, y_test


def detect_feature_types(X):
    """
    Automatically detect categorical and numerical features.
    
    Args:
        X (pd.DataFrame): Features dataframe
        
    Returns:
        tuple: (numerical_features, categorical_features)
    """
    print("Detecting feature types...")
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    return numerical_features, categorical_features


def preprocess_pipeline(filepath, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline: load → clean → split → detect features.
    
    Args:
        filepath (str): Path to CSV file
        test_size (float): Proportion of test set
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary containing all preprocessed data
    """
    print("\n" + "="*80)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*80 + "\n")
    
    # Load and clean
    df = load_dataset(filepath)
    df = handle_missing_values(df)
    df = convert_data_types(df)
    
    # Separate and split
    X, y = separate_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)
    
    # Detect features
    numerical_features, categorical_features = detect_feature_types(X_train)
    
    print("\n" + "="*80)
    print("DATA PREPROCESSING COMPLETED")
    print("="*80 + "\n")
    
    return {
        'df': df,
        'X': X,
        'y': y,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features
    }
