"""
Model Training Module

This module handles model creation, training, and hyperparameter tuning.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


def create_preprocessor(numerical_features, categorical_features):
    """
    Create a ColumnTransformer for preprocessing numerical and categorical features.
    
    Args:
        numerical_features (list): List of numerical feature names
        categorical_features (list): List of categorical feature names
        
    Returns:
        ColumnTransformer: Preprocessing transformer
    """
    print("Creating preprocessing pipelines...")
    
    # Numerical pipeline
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical pipeline
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine both
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough'
    )
    
    print(f"Preprocessor created for:")
    print(f"  - {len(numerical_features)} numerical features")
    print(f"  - {len(categorical_features)} categorical features")
    return preprocessor


def train_logistic_regression(X_train, y_train, preprocessor, cv=5):
    """
    Train Logistic Regression with GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor: ColumnTransformer for preprocessing
        cv (int): Number of cross-validation folds
        
    Returns:
        GridSearchCV: Trained grid search object
    """
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION")
    print("="*80 + "\n")
    
    # Create pipeline
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # Define hyperparameter grid
    param_grid_lr = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'liblinear']
    }
    
    # Run GridSearchCV
    print("Running GridSearchCV for Logistic Regression...")
    grid_search = GridSearchCV(
        lr_pipeline,
        param_grid_lr,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Logistic Regression Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV Score (F1-Weighted): {grid_search.best_score_:.4f}")
    
    return grid_search


def train_random_forest(X_train, y_train, preprocessor, cv=5):
    """
    Train Random Forest with GridSearchCV.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor: ColumnTransformer for preprocessing
        cv (int): Number of cross-validation folds
        
    Returns:
        GridSearchCV: Trained grid search object
    """
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST")
    print("="*80 + "\n")
    
    # Create pipeline
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    # Define hyperparameter grid
    param_grid_rf = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [10, 20, 30, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    
    # Run GridSearchCV
    print("Running GridSearchCV for Random Forest...")
    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid_rf,
        cv=cv,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest Random Forest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    print(f"Best CV Score (F1-Weighted): {grid_search.best_score_:.4f}")
    
    return grid_search


def compare_models(grid_search_lr, grid_search_rf, X_test, y_test):
    """
    Compare performance of Logistic Regression and Random Forest.
    
    Args:
        grid_search_lr: Trained LR GridSearchCV
        grid_search_rf: Trained RF GridSearchCV
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        
    Returns:
        dict: Comparison results with best model
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80 + "\n")
    
    # Predictions
    y_pred_lr = grid_search_lr.predict(X_test)
    y_pred_rf = grid_search_rf.predict(X_test)
    
    y_proba_lr = grid_search_lr.predict_proba(X_test)[:, 1]
    y_proba_rf = grid_search_rf.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics_lr = {
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr, pos_label='Yes'),
        'recall': recall_score(y_test, y_pred_lr, pos_label='Yes'),
        'f1': f1_score(y_test, y_pred_lr, pos_label='Yes'),
        'roc_auc': roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_proba_lr)
    }
    
    metrics_rf = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, pos_label='Yes'),
        'recall': recall_score(y_test, y_pred_rf, pos_label='Yes'),
        'f1': f1_score(y_test, y_pred_rf, pos_label='Yes'),
        'roc_auc': roc_auc_score(y_test.map({'No': 0, 'Yes': 1}), y_proba_rf)
    }
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Logistic Regression': metrics_lr,
        'Random Forest': metrics_rf
    })
    
    print(comparison_df.to_string())
    
    # Select best model
    best_model_name = 'Random Forest' if metrics_rf['accuracy'] > metrics_lr['accuracy'] else 'Logistic Regression'
    best_model = grid_search_rf if metrics_rf['accuracy'] > metrics_lr['accuracy'] else grid_search_lr
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {max(metrics_rf['accuracy'], metrics_lr['accuracy']):.4f}")
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'lr_metrics': metrics_lr,
        'rf_metrics': metrics_rf,
        'comparison_df': comparison_df
    }
