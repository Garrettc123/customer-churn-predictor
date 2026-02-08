#!/usr/bin/env python3
"""
Train Customer Churn Prediction Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data(n_samples=10000):
    """
    Generate synthetic customer data for training
    """
    np.random.seed(42)
    
    data = {
        'last_login_days': np.random.exponential(scale=15, size=n_samples),
        'support_tickets': np.random.poisson(lam=2, size=n_samples),
        'contract_value': np.random.lognormal(mean=7, sigma=1, size=n_samples),
        'engagement_score': np.random.beta(a=2, b=2, size=n_samples),
        'payment_failures': np.random.poisson(lam=0.5, size=n_samples),
        'feature_usage_count': np.random.poisson(lam=10, size=n_samples),
        'tenure_months': np.random.exponential(scale=12, size=n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate churn labels based on features
    churn_prob = (
        (df['last_login_days'] > 30) * 0.3 +
        (df['support_tickets'] > 5) * 0.2 +
        (df['engagement_score'] < 0.3) * 0.25 +
        (df['payment_failures'] > 2) * 0.25
    )
    
    df['churned'] = (np.random.random(n_samples) < churn_prob).astype(int)
    
    return df

def train_model(df):
    """
    Train XGBoost model
    """
    logger.info(f"Training on {len(df)} samples")
    
    # Split features and target
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train XGBoost
    logger.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_proba)
    }
    
    logger.info("Model Performance:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    return model, metrics

def main():
    """
    Main training pipeline
    """
    # Generate or load data
    logger.info("Generating sample data...")
    df = generate_sample_data(10000)
    
    # Train model
    model, metrics = train_model(df)
    
    # Save model
    import os
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/churn_model.pkl')
    logger.info("Model saved to models/churn_model.pkl")
    
    # Save metrics
    import json
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to models/metrics.json")

if __name__ == "__main__":
    main()
