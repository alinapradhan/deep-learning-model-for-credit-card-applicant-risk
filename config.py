"""
Configuration settings for the credit card fraud detection system.
"""
import os

# Model Configuration
MODEL_CONFIG = {
    'input_dim': 23,  # Number of features in the dataset
    'hidden_layers': [128, 64, 32, 16],
    'dropout_rates': [0.3, 0.3, 0.2, 0.2],
    'activation': 'relu',
    'output_activation': 'sigmoid',
    'optimizer': 'adam',
    'learning_rate': 0.001,
    'loss': 'binary_crossentropy',
    'metrics': ['accuracy', 'precision', 'recall', 'auc']
}

# Training Configuration
TRAINING_CONFIG = {
    'epochs': 100,
    'batch_size': 32,
    'validation_split': 0.1,
    'test_size': 0.2,
    'random_state': 42,
    'early_stopping_patience': 15,
    'reduce_lr_patience': 10,
    'min_lr': 1e-7
}

# Data Configuration
DATA_CONFIG = {
    'dataset_path': 'UCI_Credit_Card.csv',
    'target_column': 'default.payment.next.month',
    'use_smote': True,
    'smote_random_state': 42
}

# API Configuration
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'title': 'Credit Card Fraud Detection API',
    'description': 'Real-time credit risk assessment for credit card applicants',
    'version': '1.0.0'
}

# Model Paths
PATHS = {
    'models_dir': 'models',
    'model_file': 'models/credit_risk_model.h5',
    'scaler_file': 'models/scaler.pkl',
    'feature_names_file': 'models/feature_names.pkl',
    'plots_dir': 'models'
}

# Risk Thresholds
RISK_THRESHOLDS = {
    'low_risk_max': 0.3,
    'medium_risk_max': 0.7,
    'high_risk_min': 0.7
}

# Feature Information
FEATURE_INFO = {
    'LIMIT_BAL': 'Credit limit balance',
    'SEX': 'Gender (1=male, 2=female)',
    'EDUCATION': 'Education level (1=graduate school, 2=university, 3=high school, 4=others)',
    'MARRIAGE': 'Marriage status (1=married, 2=single, 3=others)',
    'AGE': 'Age in years',
    'PAY_0': 'Repayment status in September',
    'PAY_2': 'Repayment status in August',
    'PAY_3': 'Repayment status in July',
    'PAY_4': 'Repayment status in June',
    'PAY_5': 'Repayment status in May',
    'PAY_6': 'Repayment status in April',
    'BILL_AMT1': 'Bill statement amount September',
    'BILL_AMT2': 'Bill statement amount August',
    'BILL_AMT3': 'Bill statement amount July',
    'BILL_AMT4': 'Bill statement amount June',
    'BILL_AMT5': 'Bill statement amount May',
    'BILL_AMT6': 'Bill statement amount April',
    'PAY_AMT1': 'Payment amount September',
    'PAY_AMT2': 'Payment amount August',
    'PAY_AMT3': 'Payment amount July',
    'PAY_AMT4': 'Payment amount June',
    'PAY_AMT5': 'Payment amount May',
    'PAY_AMT6': 'Payment amount April'
}

# Environment Variables
def get_env_config():
    """Get configuration from environment variables."""
    return {
        'debug': os.getenv('DEBUG', 'False').lower() == 'true',
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        'model_path': os.getenv('MODEL_PATH', PATHS['model_file']),
        'api_host': os.getenv('API_HOST', API_CONFIG['host']),
        'api_port': int(os.getenv('API_PORT', API_CONFIG['port']))
    }