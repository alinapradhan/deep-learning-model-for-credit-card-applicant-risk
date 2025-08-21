"""
Data preprocessing module for credit card fraud detection.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def load_data(self, file_path):
        """Load the UCI Credit Card dataset."""
        df = pd.read_csv(file_path)
        return df
    
    def preprocess_features(self, df):
        """Clean and preprocess the features."""
        # Drop ID column as it's not useful for prediction
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        # Handle categorical variables
        categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
        
        for col in categorical_columns:
            if col in df.columns:
                # Convert to category and handle unknown values
                df[col] = df[col].astype(int)
                
                # Clean education values (4+ are others/unknown, set to 4)
                if col == 'EDUCATION':
                    df[col] = df[col].apply(lambda x: 4 if x > 4 or x < 1 else x)
                
                # Clean marriage values (0 and values > 3 are others, set to 3)
                if col == 'MARRIAGE':
                    df[col] = df[col].apply(lambda x: 3 if x > 3 or x < 1 else x)
        
        # Handle payment status columns (PAY_0 to PAY_6)
        pay_columns = [f'PAY_{i}' for i in range(7)]
        pay_columns[0] = 'PAY_0'  # Special case for PAY_0
        
        for col in pay_columns:
            if col in df.columns:
                # -2: No consumption, -1: Paid in full, 0: Use of revolving credit
                # 1-9: Payment delay for X months
                df[col] = df[col].astype(int)
        
        return df
    
    def handle_target(self, df, target_column='default.payment.next.month'):
        """Separate features and target."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(target_column, axis=1)
        y = df[target_column]
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        return X, y
    
    def fit_transform(self, X, y):
        """Fit preprocessor and transform data."""
        # Scale numerical features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def transform(self, X):
        """Transform new data using fitted preprocessor."""
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_scaled
    
    def split_data(self, X, y, test_size=0.2, random_state=42, stratify=True):
        """Split data into train and test sets."""
        stratify_param = y if stratify else None
        return train_test_split(X, y, test_size=test_size, 
                              random_state=random_state, 
                              stratify=stratify_param)
    
    def handle_imbalance(self, X_train, y_train, random_state=42):
        """Handle class imbalance using SMOTE."""
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return X_resampled, y_resampled
    
    def save_preprocessor(self, file_path='models'):
        """Save the fitted preprocessor."""
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        
        joblib.dump(self.scaler, os.path.join(file_path, 'scaler.pkl'))
        joblib.dump(self.feature_names, os.path.join(file_path, 'feature_names.pkl'))
        
    def load_preprocessor(self, file_path='models'):
        """Load a fitted preprocessor."""
        self.scaler = joblib.load(os.path.join(file_path, 'scaler.pkl'))
        self.feature_names = joblib.load(os.path.join(file_path, 'feature_names.pkl'))

def prepare_data(file_path='UCI_Credit_Card.csv', save_path='models'):
    """Main function to prepare data for training."""
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data(file_path)
    df_clean = preprocessor.preprocess_features(df)
    X, y = preprocessor.handle_target(df_clean)
    
    # Split data
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    
    # Fit and transform training data
    X_train_scaled, y_train = preprocessor.fit_transform(X_train, y_train)
    
    # Transform test data
    X_test_scaled = preprocessor.transform(X_test)
    
    # Handle imbalance
    X_train_balanced, y_train_balanced = preprocessor.handle_imbalance(X_train_scaled, y_train)
    
    # Save preprocessor
    preprocessor.save_preprocessor(save_path)
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, preprocessor

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, preprocessor = prepare_data()
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Class distribution in training set:")
    print(pd.Series(y_train).value_counts().sort_index())