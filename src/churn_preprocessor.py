import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

class ChurnPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.feature_names = None
        
    def fit(self, df, target_col='Churn'):
        """Fit preprocessor on training data."""
        df = df.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'CustomerID'])
        y = df[target_col]
        
        # Identify column types
        self.categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        self.numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Encode categorical features
        for col in self.categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        # Fit scaler on numerical columns
        if self.numerical_columns:
            self.scaler.fit(X[self.numerical_columns])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        return X, y
    
    def transform(self, df, target_col='Churn'):
        """Transform data using fitted preprocessor."""
        df = df.copy()
        
        # Separate features and target
        X = df.drop(columns=[target_col, 'CustomerID'], errors='ignore')
        
        # Encode categorical features
        for col in self.categorical_columns:
            if col in X.columns:
                X[col] = self.label_encoders[col].transform(X[col])
        
        # Scale numerical columns
        if self.numerical_columns:
            X[self.numerical_columns] = self.scaler.transform(X[self.numerical_columns])
        
        # Get target if available
        if target_col in df.columns:
            y = df[target_col]
            return X, y
        
        return X, None
    
    def fit_transform(self, df, target_col='Churn'):
        """Fit and transform in one step."""
        self.fit(df, target_col)
        return self.transform(df, target_col)
    
    def save(self, filepath):
        """Save preprocessor to file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath):
        """Load preprocessor from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def preprocess_churn_data(df, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for churn data.
    
    Args:
        df: Raw churn dataset
        test_size: Test set fraction
        random_state: Random seed
    
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    # Initialize preprocessor
    preprocessor = ChurnPreprocessor()
    
    # Fit on full dataset and split
    X, y = preprocessor.fit_transform(df)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, preprocessor


def get_preprocessing_info(preprocessor):
    """Get information about preprocessing."""
    return {
        'categorical_columns': preprocessor.categorical_columns,
        'numerical_columns': preprocessor.numerical_columns,
        'feature_count': len(preprocessor.feature_names),
        'feature_names': preprocessor.feature_names
    }
