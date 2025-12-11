import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pickle
import warnings

warnings.filterwarnings('ignore')

class ChurnModelTrainer:
    def __init__(self):
        self.models = {}
        self.trained_models = {}
        
    def initialize_models(self):
        """Initialize all classification models."""
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
        }
        return self.models
    
    def train_model(self, model_name, X_train, y_train):
        """
        Train a single model.
        
        Args:
            model_name: Name of the model
            X_train: Training features
            y_train: Training target
        
        Returns:
            Trained model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        
        return model
    
    def train_all_models(self, X_train, y_train):
        """
        Train all initialized models.
        
        Args:
            X_train: Training features
            y_train: Training target
        
        Returns:
            Dictionary of trained models
        """
        self.initialize_models()
        
        for model_name in self.models.keys():
            print(f"Training {model_name}...")
            self.train_model(model_name, X_train, y_train)
        
        return self.trained_models
    
    def predict(self, model_name, X):
        """
        Make predictions with a trained model.
        
        Args:
            model_name: Name of the model
            X: Features
        
        Returns:
            Predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.trained_models[model_name].predict(X)
    
    def predict_proba(self, model_name, X):
        """
        Get prediction probabilities.
        
        Args:
            model_name: Name of the model
            X: Features
        
        Returns:
            Probability predictions
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        return self.trained_models[model_name].predict_proba(X)
    
    def get_feature_importance(self, model_name, feature_names=None):
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
        
        Returns:
            Series with feature importance
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        model = self.trained_models[model_name]
        
        # Get feature importance based on model type
        if model_name == 'Logistic Regression':
            importance = np.abs(model.coef_[0])
        elif model_name in ['Random Forest', 'XGBoost']:
            importance = model.feature_importances_
        else:
            raise ValueError(f"Feature importance not available for {model_name}")
        
        if feature_names:
            return pd.Series(importance, index=feature_names).sort_values(ascending=False)
        
        return pd.Series(importance).sort_values(ascending=False)
    
    def save_model(self, model_name, filepath):
        """Save trained model to file."""
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained yet")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.trained_models[model_name], f)
    
    @staticmethod
    def load_model(filepath):
        """Load trained model from file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def train_churn_models(X_train, y_train):
    """
    Train all churn prediction models.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        ChurnModelTrainer instance with trained models
    """
    trainer = ChurnModelTrainer()
    trainer.train_all_models(X_train, y_train)
    return trainer
