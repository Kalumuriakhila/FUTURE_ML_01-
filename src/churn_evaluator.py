import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

class ChurnEvaluator:
    def __init__(self):
        self.results = {}
        self.confusion_matrices = {}
        self.feature_importance_data = {}
        
    def evaluate_model(self, model_name, y_true, y_pred, y_pred_proba=None):
        """
        Evaluate a classification model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (for ROC-AUC)
        
        Returns:
            Dictionary with evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # Add ROC-AUC if probabilities available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        # Store confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        self.confusion_matrices[model_name] = cm
        metrics['confusion_matrix'] = cm
        
        # Add detailed classification report
        metrics['classification_report'] = classification_report(
            y_true, y_pred, zero_division=0
        )
        
        self.results[model_name] = metrics
        return metrics
    
    def get_confusion_matrix(self, model_name):
        """Get confusion matrix for a model."""
        return self.confusion_matrices.get(model_name)
    
    def plot_confusion_matrix(self, model_name, figsize=(8, 6)):
        """
        Plot confusion matrix for a model.
        
        Args:
            model_name: Name of the model
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        if model_name not in self.confusion_matrices:
            raise ValueError(f"No confusion matrix for {model_name}")
        
        cm = self.confusion_matrices[model_name]
        
        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn']
        )
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        return fig
    
    def plot_feature_importance(self, model_name, feature_importance, top_n=15, figsize=(10, 6)):
        """
        Plot feature importance for a model.
        
        Args:
            model_name: Name of the model
            feature_importance: Series with feature importance
            top_n: Number of top features to show
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        top_features = feature_importance.head(top_n)
        
        fig, ax = plt.subplots(figsize=figsize)
        top_features.plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance - {model_name}')
        ax.invert_yaxis()
        
        return fig
    
    def plot_roc_curve(self, model_name, y_true, y_pred_proba, figsize=(8, 6)):
        """
        Plot ROC curve for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred_proba: Predicted probabilities
            figsize: Figure size
        
        Returns:
            Matplotlib figure
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend(loc="lower right")
        
        return fig
    
    def compare_models(self):
        """
        Compare all trained models.
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for model_name, metrics in self.results.items():
            row = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
            }
            if 'roc_auc' in metrics:
                row['ROC-AUC'] = metrics['roc_auc']
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.set_index('Model')
    
    def get_best_model(self, metric='f1'):
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Best model name and its score
        """
        scores = {name: metrics[metric] for name, metrics in self.results.items()}
        best_model = max(scores, key=scores.get)
        return best_model, scores[best_model]
    
    def get_model_summary(self, model_name):
        """Get detailed summary for a model."""
        if model_name not in self.results:
            raise ValueError(f"No results for {model_name}")
        
        metrics = self.results[model_name]
        cm = self.confusion_matrices.get(model_name)
        
        summary = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1'],
            'roc_auc': metrics.get('roc_auc', 'N/A'),
            'true_negatives': cm[0, 0] if cm is not None else None,
            'false_positives': cm[0, 1] if cm is not None else None,
            'false_negatives': cm[1, 0] if cm is not None else None,
            'true_positives': cm[1, 1] if cm is not None else None,
        }
        
        return summary


def evaluate_all_models(trainer, X_test, y_test, model_names=None):
    """
    Evaluate all trained models.
    
    Args:
        trainer: ChurnModelTrainer instance
        X_test: Test features
        y_test: Test labels
        model_names: List of model names to evaluate (all if None)
    
    Returns:
        ChurnEvaluator instance with results
    """
    evaluator = ChurnEvaluator()
    
    if model_names is None:
        model_names = list(trainer.trained_models.keys())
    
    for model_name in model_names:
        y_pred = trainer.predict(model_name, X_test)
        
        # Get probabilities if available
        y_pred_proba = None
        try:
            y_pred_proba = trainer.predict_proba(model_name, X_test)
        except:
            pass
        
        evaluator.evaluate_model(model_name, y_test, y_pred, y_pred_proba)
    
    return evaluator
