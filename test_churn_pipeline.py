#!/usr/bin/env python3
"""
Complete Churn Prediction Pipeline Test
This script demonstrates the full workflow
"""

import pandas as pd
from src.data_generator import generate_churn_dataset
from src.churn_preprocessor import preprocess_churn_data
from src.churn_models import train_churn_models
from src.churn_evaluator import evaluate_all_models

print("=" * 70)
print("🎯 CUSTOMER CHURN PREDICTION - COMPLETE PIPELINE TEST")
print("=" * 70)

# Step 1: Generate Data
print("\n[STEP 1] Generating synthetic churn dataset...")
df = generate_churn_dataset(1000)
print(f"✓ Dataset shape: {df.shape}")
print(f"✓ Churn distribution:\n{df['Churn'].value_counts()}")

# Step 2: Preprocess Data
print("\n[STEP 2] Preprocessing data...")
X_train, X_test, y_train, y_test, preprocessor = preprocess_churn_data(df, test_size=0.2)
print(f"✓ Training set: {X_train.shape}")
print(f"✓ Test set: {X_test.shape}")
print(f"✓ Features: {len(preprocessor.feature_names)}")
print(f"  - Categorical: {len(preprocessor.categorical_columns)}")
print(f"  - Numerical: {len(preprocessor.numerical_columns)}")

# Step 3: Train Models
print("\n[STEP 3] Training models...")
trainer = train_churn_models(X_train, y_train)
print(f"✓ Models trained: {list(trainer.trained_models.keys())}")

# Step 4: Evaluate Models
print("\n[STEP 4] Evaluating models...")
evaluator = evaluate_all_models(trainer, X_test, y_test)

# Display Results
print("\n" + "=" * 70)
print("📊 MODEL COMPARISON RESULTS")
print("=" * 70)

comparison = evaluator.compare_models()
print(comparison)

print("\n" + "=" * 70)
print("🏆 BEST MODELS BY METRIC")
print("=" * 70)

for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
    best_model, best_score = evaluator.get_best_model(metric)
    print(f"  {metric.upper():15} → {best_model:20} ({best_score:.4f})")

# Detailed Results for Each Model
print("\n" + "=" * 70)
print("📈 DETAILED RESULTS")
print("=" * 70)

for model_name in trainer.trained_models.keys():
    print(f"\n[{model_name}]")
    summary = evaluator.get_model_summary(model_name)
    
    print(f"  Accuracy:  {summary['accuracy']:.4f}")
    print(f"  Precision: {summary['precision']:.4f}")
    print(f"  Recall:    {summary['recall']:.4f}")
    print(f"  F1-Score:  {summary['f1_score']:.4f}")
    print(f"  ROC-AUC:   {summary['roc_auc']}")
    
    print(f"\n  Confusion Matrix:")
    print(f"    True Negatives:  {summary['true_negatives']}")
    print(f"    False Positives: {summary['false_positives']}")
    print(f"    False Negatives: {summary['false_negatives']}")
    print(f"    True Positives:  {summary['true_positives']}")
    
    # Feature Importance
    try:
        fi = trainer.get_feature_importance(model_name, preprocessor.feature_names)
        print(f"\n  Top 5 Important Features:")
        for i, (feat, importance) in enumerate(fi.head(5).items(), 1):
            print(f"    {i}. {feat:20} → {importance:.4f}")
    except:
        pass

print("\n" + "=" * 70)
print("✅ PIPELINE TEST COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nTo run the dashboard, execute:")
print("  streamlit run churn_dashboard.py")
print("=" * 70)
