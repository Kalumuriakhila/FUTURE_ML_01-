# 🎯 CUSTOMER CHURN PREDICTION SYSTEM

## Overview

A complete machine learning system to predict customer churn with multiple models, comprehensive evaluation, and interactive dashboard.

---

## 📁 Project Structure

```
FUTURE_ML_01/
├── data/
│   ├── churn_data.csv           # Generated telecom churn dataset (1000 samples)
│   └── sales_data.csv           # Sales forecasting dataset
├── src/
│   ├── data_generator.py        # Generate synthetic churn data
│   ├── churn_preprocessor.py    # Data preprocessing & encoding
│   ├── churn_models.py          # Model training (3 algorithms)
│   ├── churn_evaluator.py       # Model evaluation & metrics
│   ├── data_cleaner.py          # Sales data cleaning
│   └── forecasting_model.py     # Sales forecasting model
├── churn_dashboard.py           # Interactive Streamlit dashboard
├── dashboard.py                 # Sales forecasting dashboard
├── test_churn_pipeline.py       # Complete pipeline test
├── requirements.txt             # Dependencies
└── README.md                    # Documentation
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Churn Prediction Dashboard

```bash
streamlit run churn_dashboard.py
```

Visit: `http://localhost:8501`

### 3. Run the Test Pipeline (Optional)

```bash
python3 test_churn_pipeline.py
```

---

## 📊 STEP-BY-STEP IMPLEMENTATION

### STEP 1: DATASET

**File**: `src/data_generator.py`

```python
from src.data_generator import generate_churn_dataset

# Generate 1000 customer records
df = generate_churn_dataset(n_samples=1000)
```

**Features Generated**:
- Customer demographics (Age)
- Contract details (Tenure, Contract Type)
- Service usage (Internet, Streaming, Support)
- Charges (Monthly, Total)
- Engagement (Customer Service Calls)
- Target: Churn (Yes/No)

---

### STEP 2: PREPROCESSING

**File**: `src/churn_preprocessor.py`

#### Encode Categorical Features

```python
from src.churn_preprocessor import ChurnPreprocessor

preprocessor = ChurnPreprocessor()

# Categorical columns are automatically encoded:
# - Contract_Type, Internet_Service, Tech_Support, etc.
# - Uses LabelEncoder to convert to numeric values
```

#### Scale Numerical Values

```python
# Numerical columns are scaled using StandardScaler:
# - Age, Tenure_Months, Monthly_Charges, Total_Charges, Customer_Service_Calls
# - Mean = 0, Standard Deviation = 1
```

#### Complete Pipeline

```python
from src.churn_preprocessor import preprocess_churn_data

X_train, X_test, y_train, y_test, preprocessor = preprocess_churn_data(
    df, 
    test_size=0.2,
    random_state=42
)
```

---

### STEP 3: BUILD MODELS

**File**: `src/churn_models.py`

#### Model 1: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**When to use**: Fast, interpretable, good baseline

#### Model 2: Random Forest

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**When to use**: Better accuracy, handles non-linear patterns

#### Model 3: XGBoost

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**When to use**: Best performance, gradient boosting

#### Train All Models

```python
from src.churn_models import train_churn_models

trainer = train_churn_models(X_train, y_train)
```

---

### STEP 4: EVALUATE MODELS

**File**: `src/churn_evaluator.py`

#### Accuracy

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)
# Percentage of correct predictions (0-1)
```

#### Precision

```python
from sklearn.metrics import precision_score

precision = precision_score(y_test, predictions)
# Of predicted churners, how many actually churned
```

#### Recall

```python
from sklearn.metrics import recall_score

recall = recall_score(y_test, predictions)
# Of actual churners, how many did we catch
```

#### F1-Score

```python
from sklearn.metrics import f1_score

f1 = f1_score(y_test, predictions)
# Harmonic mean of Precision & Recall
```

#### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, predictions)
#
#        Predicted
#        No    Yes
# Actual
# No     TN    FP
# Yes    FN    TP
#
```

#### Feature Importance

```python
# Random Forest
feature_importance = model.feature_importances_

# XGBoost
feature_importance = model.feature_importances_

# Logistic Regression
feature_importance = np.abs(model.coef_[0])
```

#### Complete Evaluation

```python
from src.churn_evaluator import evaluate_all_models

evaluator = evaluate_all_models(trainer, X_test, y_test)

# Get comparison table
comparison = evaluator.compare_models()
print(comparison)

# Get best model
best_model, score = evaluator.get_best_model('f1')
print(f"Best: {best_model} (F1: {score})")

# Get confusion matrix
cm = evaluator.get_confusion_matrix('Random Forest')

# Plot feature importance
fig = evaluator.plot_feature_importance('Random Forest', feature_importance)
```

---

## 📈 Dashboard Features

### Tab 1: Overview
- Dataset statistics
- Churn distribution
- Age analysis

### Tab 2: Model Comparison
- Performance metrics table
- Best model highlighting
- Metrics explanation

### Tab 3: Detailed Analysis
- Confusion matrix visualization
- Feature importance plot
- Classification report

### Tab 4: Predictions
- Interactive customer profile
- Real-time churn probability
- Multi-model prediction

### Tab 5: Data Info
- Data types & missing values
- Statistical summary
- Raw data sample

---

## 📊 Test Results

```
====================================================
📊 MODEL COMPARISON RESULTS
====================================================

                      Accuracy  Precision   Recall  F1-Score  ROC-AUC
Logistic Regression     0.5950    0.6000   0.6408   0.6197   0.6539
Random Forest           0.6350    0.6293   0.7087   0.6667   0.6858  ← Best
XGBoost                 0.6300    0.6306   0.6796   0.6542   0.6395
```

---

## 🔧 Customization

### Use Your Own Data

Replace `data/churn_data.csv` with your data:

```csv
CustomerID,Age,Tenure_Months,Monthly_Charges,...,Churn
CUST001,35,24,65.50,...,0
```

### Adjust Model Parameters

Edit `src/churn_models.py`:

```python
# Random Forest
RandomForestClassifier(
    n_estimators=200,  # Increase for better accuracy
    max_depth=20,      # Increase for complex patterns
    min_samples_split=5,  # Lower for more splits
)
```

### Add New Features

In `src/data_generator.py`:

```python
data['new_feature'] = np.random.randint(0, 100, n_samples)
```

---

## 🎯 Interpretation Guide

### Understanding Metrics

| Metric | Best Value | Formula |
|--------|-----------|---------|
| Accuracy | 1.0 | (TP+TN)/(TP+TN+FP+FN) |
| Precision | 1.0 | TP/(TP+FP) |
| Recall | 1.0 | TP/(TP+FN) |
| F1-Score | 1.0 | 2·(Precision·Recall)/(Precision+Recall) |
| ROC-AUC | 1.0 | Area under ROC curve |

### Confusion Matrix Terms

- **TP (True Positive)**: Correctly predicted churner
- **TN (True Negative)**: Correctly predicted non-churner
- **FP (False Positive)**: Incorrectly predicted as churner
- **FN (False Negative)**: Incorrectly predicted as non-churner

### Feature Importance

Higher importance = more impact on churn prediction

Example:
1. Contract Type (0.19) - Most important
2. Monthly Charges (0.15)
3. Tenure (0.14)

---

## 💡 Best Practices

1. **Always scale numerical features** before training
2. **Encode categorical features** using LabelEncoder or OneHotEncoder
3. **Split data** into train/test (80/20 or 70/30)
4. **Handle imbalanced data** using SMOTE or class weights
5. **Validate with stratified cross-validation**
6. **Monitor for overfitting** on test set
7. **Use ensemble methods** for better predictions
8. **Interpret feature importance** to understand decisions

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'prophet'" | `pip install -r requirements.txt` |
| Dashboard won't start | Check port 8501 is available |
| Low accuracy | Add more data or tune hyperparameters |
| Slow training | Reduce n_estimators or use smaller dataset |

---

## 📚 Resources

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Customer Churn Guide](https://en.wikipedia.org/wiki/Customer_attrition)

---

## ✅ Next Steps

1. ✅ Run `test_churn_pipeline.py` to verify setup
2. ✅ Launch dashboard: `streamlit run churn_dashboard.py`
3. ✅ Explore different models and metrics
4. ✅ Make predictions on new customers
5. ✅ Deploy to production

---

**Built with ❤️ using Scikit-Learn, XGBoost & Streamlit**
