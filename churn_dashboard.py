import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from src.data_generator import generate_churn_dataset
from src.churn_preprocessor import preprocess_churn_data, ChurnPreprocessor
from src.churn_models import train_churn_models
from src.churn_evaluator import evaluate_all_models

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #d32f2f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">⚠️ Customer Churn Prediction</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ Configuration")

# Load data
@st.cache_data
def load_churn_data():
    try:
        df = pd.read_csv('data/churn_data.csv')
    except:
        df = generate_churn_dataset(1000)
        df.to_csv('data/churn_data.csv', index=False)
    return df

@st.cache_resource
def train_models_cached(X_train, y_train):
    return train_churn_models(X_train, y_train)

@st.cache_resource
def evaluate_models_cached(trainer, X_test, y_test):
    return evaluate_all_models(trainer, X_test, y_test)

try:
    # Load and preprocess data
    df = load_churn_data()
    X_train, X_test, y_train, y_test, preprocessor = preprocess_churn_data(df, test_size=0.2)
    
    # Train models
    trainer = train_models_cached(X_train, y_train)
    
    # Evaluate models
    evaluator = evaluate_models_cached(trainer, X_test, y_test)
    
    # Get feature names
    feature_names = preprocessor.feature_names
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "🎯 Model Comparison",
        "📈 Detailed Analysis",
        "🔮 Predictions",
        "📋 Data Info"
    ])
    
    # TAB 1: Overview
    with tab1:
        st.markdown("## Dataset Overview")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Customers", len(df))
        with col2:
            churn_count = (df['Churn'] == 1).sum()
            st.metric("Churned", churn_count)
        with col3:
            churn_rate = (df['Churn'].sum() / len(df)) * 100
            st.metric("Churn Rate", f"{churn_rate:.1f}%")
        with col4:
            st.metric("Training Samples", len(X_train))
        with col5:
            st.metric("Test Samples", len(X_test))
        
        st.markdown("---")
        st.markdown("## Churn Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            churn_counts = df['Churn'].value_counts()
            fig = go.Figure(data=[
                go.Pie(labels=['No Churn', 'Churned'], values=[churn_counts[0], churn_counts[1]],
                        marker=dict(colors=['#4CAF50', '#f44336']))
            ])
            fig.update_layout(title="Churn Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Age distribution by churn
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=df[df['Churn']==0]['Age'], name='No Churn', opacity=0.7))
            fig.add_trace(go.Histogram(x=df[df['Churn']==1]['Age'], name='Churned', opacity=0.7))
            fig.update_layout(title="Age Distribution by Churn", barmode='overlay', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # TAB 2: Model Comparison
    with tab2:
        st.markdown("## Model Performance Comparison")
        
        # Comparison table
        comparison_df = evaluator.compare_models()
        
        st.dataframe(
            comparison_df.style.format("{:.4f}").highlight_max(axis=0, color='lightgreen'),
            use_container_width=True
        )
        
        best_model, best_score = evaluator.get_best_model('f1')
        st.success(f"🏆 Best Model: **{best_model}** (F1-Score: {best_score:.4f})")
        
        st.markdown("---")
        st.markdown("## Metrics Explanation")
        
        metrics_info = {
            "Accuracy": "Percentage of correct predictions",
            "Precision": "Of predicted churners, how many actually churned",
            "Recall": "Of actual churners, how many were caught",
            "F1-Score": "Harmonic mean of Precision and Recall",
            "ROC-AUC": "Area under the ROC curve (0.5 = random, 1.0 = perfect)"
        }
        
        for metric, explanation in metrics_info.items():
            st.write(f"**{metric}**: {explanation}")
    
    # TAB 3: Detailed Analysis
    with tab3:
        st.markdown("## Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Select model for detailed analysis:",
            list(trainer.trained_models.keys())
        )
        
        col1, col2, col3 = st.columns(3)
        
        summary = evaluator.get_model_summary(selected_model)
        
        with col1:
            st.metric("Accuracy", f"{summary['accuracy']:.4f}")
            st.metric("Precision", f"{summary['precision']:.4f}")
        
        with col2:
            st.metric("Recall", f"{summary['recall']:.4f}")
            st.metric("F1-Score", f"{summary['f1_score']:.4f}")
        
        with col3:
            st.metric("ROC-AUC", f"{summary['roc_auc']}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        # Confusion Matrix
        with col1:
            st.markdown("### Confusion Matrix")
            fig = evaluator.plot_confusion_matrix(selected_model)
            st.pyplot(fig)
        
        # Feature Importance
        with col2:
            st.markdown("### Feature Importance")
            try:
                feature_importance = trainer.get_feature_importance(selected_model, feature_names)
                fig = evaluator.plot_feature_importance(selected_model, feature_importance, top_n=10)
                st.pyplot(fig)
            except Exception as e:
                st.info(f"Feature importance not available for {selected_model}")
        
        # Classification Report
        st.markdown("### Detailed Classification Report")
        st.text(summary.get('classification_report', summary))
    
    # TAB 4: Predictions
    with tab4:
        st.markdown("## Make Predictions")
        
        st.info("Enter customer details to predict churn probability:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=24)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
        
        with col2:
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=1500.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        
        with col3:
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
            cust_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=10, value=2)
        
        if st.button("🔮 Predict Churn"):
            # Create customer data
            customer_data = pd.DataFrame({
                'CustomerID': ['TEST001'],
                'Age': [age],
                'Tenure_Months': [tenure],
                'Monthly_Charges': [monthly_charges],
                'Total_Charges': [total_charges],
                'Contract_Type': [contract],
                'Internet_Service': [internet],
                'Online_Security': ['No'],
                'Online_Backup': ['No'],
                'Device_Protection': ['No'],
                'Tech_Support': [tech_support],
                'Streaming_TV': [streaming_tv],
                'Streaming_Movies': ['No'],
                'Paperless_Billing': ['Yes'],
                'Payment_Method': ['Electronic check'],
                'Customer_Service_Calls': [cust_service_calls],
                'Churn': [0]
            })
            
            # Preprocess
            X_customer, _ = preprocessor.transform(customer_data, target_col='Churn')
            
            st.markdown("---")
            st.markdown("### Predictions")
            
            col1, col2, col3 = st.columns(3)
            
            for i, model_name in enumerate(trainer.trained_models.keys()):
                pred = trainer.predict(model_name, X_customer)[0]
                proba = trainer.predict_proba(model_name, X_customer)[0]
                churn_prob = proba[1] * 100
                
                if i < 3:
                    with [col1, col2, col3][i]:
                        if pred == 1:
                            st.error(f"🚨 {model_name}")
                            st.metric("Churn Risk", f"{churn_prob:.1f}%")
                        else:
                            st.success(f"✅ {model_name}")
                            st.metric("Retention Prob", f"{(100-churn_prob):.1f}%")
    
    # TAB 5: Data Info
    with tab5:
        st.markdown("## Dataset Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Data Types & Missing Values")
            info_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Missing': df.isnull().sum()
            })
            st.dataframe(info_df, use_container_width=True)
        
        with col2:
            st.markdown("### Statistical Summary")
            st.dataframe(df.describe().T, use_container_width=True)
        
        st.markdown("### Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure the data and models are properly configured.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🎯 Customer Churn Prediction | Built with Streamlit, Scikit-Learn & XGBoost</p>
</div>
""", unsafe_allow_html=True)
