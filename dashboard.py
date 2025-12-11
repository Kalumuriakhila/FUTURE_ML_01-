import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from prophet import Prophet
import warnings

# Import custom modules
from src.data_cleaner import load_data, clean_data, prepare_for_prophet, get_data_summary
from src.forecasting_model import train_prophet_model, forecast_sales, get_forecast_summary, get_model_components

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
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

# Title
st.markdown('<div class="main-header">📊 Sales Forecasting Dashboard</div>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## ⚙️ Configuration")
forecast_days = st.sidebar.slider("Forecast Period (days)", min_value=7, max_value=90, value=30, step=7)
confidence_level = st.sidebar.select_slider("Confidence Interval", options=[80, 85, 90, 95, 99], value=95)

# Load and process data
@st.cache_data
def load_and_process_data():
    df = load_data('data/sales_data.csv')
    df_clean = clean_data(df)
    df_prophet = prepare_for_prophet(df_clean)
    return df_clean, df_prophet

# Train model
@st.cache_resource
def train_model(df_prophet):
    model = train_prophet_model(df_prophet, yearly_seasonality=True, weekly_seasonality=True)
    return model

try:
    df_clean, df_prophet = load_and_process_data()
    model = train_model(df_prophet)
    forecast = forecast_sales(model, periods=forecast_days)
    
    # Data Summary Section
    st.markdown("## 📋 Data Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    summary = get_data_summary(df_clean)
    
    with col1:
        st.metric("Total Records", summary['total_records'])
    with col2:
        st.metric("Avg Sales", f"${summary['avg_sales']:,.0f}")
    with col3:
        st.metric("Min Sales", f"${summary['min_sales']:,.0f}")
    with col4:
        st.metric("Max Sales", f"${summary['max_sales']:,.0f}")
    with col5:
        st.metric("Std Dev", f"${summary['std_sales']:,.0f}")
    
    st.info(f"📅 Date Range: {summary['date_range']}")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Historical Data", "Forecast", "Components", "Analysis"])
    
    with tab1:
        st.markdown("### Historical Sales Data")
        
        # Historical chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_clean['date'],
            y=df_clean['sales'],
            mode='lines+markers',
            name='Historical Sales',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        fig.update_layout(
            title="Historical Sales Trend",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        if st.checkbox("Show data table", key="historical_table"):
            st.dataframe(df_clean.tail(20), use_container_width=True)
    
    with tab2:
        st.markdown("### Sales Forecast")
        
        # Get forecast data
        forecast_summary = get_forecast_summary(forecast, periods=forecast_days)
        
        # Forecast metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Forecasted Sales", f"${forecast_summary['avg_forecasted_sales']:,.0f}")
        with col2:
            st.metric("Min Forecast", f"${forecast_summary['min_forecasted_sales']:,.0f}")
        with col3:
            st.metric("Max Forecast", f"${forecast_summary['max_forecasted_sales']:,.0f}")
        with col4:
            st.metric("Trend", forecast_summary['forecast_trend'])
        
        # Combined chart - historical + forecast
        historical_forecast = forecast.iloc[:len(df_clean)]
        future_forecast = forecast.iloc[len(df_clean):]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=df_clean['date'],
            y=df_clean['sales'],
            mode='lines',
            name='Historical Sales',
            line=dict(color='#1f77b4', width=2)
        ))
        
        # Forecasted values
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'],
            y=future_forecast['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
            y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 127, 14, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name=f'Confidence Interval ({confidence_level}%)'
        ))
        
        fig.update_layout(
            title=f"Sales Forecast - Next {forecast_days} Days",
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        if st.checkbox("Show forecast table", key="forecast_table"):
            forecast_display = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            forecast_display.columns = ['Date', 'Forecast', 'Lower Bound', 'Upper Bound']
            forecast_display['Date'] = forecast_display['Date'].dt.date
            st.dataframe(forecast_display, use_container_width=True)
    
    with tab3:
        st.markdown("### Model Components")
        
        components = get_model_components(model)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write("**Trend Type**")
            st.write(components['trend'])
        with col2:
            st.write("**Yearly Seasonality**")
            st.write("✅ Enabled" if components['yearly_seasonality'] else "❌ Disabled")
        with col3:
            st.write("**Weekly Seasonality**")
            st.write("✅ Enabled" if components['weekly_seasonality'] else "❌ Disabled")
        with col4:
            st.write("**Daily Seasonality**")
            st.write("✅ Enabled" if components['daily_seasonality'] else "❌ Disabled")
        
        st.markdown("---")
        st.markdown("### Decomposed Components")
        
        # Plot components
        fig = model.plot_components(forecast, period_num_scales=3)
        st.pyplot(fig)
    
    with tab4:
        st.markdown("### Forecast Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Key Insights")
            future_data = forecast.tail(forecast_days)
            
            # Calculate growth
            first_forecast = future_data['yhat'].iloc[0]
            last_forecast = future_data['yhat'].iloc[-1]
            growth_percent = ((last_forecast - first_forecast) / first_forecast) * 100
            
            st.write(f"""
            - **Forecast Period**: {forecast_days} days
            - **Expected Growth**: {growth_percent:.1f}%
            - **Average Daily Sales**: ${future_data['yhat'].mean():,.0f}
            - **Peak Expected**: ${future_data['yhat'].max():,.0f}
            - **Trough Expected**: ${future_data['yhat'].min():,.0f}
            """)
        
        with col2:
            st.markdown("#### Model Performance")
            
            # Calculate MAPE (Mean Absolute Percentage Error) on historical data
            historical_forecast = forecast.iloc[:len(df_clean)]
            mape = np.mean(np.abs((df_clean['sales'].values - historical_forecast['yhat'].values) / df_clean['sales'].values)) * 100
            
            st.write(f"""
            - **Model Type**: Facebook Prophet
            - **Training Data Points**: {len(df_clean)}
            - **Estimated MAPE**: {mape:.2f}%
            - **Confidence Level**: {confidence_level}%
            """)
        
        # Residuals analysis
        st.markdown("#### Residuals Distribution")
        historical_forecast = forecast.iloc[:len(df_clean)]
        residuals = df_clean['sales'].values - historical_forecast['yhat'].values
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=residuals, nbinsx=30, name='Residuals'))
        fig.update_layout(
            title="Distribution of Residuals",
            xaxis_title="Residual Value",
            yaxis_title="Frequency",
            template='plotly_white',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.info("Please ensure all required files are in place and try again.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🚀 AI-Powered Sales Forecasting Dashboard | Built with Streamlit & Facebook Prophet</p>
</div>
""", unsafe_allow_html=True)
