# 📊 AI-Powered Sales Forecasting Dashboard

A machine learning-powered sales forecasting dashboard built with **Streamlit** and **Facebook Prophet**. This application predicts future sales using time-series analysis and provides interactive visualizations for sales trends and forecasts.

## 🎯 Features

- **Historical Sales Analysis**: View and analyze past sales trends
- **Time-Series Forecasting**: Predict future sales using Prophet
- **Interactive Visualizations**: Explore data with Plotly charts
- **Confidence Intervals**: See upper and lower bounds of predictions
- **Model Components**: Understand trend and seasonality patterns
- **Forecast Analysis**: Get insights on growth, peaks, and troughs
- **Data Summary**: Statistical overview of historical data

## 📁 Project Structure

```
FUTURE_ML_01/
├── data/
│   └── sales_data.csv          # Sample sales dataset
├── src/
│   ├── data_cleaner.py         # Data loading & cleaning
│   └── forecasting_model.py    # Prophet model & forecasting
├── dashboard.py                # Streamlit dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open at `http://localhost:8501`

## 📊 Data Format

Your CSV file should contain:
```csv
date,sales
2023-01-01,15000
2023-01-02,16500
...
```

## 🔧 How It Works

### Step 1: Data Cleaning
- Converts date column to datetime format
- Handles missing values using forward/backward fill
- Sorts data chronologically

### Step 2: Model Training
- Trains Facebook Prophet model on historical data
- Includes yearly and weekly seasonality
- Estimates trend and changepoints automatically

### Step 3: Sales Forecasting
- Generates predictions for next N days (adjustable: 7-90 days)
- Provides confidence intervals (80%-99% adjustable)
- Extracts trend information (increasing/decreasing)

### Step 4: Dashboard Display
- **Historical Data Tab**: View raw sales trends
- **Forecast Tab**: See predictions with confidence bounds
- **Components Tab**: Analyze trend and seasonality
- **Analysis Tab**: Get insights on model performance

## 📈 Dashboard Sections

### Data Summary
- Total number of records
- Date range covered
- Average, min, max sales values
- Standard deviation

### Configuration (Sidebar)
- Adjust forecast period (7-90 days)
- Set confidence interval level (80%-99%)

### Four Main Tabs

1. **Historical Data**
   - Line chart of past sales
   - Optional data table view

2. **Forecast**
   - Combined historical + forecast chart
   - Confidence interval visualization
   - Key metrics for forecasted period
   - Optional forecast data table

3. **Components**
   - Model configuration details
   - Decomposed trend and seasonality plots
   - Component analysis

4. **Analysis**
   - Key insights and growth metrics
   - Model performance statistics
   - Residuals distribution

## 🛠️ Customization

### Add Your Own Data
Replace `data/sales_data.csv` with your own dataset containing `date` and `sales` columns.

### Adjust Forecast Parameters
Edit `src/forecasting_model.py`:
```python
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    changepoint_prior_scale=0.05  # Adjust sensitivity to changes
)
```

### Modify Dashboard Layout
Edit `dashboard.py` to customize colors, add/remove sections, or change visualizations.

## 📊 Sample Dataset

The project includes a sample dataset with 150 days of sales data showing an upward trend. This is perfect for testing the dashboard before using your own data.

## �� Interpreting Results

### Forecast Tab
- **Blue line**: Historical sales data
- **Orange dashed line**: Forecasted sales
- **Shaded area**: Confidence interval (uncertainty range)

### Analysis Tab
- **MAPE**: Lower percentage = better model accuracy
- **Growth %**: Positive = increasing trend, Negative = decreasing trend
- **Residuals**: Should follow a roughly normal distribution

## 📦 Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **prophet**: Time-series forecasting
- **streamlit**: Interactive web dashboard
- **plotly**: Interactive visualizations
- **scikit-learn**: Machine learning utilities

## 🎓 Learning Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Prophet Documentation](https://facebook.github.io/prophet/)
- [Plotly Documentation](https://plotly.com/python/)
- [Time Series Forecasting Guide](https://en.wikipedia.org/wiki/Time_series)

## 📝 Notes

- First run will take longer as Prophet compiles the model
- Subsequent runs will be cached for better performance
- Ensure your date column is properly formatted (YYYY-MM-DD)
- Handle any missing values before running the dashboard

## 🐛 Troubleshooting

**Issue**: "No module named 'prophet'"
- Solution: Run `pip install -r requirements.txt`

**Issue**: Dashboard loads but shows error
- Solution: Check that `data/sales_data.csv` exists with proper formatting

**Issue**: Forecast seems inaccurate
- Solution: Ensure you have at least 90-100 data points; more data = better predictions

## 📄 License

This project is open source and available for educational and commercial use.

## 🤝 Contributing

Feel free to fork, modify, and enhance this project. Share your improvements!

---

**Built with ❤️ using Streamlit & Facebook Prophet**
