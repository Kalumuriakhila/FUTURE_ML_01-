import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_churn_dataset(n_samples=1000):
    """Generate synthetic telecom churn dataset."""
    np.random.seed(42)
    
    data = {
        'CustomerID': [f'CUST{i:05d}' for i in range(1, n_samples + 1)],
        'Age': np.random.randint(18, 80, n_samples),
        'Tenure_Months': np.random.randint(1, 73, n_samples),
        'Monthly_Charges': np.random.uniform(20, 120, n_samples),
        'Total_Charges': np.random.uniform(100, 8000, n_samples),
        'Contract_Type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
        'Internet_Service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
        'Online_Security': np.random.choice(['Yes', 'No'], n_samples),
        'Online_Backup': np.random.choice(['Yes', 'No'], n_samples),
        'Device_Protection': np.random.choice(['Yes', 'No'], n_samples),
        'Tech_Support': np.random.choice(['Yes', 'No'], n_samples),
        'Streaming_TV': np.random.choice(['Yes', 'No'], n_samples),
        'Streaming_Movies': np.random.choice(['Yes', 'No'], n_samples),
        'Paperless_Billing': np.random.choice(['Yes', 'No'], n_samples),
        'Payment_Method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'Customer_Service_Calls': np.random.randint(0, 10, n_samples),
        'Churn': np.random.choice([0, 1], n_samples, p=[0.73, 0.27])
    }
    
    df = pd.DataFrame(data)
    
    # Add some correlation to churn
    short_tenure = df['Tenure_Months'] < 12
    monthly_contract = df['Contract_Type'] == 'Month-to-month'
    high_charges = df['Monthly_Charges'] > 80
    
    churn_indices = df[short_tenure | monthly_contract | high_charges].index
    df.loc[churn_indices, 'Churn'] = np.where(
        np.random.random(len(churn_indices)) < 0.5,
        1,
        df.loc[churn_indices, 'Churn']
    )
    
    return df

if __name__ == "__main__":
    df = generate_churn_dataset(1000)
    df.to_csv('data/churn_data.csv', index=False)
    print(f"✓ Generated churn dataset with {len(df)} samples")
    print(f"✓ Churn rate: {df['Churn'].mean():.2%}")
    print(f"✓ Saved to data/churn_data.csv")
