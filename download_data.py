import pandas as pd
from sklearn.datasets import fetch_openml
import os

print("Fetching Ames Housing dataset from OpenML...")
# Dataset ID 42165 is the house_prices dataset
housing = fetch_openml(data_id=42165, as_frame=True, parser='auto')
df = housing.frame

# OpenML names might be slightly different or same. 
# Kaggle uses 'SalePrice' for target. OpenML uses 'SalePrice'.
if 'SalePrice' not in df.columns and housing.target.name == 'SalePrice':
    df['SalePrice'] = housing.target
elif 'SalePrice' not in df.columns:
    # Just in case target is named differently in OpenML
    df['SalePrice'] = housing.target

os.makedirs('data', exist_ok=True)
df.to_csv('data/train.csv', index=False)
print(f"Dataset successfully saved to data/train.csv with {df.shape[0]} rows and {df.shape[1]} columns.")
