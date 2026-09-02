import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for feature engineering
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_out = X.copy()
        
        # Total Square Footage
        bsmt = X_out['TotalBsmtSF'].fillna(0)
        flr1 = X_out['1stFlrSF'].fillna(0)
        flr2 = X_out['2ndFlrSF'].fillna(0)
        X_out['TotalSF'] = bsmt + flr1 + flr2
        
        # Total Bathrooms
        full = X_out.get('FullBath', 0).fillna(0)
        half = X_out.get('HalfBath', 0).fillna(0)
        bsmt_full = X_out.get('BsmtFullBath', 0).fillna(0)
        bsmt_half = X_out.get('BsmtHalfBath', 0).fillna(0)
        X_out['TotalBathrooms'] = full + (half * 0.5) + bsmt_full + (bsmt_half * 0.5)
        
        # Total Porch SF
        open_porch = X_out.get('OpenPorchSF', 0).fillna(0)
        enclosed = X_out.get('EnclosedPorch', 0).fillna(0)
        ssn3 = X_out.get('3SsnPorch', 0).fillna(0)
        screen = X_out.get('ScreenPorch', 0).fillna(0)
        X_out['TotalPorchSF'] = open_porch + enclosed + ssn3 + screen
        
        # House Age and Remod Age
        yr_sold = X_out.get('YrSold', 2010).fillna(2010)
        yr_built = X_out.get('YearBuilt', yr_sold).fillna(yr_sold)
        yr_remod = X_out.get('YearRemodAdd', yr_sold).fillna(yr_sold)
        
        X_out['HouseAge'] = yr_sold - yr_built
        X_out['RemodAge'] = yr_sold - yr_remod
        
        # Total Finished SF
        bsmt_fin1 = X_out.get('BsmtFinSF1', 0).fillna(0)
        bsmt_fin2 = X_out.get('BsmtFinSF2', 0).fillna(0)
        X_out['TotalFinishedSF'] = bsmt_fin1 + bsmt_fin2 + flr1 + flr2
        
        # We can drop the original ones to prevent multicollinearity, but Tree models handle it fine.
        # However, to keep it clean, we will let ColumnTransformer pick what it needs.
        
        return X_out

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Builds the preprocessing pipeline using ColumnTransformer
    """
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop' # Drop columns that are not explicitly specified
    )
    
    # Complete feature engineering + preprocessing pipeline
    full_pipeline = Pipeline(steps=[
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

def get_feature_columns(df):
    """
    Given a dataframe, separate numerical and categorical columns.
    Excludes ID and target variable.
    """
    # Target and ID
    exclude = ['Id', 'SalePrice']
    features = [col for col in df.columns if col not in exclude]
    
    # We will add the engineered features to the lists so ColumnTransformer uses them
    engineered_num_cols = ['TotalSF', 'TotalBathrooms', 'TotalPorchSF', 'HouseAge', 'RemodAge', 'TotalFinishedSF']
    
    # We also need to exclude the original ones that were used to create them if desired, but we can just use all numerical
    num_cols = [col for col in features if df[col].dtype in ['int64', 'float64']] + engineered_num_cols
    # Remove duplicates if any
    num_cols = list(set(num_cols))
    
    cat_cols = [col for col in features if df[col].dtype == 'object']
    
    return num_cols, cat_cols
