import pandas as pd
import numpy as np
import joblib
import os
import sys
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from preprocessing import build_preprocessor, get_feature_columns
from evaluate_model import evaluate_predictions, print_evaluation

def train():
    data_path = '../data/train.csv'
    # Adjust path if run from root
    if not os.path.exists(data_path):
        data_path = 'data/train.csv'
    
    if not os.path.exists(data_path):
        print("Dataset not found. Please place train.csv in the data/ directory.")
        sys.exit(1)
        
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    X = df.drop(['SalePrice'], axis=1)
    y = df['SalePrice']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get columns
    num_cols, cat_cols = get_feature_columns(df)
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # Model 1: Linear Regression
    print("Training Linear Regression...")
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_preds = lr_pipeline.predict(X_test)
    lr_metrics = evaluate_predictions(y_test, lr_preds)
    print_evaluation("Linear Regression", lr_metrics)
    
    # Model 2: Random Forest Regression with tuning
    print("Training Random Forest Regression...")
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Simple tuning for speed
    param_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2],
    }
    
    # Use few CVs to speed up execution
    grid_search = GridSearchCV(rf_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    rf_preds = best_rf.predict(X_test)
    rf_metrics = evaluate_predictions(y_test, rf_preds)
    print_evaluation("Random Forest (Tuned)", rf_metrics)
    
    # Compare and save
    if rf_metrics['RMSE'] < lr_metrics['RMSE']:
        best_model = best_rf
        print("Random Forest selected as the best model.")
    else:
        best_model = lr_pipeline
        print("Linear Regression selected as the best model.")
        
    # Save metrics to a file so Streamlit app can read it
    metrics_df = pd.DataFrame([
        {'Model': 'Linear Regression', **lr_metrics},
        {'Model': 'Random Forest', **rf_metrics}
    ])
    os.makedirs('models', exist_ok=True)
    metrics_df.to_csv('models/model_metrics.csv', index=False)
    
    # Save the pipeline
    joblib.dump(best_model, 'models/house_price_pipeline.pkl')
    print("Model saved to models/house_price_pipeline.pkl")

if __name__ == '__main__':
    train()
