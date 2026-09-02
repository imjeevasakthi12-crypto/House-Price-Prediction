import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    """
    Evaluates predictions and returns MAE, MSE, RMSE, R2 Score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

def print_evaluation(model_name, metrics):
    """
    Utility to print metrics nicely.
    """
    print(f"--- Evaluation for {model_name} ---")
    print(f"MAE:  {metrics['MAE']:.2f}")
    print(f"MSE:  {metrics['MSE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2:   {metrics['R2']:.4f}\n")
