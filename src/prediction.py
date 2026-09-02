import joblib
import pandas as pd
import os

class HousePricePredictor:
    def __init__(self, model_path='models/house_price_pipeline.pkl'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
        self.model = joblib.load(model_path)
        
    def predict(self, input_data: dict) -> float:
        """
        Takes a dictionary of house features and returns a predicted price.
        """
        # Convert single dict to DataFrame
        df = pd.DataFrame([input_data])
        
        # The pipeline handles missing values, feature engineering, and encoding
        prediction = self.model.predict(df)
        
        return prediction[0]
