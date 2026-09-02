from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import sys

sys.path.append(os.path.abspath('src'))
try:
    from prediction import HousePricePredictor
except ImportError:
    pass

app = Flask(__name__)

# Initialize Predictor
predictor = None
try:
    predictor = HousePricePredictor('models/house_price_pipeline.pkl')
except Exception as e:
    print(f"Warning: Model not found. {e}")

# Load default base values to merge with user input
default_data = {}
try:
    df = pd.read_csv('data/train.csv')
    default_data = df.drop(['SalePrice', 'Id'], axis=1, errors='ignore').mode().iloc[0].to_dict()
except:
    pass

# Load metrics
rf_metrics = {'R2': 0.89, 'MAE': 17553, 'RMSE': 29177}
try:
    metrics_df = pd.read_csv('models/model_metrics.csv')
    rf_row = metrics_df[metrics_df['Model'] == 'Random Forest'].iloc[0]
    rf_metrics = {'R2': rf_row['R2'], 'MAE': rf_row['MAE'], 'RMSE': rf_row['RMSE']}
except:
    pass

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({'error': 'Model not trained yet!'}), 500
        
    user_input = request.json
    
    # Merge default values with user input
    input_data = default_data.copy()
    for key, value in user_input.items():
        if value is not None and str(value).strip() != "":
            input_data[key] = float(value)
            
    # Predict
    try:
        prediction = predictor.predict(input_data)
        
        lower_bound = prediction * 0.94
        upper_bound = prediction * 1.05
        
        return jsonify({
            'price': prediction,
            'formatted_price': f"${prediction:,.0f}",
            'price_range': f"${lower_bound:,.0f} - ${upper_bound:,.0f}",
            'metrics': {
                'r2': f"{rf_metrics['R2']:.2f}",
                'mae': f"${rf_metrics['MAE']:,.0f}",
                'rmse': f"${rf_metrics['RMSE']:,.0f}"
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import webbrowser
    import threading
    
    print("\n" + "="*60)
    print(" HOUSE PRICE PREDICTION APP STARTED SUCCESSFULLY ")
    print("="*60)
    print("\nYou can now view your application in your browser.")
    print("\nLocal URL: http://localhost:8501\n")
    print("="*60 + "\n")
    
    # Automatically open the web browser after a short delay
    threading.Timer(1.5, lambda: webbrowser.open('http://localhost:8501')).start()
    
    # Start the Flask server
    app.run(host='0.0.0.0', port=8501, debug=False)
