# 🏠 House Price Prediction

A complete, end-to-end Machine Learning web application built with Scikit-Learn and Streamlit to predict the sale price of residential properties using the Ames Housing Dataset.

## 🔗 Application URL
**Local App URL:** [http://localhost:8501](http://localhost:8501)

*(Note: The server must be running for this link to work. See instructions below).*

---

## 🚀 How to Run

1. **Install Requirements:**
   ```powershell
   pip install -r requirements.txt
   ```

2. **Train the Model:**
   *If you haven't trained the model yet, run the training script to generate the `models/house_price_pipeline.pkl`.*
   ```powershell
   python src/train_model.py
   ```

3. **Start the Web Application:**
   ```powershell
   streamlit run app.py
   ```
   *This command will automatically open the application in your default web browser.*

## 📁 Project Structure
- `app.py`: Main Streamlit application with the custom User Interface.
- `src/`: Core logic for preprocessing, training, evaluating, and predicting.
- `models/`: Saved model pipelines (`.pkl`) and evaluation metrics.
- `data/`: Contains the dataset (`train.csv`).
- `notebooks/`: Exploratory Data Analysis (EDA) notebook.

## 🤖 Models Used
- **Baseline:** Linear Regression
- **Best Model:** Random Forest Regressor (Hyperparameter Tuned via GridSearchCV)
