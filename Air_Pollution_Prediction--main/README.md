
🌫️  
## Air Pollution Prediction
This project uses machine learning models to predict air pollution levels based on historical data. It includes preprocessing, model training, and a basic frontend form for input.
📦  
## Installation
- Clone the repository
git clone https://github.com/yourusername/AirPollutionPrediction.git
cd AirPollutionPrediction
- Install required Python packages
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost joblib
- Add the dataset
Place the AIR.csv file in the root directory of the project. Ensure it contains the expected columns used in the scripts.
🚀 Usage
- Run preprocessing and model training scripts:
python data_preprocessing.py
python extra_tree_classifier.py
python random_forest_classifier.py
python xgb_classifier.py
- Open index.html in a web browser to use the input form.
(Note: Backend integration is required for full functionality.)

## Project Overview
This project analyzes and predicts air pollution levels in India using machine learning classification algorithms. It uses a dataset containing various air quality indicators such as PM2.5, PM10, NO, NO2, CO, SO2, O3, and more. The project implements multiple classifiers including Extra Tree Classifier, Random Forest, and XGBoost to predict the Air Quality Index (AQI) bucket.

## Project Modules
- **Data Preprocessing:** Cleaning data, handling missing values, encoding categorical labels, and removing duplicates.
- **Extra Tree Classifier:** Train and evaluate an Extra Tree model for AQI classification.
- **Random Forest Classifier:** Train and evaluate a Random Forest model for AQI classification.
- **XGBoost Classifier:** Train and evaluate an XGBoost model for AQI classification.
- **Web Interface:** HTML form for user input of air pollution parameters to integrate model predictions.

## Files Included
- `data_preprocessing.py` — Data cleaning and preprocessing scripts.
- `extra_tree_classifier.py` — Extra Tree Classifier training and evaluation.
- `random_forest_classifier.py` — Random Forest Classifier training and evaluation.
- `xgb_classifier.py` — XGBoost Classifier training and evaluation.
- `index.html` — Frontend form to input air pollution data.
- `AIR.csv` — Dataset containing air quality records. *(Provide this file or instructions to obtain it.)*
- `MODEL.pkl`, `XGB.pkl` — Saved model files after training.
