
````
 🌫️ Air Pollution Prediction Using ML

Predict air quality levels in India using **machine learning models**.  
This project provides a complete pipeline: **data preprocessing → model training → web interface** for input and predictions.

---

⚡ Features

- Analyze historical air pollution data  
- Predict **Air Quality Index (AQI) buckets**  
- Train multiple ML classifiers:
  - Extra Tree Classifier  
  - Random Forest Classifier  
  - XGBoost  
- Simple HTML frontend for user input  

---

 🛠️ Setup Instructions

1. Clone the repository
```bash
git clone https://github.com/yourusername/AirPollutionPrediction.git
cd AirPollutionPrediction
````

2. Install dependencies

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost joblib
```

3. Add the dataset

Place `AIR.csv` in the project root. Make sure it contains **all required columns** like PM2.5, PM10, NO, NO2, CO, SO2, O3, etc.

---

🚀 Running the Project

 1. Preprocess and train models

```bash
python data_preprocessing.py
python extra_tree_classifier.py
python random_forest_classifier.py
python xgb_classifier.py
```

 2. Use the web interface

Open `index.html` in a browser to input air pollution parameters.

> ⚠️ Note: Full backend integration is required to generate live predictions.

---

 📊 Project Overview

This project predicts **air pollution levels** in India using machine learning.
The models classify the **Air Quality Index (AQI)** based on multiple air quality indicators:

* PM2.5, PM10
* NO, NO2, CO
* SO2, O3

Classifiers implemented: **Extra Tree, Random Forest, XGBoost**.

---

 🧩 Modules

* **Data Preprocessing**: Clean data, handle missing values, encode categorical variables, remove duplicates
* **Extra Tree Classifier**: Train and evaluate Extra Tree model
* **Random Forest Classifier**: Train and evaluate Random Forest model
* **XGBoost Classifier**: Train and evaluate XGBoost model
* **Web Interface**: HTML form to input air pollution data for predictions

---

 📂 Included Files

| File                          | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `data_preprocessing.py`       | Data cleaning and preprocessing             |
| `extra_tree_classifier.py`    | Extra Tree model training and evaluation    |
| `random_forest_classifier.py` | Random Forest model training and evaluation |
| `xgb_classifier.py`           | XGBoost model training and evaluation       |
| `index.html`                  | Frontend form for user input                |
| `AIR.csv`                     | Dataset with historical air quality data    |
| `MODEL.pkl`, `XGB.pkl`        | Saved trained models                        |

