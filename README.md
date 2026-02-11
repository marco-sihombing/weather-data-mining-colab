# Weather Data Mining with Machine Learning

## 📌 Overview
This project demonstrates a data mining and machine learning pipeline for daily weather data.
The goal is to classify weather conditions (e.g., drizzle, rain, sun) based on numerical
meteorological features.

The project is designed to run seamlessly on **Google Colab** and focuses on:
- Data preprocessing
- Feature engineering
- Supervised classification
- Model evaluation

---

## 📊 Dataset
The dataset contains daily weather observations with the following fields:

| Column | Description |
|------|------------|
| date | Observation date |
| precipitation | Precipitation amount (mm) |
| temp_max | Maximum temperature (°C) |
| temp_min | Minimum temperature (°C) |
| wind | Wind speed |
| weather | Weather condition (target label) |

Example:
|index|date|precipitation|temp\_max|temp\_min|wind|weather|
|---|---|---|---|---|---|---|
|0|2012-01-01|0\.0|12\.8|5\.0|4\.7|drizzle|
|1|2012-01-02|10\.9|10\.6|2\.8|4\.5|rain|
|2|2012-01-03|0\.8|11\.7|7\.2|2\.3|rain|
|3|2012-01-04|20\.3|12\.2|5\.6|4\.7|rain|
|4|2012-01-05|1\.3|8\.9|2\.8|6\.1|rain|

---

## 🔧 Feature Engineering
The `date` column is transformed into numerical features to improve model performance:

- `year`
- `month`
- `day`

This allows the model to capture seasonal patterns without using raw date values.

---

## 🤖 Machine Learning Model
We use **Random Forest Classifier** due to its:
- Robustness to noise
- Ability to handle non-linear relationships
- Built-in feature importance analysis

Target variable:
<img src=RandomForestClassifier.png>

---

## 📈 Model Evaluation
Model performance is evaluated using:
- Accuracy score
- Precision, recall, and F1-score (classification report)

---

## 🚀 How to Run (Google Colab)
1. Open Google Colab
2. Upload the notebook
3. Upload the dataset (`seattle-weather.csv`)
4. Run all cells

---

## 🧠 Future Improvements
- Compare multiple models (SVM, XGBoost)
- Add confusion matrix visualization
- Extend to time-series forecasting
- Deploy as a REST API

---

## 📁 Repository Structure

└── 📁weather-data-mining-colab
    ├── index.py
    ├── RandomForestClassifier.png
    ├── README.md
    └── seattle-weather.csv

---