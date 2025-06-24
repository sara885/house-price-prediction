# House Prices Regression Ensemble

A project to predict house prices using an ensemble of three models:  
XGBoost, LightGBM, and CatBoost, combined using a Median Voting approach.

---

## Project Description

This code trains regression models to predict house prices based on the Kaggle dataset from the "House Prices: Advanced Regression Techniques" competition.  
The data is preprocessed by removing outliers, handling missing values, and encoding categorical features.  
Then, three different models are trained and their predictions are combined by taking the median.  
Model performance is evaluated using K-Fold Cross Validation.

---

## Requirements

- Python 3.7+
- Required libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - lightgbm
  - catboost

---

## Installation

Install the required packages using:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost
