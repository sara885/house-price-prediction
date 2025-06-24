!pip install catboost

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.base import RegressorMixin, BaseEstimator, clone

# ✅ Class for ensemble prediction using median voting
class MedianVotingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = []
        for name, est in self.estimators:
            model = clone(est)
            model.fit(X, y)
            self.fitted_.append((name, model))
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for _, model in self.fitted_
        ])
        return np.median(predictions, axis=1)

# Load training data
train_path = "/content/drive/MyDrive/fine-tuning-image-matting/house-prices-advanced-regression-techniques/train.csv"
train_df = pd.read_csv(train_path)

# Remove outliers using IQR
Q1 = train_df['SalePrice'].quantile(0.25)
Q3 = train_df['SalePrice'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
train_df = train_df[(train_df['SalePrice'] >= lower_bound) & (train_df['SalePrice'] <= upper_bound)]

# Prepare target (log1p) and features
y = np.log1p(train_df['SalePrice'])
X = train_df.drop(columns=['SalePrice', 'Id'])

# Handle missing values
for col in X.select_dtypes(include=['float64', 'int64']).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna('Missing')

# Identify categorical and numerical columns
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('num', 'passthrough', numerical_features)
])

X_processed = preprocessor.fit_transform(X)

# Define models
models = [
    ('xgb', XGBRegressor(random_state=42, n_jobs=-1, n_estimators=800, learning_rate=0.03, max_depth=4,
                         subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1)),
    ('lgbm', LGBMRegressor(random_state=42, n_estimators=800, learning_rate=0.03, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1)),
    ('catboost', CatBoostRegressor(verbose=0, random_state=42, iterations=800, learning_rate=0.03, depth=4,
                                   l2_leaf_reg=3.0))
]

# Ensemble using median voting
ensemble = MedianVotingRegressor(estimators=models)

# Setup cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_train_scores = []
r2_train_scores = []
mae_val_scores = []
r2_val_scores = []

# Training and evaluation
for fold, (train_idx, val_idx) in enumerate(kf.split(X_processed)):
    print(f"\n-- Fold {fold+1} --")
    X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    ensemble.fit(X_train_fold, y_train_fold)

    y_train_pred_log = ensemble.predict(X_train_fold)
    y_val_pred_log = ensemble.predict(X_val_fold)

    y_train_true = np.expm1(y_train_fold)
    y_val_true = np.expm1(y_val_fold)
    y_train_pred = np.expm1(y_train_pred_log)
    y_val_pred = np.expm1(y_val_pred_log)

    mae_train = mean_absolute_error(y_train_true, y_train_pred)
    r2_train = r2_score(y_train_true, y_train_pred)
    mae_val = mean_absolute_error(y_val_true, y_val_pred)
    r2_val = r2_score(y_val_true, y_val_pred)

    mae_train_scores.append(mae_train)
    r2_train_scores.append(r2_train)
    mae_val_scores.append(mae_val)
    r2_val_scores.append(r2_val)

    print(f"MAE on training data: {mae_train:.2f}")
    print(f"R² on training data: {r2_train:.4f}")
    print(f"MAE on validation data: {mae_val:.2f}")
    print(f"R² on validation data: {r2_val:.4f}")

    print("\n🔍 Comparing the first 5 predictions from this fold:")
    results_df = pd.DataFrame({
        'Actual': y_val_true[:5].values,
        'Predicted': y_val_pred[:5]
    })
    print(results_df.to_string(index=False))

# Performance summary
print("\n=== Performance Summary Across 5 Folds ===")
print(f"Average MAE on training data: {np.mean(mae_train_scores):.2f} ± {np.std(mae_train_scores):.2f}")
print(f"Average R² on training data: {np.mean(r2_train_scores):.4f} ± {np.std(r2_train_scores):.4f}")
print(f"Average MAE on validation data: {np.mean(mae_val_scores):.2f} ± {np.std(mae_val_scores):.2f}")
print(f"Average R² on validation data: {np.mean(r2_val_scores):.4f} ± {np.std(r2_val_scores):.4f}")

diff = np.mean(mae_val_scores) - np.mean(mae_train_scores)
print(f"MAE difference between training and validation: {diff:.2f}")
if diff > 10000:
    print("⚠️ Warning: Potential Overfitting detected.")
else:
    print("✅ No strong evidence of Overfitting.")
