# src/models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

class DemandForecaster:
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.model_performance = {}
    
    def prepare_features_target(self, df, feature_columns):
        """Prepare features and target variable"""
        X = df[feature_columns].fillna(0)
        y = df['Next_Day_Consumption']
        
        return X, y
    
    def train_models(self, X, y, test_size=0.2):
        """Train multiple models and compare performance"""
        # Use time-based split (if date information is available)
        # For simplicity, using random split here
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate metrics
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Mean Absolute Percentage Error (MAPE)
            train_mape = np.mean(np.abs((y_train - y_pred_train) / (y_train + 1))) * 100
            test_mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1))) * 100
            
            self.model_performance[name] = {
                'Train_MAE': train_mae,
                'Test_MAE': test_mae,
                'Train_RMSE': train_rmse,
                'Test_RMSE': test_rmse,
                'Train_R2': train_r2,
                'Test_R2': test_r2,
                'Train_MAPE': train_mape,
                'Test_MAPE': test_mape
            }
            
            self.models[name] = model
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        return self.model_performance
    
    def select_best_model(self):
        """Select the best performing model based on MAPE"""
        if not self.model_performance:
            raise ValueError("No models have been trained yet.")
            
        best_model_name = min(self.model_performance.items(), 
                            key=lambda x: x[1]['Test_MAPE'])[0]
        print(f"Best model: {best_model_name} with MAPE: {self.model_performance[best_model_name]['Test_MAPE']:.2f}%")
        return best_model_name
    
    def forecast_demand(self, X, model_name):
        """Generate demand forecasts"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
            
        model = self.models[model_name]
        return model.predict(X)
    
    def save_model(self, model_name, filepath):
        """Save trained model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found.")
            
        joblib.dump(self.models[model_name], filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, model_name, filepath):
        """Load trained model"""
        self.models[model_name] = joblib.load(filepath)