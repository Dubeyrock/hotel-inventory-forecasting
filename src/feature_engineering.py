# src/feature_engineering.py

import pandas as pd
import numpy as np
from datetime import datetime

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []
    
    def create_time_features(self, df):
        """Create time-based features"""
        df['Date'] = pd.to_datetime(df['Date'])
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['DayOfMonth'] = df['Date'].dt.day
        df['Month'] = df['Date'].dt.month
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Holiday patterns (simplified - you can enhance this)
        df['IsHolidaySeason'] = ((df['Month'] == 12) | (df['Month'] == 1)).astype(int)
        
        return df
    
    def create_lag_features(self, df, lags=[1, 2, 3, 7, 14]):
        """Create lag features for consumption patterns"""
        grouped = df.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])
        
        for lag in lags:
            df[f'Consumption_Lag_{lag}'] = grouped['Consumed (ml)'].shift(lag)
            df[f'Purchase_Lag_{lag}'] = grouped['Purchase (ml)'].shift(lag)
        
        return df
    
    def create_rolling_features(self, df, windows=[3, 7, 14]):
        """Create rolling statistics"""
        grouped = df.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])
        
        for window in windows:
            df[f'Consumption_Mean_{window}'] = grouped['Consumed (ml)'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'Consumption_Std_{window}'] = grouped['Consumed (ml)'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
        return df
    
    def create_inventory_features(self, df):
        """Create inventory-related features"""
        df['Inventory_Ratio'] = df['Closing Balance (ml)'] / (df['Consumed (ml)'] + 1)  # +1 to avoid division by zero
        df['Stockout_Risk'] = (df['Closing Balance (ml)'] / df['Consumed (ml)'].rolling(7, min_periods=1).mean()).replace([np.inf, -np.inf], 0)
        df['Inventory_Turnover'] = df['Consumed (ml)'] / (df['Opening Balance (ml)'] + 1)
        
        return df
    
    def create_product_features(self, df):
        """Create product-specific features"""
        # Alcohol type dummies
        alcohol_dummies = pd.get_dummies(df['Alcohol Type'], prefix='Alcohol')
        df = pd.concat([df, alcohol_dummies], axis=1)
        
        # Bar location dummies
        bar_dummies = pd.get_dummies(df['Bar Name'], prefix='Bar')
        df = pd.concat([df, bar_dummies], axis=1)
        
        return df
    
    def engineer_all_features(self, df):
        """Apply all feature engineering steps"""
        print("Creating time features...")
        df = self.create_time_features(df)
        
        print("Creating lag features...")
        df = self.create_lag_features(df)
        
        print("Creating rolling features...")
        df = self.create_rolling_features(df)
        
        print("Creating inventory features...")
        df = self.create_inventory_features(df)
        
        print("Creating product features...")
        df = self.create_product_features(df)
        
        # Store feature columns
        self.feature_columns = [col for col in df.columns if col not in 
                               ['Date', 'Bar Name', 'Alcohol Type', 'Brand Name', 
                                'Next_Day_Consumption', 'Consumed (ml)', 'Purchase (ml)',
                                'Opening Balance (ml)', 'Closing Balance (ml)']]
        
        return df