# src/preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class DataPreprocessor:
    def __init__(self):
        self.feature_columns = []
        
    def load_data(self, file_path):
        """Load and basic cleaning"""
        df = pd.read_excel(file_path)
        df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
        return df
    
    def handle_missing_values(self, df):
        """Handle any missing values"""
        # Check for missing values
        print(f"Missing values before handling: {df.isnull().sum().sum()}")
        
        # Forward fill for time series data
        df = df.sort_values(['Bar Name', 'Alcohol Type', 'Brand Name', 'Date Time Served'])
        numeric_cols = ['Opening Balance (ml)', 'Purchase (ml)', 'Consumed (ml)', 'Closing Balance (ml)']
        
        for col in numeric_cols:
            df[col] = df[col].fillna(0)
            
        return df
    
    def detect_anomalies(self, df):
        """Detect data anomalies"""
        # Check for negative values in consumptions/purchases
        negative_consumption = df[df['Consumed (ml)'] < 0].shape[0]
        negative_purchase = df[df['Purchase (ml)'] < 0].shape[0]
        
        print(f"Records with negative consumption: {negative_consumption}")
        print(f"Records with negative purchase: {negative_purchase}")
        
        # Remove negative values (data errors)
        df = df[df['Consumed (ml)'] >= 0]
        df = df[df['Purchase (ml)'] >= 0]
        
        return df
    
    def create_daily_aggregates(self, df):
        """Aggregate data to daily level for each bar-product combination"""
        df['Date'] = df['Date Time Served'].dt.date
        
        daily_data = df.groupby(['Date', 'Bar Name', 'Alcohol Type', 'Brand Name']).agg({
            'Consumed (ml)': 'sum',
            'Purchase (ml)': 'sum',
            'Opening Balance (ml)': 'last',  # Use last opening balance of the day
            'Closing Balance (ml)': 'last'   # Use last closing balance of the day
        }).reset_index()
        
        return daily_data
    
    def prepare_training_data(self, df):
        """Prepare final training dataset"""
        # Sort by date and create sequential data
        df = df.sort_values(['Bar Name', 'Alcohol Type', 'Brand Name', 'Date'])
        
        # Create target variable (next day's consumption)
        df['Next_Day_Consumption'] = df.groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])['Consumed (ml)'].shift(-1)
        
        # Remove rows without target (last day for each series)
        df = df.dropna(subset=['Next_Day_Consumption'])
        
        return df