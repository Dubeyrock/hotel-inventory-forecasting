# src/data_loader.py

import pandas as pd
import numpy as np

class DataLoader:
    def __init__(self):
        pass
    
    def load_excel_data(self, file_path):
        """Load data from Excel file"""
        try:
            df = pd.read_excel(file_path)
            print(f"Data loaded successfully: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def validate_data(self, df):
        """Basic data validation"""
        required_columns = ['Date Time Served', 'Bar Name', 'Alcohol Type', 'Brand Name', 
                          'Opening Balance (ml)', 'Purchase (ml)', 'Consumed (ml)', 'Closing Balance (ml)']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
        
        print("Data validation passed")
        return True