# config.py

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'lags': [1, 2, 3, 7, 14],
    'rolling_windows': [3, 7, 14]
}

# Inventory Configuration
INVENTORY_CONFIG = {
    'service_level': 0.95,
    'lead_time_days': 1,
    'review_period_days': 1
}

# Data Configuration
DATA_CONFIG = {
    'date_format': '%Y-%m-%d',
    'target_column': 'Next_Day_Consumption'
}