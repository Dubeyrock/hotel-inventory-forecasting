# src/utils.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

def save_model(model, filepath):
    """Save trained model"""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    """Load trained model"""
    return joblib.load(filepath)

def create_directories():
    """Create necessary directories"""
    import os
    directories = ['models', 'results', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)