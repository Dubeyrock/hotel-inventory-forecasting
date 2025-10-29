# src/inventory_optimizer.py

import pandas as pd
import numpy as np
from scipy import stats

class InventoryOptimizer:
    def __init__(self, service_level=0.95):
        self.service_level = service_level
        self.z_score = stats.norm.ppf(service_level)
    
    def calculate_safety_stock(self, demand_std, lead_time=1, lead_time_std=0):
        """Calculate safety stock based on demand variability"""
        safety_stock = self.z_score * np.sqrt(
            lead_time * demand_std**2 + lead_time_std**2 * demand_std**2
        )
        return max(safety_stock, 0)
    
    def calculate_reorder_point(self, forecast_demand, safety_stock, lead_time=1):
        """Calculate reorder point"""
        return forecast_demand * lead_time + safety_stock
    
    def calculate_par_level(self, forecast_demand, demand_std, lead_time=1, review_period=1):
        """Calculate par level (target inventory level)"""
        safety_stock = self.calculate_safety_stock(demand_std, lead_time)
        par_level = forecast_demand * (lead_time + review_period) + safety_stock
        return max(par_level, forecast_demand)  # Ensure at least one period demand
    
    def optimize_inventory_all_items(self, forecasts_df, historical_consumption):
        """Calculate optimal inventory levels for all items"""
        recommendations = []
        
        # Print debug info
        print(f"Forecasts_df columns: {list(forecasts_df.columns)}")
        print(f"Historical_consumption columns: {list(historical_consumption.columns)}")
        
        # Handle different column naming conventions
        bar_col = 'Bar Name' if 'Bar Name' in forecasts_df.columns else 'Bar_Name' if 'Bar_Name' in forecasts_df.columns else 'BarName'
        alcohol_col = 'Alcohol Type' if 'Alcohol Type' in forecasts_df.columns else 'Alcohol_Type' if 'Alcohol_Type' in forecasts_df.columns else 'AlcoholType'
        brand_col = 'Brand Name' if 'Brand Name' in forecasts_df.columns else 'Brand_Name' if 'Brand_Name' in forecasts_df.columns else 'BrandName'
        
        for (bar, alcohol_type, brand_name), group in historical_consumption.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']):
            # Get forecast for this item
            item_forecast = forecasts_df[
                (forecasts_df[bar_col] == bar) & 
                (forecasts_df[alcohol_col] == alcohol_type) & 
                (forecasts_df[brand_col] == brand_name)
            ]
            
            if len(item_forecast) == 0:
                continue
                
            forecast_demand = item_forecast['Forecasted_Demand'].iloc[0]
            demand_std = group['Consumed (ml)'].std()
            
            # Calculate optimal levels
            safety_stock = self.calculate_safety_stock(demand_std)
            reorder_point = self.calculate_reorder_point(forecast_demand, safety_stock)
            par_level = self.calculate_par_level(forecast_demand, demand_std)
            
            recommendations.append({
                'Bar_Name': bar,
                'Alcohol_Type': alcohol_type,
                'Brand_Name': brand_name,
                'Forecasted_Daily_Demand': forecast_demand,
                'Safety_Stock': safety_stock,
                'Reorder_Point': reorder_point,
                'Par_Level': par_level,
                'Current_Avg_Demand': group['Consumed (ml)'].mean(),
                'Demand_Variability': demand_std
            })
        
        result_df = pd.DataFrame(recommendations)
        print(f"Generated {len(result_df)} recommendations with columns: {list(result_df.columns)}")
        return result_df