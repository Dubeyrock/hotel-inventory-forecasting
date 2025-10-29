# main.py

import pandas as pd
import numpy as np
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.models import DemandForecaster
from src.inventory_optimizer import InventoryOptimizer
from src.utils import create_directories, save_model
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Inventory Forecasting System...")
    
    # Create necessary directories
    create_directories()
    
    # Step 1: Load Data
    print("\n📊 Step 1: Loading data...")
    data_loader = DataLoader()
    df = data_loader.load_excel_data('data/Consumption Dataset.xlsx')
    
    if df is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Validate data
    if not data_loader.validate_data(df):
        print("❌ Data validation failed. Exiting.")
        return
    
    # Step 2: Preprocess Data
    print("\n🔧 Step 2: Preprocessing data...")
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.handle_missing_values(df)
    df_clean = preprocessor.detect_anomalies(df_clean)
    daily_data = preprocessor.create_daily_aggregates(df_clean)
    training_data = preprocessor.prepare_training_data(daily_data)
    
    print(f"✅ Preprocessing complete. Training data: {training_data.shape}")
    
    # Step 3: Feature Engineering
    print("\n🎯 Step 3: Feature engineering...")
    feature_engineer = FeatureEngineer()
    featured_data = feature_engineer.engineer_all_features(training_data)
    
    print(f"✅ Feature engineering complete. Features: {len(feature_engineer.feature_columns)}")
    
    # Step 4: Model Training
    print("\n🤖 Step 4: Training models...")
    forecaster = DemandForecaster()
    
    # Prepare features and target
    X, y = forecaster.prepare_features_target(featured_data, feature_engineer.feature_columns)
    
    # Train models with the correct parameters
    performance = forecaster.train_models(X, y)
    
    # Print results
    print("\n📈 Model Performance:")
    for model, metrics in performance.items():
        print(f"  {model}: MAPE = {metrics['Test_MAPE']:.2f}%")
    
    # Select and save best model
    best_model_name = forecaster.select_best_model()
    model_path = f'models/best_model_{best_model_name}.pkl'
    forecaster.save_model(best_model_name, model_path)
    
    # Step 5: Generate Forecasts
    print("\n📊 Step 5: Generating forecasts...")
    
    # Get the test set predictions from the trained model
    featured_data['Date'] = pd.to_datetime(featured_data['Date'])
    split_date = featured_data['Date'].quantile(0.8)
    test_mask = featured_data['Date'] > split_date
    
    # Use the best model to generate forecasts for the test set
    X_test = X[test_mask]
    y_pred_test = forecaster.forecast_demand(X_test, best_model_name)
    
    # Create forecasts dataframe
    forecasts_df = featured_data[test_mask][['Date', 'Bar Name', 'Alcohol Type', 'Brand Name']].copy()
    forecasts_df['Forecasted_Demand'] = y_pred_test
    forecasts_df['Actual_Demand'] = featured_data[test_mask]['Next_Day_Consumption'].values
    
    # Step 6: Inventory Optimization
    print("\n📦 Step 6: Optimizing inventory...")
    optimizer = InventoryOptimizer(service_level=0.95)
    
    # For inventory optimization, we need forecasts for all items (not just test set)
    # Generate forecasts for the entire dataset to get recommendations for all items
    all_forecasts = forecaster.forecast_demand(X, best_model_name)
    all_forecasts_df = featured_data[['Date', 'Bar Name', 'Alcohol Type', 'Brand Name']].copy()
    all_forecasts_df['Forecasted_Demand'] = all_forecasts
    
    # Get recent forecasts for each item (last available forecast)
    recent_forecasts = all_forecasts_df.sort_values(['Bar Name', 'Alcohol Type', 'Brand Name', 'Date'])\
                                      .groupby(['Bar Name', 'Alcohol Type', 'Brand Name'])\
                                      .last()\
                                      .reset_index()
    
    recommendations = optimizer.optimize_inventory_all_items(recent_forecasts, daily_data)
    
    # Step 7: Save Results
    print("\n💾 Step 7: Saving results...")
    
    # Save test predictions
    forecasts_df.to_csv('results/demand_forecasts.csv', index=False)
    
    # Save inventory recommendations
    recommendations.to_csv('results/inventory_recommendations.csv', index=False)
    
    # Save engineered dataset
    featured_data.to_csv('results/engineered_dataset.csv', index=False)
    
    # Save performance summary
    performance_df = pd.DataFrame(performance).T
    performance_df.to_csv('results/model_performance.csv')
    
    # Save feature importance if available
    if hasattr(forecaster, 'feature_importance') and best_model_name in forecaster.feature_importance:
        forecaster.feature_importance[best_model_name].to_csv('results/feature_importance.csv', index=False)
    
    print("\n🎉 Inventory Forecasting Completed Successfully!")
    print(f"📁 Results saved in 'results/' folder")
    print(f"🤖 Best model: {best_model_name} (saved as {model_path})")
    print(f"📊 Test forecasts generated: {len(forecasts_df)} records")
    print(f"📦 Inventory recommendations: {len(recommendations)} items")
    
    # Final summary
    print("\n" + "="*50)
    print("📋 PROJECT SUMMARY")
    print("="*50)
    print(f"Original dataset: {df.shape}")
    print(f"Engineered features: {len(feature_engineer.feature_columns)}")
    print(f"Best model MAPE: {performance[best_model_name]['Test_MAPE']:.2f}%")
    print(f"Total inventory recommendations: {len(recommendations)}")
    
    # Debug: Print column names to see what's available
    if len(recommendations) > 0:
        print(f"\n🔍 Available columns in recommendations: {list(recommendations.columns)}")
    
    if len(recommendations) > 0:
        # Check which columns actually exist and use them
        available_columns = []
        display_columns = []
        
        # Map possible column names to what might actually exist
        column_mapping = {
            'Bar_Name': ['Bar_Name', 'Bar Name', 'BarName'],
            'Alcohol_Type': ['Alcohol_Type', 'Alcohol Type', 'AlcoholType'], 
            'Brand_Name': ['Brand_Name', 'Brand Name', 'BrandName'],
            'Par_Level': ['Par_Level', 'Par Level', 'ParLevel']
        }
        
        # Find which columns actually exist
        for display_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in recommendations.columns:
                    available_columns.append(col)
                    display_columns.append(display_col)
                    break
        
        if 'Par_Level' in [col for col in available_columns if 'Par' in col]:
            # Find the actual par level column name
            par_col = [col for col in recommendations.columns if 'Par' in col][0]
            
            print(f"\n📝 Top 5 Inventory Recommendations:")
            
            # Use available columns for display
            if len(available_columns) >= 4:  # We have all required columns
                top_items = recommendations.nlargest(5, par_col)[available_columns]
                # Rename columns for display
                top_items_display = top_items.copy()
                top_items_display.columns = display_columns
                
                for idx, row in top_items_display.iterrows():
                    print(f"   {row['Bar_Name']} - {row['Alcohol_Type']} - {row['Brand_Name']}: {row['Par_Level']:.0f} ml")
            else:
                # Fallback: just show par levels
                top_items = recommendations.nlargest(5, par_col)
                print(top_items[['Bar Name', 'Alcohol Type', 'Brand Name', par_col]].head())
        else:
            print("   Par_Level column not found in recommendations.")
            
        # Show basic statistics if we have the data
        if 'Safety_Stock' in recommendations.columns or any('Safety' in col for col in recommendations.columns):
            safety_col = [col for col in recommendations.columns if 'Safety' in col][0] if any('Safety' in col for col in recommendations.columns) else None
            par_col = [col for col in recommendations.columns if 'Par' in col][0] if any('Par' in col for col in recommendations.columns) else None
            
            if safety_col and par_col:
                print(f"   Average safety stock: {recommendations[safety_col].mean():.2f} ml")
                print(f"   Average par level: {recommendations[par_col].mean():.2f} ml")

if __name__ == "__main__":
    main()