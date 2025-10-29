# snowflake_ml_integration.py
from snowflake.snowpark import Session
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.preprocessing import StandardScaler
import pandas as pd

def train_forecasting_model(session):
    # Load data from Snowflake
    df = session.table("historical_inventory_data").to_pandas()
    
    # Prepare features and target
    features = ['day_of_week', 'month', 'is_holiday', 'location_type']
    target = 'demand'
    
    # Train model using Snowpark ML
    model = XGBRegressor(
        input_cols=features,
        label_cols=target,
        output_cols='PREDICTED_DEMAND'
    )
    
    # Fit model
    model.fit(df)
    
    return model

def make_predictions(session, model):
    # Get latest data for prediction
    latest_data = session.table("current_inventory_state").to_pandas()
    
    # Make predictions
    predictions = model.predict(latest_data)
    
    # Save predictions to Snowflake table
    predictions_df = session.create_dataframe(predictions)
    predictions_df.write.mode("overwrite").save_as_table("demand_forecasts")
