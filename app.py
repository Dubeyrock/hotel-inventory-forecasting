# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import io
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🍷 Bar Inventory Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #3498db;
        font-weight: 600;
    }
    .warning-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background: linear-gradient(135deg, #55efc4 0%, #00b894 100%);
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        border: none;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    /* Custom radio button styling */
    .stRadio > div {
        background: transparent;
    }
    .stRadio > div > label {
        background: rgba(255,255,255,0.1);
        color: white;
        padding: 0.75rem 1rem;
        margin: 0.25rem 0;
        border-radius: 10px;
        transition: all 0.3s ease;
        cursor: pointer;
        border: none !important;
    }
    .stRadio > div > label:hover {
        background: rgba(255,255,255,0.2);
        transform: translateX(5px);
    }
    .stRadio > div > label[data-baseweb="radio"] {
        background: rgba(255,255,255,0.1);
    }
    .stRadio > div > label[data-baseweb="radio"]:hover {
        background: rgba(255,255,255,0.2);
    }
    /* Selected radio button */
    .stRadio > div > [data-baseweb="radio"]:checked + label {
        background: rgba(255,255,255,0.2);
        border-left: 4px solid #3498db !important;
    }
    /* Download button styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #00b894 0%, #55efc4 100%);
        color: white;
        border: none;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stDownloadButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

class InventoryDashboard:
    def __init__(self):
        self.model = None
        self.data = None
        self.feature_columns = []
        
    def load_data(self):
        """Load the dataset"""
        try:
            df = pd.read_excel('data/Consumption Dataset.xlsx')
            df['Date Time Served'] = pd.to_datetime(df['Date Time Served'])
            
            # Add some sample data if needed for testing
            if len(df) == 0:
                st.warning("📝 Sample data loaded for demonstration")
                df = self._create_sample_data()
            else:
                st.success(f"✅ Data loaded successfully: {len(df):,} records")
                
            return df
        except FileNotFoundError:
            st.error("❌ Data file not found. Please ensure 'data/Consumption Dataset.xlsx' exists.")
            return self._create_sample_data()
        except Exception as e:
            st.error(f"❌ Data load error: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data for demonstration"""
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='D')
        bars = ["Thomas's Bar", "Taylor's Bar", "Smith's Lounge"]
        alcohol_types = ['Whiskey', 'Beer', 'Vodka', 'Rum']
        brands = {
            'Whiskey': ['Jameson', 'Jack Daniels', 'Johnnie Walker'],
            'Beer': ['Heineken', 'Budweiser', 'Corona'],
            'Vodka': ['Smirnoff', 'Absolut', 'Grey Goose'],
            'Rum': ['Bacardi', 'Captain Morgan', 'Malibu']
        }
        
        data = []
        for date in dates:
            for bar in bars:
                for alcohol in alcohol_types:
                    for brand in brands[alcohol]:
                        consumed = np.random.poisson(100) + np.random.randint(-20, 50)
                        purchase = consumed + np.random.randint(0, 30)
                        data.append({
                            'Date Time Served': date,
                            'Bar Name': bar,
                            'Alcohol Type': alcohol,
                            'Brand Name': brand,
                            'Consumed (ml)': max(0, consumed),
                            'Purchase (ml)': max(0, purchase),
                            'Opening Balance (ml)': np.random.randint(200, 500),
                            'Closing Balance (ml)': np.random.randint(100, 300)
                        })
        
        return pd.DataFrame(data)
    
    def load_model(self):
        """Load trained model"""
        try:
            import glob
            model_files = glob.glob('models/best_model_*.pkl')
            if model_files:
                self.model = joblib.load(model_files[0])
                return True
            else:
                return False
        except Exception as e:
            st.error(f"❌ Model load error: {e}")
            return False

def create_3d_consumption_plot(data):
    """Create 3D visualization of consumption patterns"""
    try:
        # Prepare data for 3D plot
        summary = data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).agg({
            'Consumed (ml)': 'sum',
            'Purchase (ml)': 'sum',
            'Closing Balance (ml)': 'mean'
        }).reset_index()
        
        # Create 3D scatter plot
        fig = px.scatter_3d(
            summary,
            x='Bar Name',
            y='Alcohol Type', 
            z='Brand Name',
            size='Consumed (ml)',
            color='Consumed (ml)',
            hover_data=['Purchase (ml)', 'Closing Balance (ml)'],
            title='🎯 3D Consumption Analysis: Bars vs Alcohol Types vs Brands',
            color_continuous_scale='Viridis',
            size_max=50
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='🏢 Bar Location',
                yaxis_title='🍸 Alcohol Type',
                zaxis_title='🏷️ Brand Name',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D plot: {e}")
        return None

def create_3d_temporal_analysis(data):
    """Create 3D temporal analysis of consumption"""
    try:
        # Extract time features
        data['Date'] = data['Date Time Served'].dt.date
        data['Hour'] = data['Date Time Served'].dt.hour
        data['DayOfWeek'] = data['Date Time Served'].dt.day_name()
        
        # Aggregate data
        temporal_data = data.groupby(['Date', 'Hour', 'Alcohol Type']).agg({
            'Consumed (ml)': 'sum'
        }).reset_index()
        
        # Create 3D surface plot
        pivot_data = temporal_data.pivot_table(
            index='Date', 
            columns='Hour', 
            values='Consumed (ml)', 
            aggfunc='sum'
        ).fillna(0)
        
        fig = go.Figure(data=[
            go.Surface(
                z=pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                colorscale='Hot'
            )
        ])
        
        fig.update_layout(
            title='🕒 3D Temporal Consumption Pattern: Date vs Hour vs Consumption',
            scene=dict(
                xaxis_title='🕐 Hour of Day',
                yaxis_title='📅 Date',
                zaxis_title='🍷 Consumption (ml)',
                camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
            ),
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D temporal plot: {e}")
        return None

def create_3d_inventory_surface(data):
    """Create 3D inventory surface plot"""
    try:
        # Prepare inventory data
        inventory_data = data.groupby(['Bar Name', 'Alcohol Type']).agg({
            'Opening Balance (ml)': 'mean',
            'Closing Balance (ml)': 'mean',
            'Consumed (ml)': 'sum'
        }).reset_index()
        
        fig = px.density_heatmap(
            inventory_data,
            x='Bar Name',
            y='Alcohol Type',
            z='Consumed (ml)',
            histfunc='avg',
            title='📊 3D Inventory Heatmap: Consumption by Bar and Alcohol Type',
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating 3D inventory plot: {e}")
        return None

def create_interactive_forecast_3d(historical, forecast_dates, forecast_values):
    """Create interactive 3D forecast visualization"""
    try:
        # Combine historical and forecast data
        historical_dates = historical.index
        historical_values = historical.values
        
        all_dates = list(historical_dates) + forecast_dates
        all_values = list(historical_values) + forecast_values
        types = ['Historical'] * len(historical_dates) + ['Forecast'] * len(forecast_dates)
        
        plot_data = pd.DataFrame({
            'Date': all_dates,
            'Consumption': all_values,
            'Type': types
        })
        
        # Create 3D line plot
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter3d(
            x=historical_dates,
            y=[0] * len(historical_dates),
            z=historical_values,
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue', width=4),
            marker=dict(size=3)
        ))
        
        # Forecast data
        fig.add_trace(go.Scatter3d(
            x=forecast_dates,
            y=[1] * len(forecast_dates),
            z=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', width=4, dash='dash'),
            marker=dict(size=3)
        ))
        
        fig.update_layout(
            title='🚀 3D Demand Forecast: Historical vs Predicted',
            scene=dict(
                xaxis_title='📅 Timeline',
                yaxis_title='Data Type',
                zaxis_title='🍷 Consumption (ml)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=600
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating 3D forecast plot: {e}")
        return None

def create_navigation():
    """Create enhanced navigation sidebar using Streamlit components"""
    with st.sidebar:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; color: white; text-align: center; margin-bottom: 2rem;'>
            <h2>🍷 Bar Intelligence</h2>
            <p>AI-Powered Inventory Management</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation options
        nav_options = [
            "🏠 Executive Dashboard",
            "📈 3D Data Analysis", 
            "🤖 AI Forecasting",
            "📦 Smart Recommendations",
            "📋 Report Summary",
            "ℹ️ About Project"
        ]
        
        # Create navigation using radio buttons
        selected_nav = st.radio(
            "NAVIGATION",
            options=nav_options,
            index=0,
            key="navigation"
        )
        
        # Map selection to page key
        page_mapping = {
            "🏠 Executive Dashboard": "dashboard",
            "📈 3D Data Analysis": "analysis", 
            "🤖 AI Forecasting": "forecasting",
            "📦 Smart Recommendations": "recommendations",
            "📋 Report Summary": "reports",
            "ℹ️ About Project": "about"
        }
        
        # Update session state
        st.session_state.current_page = page_mapping[selected_nav]
        
        st.markdown("---")
        
        # Quick stats in sidebar
        st.markdown("### 📊 Quick Stats")
        if 'data' in st.session_state and st.session_state.data is not None:
            data = st.session_state.data
            total_consumption = data['Consumed (ml)'].sum()
            total_bars = data['Bar Name'].nunique()
            st.metric("Total Consumption", f"{total_consumption:,.0f} ml")
            st.metric("Active Bars", total_bars)
        
        st.markdown("---")
        
        # Footer
        st.markdown("""
        <div style='text-align: center; color: #bdc3c7; padding: 1rem;'>
            <p><strong>Bar Inventory Intelligence v4.0</strong></p>
            <p>Powered by AI & 3D Analytics</p>
            <p style='font-size: 0.8rem;'>© 2024 Inventory Analytics Suite</p>
        </div>
        """, unsafe_allow_html=True)

def show_3d_forecast_results(dashboard, data, bar_name, alcohol_type, brand_name, forecast_days, confidence_level):
    """Show 3D forecast results"""
    try:
        # Get historical data
        item_data = data[
            (data['Bar Name'] == bar_name) & 
            (data['Alcohol Type'] == alcohol_type) & 
            (data['Brand Name'] == brand_name)
        ].copy()
        
        if len(item_data) == 0:
            st.error("❌ No historical data found for selected product")
            return
        
        # Process data
        item_data['Date'] = item_data['Date Time Served'].dt.date
        daily_data = item_data.groupby('Date')['Consumed (ml)'].sum().tail(60)
        
        if len(daily_data) == 0:
            st.error("❌ No daily consumption data available")
            return
        
        # Generate forecast
        last_consumption = daily_data.iloc[-1]
        base_forecast = [last_consumption * (1 + 0.02 * i) for i in range(forecast_days)]
        
        # Add some randomness for demo
        np.random.seed(42)
        forecast_values = [max(0, x * np.random.uniform(0.9, 1.1)) for x in base_forecast]
        
        # Forecast dates
        last_date = pd.to_datetime(daily_data.index[-1])
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(forecast_days)]
        
        # 3D Forecast Visualization
        st.markdown('<div class="section-header">🚀 3D Forecast Visualization</div>', unsafe_allow_html=True)
        
        # Create 3D forecast plot
        fig_3d_forecast = create_interactive_forecast_3d(daily_data, forecast_dates, forecast_values)
        if fig_3d_forecast:
            st.plotly_chart(fig_3d_forecast, use_container_width=True)
        
        # Additional 2D forecast for comparison
        st.markdown("### 📊 Traditional 2D Forecast View")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_data.index[-30:], daily_data.values[-30:], 
               label='Historical', marker='o', linewidth=2, color='#3498db', markersize=4)
        ax.plot(forecast_dates, forecast_values, 
               label='Forecast', marker='s', linestyle='--', linewidth=2, color='#e74c3c', markersize=4)
        ax.set_title(f'2D Forecast: {brand_name} at {bar_name}', fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Consumption (ml)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        # Forecast insights
        st.markdown("### 💡 Forecast Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_forecast = np.mean(forecast_values)
            st.metric("📊 Average Forecast", f"{avg_forecast:.0f} ml")
        
        with col2:
            growth = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
            st.metric("📈 Growth Trend", f"{growth:+.1f}%")
        
        with col3:
            volatility = np.std(forecast_values) / avg_forecast * 100
            st.metric("⚡ Volatility", f"{volatility:.1f}%")
        
        # Forecast details
        st.markdown("### 📅 Forecast Details")
        forecast_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in forecast_dates],
            'Forecasted Demand (ml)': [f"{val:.0f}" for val in forecast_values],
            'Confidence': f"{confidence_level}%"
        })
        st.dataframe(forecast_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Forecasting error: {str(e)}")

def show_3d_recommendation_results(data, bar_name, alcohol_type, brand_name, service_level, lead_time):
    """Show 3D inventory recommendation results"""
    try:
        # Get item data
        item_data = data[
            (data['Bar Name'] == bar_name) & 
            (data['Alcohol Type'] == alcohol_type) & 
            (data['Brand Name'] == brand_name)
        ]
        
        if len(item_data) == 0:
            st.error("❌ No data found for selected product")
            return
        
        # Calculate statistics
        valid_consumption = item_data['Consumed (ml)'].dropna()
        if len(valid_consumption) == 0:
            st.error("❌ No consumption data available")
            return
        
        avg_daily = valid_consumption.mean()
        std_daily = valid_consumption.std()
        
        # Handle low variability
        if np.isnan(std_daily) or std_daily == 0:
            std_daily = avg_daily * 0.2
        
        # Calculate recommendations
        from scipy import stats
        z_score = stats.norm.ppf(service_level/100)
        
        safety_stock = z_score * std_daily * np.sqrt(lead_time)
        reorder_point = (avg_daily * lead_time) + safety_stock
        par_level = (avg_daily * (lead_time + 7)) + safety_stock
        
        # Ensure non-negative
        safety_stock = max(0, safety_stock)
        reorder_point = max(0, reorder_point)
        par_level = max(0, par_level)
        
        # Display recommendations in 3D-style cards
        st.markdown('<div class="section-header">🎯 3D Recommended Inventory Levels</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics_data = [
            (avg_daily, "📊 Avg Daily", "#74b9ff", f"Based on {len(valid_consumption)} records"),
            (safety_stock, "🛡️ Safety Stock", "#fd79a8", f"For {service_level}% service level"),
            (reorder_point, "⚠️ Reorder Point", "#55efc4", "Trigger new orders"),
            (par_level, "🎯 Par Level", "#a29bfe", "Target inventory")
        ]
        
        for (value, title, color, subtitle), col in zip(metrics_data, [col1, col2, col3, col4]):
            with col:
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, {color} 0%, {color}44 100%); 
                          padding: 1.5rem; border-radius: 15px; color: white; text-align: center;
                          box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                    <h3>{title}</h3>
                    <h2>{value:.0f} ml</h2>
                    <p>{subtitle}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 3D Visualization of recommendations
        st.markdown("### 📊 3D Recommendation Analysis")
        
        # Create a 3D scatter plot for recommendations
        recommendation_data = pd.DataFrame({
            'Metric': ['Avg Daily', 'Safety Stock', 'Reorder Point', 'Par Level'],
            'Value': [avg_daily, safety_stock, reorder_point, par_level],
            'Type': ['Demand', 'Buffer', 'Trigger', 'Target']
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter3d(
            x=recommendation_data['Metric'],
            y=recommendation_data['Type'],
            z=recommendation_data['Value'],
            mode='markers',
            marker=dict(
                size=recommendation_data['Value'] / max(recommendation_data['Value']) * 50,
                color=recommendation_data['Value'],
                colorscale='Viridis',
                opacity=0.8,
                line=dict(width=2, color='darkblue')
            ),
            text=recommendation_data['Metric'] + ': ' + recommendation_data['Value'].astype(int).astype(str) + ' ml',
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title='🎯 3D Inventory Recommendation Analysis',
            scene=dict(
                xaxis_title='Inventory Metric',
                yaxis_title='Metric Type', 
                zaxis_title='Value (ml)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Additional insights
        st.markdown("### 💡 Smart Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid #3498db;'>
                <h4>📋 Recommendation Guide</h4>
                <p><strong>Safety Stock:</strong> {safety_stock:.0f} ml buffer for demand variability</p>
                <p><strong>Reorder Point:</strong> Order when inventory reaches {reorder_point:.0f} ml</p>
                <p><strong>Par Level:</strong> Maintain {par_level:.0f} ml as target inventory</p>
                <p><strong>Service Level:</strong> {service_level}% probability of avoiding stockouts</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Demand variability insight
            cv = std_daily / avg_daily if avg_daily > 0 else 0
            if cv > 0.5:
                insight = "High demand variability - monitor frequently"
                color = "#e74c3c"
            elif cv > 0.2:
                insight = "Moderate variability - weekly monitoring recommended"
                color = "#f39c12"
            else:
                insight = "Stable demand pattern - reliable forecasts"
                color = "#27ae60"
            
            st.markdown(f"""
            <div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 5px solid {color};'>
                <h4>📈 Demand Analysis</h4>
                <p><strong>Variability:</strong> {cv:.2f} coefficient of variation</p>
                <p><strong>Insight:</strong> {insight}</p>
                <p><strong>Data Quality:</strong> {len(valid_consumption)} historical records</p>
                <p><strong>Lead Time:</strong> {lead_time} days for new orders</p>
            </div>
            """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error generating 3D recommendations: {str(e)}")

def generate_text_report(data):
    """Generate a comprehensive text report"""
    report_content = f"""
BAR INVENTORY INTELLIGENCE REPORT
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
-----------------
Total Records: {len(data):,}
Total Consumption: {data['Consumed (ml)'].sum():,} ml
Active Bars: {data['Bar Name'].nunique()}
Unique Products: {data['Brand Name'].nunique()}
Data Period: {(data['Date Time Served'].max() - data['Date Time Served'].min()).days} days

TOP PERFORMERS
--------------
Top Alcohol Types by Consumption:
"""
    
    # Add top alcohol types
    alcohol_consumption = data.groupby('Alcohol Type')['Consumed (ml)'].sum().sort_values(ascending=False)
    for i, (alcohol, consumption) in enumerate(alcohol_consumption.head().items(), 1):
        report_content += f"{i}. {alcohol}: {consumption:,.0f} ml\n"
    
    report_content += "\nTop Brands by Consumption:\n"
    brand_consumption = data.groupby('Brand Name')['Consumed (ml)'].sum().sort_values(ascending=False)
    for i, (brand, consumption) in enumerate(brand_consumption.head().items(), 1):
        report_content += f"{i}. {brand}: {consumption:,.0f} ml\n"
    
    report_content += "\nBar Performance Ranking:\n"
    bar_performance = data.groupby('Bar Name')['Consumed (ml)'].sum().sort_values(ascending=False)
    for i, (bar, consumption) in enumerate(bar_performance.items(), 1):
        report_content += f"{i}. {bar}: {consumption:,.0f} ml\n"
    
    report_content += """
KEY RECOMMENDATIONS
-------------------
1. Optimize Stock Levels: Implement dynamic par levels based on consumption patterns
2. Improve Turnover: Focus on high-consumption products for better inventory rotation
3. Enhanced Monitoring: Use real-time tracking for popular items
4. AI Integration: Implement machine learning for demand forecasting
5. Safety Stock: Maintain optimal safety stock levels for high-variability products

---
Report generated by Bar Inventory Intelligence v4.0
Powered by AI & 3D Analytics
"""
    
    return report_content

def main():
    # Initialize dashboard
    dashboard = InventoryDashboard()
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "dashboard"
    
    # Create enhanced navigation
    create_navigation()
    
    # Header with better styling
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">🍷 Bar Inventory Intelligence</h1>', unsafe_allow_html=True)
        st.markdown("### AI-Powered Inventory Optimization & 3D Analytics")
    
    # Load data
    if 'data' not in st.session_state:
        with st.spinner('🚀 Loading intelligent data system...'):
            st.session_state.data = dashboard.load_data()
    
    # Main content based on selected mode
    current_page = st.session_state.current_page
    
    if current_page == "dashboard":
        show_executive_dashboard(dashboard, st.session_state.data)
    elif current_page == "analysis":
        show_3d_data_analysis(st.session_state.data)
    elif current_page == "forecasting":
        show_demand_forecasting(dashboard, st.session_state.data)
    elif current_page == "recommendations":
        show_inventory_recommendations(dashboard, st.session_state.data)
    elif current_page == "reports":
        show_report_summary(st.session_state.data)
    elif current_page == "about":
        show_about_section()
    else:
        show_executive_dashboard(dashboard, st.session_state.data)

def show_executive_dashboard(dashboard, data):
    """Show enhanced executive dashboard with 3D elements"""
    
    st.markdown('<div class="section-header">📈 Executive Overview</div>', unsafe_allow_html=True)
    
    # Key metrics in cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consumption = data['Consumed (ml)'].sum()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Consumption</h3>
            <h2>{total_consumption:,.0f} ml</h2>
            <p>📊 All Time</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        total_bars = data['Bar Name'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Active Bars</h3>
            <h2>{total_bars}</h2>
            <p>🏢 Locations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_products = data['Brand Name'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Products</h3>
            <h2>{total_products}</h2>
            <p>🍾 Unique Items</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_daily = data.groupby(data['Date Time Served'].dt.date)['Consumed (ml)'].sum().mean()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Avg Daily</h3>
            <h2>{avg_daily:,.0f} ml</h2>
            <p>📅 Per Day</p>
        </div>
        """, unsafe_allow_html=True)
    
    # 3D Visualization Section
    st.markdown('<div class="section-header">🎯 3D Consumption Analysis</div>', unsafe_allow_html=True)
    
    # Interactive 3D Plot
    st.info("🔄 **Interactive 3D Chart**: Rotate, zoom, and hover for details!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Consumption Analysis
        fig_3d = create_3d_consumption_plot(data)
        if fig_3d:
            st.plotly_chart(fig_3d, use_container_width=True)
    
    with col2:
        # 3D Temporal Analysis
        fig_temporal = create_3d_temporal_analysis(data)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Additional metrics
    st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        peak_hour = data['Date Time Served'].dt.hour.mode()[0]
        st.metric("🏆 Peak Hour", f"{peak_hour}:00")
    
    with col2:
        top_alcohol = data.groupby('Alcohol Type')['Consumed (ml)'].sum().idxmax()
        st.metric("🥇 Top Type", top_alcohol)
    
    with col3:
        top_brand = data.groupby('Brand Name')['Consumed (ml)'].sum().idxmax()
        st.metric("🎯 Top Brand", top_brand[:15] + "..." if len(top_brand) > 15 else top_brand)
    
    with col4:
        data_days = (data['Date Time Served'].max() - data['Date Time Served'].min()).days
        st.metric("📅 Data Period", f"{data_days} days")

def show_3d_data_analysis(data):
    """Show advanced 3D data analysis"""
    
    st.markdown('<div class="section-header">🔍 3D Advanced Data Analysis</div>', unsafe_allow_html=True)
    
    # Enhanced filters with better layout
    st.markdown("### 🎯 Filter Data for 3D Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bar_options = ['All Bars'] + sorted(data['Bar Name'].unique().tolist())
        selected_bar = st.selectbox("🏢 Select Bar", bar_options, key='3d_bar_filter')
    
    with col2:
        if selected_bar == 'All Bars':
            alcohol_options = ['All Types'] + sorted(data['Alcohol Type'].unique().tolist())
        else:
            bar_data = data[data['Bar Name'] == selected_bar]
            alcohol_options = ['All Types'] + sorted(bar_data['Alcohol Type'].unique().tolist())
        selected_alcohol = st.selectbox("🍸 Alcohol Type", alcohol_options, key='3d_alcohol_filter')
    
    with col3:
        if selected_bar == 'All Bars' and selected_alcohol == 'All Types':
            brand_options = ['All Brands'] + sorted(data['Brand Name'].unique().tolist())
        elif selected_bar == 'All Bars':
            brand_options = ['All Brands'] + sorted(data[data['Alcohol Type'] == selected_alcohol]['Brand Name'].unique().tolist())
        elif selected_alcohol == 'All Types':
            brand_options = ['All Brands'] + sorted(data[data['Bar Name'] == selected_bar]['Brand Name'].unique().tolist())
        else:
            brand_data = data[(data['Bar Name'] == selected_bar) & (data['Alcohol Type'] == selected_alcohol)]
            brand_options = ['All Brands'] + sorted(brand_data['Brand Name'].unique().tolist())
        selected_brand = st.selectbox("🏷️ Brand", brand_options, key='3d_brand_filter')
    
    # Apply filters
    filtered_data = data.copy()
    
    if selected_bar != 'All Bars':
        filtered_data = filtered_data[filtered_data['Bar Name'] == selected_bar]
    
    if selected_alcohol != 'All Types':
        filtered_data = filtered_data[filtered_data['Alcohol Type'] == selected_alcohol]
    
    if selected_brand != 'All Brands':
        filtered_data = filtered_data[filtered_data['Brand Name'] == selected_brand]
    
    # Show filter summary
    filter_summary = f"Showing: {selected_bar} | {selected_alcohol} | {selected_brand}"
    st.markdown(f"**🔍 Active Filters:** {filter_summary}")
    st.markdown(f"**📊 Records Found:** {len(filtered_data):,} records")
    
    if len(filtered_data) == 0:
        st.markdown("""
        <div class="warning-box">
            <h3>⚠️ No Data Available</h3>
            <p>No records found for the selected filter combination. Please adjust your filters.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # 3D Visualizations
    st.markdown('<div class="section-header">🎯 Interactive 3D Visualizations</div>', unsafe_allow_html=True)
    
    # Main 3D Scatter Plot
    st.info("🔄 **Tip**: Rotate the 3D charts by clicking and dragging. Zoom with scroll wheel.")
    
    fig_3d_main = create_3d_consumption_plot(filtered_data)
    if fig_3d_main:
        st.plotly_chart(fig_3d_main, use_container_width=True)
    
    # Additional 3D Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Temporal Analysis
        fig_temporal = create_3d_temporal_analysis(filtered_data)
        if fig_temporal:
            st.plotly_chart(fig_temporal, use_container_width=True)
    
    with col2:
        # Interactive Sunburst Chart
        try:
            if len(filtered_data) > 0:
                sunburst_data = filtered_data.groupby(['Bar Name', 'Alcohol Type', 'Brand Name']).agg({
                    'Consumed (ml)': 'sum'
                }).reset_index()
                
                fig_sunburst = px.sunburst(
                    sunburst_data,
                    path=['Bar Name', 'Alcohol Type', 'Brand Name'],
                    values='Consumed (ml)',
                    title='🌞 Consumption Hierarchy',
                    color='Consumed (ml)',
                    color_continuous_scale='RdBu'
                )
                fig_sunburst.update_layout(height=500)
                st.plotly_chart(fig_sunburst, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating sunburst chart: {e}")

def show_demand_forecasting(dashboard, data):
    """Show enhanced demand forecasting with 3D"""
    
    st.markdown('<div class="section-header">🤖 AI Demand Forecasting</div>', unsafe_allow_html=True)
    
    if not dashboard.load_model():
        st.markdown("""
        <div class="warning-box">
            <h3>ℹ️ Demo Mode</h3>
            <p>Using intelligent forecasting algorithms. For enhanced accuracy, train the AI model first.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced selection interface
    st.markdown("### 🎯 Select Product for 3D Forecasting")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bar_name = st.selectbox("🏢 Bar Location", data['Bar Name'].unique(), key='forecast_bar')
    
    with col2:
        bar_data = data[data['Bar Name'] == bar_name]
        alcohol_type = st.selectbox("🍸 Alcohol Type", bar_data['Alcohol Type'].unique(), key='forecast_alcohol')
    
    with col3:
        brand_data = data[(data['Bar Name'] == bar_name) & (data['Alcohol Type'] == alcohol_type)]
        if len(brand_data) == 0:
            st.warning("No brands available for this combination")
            return
        brand_name = st.selectbox("🏷️ Brand", brand_data['Brand Name'].unique(), key='forecast_brand')
    
    # Forecast settings
    st.markdown("### ⚙️ 3D Forecast Settings")
    col1, col2 = st.columns(2)
    
    with col1:
        forecast_days = st.slider("📅 Forecast Period (days)", 1, 30, 14, 
                                 help="Number of days to forecast into the future")
    
    with col2:
        confidence_level = st.slider("🎯 Confidence Level", 80, 95, 90,
                                    help="Statistical confidence level for predictions")
    
    if st.button("🚀 Generate 3D AI Forecast", use_container_width=True):
        with st.spinner('🤖 AI is generating 3D intelligent forecasts...'):
            show_3d_forecast_results(dashboard, data, bar_name, alcohol_type, brand_name, forecast_days, confidence_level)

def show_inventory_recommendations(dashboard, data):
    """Show enhanced inventory recommendations with 3D elements"""
    
    st.markdown('<div class="section-header">📦 Smart Inventory Recommendations</div>', unsafe_allow_html=True)
    
    # Product selection
    st.markdown("### 🎯 Select Product for 3D Optimization")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bar_name = st.selectbox("🏢 Bar Location", data['Bar Name'].unique(), key='inv_bar')
    
    with col2:
        bar_data = data[data['Bar Name'] == bar_name]
        alcohol_type = st.selectbox("🍸 Alcohol Type", bar_data['Alcohol Type'].unique(), key='inv_alcohol')
    
    with col3:
        brand_data = data[(data['Bar Name'] == bar_name) & (data['Alcohol Type'] == alcohol_type)]
        if len(brand_data) == 0:
            st.warning("No brands available for this combination")
            return
        brand_name = st.selectbox("🏷️ Brand", brand_data['Brand Name'].unique(), key='inv_brand')
    
    # Optimization settings
    st.markdown("### ⚙️ 3D Optimization Parameters")
    col1, col2 = st.columns(2)
    
    with col1:
        service_level = st.slider("🎯 Service Level Target", 80, 99, 95,
                                 help="Probability of avoiding stockouts")
        st.info(f"Target: {service_level}% service level")
    
    with col2:
        lead_time = st.slider("⏱️ Lead Time (days)", 1, 7, 2,
                             help="Expected delivery time for new orders")
        st.info(f"Lead time: {lead_time} days")
    
    if st.button("🔍 Generate 3D Smart Recommendations", use_container_width=True):
        with st.spinner('🔄 Calculating 3D optimal inventory levels...'):
            show_3d_recommendation_results(data, bar_name, alcohol_type, brand_name, service_level, lead_time)

def show_report_summary(data):
    """Show comprehensive report summary with enhanced export functionality"""
    
    st.markdown('<div class="section-header">📋 Executive Report Summary</div>', unsafe_allow_html=True)
    
    # Report generation date
    st.markdown(f"**Report Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Key Performance Indicators
    st.markdown("### 📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_consumption = data['Consumed (ml)'].sum()
        st.metric("Total Consumption", f"{total_consumption:,.0f} ml")
    
    with col2:
        total_bars = data['Bar Name'].nunique()
        st.metric("Active Bars", total_bars)
    
    with col3:
        total_products = data['Brand Name'].nunique()
        st.metric("Unique Products", total_products)
    
    with col4:
        avg_daily = data.groupby(data['Date Time Served'].dt.date)['Consumed (ml)'].sum().mean()
        st.metric("Avg Daily Consumption", f"{avg_daily:,.0f} ml")
    
    # Consumption Analysis
    st.markdown("### 🍷 Consumption Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top alcohol types
        alcohol_consumption = data.groupby('Alcohol Type')['Consumed (ml)'].sum().sort_values(ascending=False)
        st.markdown("**Top Alcohol Types by Consumption:**")
        for alcohol, consumption in alcohol_consumption.head().items():
            st.write(f"- {alcohol}: {consumption:,.0f} ml")
    
    with col2:
        # Top brands
        brand_consumption = data.groupby('Brand Name')['Consumed (ml)'].sum().sort_values(ascending=False)
        st.markdown("**Top Brands by Consumption:**")
        for brand, consumption in brand_consumption.head().items():
            st.write(f"- {brand}: {consumption:,.0f} ml")
    
    # Bar Performance
    st.markdown("### 🏢 Bar Performance Ranking")
    
    bar_performance = data.groupby('Bar Name')['Consumed (ml)'].sum().sort_values(ascending=False)
    
    for i, (bar, consumption) in enumerate(bar_performance.items(), 1):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"**{i}. {bar}**")
        with col2:
            st.write(f"{consumption:,.0f} ml")
    
    # Recommendations Summary
    st.markdown("### 💡 Key Recommendations")
    
    recommendations = [
        "📈 **Optimize Stock Levels**: Implement dynamic par levels based on consumption patterns",
        "🔄 **Improve Turnover**: Focus on high-consumption products for better inventory rotation",
        "📊 **Enhanced Monitoring**: Use real-time tracking for popular items",
        "🤖 **AI Integration**: Implement machine learning for demand forecasting",
        "📦 **Safety Stock**: Maintain optimal safety stock levels for high-variability products"
    ]
    
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    # Enhanced Export Options
    st.markdown("### 📤 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📄 Report Export")
        
        # Text Report Download
        text_report = generate_text_report(data)
        st.download_button(
            label="📝 Download Text Report",
            data=text_report,
            file_name=f"bar_inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        # PDF Report (if reportlab is available)
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            def generate_pdf_report(data):
                buffer = io.BytesIO()
                p = canvas.Canvas(buffer, pagesize=letter)
                
                # Add content to PDF
                p.drawString(100, 750, "Bar Inventory Intelligence Report")
                p.drawString(100, 730, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                p.drawString(100, 710, f"Total Records: {len(data):,}")
                p.drawString(100, 690, f"Total Consumption: {data['Consumed (ml)'].sum():,} ml")
                p.drawString(100, 670, f"Number of Bars: {data['Bar Name'].nunique()}")
                p.drawString(100, 650, f"Number of Products: {data['Brand Name'].nunique()}")
                
                p.save()
                buffer.seek(0)
                return buffer
            
            if st.button("📊 Generate PDF Report", use_container_width=True):
                pdf_buffer = generate_pdf_report(data)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=pdf_buffer,
                    file_name=f"bar_inventory_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                st.success("PDF report generated successfully!")
        except ImportError:
            st.info("📚 Install reportlab for PDF export: `pip install reportlab`")
    
    with col2:
        st.markdown("#### 📊 Data Export")
        
        # CSV Download
        @st.cache_data
        def convert_to_csv(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv_data = convert_to_csv(data)
        
        st.download_button(
            label="💾 Download CSV",
            data=csv_data,
            file_name=f"bar_inventory_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Excel Download
        @st.cache_data
        def convert_to_excel(df):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='InventoryData')
                
                # Create summary sheet
                summary_data = {
                    'Metric': ['Total Records', 'Total Consumption (ml)', 'Number of Bars', 'Number of Products', 'Average Daily Consumption (ml)'],
                    'Value': [
                        len(df),
                        df['Consumed (ml)'].sum(),
                        df['Bar Name'].nunique(),
                        df['Brand Name'].nunique(),
                        df.groupby(df['Date Time Served'].dt.date)['Consumed (ml)'].sum().mean()
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, index=False, sheet_name='Summary')
            
            return output.getvalue()
        
        excel_data = convert_to_excel(data)
        
        st.download_button(
            label="📗 Download Excel",
            data=excel_data,
            file_name=f"bar_inventory_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        st.markdown("#### 🔄 System")
        
        if st.button("🔄 Refresh Report", use_container_width=True):
            st.rerun()
        
        # Data Summary
        st.markdown("---")
        st.markdown("**📈 Data Summary:**")
        st.markdown(f"- Records: {len(data):,}")
        st.markdown(f"- Date Range: {data['Date Time Served'].min().strftime('%Y-%m-%d')} to {data['Date Time Served'].max().strftime('%Y-%m-%d')}")
        st.markdown(f"- Total Consumption: {data['Consumed (ml)'].sum():,} ml")
        st.markdown(f"- Last Updated: {datetime.now().strftime('%H:%M:%S')}")

def show_about_section():
    """Show about section with project details"""
    
    st.markdown('<div class="section-header">ℹ️ About Bar Inventory Intelligence</div>', unsafe_allow_html=True)
    
    # Project Overview
    st.markdown("""
    <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>Bar Inventory Intelligence is an AI-powered inventory management system designed specifically for bar and beverage businesses. 
        It combines machine learning forecasting with interactive 3D analytics to optimize inventory levels and reduce costs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.markdown("### 🚀 Key Features")
    
    features = [
        ("📊 3D Analytics", "Interactive 3D visualizations for multi-dimensional data analysis"),
        ("🤖 AI Forecasting", "Machine learning models for accurate demand prediction"),
        ("📦 Smart Recommendations", "AI-driven inventory optimization suggestions"),
        ("📈 Real-time Monitoring", "Live dashboards with key performance indicators"),
        ("📋 Report Generation", "Automated executive reports and insights")
    ]
    
    for icon, description in features:
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"**{icon}**")
        with col2:
            st.markdown(description)
    
    # Technology Stack
    st.markdown("### 🔧 Technology Stack")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Frontend:**
        - Streamlit
        - Plotly
        - Matplotlib
        """)
    
    with col2:
        st.markdown("""
        **Backend:**
        - Python
        - Pandas
        - Scikit-learn
        """)
    
    with col3:
        st.markdown("""
        **AI/ML:**
        - XGBoost
        - Random Forest
        - Time Series Analysis
        """)
    '''
    
    # Team Information
    st.markdown("### 👥 Development Team")
    
    team_members = [
        {"Shivam Dubey": "AI Engineer", "role": "Machine Learning & Forecasting",.}
        ]
    
    for member in team_members:
        st.markdown(f"- **{member['name']}**: {member['role']}")
    
    
    # Contact & Support
    st.markdown("### 📞 Support & Contact")
    
    st.markdown("""
    For technical support, feature requests, or bug reports, please contact:
    
    **Email**: support@barinventory.ai  
    **Documentation**: [Bar Inventory Docs](https://docs.barinventory.ai)  
    **GitHub**: [Bar Intelligence Repository](https://github.com/bar-inventory)
    """)
'''

    # Version Info
    st.markdown("### 🔄 Version Information")
    
    st.markdown(f"""
    - **Current Version**: 4.0.0
    - **Last Updated**: {datetime.now().strftime('%Y-%m-%d')}
    - **Python Version**: 3.8+
    - **Streamlit Version**: 1.28.0+
    """)

if __name__ == "__main__":
    main()