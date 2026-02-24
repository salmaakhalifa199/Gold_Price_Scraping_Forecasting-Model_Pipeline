"""
Gold Price Analysis & Forecasting Dashboard
Complete interactive dashboard with historical data and AI predictions
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
import numpy as np

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../include/utils'))
from db_connector import DatabaseConnector

# Page configuration
st.set_page_config(
    page_title="Gold Price Forecasting Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db_connection():
    return DatabaseConnector()  # lightweight — no persistent conn inside


@st.cache_data(ttl=300)
def load_historical_data(_db):
    """Load historical price data"""
    return _db.get_all_gold_prices()


@st.cache_data(ttl=300)
def load_forecasts(_db, model_name=None):
    """Load forecast data"""
    return _db.get_forecasts(model_name)


@st.cache_data(ttl=300)
def load_model_performance(_db):
    """Load model performance metrics"""
    return _db.get_model_performance()


@st.cache_data(ttl=300)
def load_summary_stats(_db):
    """Load summary statistics"""
    return _db.get_summary_stats()


def calculate_kpis(df):
    """Calculate key performance indicators"""
    if df.empty or len(df) < 2:
        return {}
    
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else latest
    first = df.iloc[0]
    
    daily_change = latest['close'] - previous['close']
    daily_change_pct = (daily_change / previous['close']) * 100
    total_return = ((latest['close'] - first['close']) / first['close']) * 100
    
    if len(df) >= 30:
        returns = df['close'].pct_change().tail(30)
        volatility = returns.std() * 100
    else:
        volatility = 0
    
    return {
        'current_price': latest['close'],
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'all_time_high': df['high'].max(),
        'all_time_low': df['low'].min(),
        'total_return': total_return,
        'volatility_30d': volatility,
        'avg_volume': df['volume'].mean(),
        'latest_date': latest['date']
    }


def plot_historical_with_forecast(historical_df, forecast_df):
    """Plot historical prices with forecasts"""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_df['date'],
        y=historical_df['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Date: %{x}<br>Price: $%{y:,.2f}<extra></extra>'
    ))
    
    if not forecast_df.empty:
        # Forecast line
        fig.add_trace(go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['predicted_price'],
            mode='lines',
            name='AI Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            hovertemplate='Date: %{x}<br>Forecast: $%{y:,.2f}<extra></extra>'
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['upper_bound'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df['forecast_date'],
            y=forecast_df['lower_bound'],
            mode='lines',
            name='95% Confidence Interval',
            fill='tonexty',
            fillcolor='rgba(255,127,14,0.2)',
            line=dict(width=0),
            hovertemplate='Lower: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Gold Price: Historical Data + AI Forecast',
            'font': {'size': 20, 'color': '#2c3e50'}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)')
    )
    
    return fig


def plot_price_comparison(df):
    """Plot OHLC comparison"""
    fig = go.Figure()
    
    colors = {
        'open': '#3498db',
        'high': '#2ecc71',
        'low': '#e74c3c',
        'close': '#f39c12'
    }
    
    for col in ['open', 'high', 'low', 'close']:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df[col],
            mode='lines',
            name=col.capitalize(),
            line=dict(width=2, color=colors[col]),
            hovertemplate='%{y:$,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        title={
            'text': 'Price Comparison (Open, High, Low, Close)',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig


def plot_candlestick(df):
    """Create candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='Gold Price',
        increasing_line_color='#2ecc71',
        decreasing_line_color='#e74c3c'
    )])
    
    fig.update_layout(
        title={
            'text': 'Gold Price Candlestick Chart',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        template='plotly_white',
        height=500,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_volume_analysis(df):
    """Plot volume analysis"""
    colors = ['#e74c3c' if df.iloc[i]['close'] < df.iloc[i]['open'] else '#2ecc71' 
              for i in range(len(df))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=df['date'],
            y=df['volume'],
            marker_color=colors,
            name='Volume',
            hovertemplate='Volume: %{y:,.0f}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title={
            'text': 'Trading Volume Analysis',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Volume',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_forecast_vs_actual(historical_df, forecast_df, test_start_date):
    """Plot forecast accuracy on test set"""
    # Filter test period from historical data
    test_actual = historical_df[historical_df['date'] >= pd.to_datetime(test_start_date)].copy()
    
    if test_actual.empty or forecast_df.empty:
        return None
    
    fig = go.Figure()
    
    # Actual test data
    fig.add_trace(go.Scatter(
        x=test_actual['date'],
        y=test_actual['close'],
        mode='lines+markers',
        name='Actual Price',
        line=dict(color='#2ecc71', width=2),
        marker=dict(size=6),
        hovertemplate='Actual: $%{y:,.2f}<extra></extra>'
    ))
    
    # Predicted values
    fig.add_trace(go.Scatter(
        x=forecast_df['forecast_date'],
        y=forecast_df['predicted_price'],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        marker=dict(size=6),
        hovertemplate='Predicted: $%{y:,.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Forecast vs Actual Prices (Test Set)',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
    
    return fig


def plot_moving_averages(df):
    """Plot price with moving averages"""
    df = df.copy()
    df['MA_7'] = df['close'].rolling(window=7).mean()
    df['MA_30'] = df['close'].rolling(window=30).mean()
    df['MA_90'] = df['close'].rolling(window=90).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['close'],
        mode='lines', name='Close Price',
        line=dict(color='lightgray', width=1), opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA_7'],
        mode='lines', name='7-Day MA',
        line=dict(color='#3498db', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA_30'],
        mode='lines', name='30-Day MA',
        line=dict(color='#f39c12', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['MA_90'],
        mode='lines', name='90-Day MA',
        line=dict(color='#e74c3c', width=2)
    ))
    
    fig.update_layout(
        title={
            'text': 'Price Trends with Moving Averages',
            'font': {'size': 18}
        },
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    return fig

def display_model_metrics(metrics_df,df_forecasts=None):
    """Display model performance metrics"""
    if metrics_df.empty:
        st.warning("⚠️ No model performance data available")
        return
    
    latest_metrics = metrics_df.iloc[0]
    

    if 'train_size' not in latest_metrics:
        if 'df_historical' in globals():
            forecast_horizon = df_forecasts['forecast_date'].nunique() if not df_forecasts.empty else 0
            train_size = max(len(df_historical) - forecast_horizon, 0)
        else:
            train_size = 0
    else:
        train_size = latest_metrics['train_size']
    
    st.subheader("🤖 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 RMSE",
            value=f"${latest_metrics['rmse']:.2f}",
            help="Root Mean Square Error - Lower is better"
        )
    
    with col2:
        st.metric(
            label="📉 MAE",
            value=f"${latest_metrics['mae']:.2f}",
            help="Mean Absolute Error - Average prediction error"
        )
    
    with col3:
        st.metric(
            label="📈 MAPE",
            value=f"{latest_metrics['mape']:.2f}%",
            help="Mean Absolute Percentage Error"
        )
    
    with col4:
        st.metric(
            label="🎯 Model",
            value=latest_metrics['model_name'],
            help="Forecasting model used"
        )
    
    with st.expander("📋 Detailed Model Information"):
     col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Training Information:**")
        st.write(f"- Training Size: {train_size:,} records")
        st.write(f"- Training Date: {latest_metrics.get('training_date', 'N/A')}")
        st.write(f"- Model: {latest_metrics['model_name']}")
    
    with col2:
        st.write("**Test Period:**")
        # These columns don't exist in DB, use forecast data instead
        if not df_forecasts.empty:
            st.write(f"- Start Date: {df_forecasts['forecast_date'].min().strftime('%Y-%m-%d')}")
            st.write(f"- End Date: {df_forecasts['forecast_date'].max().strftime('%Y-%m-%d')}")
        else:
            st.write("- Start Date: N/A")
            st.write("- End Date: N/A")
    
    # Model comparison if multiple models exist
    if len(metrics_df) > 1:
        st.subheader("📊 Model Comparison")
        
        comparison_df = metrics_df[['model_name', 'rmse', 'mae', 'mape', 'training_date']].copy()
        comparison_df.columns = ['Model', 'RMSE', 'MAE', 'MAPE (%)', 'Training Date']
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 Gold Price Forecasting Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align: center; color: gray; font-size: 1.1rem;">'
        'Powered by AI | Real-time Data Analysis & Predictions'
        '</p>',
        unsafe_allow_html=True
    )
    
    # Initialize database
    db = get_db_connection()
    
    # Sidebar
    st.sidebar.title("⚙️ Dashboard Controls")
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    # Load data
    with st.spinner("📊 Loading data from database..."):
        df_historical = load_historical_data(db)
        df_forecasts = load_forecasts(db)
        df_metrics = load_model_performance(db)
        summary_stats = load_summary_stats(db)
    
    if df_historical.empty:
        st.error("❌ No data available in the database. Please run the Airflow pipeline first.")
        st.info("💡 Run the pipeline to scrape and store gold price data.")
        return
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range Filter")
    
    min_date = df_historical['date'].min().date()
    max_date = df_historical['date'].max().date()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=max_date - timedelta(days=180),
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Filter data
    df_filtered = df_historical[
        (df_historical['date'].dt.date >= start_date) & 
        (df_historical['date'].dt.date <= end_date)
    ].copy()
    
    st.sidebar.info(f"📊 Showing **{len(df_filtered):,}** records")
    
    # Chart selection
    st.sidebar.subheader("📈 Chart Options")
    show_forecast = st.sidebar.checkbox("Show AI Forecast", value=True)
    show_candlestick = st.sidebar.checkbox("Candlestick Chart", value=False)
    show_volume = st.sidebar.checkbox("Volume Analysis", value=False)
    show_ma = st.sidebar.checkbox("Moving Averages", value=False)
    show_comparison = st.sidebar.checkbox("OHLC Comparison", value=True)
    
    # Calculate KPIs
    kpis = calculate_kpis(df_filtered)
    
    # KPI Dashboard
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        delta_color = "normal" if kpis.get('daily_change', 0) >= 0 else "inverse"
        st.metric(
            label="Current Price",
            value=f"${kpis.get('current_price', 0):,.2f}",
            delta=f"{kpis.get('daily_change_pct', 0):.2f}%",
            delta_color=delta_color
        )
    
    with col2:
        st.metric(
            label="All-Time High",
            value=f"${kpis.get('all_time_high', 0):,.2f}",
            delta=None
        )
    
    with col3:
        st.metric(
            label="All-Time Low",
            value=f"${kpis.get('all_time_low', 0):,.2f}",
            delta=None
        )
    
    with col4:
        st.metric(
            label="Total Return",
            value=f"{kpis.get('total_return', 0):.2f}%",
            delta=None,
            delta_color="normal" if kpis.get('total_return', 0) >= 0 else "inverse"
        )
    
    with col5:
        st.metric(
            label="30-Day Volatility",
            value=f"{kpis.get('volatility_30d', 0):.2f}%",
            delta=None
        )
    
    # Additional KPIs
    col6, col7, col8 = st.columns(3)
    
    with col6:
        st.metric(
            label="Average Volume",
            value=f"{kpis.get('avg_volume', 0):,.0f}"
        )
    
    with col7:
        st.metric(
            label="Total Records",
            value=f"{summary_stats.get('total_records', 0):,}"
        )
    
    with col8:
        st.metric(
            label="Data Range",
            value=f"{(max_date - min_date).days} days"
        )
    
    st.markdown("---")
    
    # Main Chart: Historical + Forecast
    st.subheader("📈 Historical Data & AI Forecast")
    
    if show_forecast and not df_forecasts.empty:
        fig_main = plot_historical_with_forecast(df_filtered, df_forecasts)
    else:
        fig_main = plot_historical_with_forecast(df_filtered, pd.DataFrame())
    
    st.plotly_chart(fig_main, use_container_width=True)
    
    # Model Performance Metrics
    if not df_metrics.empty:
        st.markdown("---")
        display_model_metrics(df_metrics, df_forecasts)
    
        # Forecast vs Actual (if metrics available)
   # Forecast vs Actual (if metrics available)
    if not df_metrics.empty and not df_forecasts.empty:
        st.markdown("---")
        st.subheader("🎯 Forecast Accuracy Analysis")
        
        # Use earliest forecast date as test start (no test_start_date column in DB)
        test_start = df_forecasts['forecast_date'].min()
        fig_accuracy = plot_forecast_vs_actual(df_historical, df_forecasts, test_start)
        
        if fig_accuracy:
            st.plotly_chart(fig_accuracy, use_container_width=True)
    
    # Additional Charts
    st.markdown("---")
    
    if show_comparison:
        st.subheader("📊 Price Comparison (OHLC)")
        fig_comparison = plot_price_comparison(df_filtered)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Two-column layout for additional charts
    col_left, col_right = st.columns(2)
    
    with col_left:
        if show_candlestick:
            st.subheader("🕯️ Candlestick Chart")
            df_candle = df_filtered.tail(90)  # Last 90 days
            fig_candle = plot_candlestick(df_candle)
            st.plotly_chart(fig_candle, use_container_width=True)
        
        if show_volume:
            st.subheader("📊 Trading Volume")
            fig_volume = plot_volume_analysis(df_filtered)
            st.plotly_chart(fig_volume, use_container_width=True)
    
    with col_right:
        if show_ma:
            st.subheader("📈 Moving Averages")
            fig_ma = plot_moving_averages(df_filtered)
            st.plotly_chart(fig_ma, use_container_width=True)
    
    # Data Table
    st.markdown("---")
    st.subheader("📋 Raw Data Table")
    
    if st.checkbox("Show data table"):
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            rows_to_show = st.selectbox("Rows to display:", [10, 25, 50, 100], index=1)
        with col2:
            sort_order = st.radio("Sort order:", ["Newest First", "Oldest First"], horizontal=True)
        
        # Prepare display data
        display_df = df_filtered[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        if sort_order == "Newest First":
            display_df = display_df.sort_values('date', ascending=False)
        else:
            display_df = display_df.sort_values('date', ascending=True)
        
        display_df = display_df.head(rows_to_show)
        
        # Format for display
        display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
        display_df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Complete Dataset as CSV",
            data=csv,
            file_name=f"gold_prices_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Forecast Table
    if not df_forecasts.empty:
        st.markdown("---")
        st.subheader("🔮 Forecast Data")
        
        if st.checkbox("Show forecast table"):
            forecast_display = df_forecasts[['forecast_date', 'predicted_price', 'lower_bound', 'upper_bound', 'model_name']].copy()
            forecast_display['forecast_date'] = forecast_display['forecast_date'].dt.strftime('%Y-%m-%d')
            forecast_display.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound', 'Model']
            
            st.dataframe(
                forecast_display,
                use_container_width=True,
                hide_index=True
            )
            
            # Download forecast
            csv_forecast = df_forecasts.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Forecast Data as CSV",
                data=csv_forecast,
                file_name=f"gold_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray; padding: 2rem;'>
            <h4>Gold Price Analysis & Forecasting Dashboard</h4>
            <p>📊 Data Source: yfinance (Gold Futures GC=F) | 🗄️ Database: PostgreSQL | 🤖 Model: ARIMA/SARIMA</p>
            <p>🔄 Pipeline: Apache Airflow | 📈 Visualization: Plotly & Streamlit</p>
            <p style='margin-top: 1rem;'>Last Updated: {}</p>
        </div>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.info("💡 Please check your database connection and ensure the pipeline has run successfully.")
        import traceback
        with st.expander("🔍 Error Details"):
            st.code(traceback.format_exc())