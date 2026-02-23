"""
Gold Price Forecasting Model - BEST VERSION
=============================================
Complete implementation with robust error handling and flexible folder creation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
# Suppress all warnings
warnings.filterwarnings('ignore')
import os
from pathlib import Path
import logging
from pathlib import Path
# Suppress statsmodels convergence warnings
logging.getLogger('statsmodels').setLevel(logging.ERROR)

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import math

# Styling
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# CREATE OUTPUT FOLDER - FLEXIBLE AND ROBUST
# ============================================================================

def create_output_folder():
    """
    Create output folder inside airflow/include/data/forecasting_output
    regardless of where the script is executed from
    """
    try:
        # Absolute path of this file
        current_file = Path(__file__).resolve()

        # Go up to 'include'
        include_root = current_file.parent.parent

        # Build correct output path
        output_path = include_root / "data" / "forecasting_output"

        # Create folder if not exists
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"📁 Output directory: {output_path}\n")
        return output_path

    except Exception as e:
        print(f"Error creating folder: {e}")
        raise
OUTPUT_DIR = create_output_folder()


# ============================================================================
# STEP 1: DATA LOADING AND PREPARATION
# ============================================================================

def generate_sample_data():
    """Generate realistic sample gold price data for demonstration"""
    print("Generating sample gold price data...")
    dates = pd.date_range(start='2020-01-01', end='2024-02-20', freq='D')
    np.random.seed(42)
    
    # Create realistic gold price data with trend and seasonality
    trend = np.linspace(1700, 2050, len(dates))
    seasonal = 50 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
    noise = np.random.normal(0, 10, len(dates))
    closing_price = trend + seasonal + noise
    
    df = pd.DataFrame({
        'closing_price': closing_price
    }, index=dates)
    
    return df


def load_gold_prices_from_db():
    """
    Load gold price data from PostgreSQL database
    Falls back to sample data if connection fails
    """
    try:
        import psycopg2
        
        print("Attempting to connect to PostgreSQL...")
        conn = psycopg2.connect(
            host="localhost",
            database="gold_prices_db",
            user="your_username",
            password="your_password"
        )
        
        query = "SELECT date, closing_price FROM gold_prices ORDER BY date ASC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        print("✓ Data loaded from PostgreSQL successfully!")
        return df
    
    except Exception as e:
        print(f"✗ Could not connect to PostgreSQL: {type(e).__name__}")
        print(f"  Error: {str(e)[:80]}...")
        print("\nFalling back to sample data for demonstration...\n")
        return generate_sample_data()


def prepare_time_series_data(df, column='closing_price', test_size=0.2):
    """Prepare data for time series modeling"""
    
    ts_data = df[column].dropna()
    
    if ts_data.isnull().sum() > 0:
        print(f"Warning: {ts_data.isnull().sum()} null values found. Using forward fill.")
        ts_data = ts_data.fillna(method='ffill')
    
    train_size = int(len(ts_data) * (1 - test_size))
    train_data = ts_data[:train_size]
    test_data = ts_data[train_size:]
    
    print(f"Data Preparation Complete:")
    print(f"  ├─ Total observations: {len(ts_data)}")
    print(f"  ├─ Training set: {len(train_data)} ({100*(1-test_size):.0f}%)")
    print(f"  ├─ Testing set: {len(test_data)} ({100*test_size:.0f}%)")
    print(f"  ├─ Train period: {train_data.index.min()} to {train_data.index.max()}")
    print(f"  └─ Test period: {test_data.index.min()} to {test_data.index.max()}\n")
    
    # Visualize the split
    try:
        plt.figure(figsize=(14, 5))
        plt.plot(train_data.index, train_data, label='Training Data', color='blue', linewidth=2)
        plt.plot(test_data.index, test_data, label='Testing Data', color='red', linewidth=2)
        plt.xlabel('Date', fontsize=11)
        plt.ylabel('Gold Price (USD/oz)', fontsize=11)
        plt.title('Train-Test Split of Gold Prices', fontsize=13, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '01_train_test_split.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save plot - {e}")
    
    return train_data, test_data, ts_data


# ============================================================================
# STEP 2: STATIONARITY TESTING
# ============================================================================

def check_stationarity(timeseries, name="Series"):
    """Perform Augmented Dickey-Fuller test for stationarity"""
    
    adf_result = adfuller(timeseries, autolag='AIC')
    
    print(f"\n{'='*70}")
    print(f"Stationarity Test: {name}")
    print(f"{'='*70}")
    print(f"ADF Statistic:        {adf_result[0]:>15.6f}")
    print(f"p-value:              {adf_result[1]:>15.6f}")
    print(f"Number of Lags:       {adf_result[2]:>15d}")
    print(f"Number of Observations: {adf_result[3]:>10d}")
    print(f"\nCritical Values:")
    for key, value in adf_result[4].items():
        print(f"   {key:>3}: {value:>10.3f}")
    
    if adf_result[1] <= 0.05:
        print(f"\n✓ STATIONARY (p-value = {adf_result[1]:.6f} ≤ 0.05)")
        print("  → d=0 parameter for ARIMA\n")
        return True
    else:
        print(f"\n✗ NON-STATIONARY (p-value = {adf_result[1]:.6f} > 0.05)")
        print("  → d≥1 parameter needed for ARIMA\n")
        return False


# ============================================================================
# STEP 3: ACF/PACF VISUALIZATION
# ============================================================================

def plot_acf_pacf(timeseries, lags=40):
    """Plot ACF and PACF for parameter identification"""
    
    try:
        fig, axes = plt.subplots(2, 1, figsize=(13, 8))
        
        plot_acf(timeseries, lags=lags, ax=axes[0])
        axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Lag')
        axes[0].set_ylabel('ACF')
        
        plot_pacf(timeseries, lags=lags, ax=axes[1], method='ywm')
        axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Lag')
        axes[1].set_ylabel('PACF')
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '02_acf_pacf.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save ACF/PACF plot - {e}")


# ============================================================================
# STEP 4: OPTIMAL PARAMETER SEARCH
# ============================================================================

def find_optimal_arima_parameters(timeseries, max_p=5, max_d=2, max_q=5):
    """Grid search for optimal ARIMA parameters using AIC"""
    
    results = []
    print("\nSearching for optimal ARIMA parameters...")
    
    total = (max_p + 1) * (max_d + 1) * (max_q + 1)
    tested = 0
    
    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(timeseries, order=(p, d, q))
                    fitted_model = model.fit()
                    
                    results.append({
                        'order': (p, d, q),
                        'AIC': fitted_model.aic,
                        'BIC': fitted_model.bic
                    })
                except:
                    pass
                
                tested += 1
                if tested % 40 == 0:
                    print(f"  Progress: {tested}/{total} combinations tested", flush=True)
    
    print(f"  Total: {tested} combinations tested successfully.\n")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC').reset_index(drop=True)
    
    print("Top 10 ARIMA models by AIC:")
    print("─" * 50)
    print(results_df.head(10).to_string(index=False))
    print("─" * 50)
    
    best_order = results_df.iloc[0]['order']
    best_aic = results_df.iloc[0]['AIC']
    
    print(f"\n✓ Best ARIMA order: {best_order}")
    print(f"  AIC score: {best_aic:.2f}\n")
    
    return best_order, results_df

def find_optimal_sarima_parameters(timeseries,
                                   max_p=2, max_d=1, max_q=2,
                                   max_P=1, max_D=1, max_Q=1,
                                   seasonal_period=12):
    """
    Grid search for SARIMA parameters using AIC
    """

    print("\nSearching for optimal SARIMA parameters...")
    results = []

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(
                                    timeseries,
                                    order=(p, d, q),
                                    seasonal_order=(P, D, Q, seasonal_period),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False
                                )

                                fitted = model.fit(disp=False)

                                results.append({
                                    'order': (p, d, q),
                                    'seasonal_order': (P, D, Q, seasonal_period),
                                    'AIC': fitted.aic
                                })

                            except:
                                continue

    results_df = pd.DataFrame(results).sort_values("AIC").reset_index(drop=True)

    print("\nTop 5 SARIMA models:")
    print(results_df.head())

    best_order = results_df.iloc[0]["order"]
    best_seasonal = results_df.iloc[0]["seasonal_order"]

    print(f"\n✓ Best SARIMA: {best_order} x {best_seasonal}")

    return best_order, best_seasonal

# ============================================================================
# STEP 5: SEASONALITY CHECK
# ============================================================================

def check_seasonality(timeseries, period=12):
    """Check for seasonality using decomposition"""
    
    try:
        decomposition = seasonal_decompose(timeseries, model='additive', period=period)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10))
        
        axes[0].plot(decomposition.observed, color='blue')
        axes[0].set_ylabel('Observed', fontsize=10)
        axes[0].set_title('Time Series Decomposition', fontsize=13, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(decomposition.trend, color='green')
        axes[1].set_ylabel('Trend', fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(decomposition.seasonal, color='orange')
        axes[2].set_ylabel('Seasonal', fontsize=10)
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(decomposition.resid, color='red')
        axes[3].set_ylabel('Residual', fontsize=10)
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '03_seasonality_decomposition.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
        
        seasonal_strength = 1 - (np.var(decomposition.resid, ddof=1) / 
                                 np.var(decomposition.seasonal + decomposition.resid, ddof=1))
        
        print(f"\nSeasonal Strength Score: {seasonal_strength:.4f}")
        
        if seasonal_strength > 0.1:
            print(f"✓ STRONG SEASONALITY DETECTED")
            print("  Recommendation: Use SARIMA\n")
            return True
        else:
            print(f"✗ WEAK SEASONALITY")
            print("  Recommendation: ARIMA should be sufficient\n")
            return False
            
    except Exception as e:
        print(f"Error in decomposition: {e}")
        print("Proceeding with ARIMA (non-seasonal)\n")
        return False


# ============================================================================
# STEP 6: MODEL TRAINING
# ============================================================================

def train_arima_model(train_data, order):
    """Train ARIMA model with specified parameters"""
    
    print(f"\nTraining ARIMA{order} model...")
    print("─" * 70)
    
    model = ARIMA(train_data, order=order)
    fitted_model = model.fit()
    
    print(fitted_model.summary())
    
    return fitted_model

def train_sarima_model(train_data, order, seasonal_order):

    print(f"\nTraining SARIMA{order} x {seasonal_order} model...")
    print("─" * 70)

    model = SARIMAX(
        train_data,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted_model = model.fit(disp=False)
    print(fitted_model.summary())

    return fitted_model
# ============================================================================
# STEP 7: PREDICTIONS
# ============================================================================

def make_predictions(fitted_model, test_data):
    """Make predictions on test set"""
    
    print("\nGenerating predictions for test set...")
    
    forecast = fitted_model.get_forecast(steps=len(test_data))
    forecast_df = forecast.summary_frame()
    
    predictions = forecast_df['mean'].values  # Convert to numpy array
    conf_lower = forecast_df['mean_ci_lower'].values
    conf_upper = forecast_df['mean_ci_upper'].values
    
    # Convert to pandas Series with proper index
    predictions = pd.Series(predictions, index=test_data.index)
    conf_lower = pd.Series(conf_lower, index=test_data.index)
    conf_upper = pd.Series(conf_upper, index=test_data.index)
    
    return predictions, conf_lower, conf_upper


# ============================================================================
# STEP 8: MODEL EVALUATION
# ============================================================================

def evaluate_model(actual, predicted):
    """Calculate evaluation metrics"""
    
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    actual_direction = np.diff(actual) > 0
    predicted_direction = np.diff(predicted) > 0
    directional_accuracy = np.sum(actual_direction == predicted_direction) / len(actual_direction)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }


def display_metrics(metrics, model_type):
    """Display model metrics"""
    
    print("\n" + "="*70)
    print(f"{model_type} Model - Evaluation Metrics")
    print("="*70)
    print(f"Mean Squared Error (MSE):           {metrics['MSE']:>12.4f}")
    print(f"Root Mean Squared Error (RMSE):     {metrics['RMSE']:>12.4f}")
    print(f"Mean Absolute Error (MAE):          {metrics['MAE']:>12.4f}")
    print(f"Mean Absolute Percentage Error:     {metrics['MAPE']:>12.2f}%")
    print(f"Directional Accuracy:               {metrics['Directional_Accuracy']*100:>12.2f}%")
    print("="*70 + "\n")


# ============================================================================
# STEP 9: VISUALIZATION
# ============================================================================

def plot_forecast_results(train_data, test_data, predictions, conf_lower, conf_upper, model_type):
    """Plot forecast results with confidence intervals"""
    
    try:
        plt.figure(figsize=(16, 7))
        
        plt.plot(train_data.index, train_data, label='Training Data', 
                 color='blue', linewidth=2, alpha=0.7)
        
        plt.plot(test_data.index, test_data, label='Actual Test Data', 
                 color='green', linewidth=2, alpha=0.8)
        
        plt.plot(predictions.index, predictions, label='Predictions', 
                 color='red', linewidth=2, linestyle='--', alpha=0.8)
        
        plt.fill_between(conf_lower.index, conf_lower, conf_upper, 
                         alpha=0.2, color='red', label='95% Confidence Interval')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Gold Price (USD/oz)', fontsize=12)
        plt.title(f'{model_type} Model: Gold Price Forecast on Test Set', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '04_forecast_results.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save forecast plot - {e}")


# ============================================================================
# STEP 10: RESIDUAL ANALYSIS
# ============================================================================

def analyze_residuals(fitted_model, model_type):
    """Analyze residuals to check model assumptions"""
    
    try:
        residuals = fitted_model.resid
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        
        axes[0, 0].plot(residuals, color='blue')
        axes[0, 0].set_title('Residuals Over Time', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(residuals, bins=30, edgecolor='black', color='green', alpha=0.7)
        axes[0, 1].set_title('Distribution of Residuals', fontweight='bold')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        plot_acf(residuals, lags=40, ax=axes[1, 1])
        axes[1, 1].set_title('ACF of Residuals', fontweight='bold')
        
        plt.suptitle(f'{model_type} Model - Residual Analysis', 
                     fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '05_residual_analysis.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
        
        print("\nResidual Statistics:")
        print(f"  Mean:     {residuals.mean():.6f} (should be ≈ 0)")
        print(f"  Std Dev:  {residuals.std():.6f}")
        print(f"  Min:      {residuals.min():.6f}")
        print(f"  Max:      {residuals.max():.6f}\n")
        
    except Exception as e:
        print(f"Warning: Could not analyze residuals - {e}")


# ============================================================================
# STEP 11: FUTURE FORECASTING
# ============================================================================

def forecast_future_prices(model, last_data, periods=30):
    """
    Forecast future prices and return a DataFrame
    """

    import pandas as pd
    
    # Ensure last_data index is datetime
    if not pd.api.types.is_datetime64_any_dtype(last_data.index):
        last_data.index = pd.to_datetime(last_data.index)
    
    # Get last date
    last_date = last_data.index[-1]
    
    # Make forecast
    forecast_res = model.get_forecast(steps=periods)
    forecast_df = forecast_res.summary_frame()
    
    # Generate future dates correctly
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, periods+1)]
    
    forecast_df = forecast_df[['mean', 'mean_ci_lower', 'mean_ci_upper']].copy()
    forecast_df.index = future_dates
    forecast_df.rename(columns={'mean': 'predicted_price',
                                'mean_ci_lower': 'lower_bound',
                                'mean_ci_upper': 'upper_bound'}, inplace=True)
    
    return forecast_df


def plot_complete_forecast(train_data, test_data, predictions, conf_lower, conf_upper,
                          future_forecast, model_type):
    """Plot historical + test predictions + future forecast"""
    
    try:
        plt.figure(figsize=(18, 7))
        
        plt.plot(train_data.index, train_data, label='Historical Training Data', 
                 color='blue', linewidth=2, alpha=0.7)
        
        plt.plot(test_data.index, test_data, label='Actual Test Data', 
                 color='green', linewidth=2, alpha=0.8)
        
        plt.plot(predictions.index, predictions, label='Test Predictions', 
                 color='red', linewidth=2, linestyle='--', alpha=0.8)
        plt.fill_between(conf_lower.index, conf_lower, conf_upper, 
                         alpha=0.15, color='red')
        
        plt.plot(future_forecast.index, future_forecast['mean'], label='Future Forecast', 
                 color='orange', linewidth=2, linestyle='--', alpha=0.8)
        plt.fill_between(future_forecast.index, future_forecast['mean_ci_lower'], 
                         future_forecast['mean_ci_upper'], alpha=0.15, color='orange')
        
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Gold Price (USD/oz)', fontsize=12)
        plt.title(f'{model_type} Model: Complete Forecast (Historical + Future)', 
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11, loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = OUTPUT_DIR / '06_complete_forecast.png'
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"✓ Saved: {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Warning: Could not save complete forecast plot - {e}")

#forecasting_model.py pipeline


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*70)
    print("GOLD PRICE FORECASTING MODEL - ARIMA vs SARIMA")
    print("="*70 + "\n")

    # Step 1: Load data
    print("[STEP 1] Loading Data...")
    df = load_gold_prices_from_db()
    print(f"Data loaded! Shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}\n")

    # Step 2: Prepare data
    print("[STEP 2] Preparing Data...")
    train_data, test_data, full_series = prepare_time_series_data(
        df, column='closing_price', test_size=0.2
    )

    # Step 3: Stationarity
    print("[STEP 3] Checking Stationarity...")
    is_stationary = check_stationarity(train_data, "Original Series")

    if not is_stationary:
        diff_series = train_data.diff().dropna()
        check_stationarity(diff_series, "First Difference")

    # Step 4: ACF/PACF
    print("[STEP 4] Generating ACF/PACF Plots...")
    plot_acf_pacf(train_data, lags=40)

    # Step 5: Seasonality
    print("[STEP 5] Checking for Seasonality...")
    check_seasonality(train_data, period=12)

    # =====================================================
    # ==================== ARIMA ==========================
    # =====================================================

    print("\n[STEP 6A] Searching ARIMA Parameters...")
    best_arima_order, _ = find_optimal_arima_parameters(train_data)

    print("[STEP 7A] Training ARIMA...")
    arima_model = train_arima_model(train_data, best_arima_order)

    print("[STEP 8A] Predicting with ARIMA...")
    arima_pred, arima_lower, arima_upper = make_predictions(arima_model, test_data)

    print("[STEP 9A] Evaluating ARIMA...")
    arima_metrics = evaluate_model(test_data.values, arima_pred.values)
    display_metrics(arima_metrics, "ARIMA")

    # =====================================================
    # ==================== SARIMA =========================
    # =====================================================

    print("\n[STEP 6B] Searching SARIMA Parameters...")
    best_sarima_order, best_sarima_seasonal = find_optimal_sarima_parameters(train_data)

    print("[STEP 7B] Training SARIMA...")
    sarima_model = train_sarima_model(
        train_data,
        best_sarima_order,
        best_sarima_seasonal
    )

    print("[STEP 8B] Predicting with SARIMA...")
    sarima_pred, sarima_lower, sarima_upper = make_predictions(sarima_model, test_data)

    print("[STEP 9B] Evaluating SARIMA...")
    sarima_metrics = evaluate_model(test_data.values, sarima_pred.values)
    display_metrics(sarima_metrics, "SARIMA")

    # =====================================================
    # ==================== COMPARISON =====================
    # =====================================================

    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70)

    comparison_df = pd.DataFrame({
        "Model": ["ARIMA", "SARIMA"],
        "AIC": [arima_model.aic, sarima_model.aic],
        "RMSE": [arima_metrics["RMSE"], sarima_metrics["RMSE"]],
        "MAE": [arima_metrics["MAE"], sarima_metrics["MAE"]],
        "MAPE": [arima_metrics["MAPE"], sarima_metrics["MAPE"]],
        "Directional Accuracy": [
            arima_metrics["Directional_Accuracy"],
            sarima_metrics["Directional_Accuracy"]
        ]
    })

    print(comparison_df)

    best_model_name = comparison_df.sort_values("RMSE").iloc[0]["Model"]
    print(f"\n🏆 Best Model Based on RMSE: {best_model_name}")

    # =====================================================
    # Use Best Model For Final Forecast
    # =====================================================

    if best_model_name == "ARIMA":
        best_model = arima_model
        best_pred = arima_pred
        best_lower = arima_lower
        best_upper = arima_upper
        best_metrics = arima_metrics
    else:
        best_model = sarima_model
        best_pred = sarima_pred
        best_lower = sarima_lower
        best_upper = sarima_upper
        best_metrics = sarima_metrics

    # Step 10: Visualization
    print("\n[STEP 10] Visualizing Best Model...")
    plot_forecast_results(
        train_data, test_data,
        best_pred, best_lower, best_upper,
        best_model_name
    )

    # Step 11: Residuals
    print("[STEP 11] Residual Analysis...")
    analyze_residuals(best_model, best_model_name)

    # Step 12: Future Forecast
    print("[STEP 12] Forecasting Future (30 days)...")
    future_forecast = forecast_future_prices(best_model, test_data, periods=30)

    print(future_forecast[['mean', 'mean_ci_lower', 'mean_ci_upper']].head(10))

    # Step 13: Complete Plot
    print("[STEP 13] Complete Forecast Plot...")
    plot_complete_forecast(
        train_data, test_data,
        best_pred, best_lower, best_upper,
        future_forecast,
        best_model_name
    )

    print("\n" + "="*70)
    print("FORECASTING COMPLETE")
    print("="*70)

    return {
        "best_model_name": best_model_name,
        "metrics": best_metrics,
        "comparison": comparison_df,
        "future_forecast": future_forecast
    }

if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        print(f"\n✗ Error occurred: {type(e).__name__}")
        print(f"  Message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()

def save_forecast_to_db(forecast_df, model_metrics, model_name='ARIMA'):
    """
    Save forecast results and model performance to PostgreSQL
    
    Args:
        forecast_df: DataFrame with columns [date, predicted_price, lower_bound, upper_bound]
        model_metrics: dict with keys [rmse, mae, mape, train_size, test_size, test_start, test_end]
        model_name: str, name of the model
    """
    import psycopg2
    from datetime import datetime
    
    DB_PARAMS = {
        'host': 'localhost',
        'port': 5432,
        'database': 'gold_prices_db',
        'user': 'postgres',
        'password': 'postgres'
    }
    
    conn = psycopg2.connect(**DB_PARAMS)
    cursor = conn.cursor()
    
    try:
        # Save forecasts
        for _, row in forecast_df.iterrows():
            cursor.execute("""
                INSERT INTO gold_forecasts (forecast_date, predicted_price, lower_bound, upper_bound, model_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row['date'],
                float(row['predicted_price']),
                float(row['lower_bound']),
                float(row['upper_bound']),
                model_name
            ))
        
        # Save model performance
        cursor.execute("""
            INSERT INTO model_performance 
            (model_name, rmse, mae, mape, training_date, test_start_date, test_end_date, train_size, test_size)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            model_name,
            float(model_metrics['rmse']),
            float(model_metrics['mae']),
            float(model_metrics['mape']),
            datetime.now().date(),
            model_metrics.get('test_start'),
            model_metrics.get('test_end'),
            model_metrics.get('train_size'),
            model_metrics.get('test_size')
        ))
        
        conn.commit()
        print(f"✅ Saved {len(forecast_df)} forecasts and model metrics to database")
        
    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving to database: {e}")
        raise
    finally:
        cursor.close()
        conn.close()