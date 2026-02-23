"""
Gold Price Scraper DAG
Schedules daily gold price scraping with error handling and monitoring
"""
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import logging
from io import StringIO

# Import from the utils package and the scrape function
sys.path.insert(0, '/usr/local/airflow/include/utils')
from scrape_gold import scrape_gold_prices
from data_cleaner import clean_gold_data
from eda_analyzer import perform_eda
from postgres_handler import PostgresHandler
from forecasting_model import (
    prepare_time_series_data,
    train_arima_model,
    train_sarima_model,
    make_predictions,
    evaluate_model,
    forecast_future_prices,
    find_optimal_arima_parameters,
    find_optimal_sarima_parameters,
    check_stationarity,
    check_seasonality,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── DB connection params (shared across tasks) ────────────────────────────────
DB_PARAMS = {
    'host': 'host.docker.internal',
    'port': 5432,
    'database': 'gold_prices_db',
    'user': 'postgres',
    'password': 'postgres'
}

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Define the DAG
dag = DAG(
    'gold_price_scraper',
    default_args=default_args,
    description='Complete gold price pipeline with forecasting',
    schedule='@daily',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['scraping', 'gold', 'finance', 'data-analysis', 'forecasting', 'ml'],
)

# =============================================================================
# TASK 1 — SCRAPE
# =============================================================================

def scrape_task_wrapper(**context):
    """Scrape gold prices"""
    logger.info("Starting gold price scraping task...")

    try:
        output_dir = '/usr/local/airflow/include/data'
        filepath = scrape_gold_prices(days=365, output_dir=output_dir)
        df = pd.read_csv(filepath)

        context['ti'].xcom_push(key='filepath', value=filepath)
        context['ti'].xcom_push(key='dataframe', value=df.to_json(orient='split', date_format='iso'))

        logger.info(f"✅ Scraping completed: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 2 — CLEAN
# =============================================================================

def clean_data_task(**context):
    """Clean and preprocess data"""
    logger.info("Starting data cleaning task...")

    try:
        df_json = context['ti'].xcom_pull(key='dataframe', task_ids='scrape_gold_prices')
        df = pd.read_json(StringIO(df_json), orient='split')

        cleaned_df, cleaning_report = clean_gold_data(df)

        output_dir = '/usr/local/airflow/include/data'
        cleaned_filename = f"gold_prices_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cleaned_filepath = os.path.join(output_dir, cleaned_filename)
        cleaned_df.to_csv(cleaned_filepath, index=False)

        context['ti'].xcom_push(key='cleaned_filepath', value=cleaned_filepath)
        context['ti'].xcom_push(key='cleaned_dataframe', value=cleaned_df.to_json(orient='split', date_format='iso'))
        context['ti'].xcom_push(key='cleaning_report', value=cleaning_report)

        logger.info(f"✅ Cleaned data saved: {cleaned_filepath}")
        return cleaned_filepath
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 3 — INSERT TO POSTGRES
# =============================================================================

def insert_to_postgres_task(**context):
    """Insert cleaned data to PostgreSQL"""
    logger.info("Starting PostgreSQL insertion...")

    try:
        import psycopg2

        df_json = context['ti'].xcom_pull(key='cleaned_dataframe', task_ids='clean_data')
        df = pd.read_json(StringIO(df_json), orient='split')

        df.columns = df.columns.str.lower()
        df['date'] = pd.to_datetime(df['date']).dt.date

        conn = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        inserted = 0
        updated = 0

        for _, row in df.iterrows():
            cursor.execute("""
                INSERT INTO gold_prices (
                    price_date, open_price, high_price, low_price, close_price, volume
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (price_date) DO UPDATE SET
                    open_price  = EXCLUDED.open_price,
                    high_price  = EXCLUDED.high_price,
                    low_price   = EXCLUDED.low_price,
                    close_price = EXCLUDED.close_price,
                    volume      = EXCLUDED.volume,
                    updated_at  = CURRENT_TIMESTAMP
                RETURNING (xmax = 0) AS inserted
            """, (
                row['date'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))

            result = cursor.fetchone()
            if result[0]:
                inserted += 1
            else:
                updated += 1

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ Database updated: {inserted} inserted, {updated} updated")
        context['ti'].xcom_push(key='db_inserted', value=inserted)
        context['ti'].xcom_push(key='db_updated', value=updated)

        return f"Inserted: {inserted}, Updated: {updated}"

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 4 — EDA
# =============================================================================

def eda_task(**context):
    """Perform exploratory data analysis"""
    logger.info("Starting EDA task...")

    try:
        df_json = context['ti'].xcom_pull(key='cleaned_dataframe', task_ids='clean_data')
        logger.info(f"type(df_json): {type(df_json)}")
        logger.info(f"Preview: {df_json[:500]}")

        df = pd.read_json(StringIO(df_json), orient='split')

        output_dir = '/usr/local/airflow/include/data/eda'
        stats_report, plot_paths = perform_eda(df, output_dir)

        context['ti'].xcom_push(key='eda_stats', value=stats_report)
        context['ti'].xcom_push(key='eda_plots', value=plot_paths)

        logger.info(f"✅ EDA complete! Generated {len(plot_paths)} visualizations")
        return stats_report

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 5 — VALIDATE
# =============================================================================

def validate_data_task(**context):
    """Validate cleaned data"""
    logger.info("Starting validation...")

    try:
        filepath = context['ti'].xcom_pull(key='cleaned_filepath', task_ids='clean_data')
        df = pd.read_csv(filepath)

        if len(df) == 0:
            raise ValueError("Empty dataset")

        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        logger.info(f"✅ Validation passed: {len(df)} records")
        return "Validation successful"
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 6 — FORECASTING  (upgraded: ARIMA vs SARIMA grid search + auto-select)
# =============================================================================

def run_forecasting(**context):
    """
    Run ARIMA and SARIMA models, compare on RMSE, and push the best model's
    results (test-set predictions + 30-day future forecast + metrics) via XCom.

    Changes vs original:
      - Added SARIMA grid search using find_optimal_sarima_parameters()
      - Auto-selects between ARIMA and SARIMA by RMSE
      - Pushes 'best_model_name' so save_forecast_to_db_task can label rows correctly
      - Fixed XCom keys so save_forecast_to_db_task can find 'forecast_data'
        and 'model_metrics' (keys the save task already expects)
    """
    import warnings
    import math
    import numpy as np
    warnings.filterwarnings('ignore')

    logger.info("Starting forecasting task...")

    # ── Pull cleaned data ─────────────────────────────────────────────
    df_json = context['ti'].xcom_pull(key='cleaned_dataframe', task_ids='clean_data')
    df = pd.read_json(StringIO(df_json), orient='split')

    # ── Set DatetimeIndex ─────────────────────────────────────────────
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.index = pd.DatetimeIndex(df.index).to_period('D').to_timestamp()

    logger.info(f"Columns after setting index: {df.columns.tolist()}")

    # ── Pick price column ─────────────────────────────────────────────
    for candidate in ('closing_price', 'Close', 'close'):
        if candidate in df.columns:
            column_to_use = candidate
            break
    else:
        column_to_use = df.columns[0]
        logger.warning(f"No standard price column found; using '{column_to_use}'")

    logger.info(f"Forecasting on column: '{column_to_use}'")

    # ── Train / test split ────────────────────────────────────────────
    train_data, test_data, full_series = prepare_time_series_data(df, column=column_to_use)

    # ── Stationarity & seasonality checks ────────────────────────────
    is_stationary = check_stationarity(train_data, "Training Series")
    has_seasonality = check_seasonality(train_data, period=12)

    # ── ARIMA: grid search + fit ──────────────────────────────────────
    logger.info("Grid-searching ARIMA parameters...")
    best_arima_order, _ = find_optimal_arima_parameters(train_data)

    logger.info(f"Fitting ARIMA{best_arima_order}...")
    arima_fitted = train_arima_model(train_data, order=best_arima_order)
    arima_pred, arima_lower, arima_upper = make_predictions(arima_fitted, test_data)
    arima_metrics = evaluate_model(test_data.values, arima_pred.values)
    logger.info(f"ARIMA  → RMSE={arima_metrics['RMSE']:.4f}  MAE={arima_metrics['MAE']:.4f}  MAPE={arima_metrics['MAPE']:.2f}%")

    # ── SARIMA: grid search + fit (only when seasonality detected) ────
    if has_seasonality:
        logger.info("Grid-searching SARIMA parameters...")
        best_sarima_order, best_sarima_seasonal = find_optimal_sarima_parameters(train_data)

        logger.info(f"Fitting SARIMA{best_sarima_order}x{best_sarima_seasonal}...")
        sarima_fitted = train_sarima_model(train_data, best_sarima_order, best_sarima_seasonal)
        sarima_pred, sarima_lower, sarima_upper = make_predictions(sarima_fitted, test_data)
        sarima_metrics = evaluate_model(test_data.values, sarima_pred.values)
        logger.info(f"SARIMA → RMSE={sarima_metrics['RMSE']:.4f}  MAE={sarima_metrics['MAE']:.4f}  MAPE={sarima_metrics['MAPE']:.2f}%")
    else:
        logger.info("No significant seasonality detected — skipping SARIMA.")
        sarima_metrics = None

    # ── Select best model ─────────────────────────────────────────────
    if sarima_metrics and sarima_metrics['RMSE'] < arima_metrics['RMSE']:
        best_model_name = 'SARIMA'
        best_fitted     = sarima_fitted
        best_pred       = sarima_pred
        best_lower      = sarima_lower
        best_upper      = sarima_upper
        best_metrics    = sarima_metrics
    else:
        best_model_name = 'ARIMA'
        best_fitted     = arima_fitted
        best_pred       = arima_pred
        best_lower      = arima_lower
        best_upper      = arima_upper
        best_metrics    = arima_metrics

    logger.info(f"🏆 Best model: {best_model_name}")

    # ── 30-day future forecast ────────────────────────────────────────
    future_forecast_df = forecast_future_prices(best_fitted, last_data=test_data, periods=30)

    # ── Build forecast DataFrame for the save task ────────────────────
    # Combines test-period predictions + future forecast into one table
    test_forecast_df = pd.DataFrame({
        'date':            test_data.index,
        'predicted_price': best_pred.values,
        'lower_bound':     best_lower.values,
        'upper_bound':     best_upper.values,
    })

    future_rows = pd.DataFrame({
        'date':            future_forecast_df.index,
        'predicted_price': future_forecast_df['predicted_price'].values,
        'lower_bound':     future_forecast_df['lower_bound'].values,
        'upper_bound':     future_forecast_df['upper_bound'].values,
    })

    combined_forecast = pd.concat([test_forecast_df, future_rows], ignore_index=True)
    combined_forecast['date'] = combined_forecast['date'].astype(str)

    # ── Model metrics dict (matches keys expected by save task) ───────
    model_metrics = {
        'rmse':       round(best_metrics['RMSE'],  4),
        'mae':        round(best_metrics['MAE'],   4),
        'mape':       round(best_metrics['MAPE'],  4),
        'train_size': len(train_data),
        'test_size':  len(test_data),
        'test_start': str(test_data.index[0].date()),
        'test_end':   str(test_data.index[-1].date()),
    }

    # ── XCom push ─────────────────────────────────────────────────────
    context['ti'].xcom_push(key='forecast_data',    value=combined_forecast.to_json(orient='split'))
    context['ti'].xcom_push(key='model_metrics',    value=model_metrics)
    context['ti'].xcom_push(key='best_model_name',  value=best_model_name)
    context['ti'].xcom_push(key='forecast_metrics', value=model_metrics)   # backward-compat alias
    context['ti'].xcom_push(key='future_forecast',  value=future_forecast_df.to_json(orient='split'))

    logger.info("✅ Forecasting task complete")

# =============================================================================
# TASK 7 — SAVE FORECAST TO DB
# =============================================================================

def save_forecast_to_db_task(**context):
    """Save forecast results to PostgreSQL"""
    logger.info("Saving forecast to database...")

    try:
        import psycopg2

        forecast_json  = context['ti'].xcom_pull(key='forecast_data',   task_ids='run_forecasting')
        model_metrics  = context['ti'].xcom_pull(key='model_metrics',   task_ids='run_forecasting')
        best_model_name = context['ti'].xcom_pull(key='best_model_name', task_ids='run_forecasting') or 'ARIMA'

        forecast_df = pd.read_json(StringIO(forecast_json), orient='split')
        forecast_df['date'] = pd.to_datetime(forecast_df['date']).dt.date

        conn   = psycopg2.connect(**DB_PARAMS)
        cursor = conn.cursor()

        # Clear old forecasts for this model before inserting fresh ones
        cursor.execute("DELETE FROM gold_forecasts WHERE model_name = %s", (best_model_name,))

        for _, row in forecast_df.iterrows():
            cursor.execute("""
                INSERT INTO gold_forecasts (forecast_date, predicted_price, lower_bound, upper_bound, model_name)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                row['date'],
                float(row['predicted_price']),
                float(row['lower_bound']),
                float(row['upper_bound']),
                best_model_name
            ))

        cursor.execute("""
            INSERT INTO model_performance
                (model_name, rmse, mae, mape, training_date)
            VALUES (%s, %s, %s, %s, %s)
        """, (
            best_model_name,
            float(model_metrics['rmse']),
            float(model_metrics['mae']),
            float(model_metrics['mape']),
            datetime.now().date(),
        ))

        conn.commit()
        cursor.close()
        conn.close()

        logger.info(f"✅ Saved {len(forecast_df)} forecasts and model metrics ({best_model_name}) to database")
        return f"Saved {len(forecast_df)} forecasts"

    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 8 — SUMMARY
# =============================================================================

def generate_summary_task(**context):
    """Generate comprehensive summary"""
    logger.info("Generating final summary report...")

    try:
        cleaning_report  = context['ti'].xcom_pull(key='cleaning_report',  task_ids='clean_data')
        eda_stats        = context['ti'].xcom_pull(key='eda_stats',         task_ids='perform_eda')
        model_metrics    = context['ti'].xcom_pull(key='model_metrics',     task_ids='run_forecasting')
        best_model_name  = context['ti'].xcom_pull(key='best_model_name',   task_ids='run_forecasting') or 'ARIMA'
        db_inserted      = context['ti'].xcom_pull(key='db_inserted',       task_ids='insert_to_postgres')
        db_updated       = context['ti'].xcom_pull(key='db_updated',        task_ids='insert_to_postgres')

        logger.info("=" * 70)
        logger.info("COMPLETE PIPELINE SUMMARY")
        logger.info("=" * 70)
        logger.info(f"1. DATA CLEANING:")
        logger.info(f"   - Duplicates Removed:     {cleaning_report['duplicates_removed']}")
        logger.info(f"   - Missing Values Handled: {cleaning_report['missing_values_handled']}")
        logger.info(f"   - Outliers Detected:      {cleaning_report['outliers_detected']}")
        logger.info(f"2. DATABASE STORAGE:")
        logger.info(f"   - Records Inserted: {db_inserted}")
        logger.info(f"   - Records Updated:  {db_updated}")
        logger.info(f"3. EDA RESULTS:")
        logger.info(f"   - Total Records: {eda_stats['summary']['count']}")
        logger.info(f"   - Mean Price:    ${eda_stats['summary']['price_statistics']['mean']:,.2f}")
        logger.info(f"   - Total Return:  {eda_stats['trends']['total_return_pct']:.2f}%")
        logger.info(f"4. FORECASTING MODEL ({best_model_name}):")
        logger.info(f"   - RMSE:          ${model_metrics['rmse']:.2f}")
        logger.info(f"   - MAE:           ${model_metrics['mae']:.2f}")
        logger.info(f"   - MAPE:          {model_metrics['mape']:.2f}%")
        logger.info(f"   - Training Size: {model_metrics['train_size']} records")
        logger.info(f"   - Test Size:     {model_metrics['test_size']} records")
        logger.info("=" * 70)

        return {
            'cleaning':    cleaning_report,
            'database':    {'inserted': db_inserted, 'updated': db_updated},
            'eda':         eda_stats,
            'forecasting': model_metrics,
        }
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        raise

# =============================================================================
# TASK 9 — CLEANUP
# =============================================================================

def cleanup_old_files_task(**context):
    """Cleanup CSV files older than 30 days"""
    logger.info("Starting cleanup...")

    data_dir    = '/usr/local/airflow/include/data'
    cutoff_date = datetime.now() - timedelta(days=30)
    deleted_count = 0

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            filepath  = os.path.join(data_dir, filename)
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                os.remove(filepath)
                deleted_count += 1

    logger.info(f"✅ Cleaned up {deleted_count} old files")
    return f"Cleaned up {deleted_count} files"

# =============================================================================
# OPERATOR DEFINITIONS
# =============================================================================

scrape_task = PythonOperator(
    task_id='scrape_gold_prices',
    python_callable=scrape_task_wrapper,
    dag=dag,
)

clean_task = PythonOperator(
    task_id='clean_data',
    python_callable=clean_data_task,
    dag=dag,
)

postgres_task = PythonOperator(
    task_id='insert_to_postgres',
    python_callable=insert_to_postgres_task,
    dag=dag,
)

eda_task_op = PythonOperator(
    task_id='perform_eda',
    python_callable=eda_task,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
    dag=dag,
)

forecasting_task = PythonOperator(
    task_id='run_forecasting',
    python_callable=run_forecasting,
    execution_timeout=timedelta(minutes=90),  # grid search can be slow
    retries=3,
    retry_delay=timedelta(minutes=5),
    dag=dag,
)

save_forecast_task = PythonOperator(
    task_id='save_forecast_to_db',
    python_callable=save_forecast_to_db_task,
    dag=dag,
)

summary_task = PythonOperator(
    task_id='generate_summary',
    python_callable=generate_summary_task,
    dag=dag,
)

cleanup_task = PythonOperator(
    task_id='cleanup_old_files',
    python_callable=cleanup_old_files_task,
    dag=dag,
)

# =============================================================================
# TASK DEPENDENCIES
# =============================================================================

scrape_task >> clean_task >> postgres_task
clean_task >> [eda_task_op, validate_task, forecasting_task]
forecasting_task >> save_forecast_task
[postgres_task, eda_task_op, validate_task, save_forecast_task] >> summary_task >> cleanup_task