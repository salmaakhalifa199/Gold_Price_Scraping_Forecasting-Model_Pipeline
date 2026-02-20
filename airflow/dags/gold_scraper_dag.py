"""
Gold Price Scraper DAG
Schedules daily gold price scraping with error handling and monitoring
"""
from multiprocessing import context

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
import pandas as pd
import yfinance as yf
import logging

# Import from the utils package and the scrape function
import sys
sys.path.insert(0, '/usr/local/airflow/include/utils')
from scrape_gold import scrape_gold_prices
from data_cleaner import clean_gold_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    description='Scrape gold prices daily and save to CSV',
    schedule='@daily',  # Run daily at 9 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['scraping', 'gold', 'finance', 'data-cleaning'],
)

def scrape_task_wrapper(**context):
    """
    Wrapper function that calls the existing scrape_gold_prices function
    """
    logger.info("Starting gold price scraping task...")
    
    try:
        # Use the existing scrape function
        output_dir = '/usr/local/airflow/include/data'
        filepath = scrape_gold_prices(days=365, output_dir=output_dir)
        
        # Read the CSV to get data for database insertion
        df = pd.read_csv(filepath)
        
        # Push data to XCom for downstream tasks
        context['ti'].xcom_push(key='filepath', value=filepath)
        context['ti'].xcom_push(key='record_count', value=len(df))
        context['ti'].xcom_push(key='dataframe', value=df.to_json())
        
        logger.info(f"✅ Scraping completed: {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Error in scraping task: {str(e)}")
        raise

def clean_data_task(**context):
    """Clean and preprocess the scraped data"""
    logger.info("Starting data cleaning task...")
    
    try:
        # Get dataframe from previous task
        df_json = context['ti'].xcom_pull(key='dataframe', task_ids='scrape_gold_prices')
        df = pd.read_json(df_json)
        
        # Clean the data
        cleaned_df, cleaning_report = clean_gold_data(df)
        
        # Save cleaned data to CSV
        output_dir = '/usr/local/airflow/include/data'
        cleaned_filename = f"gold_prices_cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cleaned_filepath = os.path.join(output_dir, cleaned_filename)
        cleaned_df.to_csv(cleaned_filepath, index=False)
        
        logger.info(f"✅ Cleaned data saved to: {cleaned_filepath}")
        
        # Push cleaned data to XCom
        context['ti'].xcom_push(key='cleaned_filepath', value=cleaned_filepath)
        context['ti'].xcom_push(key='cleaned_dataframe', value=cleaned_df.to_json())
        context['ti'].xcom_push(key='cleaning_report', value=cleaning_report)
        
        return cleaned_filepath
        
    except Exception as e:
        logger.error(f"❌ Error in cleaning task: {str(e)}")
        raise e

def validate_data_task(**context):
    """Validate the cleaned data"""
    logger.info("Starting data validation...")
    
    try:
        # Get cleaned filepath
        filepath = context['ti'].xcom_pull(key='cleaned_filepath', task_ids='clean_data')
        
        if not filepath or not os.path.exists(filepath):
            raise ValueError(f"Cleaned file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        # Validation checks
        if len(df) == 0:
            raise ValueError("Cleaned CSV file is empty")
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Data quality checks
        null_counts = df[['Open', 'High', 'Low', 'Close']].isnull().sum()
        if null_counts.sum() > 0:
            raise ValueError(f"Found null values after cleaning: {null_counts.to_dict()}")
        
        if (df['High'] < df['Low']).any():
            raise ValueError("Data quality issue: High price is less than Low price")
        
        if (df['Close'] < 0).any() or (df['Open'] < 0).any():
            raise ValueError("Data quality issue: Negative prices found")
        
        logger.info(f"✅ Validation passed: {len(df)} clean records")
        logger.info(f"Date range: {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
        
        return f"Validation successful for {filepath}"
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {str(e)}")
        raise

def generate_summary_task(**context):
    """Generate summary report"""
    logger.info("Generating summary report...")
    
    try:
        # Get cleaned data
        filepath = context['ti'].xcom_pull(key='cleaned_filepath', task_ids='clean_data')
        cleaning_report = context['ti'].xcom_pull(key='cleaning_report', task_ids='clean_data')
        
        df = pd.read_csv(filepath)
        
        summary = {
            'total_records': len(df),
            'date_range': f"{df['Date'].iloc[0]} to {df['Date'].iloc[-1]}",
            'lowest_price': float(df['Low'].min()),
            'highest_price': float(df['High'].max()),
            'current_price': float(df['Close'].iloc[-1]),
            'avg_price': float(df['Close'].mean()),
            'price_change': float(df['Close'].iloc[-1] - df['Close'].iloc[0]),
            'price_change_pct': float(((df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0]) * 100),
            'cleaning_stats': cleaning_report
        }
        
        logger.info("=" * 50)
        logger.info("GOLD PRICE SUMMARY REPORT")
        logger.info("=" * 50)
        logger.info(f"Total Records: {summary['total_records']}")
        logger.info(f"Date Range: {summary['date_range']}")
        logger.info(f"Lowest Price: ${summary['lowest_price']:,.2f}")
        logger.info(f"Highest Price: ${summary['highest_price']:,.2f}")
        logger.info(f"Current Price: ${summary['current_price']:,.2f}")
        logger.info(f"Average Price: ${summary['avg_price']:,.2f}")
        logger.info(f"Price Change: ${summary['price_change']:,.2f} ({summary['price_change_pct']:.2f}%)")
        logger.info("=" * 50)
        logger.info("CLEANING STATISTICS")
        logger.info("=" * 50)
        logger.info(f"Duplicates Removed: {cleaning_report['duplicates_removed']}")
        logger.info(f"Missing Values Handled: {cleaning_report['missing_values_handled']}")
        logger.info(f"Outliers Detected: {cleaning_report['outliers_detected']}")
        logger.info("=" * 50)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Error generating summary: {str(e)}")
        raise

def cleanup_old_files_task(**context):
    """
    Cleanup files older than 30 days
    """
    logger.info("Starting cleanup of old files...")
    
    data_dir = '/usr/local/airflow/include/data'
    
    if not os.path.exists(data_dir):
        logger.info("No data directory to clean")
        return "No data directory to clean"
    
    cutoff_date = datetime.now() - timedelta(days=30)
    deleted_count = 0
    
    for filename in os.listdir(data_dir):
        if not filename.startswith('gold_prices_'):
            continue
            
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            if file_time < cutoff_date:
                os.remove(filepath)
                deleted_count += 1
                logger.info(f"Deleted old file: {filename}")
    
    logger.info(f"✅ Cleaned up {deleted_count} old files")
    return f"Cleaned up {deleted_count} old files"

# Define tasks - REMOVED provide_context=True
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

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data_task,
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

# Set task dependencies
scrape_task >> clean_task >> validate_task >> summary_task >> cleanup_task