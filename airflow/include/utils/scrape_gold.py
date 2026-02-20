"""
Gold Price Scraper for Airflow
"""
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_gold_prices(days=365, output_dir='data'):
    """
    Scrape gold prices using yfinance
    
    Args:
        days (int): Number of days to scrape
        output_dir (str): Directory to save output
    
    Returns:
        str: Path to saved file
    """
    logger.info(f"Starting gold price scraping for {days} days...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Download gold futures data
        logger.info("Fetching data from yfinance...")
        gold = yf.Ticker("GC=F")
        df = gold.history(start=start_date, end=end_date)
        
        if df.empty:
            raise ValueError("No data retrieved from yfinance")
        
        # Process data
        df = df.reset_index()
        df = df.rename(columns={
            'Date': 'Date',
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to CSV
        filename = f"gold_prices_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Successfully scraped {len(df)} records")
        logger.info(f"✅ Data saved to: {filepath}")
        
        # Log statistics
        logger.info(f"Price Range - Low: ${df['Low'].min():,.2f}, High: ${df['High'].max():,.2f}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"❌ Error during scraping: {str(e)}")
        raise

if __name__ == "__main__":
    scrape_gold_prices()