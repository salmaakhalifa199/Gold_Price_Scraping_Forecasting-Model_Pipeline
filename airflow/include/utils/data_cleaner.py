"""
Data Cleaning and Preprocessing for Gold Price Data
"""
import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldDataCleaner:
    """Handle data cleaning and preprocessing for gold price data"""
    
    def __init__(self, df):
        """
        Initialize with a pandas DataFrame
        
        Args:
            df (pd.DataFrame): Raw gold price data
        """
        self.df = df.copy()
        self.cleaning_report = {
            'original_rows': len(df),
            'duplicates_removed': 0,
            'missing_values_handled': 0,
            'outliers_detected': 0,
            'final_rows': 0
        }
    
    def clean(self):
        """
        Main cleaning pipeline - executes all cleaning steps
        
        Returns:
            pd.DataFrame: Cleaned data
        """
        logger.info("Starting data cleaning pipeline...")
        
        self._convert_data_types()
        self._handle_duplicates()
        self._handle_missing_values()
        self._validate_price_relationships()
        self._detect_outliers()
        self._sort_data()
        
        self.cleaning_report['final_rows'] = len(self.df)
        
        logger.info("=" * 50)
        logger.info("DATA CLEANING REPORT")
        logger.info("=" * 50)
        logger.info(f"Original rows: {self.cleaning_report['original_rows']}")
        logger.info(f"Duplicates removed: {self.cleaning_report['duplicates_removed']}")
        logger.info(f"Missing values handled: {self.cleaning_report['missing_values_handled']}")
        logger.info(f"Outliers detected: {self.cleaning_report['outliers_detected']}")
        logger.info(f"Final rows: {self.cleaning_report['final_rows']}")
        logger.info("=" * 50)
        
        return self.df
    
    def _convert_data_types(self):
        """Convert columns to appropriate data types"""
        logger.info("Converting data types...")
        
        # Convert date column (handle both 'Date' and 'date')
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        
        # Ensure price columns are numeric
        price_columns = ['Open', 'High', 'Low', 'Close'] if 'Open' in self.df.columns else ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Ensure volume is integer
        vol_col = 'Volume' if 'Volume' in self.df.columns else 'volume'
        if vol_col in self.df.columns:
            self.df[vol_col] = pd.to_numeric(self.df[vol_col], errors='coerce').fillna(0).astype(np.int64)
        
        logger.info("✅ Data types converted successfully")
    
    def _handle_duplicates(self):
        """Remove duplicate records based on date"""
        logger.info("Checking for duplicates...")
        
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        initial_count = len(self.df)
        
        # Remove duplicates, keeping the last occurrence (most recent data)
        self.df = self.df.drop_duplicates(subset=[date_col], keep='last')
        
        duplicates_removed = initial_count - len(self.df)
        self.cleaning_report['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.warning(f"⚠️  Removed {duplicates_removed} duplicate records")
        else:
            logger.info("✅ No duplicates found")
    
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")
        
        price_columns = ['Open', 'High', 'Low', 'Close'] if 'Open' in self.df.columns else ['open', 'high', 'low', 'close']
        
        # Count missing values
        missing_counts = self.df[price_columns].isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            logger.warning(f"⚠️  Found {total_missing} missing values")
            
            # Strategy 1: Forward fill (use previous day's value)
            self.df[price_columns] = self.df[price_columns].fillna(method='ffill')
            
            # Strategy 2: Backward fill for any remaining NaNs at the start
            self.df[price_columns] = self.df[price_columns].fillna(method='bfill')
            
            # Strategy 3: Drop rows if still NaN (shouldn't happen, but safety)
            rows_before = len(self.df)
            self.df = self.df.dropna(subset=price_columns)
            rows_after = len(self.df)
            
            self.cleaning_report['missing_values_handled'] = total_missing
            
            if rows_before != rows_after:
                logger.warning(f"⚠️  Dropped {rows_before - rows_after} rows with unrecoverable missing values")
        else:
            logger.info("✅ No missing values found")
    
    def _validate_price_relationships(self):
        """Validate that High >= Low and other price relationships"""
        logger.info("Validating price relationships...")
        
        price_cols = ['Open', 'High', 'Low', 'Close'] if 'Open' in self.df.columns else ['open', 'high', 'low', 'close']
        open_col, high_col, low_col, close_col = price_cols
        
        # Check High >= Low
        invalid_high_low = self.df[self.df[high_col] < self.df[low_col]]
        
        if len(invalid_high_low) > 0:
            logger.warning(f"⚠️  Found {len(invalid_high_low)} rows where High < Low. Fixing...")
            
            # Swap High and Low values
            self.df.loc[self.df[high_col] < self.df[low_col], [high_col, low_col]] = \
                self.df.loc[self.df[high_col] < self.df[low_col], [low_col, high_col]].values
        
        # Check that Open, Close are within High-Low range
        invalid_open = self.df[(self.df[open_col] > self.df[high_col]) | (self.df[open_col] < self.df[low_col])]
        invalid_close = self.df[(self.df[close_col] > self.df[high_col]) | (self.df[close_col] < self.df[low_col])]
        
        if len(invalid_open) > 0:
            logger.warning(f"⚠️  Found {len(invalid_open)} rows where Open is outside High-Low range")
        
        if len(invalid_close) > 0:
            logger.warning(f"⚠️  Found {len(invalid_close)} rows where Close is outside High-Low range")
        
        logger.info("✅ Price relationships validated")
    
    def _detect_outliers(self):
        """Detect outliers using IQR method"""
        logger.info("Detecting outliers...")
        
        close_col = 'Close' if 'Close' in self.df.columns else 'close'
        
        # Calculate IQR for Close price
        Q1 = self.df[close_col].quantile(0.25)
        Q3 = self.df[close_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        
        # Identify outliers (but don't remove them - just flag)
        outliers = self.df[(self.df[close_col] < lower_bound) | (self.df[close_col] > upper_bound)]
        
        self.cleaning_report['outliers_detected'] = len(outliers)
        
        if len(outliers) > 0:
            logger.warning(f"⚠️  Detected {len(outliers)} potential outliers (not removed)")
            logger.info(f"   Outlier bounds: ${lower_bound:.2f} - ${upper_bound:.2f}")
        else:
            logger.info("✅ No outliers detected")
    
    def _sort_data(self):
        """Sort data by date in ascending order"""
        logger.info("Sorting data by date...")
        
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        self.df = self.df.sort_values(by=date_col).reset_index(drop=True)
        
        logger.info("✅ Data sorted successfully")
    
    def get_cleaning_report(self):
        """Return the cleaning report"""
        return self.cleaning_report


def clean_gold_data(df):
    """
    Convenience function to clean gold price data
    
    Args:
        df (pd.DataFrame): Raw gold price data
    
    Returns:
        tuple: (cleaned_df, cleaning_report)
    """
    cleaner = GoldDataCleaner(df)
    cleaned_df = cleaner.clean()
    report = cleaner.get_cleaning_report()
    
    return cleaned_df, report


# Test the cleaner
if __name__ == "__main__":
    # Create sample data with issues
    test_data = pd.DataFrame({
        'Date': ['2024-01-01', '2024-01-02', '2024-01-02', '2024-01-03', '2024-01-04'],
        'Open': [2000, 2010, 2010, None, 2030],
        'High': [2020, 2025, 2025, 2040, 2035],
        'Low': [1990, 1995, 1995, 2015, 2025],
        'Close': [2005, 2015, 2015, 2035, 2028],
        'Volume': [100000, 110000, 110000, 120000, 115000]
    })
    
    print("Original Data:")
    print(test_data)
    print("\n")
    
    cleaned_df, report = clean_gold_data(test_data)
    
    print("\nCleaned Data:")
    print(cleaned_df)