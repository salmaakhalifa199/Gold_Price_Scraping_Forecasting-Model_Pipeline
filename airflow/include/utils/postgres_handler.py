import psycopg2
from psycopg2 import sql
import pandas as pd
import os

class PostgresHandler:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432"),
            database=os.getenv("DB_NAME", "gold_prices_db"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", "postgres"),
            sslmode=os.getenv("DB_SSLMODE", "disable")
        )
        self.conn.autocommit = True
        self.cursor = self.conn.cursor()

    def insert_gold_prices(self, data_list):
        """Insert gold price data"""
        insert_query = """
            INSERT INTO gold_prices (
                date,
                open,
                high,
                low,
                close,
                volume
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP;
        """

        for data in data_list:
            self.cursor.execute(insert_query, (
                data['date'],
                data.get('open'),
                data.get('high'),
                data.get('low'),
                data['close'],
                data.get('volume')
            ))

        self.conn.commit()
        print(f"✓ Inserted {len(data_list)} records")
    
    def get_all_gold_prices(self):
        """Retrieve all gold prices as DataFrame"""
        query = """
            SELECT 
                date,
                open,
                high,
                low,
                close,
                volume,
                created_at,
                updated_at
            FROM gold_prices
            ORDER BY date ASC
        """
        df = pd.read_sql(query, self.conn)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def get_forecasts(self, model_name=None):
        """Get forecast data"""
        if model_name:
            query = """
                SELECT 
                    forecast_date,
                    predicted_price,
                    lower_bound,
                    upper_bound,
                    model_name,
                    created_at
                FROM gold_forecasts
                WHERE model_name = %s
                ORDER BY forecast_date ASC
            """
            df = pd.read_sql(query, self.conn, params=(model_name,))
        else:
            query = """
                SELECT 
                    forecast_date,
                    predicted_price,
                    lower_bound,
                    upper_bound,
                    model_name,
                    created_at
                FROM gold_forecasts
                ORDER BY forecast_date ASC
            """
            df = pd.read_sql(query, self.conn)
        
        df['forecast_date'] = pd.to_datetime(df['forecast_date'])
        return df
    
    def get_model_performance(self):
        """Get model performance metrics"""
        query = """
            SELECT 
                model_name,
                rmse,
                mae,
                mape,
                training_date,
                test_start_date,
                test_end_date,
                train_size,
                test_size,
                created_at
            FROM model_performance
            ORDER BY created_at DESC
        """
        return pd.read_sql(query, self.conn)
    
    def get_summary_stats(self):
        """Get summary statistics"""
        query = """
            SELECT 
                COUNT(*) as total_records,
                MIN(date) as first_date,
                MAX(date) as last_date,
                MIN(low) as all_time_low,
                MAX(high) as all_time_high,
                AVG(close) as average_price
            FROM gold_prices
        """
        df = pd.read_sql(query, self.conn)
        return df.iloc[0].to_dict()
    
    def get_latest_price(self):
        """Get the most recent gold price"""
        query = """
            SELECT 
                date,
                open,
                high,
                low,
                close,
                volume
            FROM gold_prices
            ORDER BY date DESC
            LIMIT 1
        """
        df = pd.read_sql(query, self.conn)
        if not df.empty:
            return df.iloc[0].to_dict()
        return {}
    
    def close(self):
        """Close database connection"""
        self.cursor.close()
        self.conn.close()