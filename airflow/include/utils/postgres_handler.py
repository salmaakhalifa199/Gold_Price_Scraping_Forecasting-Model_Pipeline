import psycopg2
from psycopg2 import sql
import os

class PostgresHandler:
    def __init__(self):
        self.conn = psycopg2.connect(
            host="localhost",
            port="5432",
            database="gold_prices_db",
            user="postgres",
            password=os.getenv("DB_PASSWORD", "postgres")
        )
        self.cursor = self.conn.cursor()
    
    def insert_gold_prices(self, data_list):
        """
        Insert gold price data
        data_list: [{'date': '2024-01-01', 'open': 2000.0, 'high': 2010.0, 
                     'low': 1990.0, 'close': 2005.0, 'volume': 1000000}, ...]
        """
        insert_query = """
            INSERT INTO gold_prices 
            (price_date, open_price, close_price, high_price, low_price, volume)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (price_date) DO UPDATE SET
                open_price = EXCLUDED.open_price,
                close_price = EXCLUDED.close_price,
                high_price = EXCLUDED.high_price,
                low_price = EXCLUDED.low_price,
                volume = EXCLUDED.volume,
                updated_at = CURRENT_TIMESTAMP;
        """
        
        for data in data_list:
            self.cursor.execute(insert_query, (
                data['date'],
                data.get('open'),
                data['close'],
                data.get('high'),
                data.get('low'),
                data.get('volume')
            ))
        
        self.conn.commit()
        print(f"✓ Inserted {len(data_list)} records")
    
    def get_all_prices(self):
        """Retrieve all gold prices"""
        self.cursor.execute("SELECT * FROM gold_prices ORDER BY price_date DESC")
        return self.cursor.fetchall()
    
    def close(self):
        self.cursor.close()
        self.conn.close()

# Test it
if __name__ == "__main__":
    db = PostgresHandler()
    
    # Test insert
    test_data = [{
        'date': '2024-02-15',
        'open': 2050.00,
        'high': 2060.00,
        'low': 2045.00,
        'close': 2055.00,
        'volume': 500000
    }]
    
    db.insert_gold_prices(test_data)
    
    # Test retrieve
    prices = db.get_all_prices()
    print(f"Total records: {len(prices)}")
    
    db.close()