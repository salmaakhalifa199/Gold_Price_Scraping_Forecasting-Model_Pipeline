import psycopg2
import pandas as pd


class DatabaseConnector:
    def __init__(self, connection_string=None):
        if connection_string is None:
            try:
                import streamlit as st
                connection_string = (
                f"postgresql://{st.secrets['database']['user']}:"
                f"{st.secrets['database']['password']}@"
                f"{st.secrets['database']['host']}:"
                f"{st.secrets['database']['port']}/"
                f"{st.secrets['database']['database']}"
                f"?sslmode={st.secrets['database']['sslmode']}"
            )
            except Exception:
                # Fallback (local dev)
                connection_string = (
                    "postgresql://postgres:postgres@localhost:5432/gold_prices_db"
                )

        self.conn = psycopg2.connect(connection_string)

    # =========================
    # Gold Prices
    # =========================
    def get_all_gold_prices(self):
        """
        Returns historical gold prices
        Columns expected by dashboard:
        date, open, high, low, close, volume
        """
        query = """
            SELECT
                price_date        AS date,
                open_price        AS open,
                high_price        AS high,
                low_price         AS low,
                close_price       AS close,
                volume,
                change_percent
            FROM gold_prices
            ORDER BY price_date
        """
        df = pd.read_sql(query, self.conn)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # =========================
    # Forecasts
    # =========================
    def get_forecasts(self, model_name=None):
        """
        Returns forecast data
        """
        base_query = """
            SELECT
                forecast_date,
                predicted_price,
                lower_bound,
                upper_bound,
                model_name
            FROM gold_forecasts
        """

        if model_name:
            query = base_query + " WHERE model_name = %s ORDER BY forecast_date"
            df = pd.read_sql(query, self.conn, params=[model_name])
        else:
            query = base_query + " ORDER BY forecast_date"
            df = pd.read_sql(query, self.conn)

        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        return df

    # =========================
    # Model Performance
    # =========================
    def get_model_performance(self):
        """
        Returns model evaluation metrics
        """
        query = """
            SELECT
                model_name,
                rmse,
                mae,
                mape,
                training_date,
                created_at
            FROM model_performance
            ORDER BY created_at DESC
        """
        df = pd.read_sql(query, self.conn)
        df["training_date"] = pd.to_datetime(df["training_date"])
        return df

    # =========================
    # Summary Stats
    # =========================
    def get_summary_stats(self):
        """
        Returns summary statistics for KPIs
        """
        query = """
            SELECT
                COUNT(*)           AS total_records,
                MIN(price_date)    AS min_date,
                MAX(price_date)    AS max_date,
                AVG(close_price)   AS avg_close_price
            FROM gold_prices
        """
        df = pd.read_sql(query, self.conn)

        return {
            "total_records": int(df.iloc[0]["total_records"]),
            "min_date": df.iloc[0]["min_date"],
            "max_date": df.iloc[0]["max_date"],
            "avg_close_price": float(df.iloc[0]["avg_close_price"])
        }