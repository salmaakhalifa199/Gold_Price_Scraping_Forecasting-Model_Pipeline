import psycopg2
import pandas as pd
import streamlit as st


def _get_connection_string():
    try:
        return (
            f"postgresql://{st.secrets['database']['user']}:"
            f"{st.secrets['database']['password']}@"
            f"{st.secrets['database']['host']}:"
            f"{st.secrets['database']['port']}/"
            f"{st.secrets['database']['database']}"
            f"?sslmode={st.secrets['database']['sslmode']}"
        )
    except Exception:
        return "postgresql://postgres:postgres@localhost:5432/gold_prices_db"


def _new_conn():
    """Always return a fresh, autocommit connection."""
    conn = psycopg2.connect(_get_connection_string())
    conn.autocommit = True
    return conn


class DatabaseConnector:
    def __init__(self):
        self._conn_string = _get_connection_string()

    def _query(self, sql, params=None):
        """Execute a query on a fresh connection and return a DataFrame."""
        conn = _new_conn()
        try:
            if params:
                return pd.read_sql(sql, conn, params=params)
            return pd.read_sql(sql, conn)
        finally:
            conn.close()

    # =========================
    # Gold Prices
    # =========================
    def get_all_gold_prices(self):
        df = self._query("""
            SELECT
                price_date   AS date,
                open_price   AS open,
                high_price   AS high,
                low_price    AS low,
                close_price  AS close,
                volume,
                change_percent
            FROM gold_prices
            ORDER BY price_date
        """)
        df["date"] = pd.to_datetime(df["date"])
        return df

    # =========================
    # Forecasts
    # =========================
    def get_forecasts(self, model_name=None):
        if model_name:
            df = self._query("""
                SELECT forecast_date, predicted_price, lower_bound, upper_bound, model_name
                FROM gold_forecasts
                WHERE model_name = %s
                ORDER BY forecast_date
            """, params=[model_name])
        else:
            df = self._query("""
                SELECT forecast_date, predicted_price, lower_bound, upper_bound, model_name
                FROM gold_forecasts
                ORDER BY forecast_date
            """)
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        return df

    # =========================
    # Model Performance
    # =========================
    def get_model_performance(self):
        df = self._query("""
            SELECT model_name, rmse, mae, mape, training_date, created_at
            FROM model_performance
            ORDER BY created_at DESC
        """)
        df["training_date"] = pd.to_datetime(df["training_date"])
        return df

    # =========================
    # Summary Stats
    # =========================
    def get_summary_stats(self):
        df = self._query("""
            SELECT
                COUNT(*)        AS total_records,
                MIN(price_date) AS min_date,
                MAX(price_date) AS max_date,
                AVG(close_price) AS avg_close_price
            FROM gold_prices
        """)
        return {
            "total_records":    int(df.iloc[0]["total_records"]),
            "min_date":         df.iloc[0]["min_date"],
            "max_date":         df.iloc[0]["max_date"],
            "avg_close_price":  float(df.iloc[0]["avg_close_price"]),
        }