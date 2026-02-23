"""
Exploratory Data Analysis for Gold Price Data
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Airflow
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class GoldPriceEDA:
    """Perform comprehensive EDA on gold price data"""
    
    def __init__(self, df, output_dir='./airflow/include/data/eda'):
        """
        Initialize EDA analyzer
        
        Args:
            df (pd.DataFrame): Gold price data
            output_dir (str): Directory to save plots
        """
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Ensure date column is datetime
        date_col = 'Date' if 'Date' in self.df.columns else 'date'
        self.df[date_col] = pd.to_datetime(self.df[date_col])
        self.df = self.df.sort_values(date_col)
        
        # Standardize column names for easier access
        self.df.columns = self.df.columns.str.lower()
        
        self.stats_report = {}
        self.plots = []
    
    def run_full_analysis(self):
        """Run complete EDA pipeline"""
        logger.info(f"Rows after cleaning: {len(self.df)}")
        
        self._compute_summary_statistics()
        self._analyze_trends()
        self._detect_seasonality()
        self._analyze_volatility()
        self._detect_anomalies()
        
        # Generate all plots
        self._plot_price_trends()
        self._plot_distribution()
        self._plot_moving_averages()
        self._plot_daily_returns()
        self._plot_seasonal_patterns()
        self._plot_correlation_heatmap()
        self._plot_candlestick_style()
        
        logger.info(f"✅ EDA complete! Generated {len(self.plots)} visualizations")
        
        return self.stats_report, self.plots
    
    def _compute_summary_statistics(self):
        """Compute comprehensive summary statistics"""
        logger.info("Computing summary statistics...")
        
        close_prices = self.df['close']
        
        stats = {
            'count': len(self.df),
            'date_range': {
                'start': str(self.df['date'].min().date()),
                'end': str(self.df['date'].max().date()),
                'days': (self.df['date'].max() - self.df['date'].min()).days
            },
            'price_statistics': {
                'mean': float(close_prices.mean()),
                'median': float(close_prices.median()),
                'std': float(close_prices.std()),
                'min': float(close_prices.min()),
                'max': float(close_prices.max()),
                'range': float(close_prices.max() - close_prices.min()),
                'q1': float(close_prices.quantile(0.25)),
                'q3': float(close_prices.quantile(0.75)),
                'iqr': float(close_prices.quantile(0.75) - close_prices.quantile(0.25))
            },
            'daily_changes': {
                'mean_change': float(self.df['close'].diff().mean()),
                'mean_pct_change': float(self.df['close'].pct_change().mean() * 100),
                'std_pct_change': float(self.df['close'].pct_change().std() * 100)
            }
        }
        
        self.stats_report['summary'] = stats
        
        logger.info("=" * 60)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 60)
        logger.info(f"Total Records: {stats['count']}")
        logger.info(f"Date Range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        logger.info(f"Mean Price: ${stats['price_statistics']['mean']:,.2f}")
        logger.info(f"Median Price: ${stats['price_statistics']['median']:,.2f}")
        logger.info(f"Std Dev: ${stats['price_statistics']['std']:,.2f}")
        logger.info(f"Price Range: ${stats['price_statistics']['min']:,.2f} - ${stats['price_statistics']['max']:,.2f}")
        logger.info("=" * 60)
    
    def _analyze_trends(self):
        """Analyze overall trends"""
        logger.info(f"Rows after return alignment: {len(self.df)}")
        
        # Ensure proper ordering
        self.df = self.df.sort_values("date").reset_index(drop=True)

        # Compute returns
        returns = self.df['close'].pct_change()

        # Drop NaNs and ALIGN
        returns = returns.dropna()
        self.df = self.df.loc[returns.index].copy()

        # Assign back
        self.df['daily_return'] = returns.values * 100
        self.df['cumulative_return'] = (1 + returns.values).cumprod() - 1
        
        # Calculate moving averages
        self.df['ma_7'] = self.df['close'].rolling(7).mean()
        self.df['ma_30'] = self.df['close'].rolling(30).mean()
        self.df['ma_90'] = self.df['close'].rolling(90).mean()

        # Drop rows where rolling windows are invalid
        self.df = self.df.dropna().reset_index(drop=True)
        
        # Trend statistics
        total_return = float((self.df['close'].iloc[-1] / self.df['close'].iloc[0] - 1) * 100)
        
        # Count up and down days
        up_days = (self.df['daily_return'] > 0).sum()
        down_days = (self.df['daily_return'] < 0).sum()
        
        trends = {
            'total_return_pct': total_return,
            'up_days': int(up_days),
            'down_days': int(down_days),
            'up_days_pct': float(up_days / len(self.df) * 100),
            'best_day': {
                'date': str(self.df.loc[self.df['daily_return'].idxmax(), 'date'].date()),
                'return': float(self.df['daily_return'].max())
            },
            'worst_day': {
                'date': str(self.df.loc[self.df['daily_return'].idxmin(), 'date'].date()),
                'return': float(self.df['daily_return'].min())
            }
        }
        
        self.stats_report['trends'] = trends
        
        logger.info("=" * 60)
        logger.info("TREND ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total Return: {total_return:.2f}%")
        logger.info(f"Up Days: {up_days} ({trends['up_days_pct']:.1f}%)")
        logger.info(f"Down Days: {down_days}")
        logger.info(f"Best Day: {trends['best_day']['date']} (+{trends['best_day']['return']:.2f}%)")
        logger.info(f"Worst Day: {trends['worst_day']['date']} ({trends['worst_day']['return']:.2f}%)")
        logger.info("=" * 60)
    
    def _detect_seasonality(self):
        """Detect seasonal patterns"""
        logger.info("Analyzing seasonal patterns...")
        
        self.df = self.df.dropna(subset=['daily_return'])
        # Extract temporal features
        self.df['month'] = self.df['date'].dt.month
        self.df['quarter'] = self.df['date'].dt.quarter
        self.df['day_of_week'] = self.df['date'].dt.dayofweek
        self.df['day_name'] = self.df['date'].dt.day_name()
        
        # Monthly statistics
        monthly_avg = self.df.groupby('month')['close'].mean()
        monthly_returns = self.df.groupby('month')['daily_return'].mean()
        
        # Day of week statistics
        dow_avg = self.df.groupby('day_name')['daily_return'].mean()
        
        seasonality = {
            'best_month': {
                'month': int(monthly_returns.idxmax()),
                'avg_return': float(monthly_returns.max())
            },
            'worst_month': {
                'month': int(monthly_returns.idxmin()),
                'avg_return': float(monthly_returns.min())
            },
            'best_day': str(dow_avg.idxmax()),
            'worst_day': str(dow_avg.idxmin())
        }
        
        self.stats_report['seasonality'] = seasonality
        
        logger.info("=" * 60)
        logger.info("SEASONALITY ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Best Month: {seasonality['best_month']['month']} (Avg return: {seasonality['best_month']['avg_return']:.3f}%)")
        logger.info(f"Worst Month: {seasonality['worst_month']['month']} (Avg return: {seasonality['worst_month']['avg_return']:.3f}%)")
        logger.info(f"Best Day of Week: {seasonality['best_day']}")
        logger.info(f"Worst Day of Week: {seasonality['worst_day']}")
        logger.info("=" * 60)
    
    def _analyze_volatility(self):
        """Analyze price volatility"""
        logger.info("Analyzing volatility...")
        
        # Calculate volatility (rolling standard deviation of returns)
        self.df['volatility_30d'] = self.df['daily_return'].rolling(window=30).std()
        
        volatility = {
            'current_volatility': float(self.df['volatility_30d'].iloc[-1]),
            'avg_volatility': float(self.df['volatility_30d'].mean()),
            'max_volatility': float(self.df['volatility_30d'].max()),
            'min_volatility': float(self.df['volatility_30d'].min())
        }
        
        self.stats_report['volatility'] = volatility
        
        logger.info("=" * 60)
        logger.info("VOLATILITY ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Current 30-day Volatility: {volatility['current_volatility']:.2f}%")
        logger.info(f"Average Volatility: {volatility['avg_volatility']:.2f}%")
        logger.info(f"Max Volatility: {volatility['max_volatility']:.2f}%")
        logger.info("=" * 60)
    
    def _detect_anomalies(self):
        """Detect anomalies using statistical methods"""
        logger.info("Detecting anomalies...")

        # Work only on valid returns
        returns_df = self.df[['date', 'daily_return']].dropna().copy()

        # Z-score
        returns_df['z_score'] = np.abs(stats.zscore(returns_df['daily_return']))

        # Anomalies threshold
        anomalies = returns_df[returns_df['z_score'] > 3]

        anomaly_info = {
            'count': int(len(anomalies)),
            'dates': [str(d.date()) for d in anomalies['date'].head(5)]
        }
        self.stats_report['anomalies'] = anomaly_info

        logger.info("=" * 60)
        logger.info("ANOMALY DETECTION")
        logger.info("=" * 60)
        logger.info(f"Anomalies Detected: {anomaly_info['count']}")
        if anomaly_info['dates']:
            logger.info(f"Sample Dates: {', '.join(anomaly_info['dates'][:3])}")
        logger.info("=" * 60)
    
    def _plot_price_trends(self):
        """Plot price trends over time"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(self.df['date'], self.df['close'], label='Close Price', linewidth=2)
        ax.fill_between(self.df['date'], self.df['low'], self.df['high'], alpha=0.3, label='High-Low Range')
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Gold Price Trends Over Time', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'price_trends.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: price_trends.png")
    
    def _plot_distribution(self):
        """Plot price distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(self.df['close'], bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(self.df['close'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
        axes[0].axvline(self.df['close'].median(), color='green', linestyle='--', linewidth=2, label='Median')
        axes[0].set_xlabel('Price (USD)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Price Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot([self.df['close']], labels=['Close Price'])
        axes[1].set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        axes[1].set_title('Price Box Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'price_distribution.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: price_distribution.png")
    
    def _plot_moving_averages(self):
        """Plot moving averages"""
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(self.df['date'], self.df['close'], label='Close Price', linewidth=1.5, alpha=0.7)
        ax.plot(self.df['date'], self.df['ma_7'], label='7-Day MA', linewidth=2)
        ax.plot(self.df['date'], self.df['ma_30'], label='30-Day MA', linewidth=2)
        ax.plot(self.df['date'], self.df['ma_90'], label='90-Day MA', linewidth=2)
        
        ax.set_xlabel('Date', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Gold Price with Moving Averages', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'moving_averages.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: moving_averages.png")
    
    def _plot_daily_returns(self):
        """Plot daily returns"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        plot_df = self.df.dropna(subset=['daily_return'])
        # Returns over time
        axes[0].plot(plot_df['date'], plot_df['daily_return'], linewidth=1, alpha=0.7)
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Date', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Daily Return (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Daily Returns Over Time', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Returns distribution
        axes[1].hist(self.df['daily_return'].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Daily Return (%)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[1].set_title('Daily Returns Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'daily_returns.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: daily_returns.png")
    
    def _plot_seasonal_patterns(self):
        """Plot seasonal patterns"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Monthly average returns
        monthly_returns = self.df.groupby('month')['daily_return'].mean()
        axes[0].bar(monthly_returns.index, monthly_returns.values, color='skyblue', edgecolor='black')
        axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[0].set_xlabel('Month', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Avg Daily Return (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Average Returns by Month', fontsize=14, fontweight='bold')
        axes[0].set_xticks(range(1, 13))
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Day of week average returns
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        dow_returns = self.df.groupby('day_name')['daily_return'].mean().reindex(day_order)
        axes[1].bar(range(len(dow_returns)), dow_returns.values, color='lightcoral', edgecolor='black')
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Day of Week', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Avg Daily Return (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Average Returns by Day of Week', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(len(day_order)))
        axes[1].set_xticklabels(day_order, rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'seasonal_patterns.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: seasonal_patterns.png")
    
    def _plot_correlation_heatmap(self):
        """Plot correlation heatmap"""
        # Select numeric columns
        corr_cols = ['open', 'high', 'low', 'close', 'volume']
        corr_matrix = self.df[corr_cols].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        ax.set_title('Price Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'correlation_heatmap.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: correlation_heatmap.png")
    
    def _plot_candlestick_style(self):
        """Plot candlestick-style chart"""
        # Sample last 60 days for readability
        recent_data = self.df.tail(60)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Determine up/down days
        colors = ['green' if row['close'] >= row['open'] else 'red' 
                  for _, row in recent_data.iterrows()]
        
        # Plot high-low lines
        for idx, (_, row) in enumerate(recent_data.iterrows()):
            ax.plot([idx, idx], [row['low'], row['high']], color='black', linewidth=1)
            ax.plot([idx, idx], [row['open'], row['close']], color=colors[idx], linewidth=4)
        
        ax.set_xlabel('Days (Last 60)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Price (USD)', fontsize=12, fontweight='bold')
        ax.set_title('Gold Price - Last 60 Days (Candlestick Style)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, 'candlestick_chart.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.plots.append(filepath)
        logger.info(f"✅ Generated: candlestick_chart.png")
    
    def get_stats_report(self):
        """Return the statistics report"""
        return self.stats_report
    
    def get_plot_paths(self):
        """Return list of generated plot paths"""
        return self.plots


def perform_eda(df, output_dir='data/eda'):
    """
    Convenience function to perform EDA
    
    Args:
        df (pd.DataFrame): Gold price data
        output_dir (str): Directory to save plots
    
    Returns:
        tuple: (stats_report, plot_paths)
    """
    analyzer = GoldPriceEDA(df, output_dir)
    stats_report, plots = analyzer.run_full_analysis()
    
    return stats_report, plots


# Test the analyzer
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    prices = 2000 + np.cumsum(np.random.randn(len(dates)) * 10)
    
    test_data = pd.DataFrame({
        'date': dates,
        'open': prices + np.random.randn(len(dates)) * 5,
        'high': prices + np.abs(np.random.randn(len(dates)) * 10),
        'low': prices - np.abs(np.random.randn(len(dates)) * 10),
        'close': prices,
        'volume': np.random.randint(100000, 200000, len(dates))
    })
    
    stats, plots = perform_eda(test_data, 'test_eda')
    print(f"Generated {len(plots)} plots")