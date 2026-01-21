#!/usr/bin/env python3
"""
Advanced sales analysis utilities: 
- Time series forecasting
- Seasonality decomposition
- Category performance
- Product profitability
- Market basket analysis
"""
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


def sales_trends_analysis(df, freq='M'):
    """
    Analyze sales trends over time. 
    
    Args:
        df: DataFrame with transaction data
        freq: Frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
    
    Returns:
        DataFrame with aggregated sales metrics
    """
    df_time = df.set_index("transaction_date")
    
    trends = df_time.resample(freq).agg({
        "total_amount": "sum",
        "profit": "sum",
        "transaction_id": "count",
        "customer_id": "nunique",
        "quantity": "sum"
    })
    
    trends.columns = ["Revenue", "Profit", "Transactions", "Unique_Customers", "Units_Sold"]
    
    # Calculate moving averages
    trends["Revenue_MA_3"] = trends["Revenue"].rolling(window=3).mean()
    trends["Revenue_MA_6"] = trends["Revenue"].rolling(window=6).mean()
    
    # Calculate growth rates
    trends["Revenue_Growth_Pct"] = trends["Revenue"].pct_change() * 100
    
    return trends. reset_index()


def category_performance_analysis(df):
    """
    Analyze performance by product category. 
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        DataFrame with category metrics
    """
    category_perf = df.groupby("category").agg({
        "total_amount": "sum",
        "profit": "sum",
        "transaction_id": "count",
        "customer_id": "nunique",
        "quantity": "sum",
        "discount":  "sum"
    })
    
    category_perf.columns = [
        "Revenue",
        "Profit",
        "Transactions",
        "Unique_Customers",
        "Units_Sold",
        "Total_Discounts"
    ]
    
    # Calculate additional metrics
    category_perf["Profit_Margin_Pct"] = (
        (category_perf["Profit"] / category_perf["Revenue"]) * 100
    ).round(2)
    
    category_perf["Avg_Transaction_Value"] = (
        category_perf["Revenue"] / category_perf["Transactions"]
    ).round(2)
    
    category_perf["Discount_Rate_Pct"] = (
        (category_perf["Total_Discounts"] / (category_perf["Revenue"] + category_perf["Total_Discounts"])) * 100
    ).round(2)
    
    return category_perf.sort_values("Revenue", ascending=False).reset_index()


def product_profitability_analysis(df):
    """
    Analyze profitability at product level.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        DataFrame with product profitability metrics
    """
    product_perf = df.groupby("product_name").agg({
        "total_amount": "sum",
        "profit": "sum",
        "quantity": "sum",
        "transaction_id": "count",
        "discount": "sum"
    })
    
    product_perf. columns = ["Revenue", "Profit", "Units_Sold", "Transactions", "Total_Discounts"]
    
    # Calculate metrics
    product_perf["Profit_Margin_Pct"] = (
        (product_perf["Profit"] / product_perf["Revenue"]) * 100
    ).round(2)
    
    product_perf["Avg_Unit_Price"] = (
        product_perf["Revenue"] / product_perf["Units_Sold"]
    ).round(2)
    
    product_perf["ROI_Pct"] = (
        (product_perf["Profit"] / (product_perf["Revenue"] - product_perf["Profit"])) * 100
    ).round(2)
    
    return product_perf.sort_values("Profit", ascending=False).reset_index()


def market_basket_analysis(df, min_support=0.01, min_confidence=0.3):
    """
    Perform market basket analysis to find product associations.
    
    Args:
        df: DataFrame with transaction data
        min_support: Minimum support threshold
        min_confidence:  Minimum confidence threshold
    
    Returns:
        DataFrame with association rules
    """
    # Group products by transaction
    baskets = df.groupby("transaction_id")["product_name"].apply(list).values
    
    # Find frequent itemsets (pairs)
    product_counts = Counter()
    pair_counts = Counter()
    total_transactions = len(baskets)
    
    for basket in baskets:
        unique_items = list(set(basket))
        for item in unique_items:
            product_counts[item] += 1
        
        if len(unique_items) >= 2:
            for pair in combinations(sorted(unique_items), 2):
                pair_counts[pair] += 1
    
    # Calculate metrics
    associations = []
    
    for (item1, item2), pair_count in pair_counts.items():
        support = pair_count / total_transactions
        
        if support >= min_support: 
            conf_1_to_2 = pair_count / product_counts[item1]
            conf_2_to_1 = pair_count / product_counts[item2]
            
            if conf_1_to_2 >= min_confidence or conf_2_to_1 >= min_confidence:
                # Calculate lift
                expected = (product_counts[item1] / total_transactions) * (product_counts[item2] / total_transactions)
                lift = support / expected if expected > 0 else 0
                
                associations.append({
                    "Item_1": item1,
                    "Item_2": item2,
                    "Support": support,
                    "Confidence_1_to_2":  conf_1_to_2,
                    "Confidence_2_to_1": conf_2_to_1,
                    "Lift": lift
                })
    
    if not associations:
        # Return empty DataFrame with correct columns if no associations found
        return pd.DataFrame(columns=["Item_1", "Item_2", "Support", "Confidence_1_to_2", "Confidence_2_to_1", "Lift"])
    
    return pd.DataFrame(associations).sort_values("Lift", ascending=False)


def payment_shipping_analysis(df):
    """
    Analyze performance by payment method and shipping type.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        Tuple of (payment_analysis, shipping_analysis) DataFrames
    """
    # Payment method analysis
    payment_perf = df.groupby("payment_method").agg({
        "total_amount": ["sum", "mean", "count"],
        "profit": "sum",
        "discount": "sum"
    })
    
    payment_perf.columns = ["Total_Revenue", "Avg_Transaction", "Transaction_Count", "Total_Profit", "Total_Discounts"]
    payment_perf = payment_perf.reset_index()
    
    # Shipping method analysis
    shipping_perf = df.groupby("shipping_method").agg({
        "total_amount": ["sum", "mean", "count"],
        "profit": "sum"
    })
    
    shipping_perf.columns = ["Total_Revenue", "Avg_Transaction", "Transaction_Count", "Total_Profit"]
    shipping_perf = shipping_perf.reset_index()
    
    return payment_perf, shipping_perf


def arima_sales_forecast(df, freq='M', periods=6, order=(1, 1, 1)):
    """
    Perform ARIMA time series forecasting on sales data.
    
    Args:
        df: DataFrame with transaction data
        freq: Frequency for resampling ('D', 'W', 'M')
        periods: Number of periods to forecast
        order: ARIMA order (p, d, q)
    
    Returns:
        Tuple of (historical_data, forecast_data, model_summary)
    """
    from statsmodels.tsa.arima.model import ARIMA
    
    try:
        # Aggregate sales by time period
        df_time = df. set_index("transaction_date")
        revenue_series = df_time.resample(freq)["total_amount"].sum()
        
        # Remove any zero or missing values
        revenue_series = revenue_series[revenue_series > 0]
        
        if len(revenue_series) < 10:
            raise ValueError("Insufficient data points for ARIMA modeling")
        
        # Fit ARIMA model
        model = ARIMA(revenue_series, order=order)
        fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=periods)
        
        # Get confidence intervals
        forecast_result = fitted_model.get_forecast(steps=periods)
        forecast_ci = forecast_result.conf_int()
        
        # Prepare historical data
        historical = pd.DataFrame({
            'Date': revenue_series.index,
            'Revenue': revenue_series.values,
            'Type': 'Historical'
        })
        
        # Prepare forecast data
        last_date = revenue_series.index[-1]
        if freq == 'M':
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
        elif freq == 'W':
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='W')[1:]
        else: 
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:]
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Revenue': forecast. values,
            'Lower_CI': forecast_ci. iloc[:, 0]. values,
            'Upper_CI': forecast_ci.iloc[:, 1].values,
            'Type': 'Forecast'
        })
        
        # Model summary
        summary = {
            'AIC': fitted_model.aic,
            'BIC': fitted_model.bic,
            'RMSE': np.sqrt(fitted_model.mse),
            'Order': order,
            'Observations': len(revenue_series)
        }
        
        return historical, forecast_df, summary
        
    except Exception as e: 
        # Return empty results if ARIMA fails
        print(f"ARIMA modeling failed: {str(e)}")
        return None, None, None


def exponential_smoothing_forecast(df, freq='M', periods=6):
    """
    Simple exponential smoothing forecast as fallback.
    
    Args:
        df: DataFrame with transaction data
        freq: Frequency for resampling
        periods: Number of periods to forecast
    
    Returns:
        Tuple of (historical_data, forecast_data)
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try: 
        # Aggregate sales
        df_time = df.set_index("transaction_date")
        revenue_series = df_time.resample(freq)["total_amount"].sum()
        revenue_series = revenue_series[revenue_series > 0]
        
        # Fit model
        model = ExponentialSmoothing(revenue_series, seasonal=None, trend='add')
        fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        
        # Prepare data
        historical = pd.DataFrame({
            'Date': revenue_series.index,
            'Revenue':  revenue_series.values,
            'Type': 'Historical'
        })
        
        # Forecast dates
        last_date = revenue_series.index[-1]
        if freq == 'M':
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='MS')[1:]
        elif freq == 'W':
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='W')[1:]
        else:
            forecast_dates = pd.date_range(start=last_date, periods=periods + 1, freq='D')[1:]
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'Revenue': forecast.values,
            'Type': 'Forecast'
        })
        
        return historical, forecast_df
        
    except Exception as e:
        print(f"Exponential smoothing failed: {str(e)}")
        return None, None