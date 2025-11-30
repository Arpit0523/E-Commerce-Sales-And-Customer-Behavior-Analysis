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
warnings. filterwarnings('ignore')


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
    
    return trends.reset_index()


def category_performance_analysis(df):
    """
    Analyze performance by product category.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        DataFrame with category metrics
    """
    category_perf = df.groupby("category"). agg({
        "total_amount": "sum",
        "profit": "sum",
        "transaction_id": "count",
        "customer_id": "nunique",
        "quantity": "sum",
        "discount": "sum"
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
    ). round(2)
    
    category_perf["Avg_Transaction_Value"] = (
        category_perf["Revenue"] / category_perf["Transactions"]
    ).round(2)
    
    category_perf["Discount_Rate_Pct"] = (
        (category_perf["Total_Discounts"] / (category_perf["Revenue"] + category_perf["Total_Discounts"])) * 100
    ).round(2)
    
    return category_perf. sort_values("Revenue", ascending=False).reset_index()


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
    
    product_perf["Revenue_Rank"] = product_perf["Revenue"].rank(ascending=False, method="dense")
    product_perf["Profit_Rank"] = product_perf["Profit"].rank(ascending=False, method="dense")
    
    return product_perf.sort_values("Revenue", ascending=False).reset_index()


def market_basket_analysis(df, min_support=0.01, min_confidence=0.3):
    """
    Perform market basket analysis to find product associations.
    
    Args:
        df: DataFrame with transaction data
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
    
    Returns:
        DataFrame with association rules
    """
    # Group products by transaction
    baskets = df.groupby("transaction_id")["category"].apply(list).values
    
    # Find frequent itemsets (category pairs)
    category_pairs = []
    for basket in baskets:
        if len(basket) >= 2:
            for pair in combinations(set(basket), 2):
                category_pairs.append(tuple(sorted(pair)))
    
    # Count frequencies
    pair_counts = Counter(category_pairs)
    total_transactions = df["transaction_id"].nunique()
    
    # Calculate support, confidence, and lift
    rules = []
    for (item1, item2), count in pair_counts.items():
        support = count / total_transactions
        
        if support >= min_support:
            # Calculate confidence and lift
            item1_count = sum(1 for basket in baskets if item1 in basket)
            item2_count = sum(1 for basket in baskets if item2 in basket)
            
            confidence_1_to_2 = count / item1_count if item1_count > 0 else 0
            confidence_2_to_1 = count / item2_count if item2_count > 0 else 0
            
            lift = (count * total_transactions) / (item1_count * item2_count) if item1_count > 0 and item2_count > 0 else 0
            
            if confidence_1_to_2 >= min_confidence or confidence_2_to_1 >= min_confidence:
                rules.append({
                    "Item_1": item1,
                    "Item_2": item2,
                    "Support": round(support, 4),
                    "Confidence_1_to_2": round(confidence_1_to_2, 4),
                    "Confidence_2_to_1": round(confidence_2_to_1, 4),
                    "Lift": round(lift, 4),
                    "Count": count
                })
    
    rules_df = pd.DataFrame(rules)
    if not rules_df.empty:
        rules_df = rules_df.sort_values("Lift", ascending=False)
    
    return rules_df


def seasonal_decomposition_simple(df, freq='M'):
    """
    Simple seasonal decomposition of sales data.
    
    Args:
        df: DataFrame with transaction data
        freq: Frequency for aggregation
    
    Returns:
        DataFrame with trend and seasonal components
    """
    sales_ts = df.set_index("transaction_date"). resample(freq)["total_amount"].sum()
    
    # Calculate moving average (trend)
    window = 12 if freq == 'M' else 7
    trend = sales_ts.rolling(window=window, center=True).mean()
    
    # Detrend
    detrended = sales_ts - trend
    
    # Calculate seasonal component (average by period)
    if freq == 'M':
        seasonal = detrended.groupby(detrended.index.month).mean()
    elif freq == 'W':
        seasonal = detrended.groupby(detrended.index.isocalendar().week).mean()
    else:
        seasonal = detrended.groupby(detrended.index.dayofweek).mean()
    
    # Map back to original index
    if freq == 'M':
        seasonal_full = detrended.index.month.map(seasonal)
    elif freq == 'W':
        seasonal_full = detrended.index.isocalendar().week.map(seasonal)
    else:
        seasonal_full = detrended.index. dayofweek.map(seasonal)
    
    # Residual
    residual = sales_ts - trend - seasonal_full
    
    result = pd.DataFrame({
        "Observed": sales_ts,
        "Trend": trend,
        "Seasonal": seasonal_full,
        "Residual": residual
    })
    
    return result.reset_index()


def payment_shipping_analysis(df):
    """
    Analyze payment methods and shipping preferences.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        Tuple of (payment_analysis, shipping_analysis) DataFrames
    """
    payment_analysis = df.groupby("payment_method").agg({
        "total_amount": "sum",
        "transaction_id": "count",
        "customer_id": "nunique"
    })
    payment_analysis.columns = ["Revenue", "Transactions", "Unique_Customers"]
    payment_analysis["Avg_Transaction_Value"] = (
        payment_analysis["Revenue"] / payment_analysis["Transactions"]
    ).round(2)
    
    shipping_analysis = df.groupby("shipping_method").agg({
        "total_amount": "sum",
        "transaction_id": "count",
        "customer_id": "nunique"
    })
    shipping_analysis.columns = ["Revenue", "Transactions", "Unique_Customers"]
    shipping_analysis["Avg_Transaction_Value"] = (
        shipping_analysis["Revenue"] / shipping_analysis["Transactions"]
    ).round(2)
    
    return payment_analysis. reset_index(), shipping_analysis.reset_index()


if __name__ == "__main__":
    from src.analysis import load_master
    
    df = load_master()
    
    print("=" * 60)
    print("SALES TRENDS ANALYSIS")
    print("=" * 60)
    trends = sales_trends_analysis(df, freq='M')
    print(trends.tail())
    
    print("\n" + "=" * 60)
    print("CATEGORY PERFORMANCE")
    print("=" * 60)
    categories = category_performance_analysis(df)
    print(categories)
    
    print("\n" + "=" * 60)
    print("MARKET BASKET ANALYSIS")
    print("=" * 60)
    mba = market_basket_analysis(df)
    print(mba.head(10))
    
    print("\n" + "=" * 60)
    print("PAYMENT & SHIPPING ANALYSIS")
    print("=" * 60)
    payment, shipping = payment_shipping_analysis(df)
    print("Payment Methods:")
    print(payment)
    print("\nShipping Methods:")
    print(shipping)