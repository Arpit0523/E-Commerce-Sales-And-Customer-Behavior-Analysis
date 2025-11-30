#!/usr/bin/env python3
"""
Additional analysis functions specific to your real dataset
Includes device analysis, session analysis, and rating analysis
"""
import pandas as pd
import numpy as np

def device_performance_analysis(df):
    """
    Analyze performance by device type
    """
    device_stats = df.groupby('device_type').agg({
        'total_amount': ['sum', 'mean'],
        'transaction_id': 'count',
        'session_duration': 'mean',
        'pages_viewed': 'mean',
        'rating': 'mean',
        'is_returning': lambda x: (x == True).sum()
    }).round(2)
    
    device_stats.columns = [
        'Total_Revenue', 'Avg_Order_Value', 'Transactions',
        'Avg_Session_Minutes', 'Avg_Pages_Viewed', 'Avg_Rating',
        'Returning_Customers'
    ]
    
    device_stats['Conversion_Quality'] = (
        device_stats['Avg_Rating'] * device_stats['Avg_Order_Value'] / 100
    ).round(2)
    
    return device_stats. sort_values('Total_Revenue', ascending=False). reset_index()

def session_behavior_analysis(df):
    """
    Analyze customer session behavior
    """
    # Segment by session duration
    df['session_segment'] = pd.cut(
        df['session_duration'],
        bins=[0, 5, 15, 30, 120],
        labels=['Quick (0-5m)', 'Medium (5-15m)', 'Long (15-30m)', 'Very Long (30m+)']
    )
    
    session_stats = df.groupby('session_segment').agg({
        'total_amount': ['mean', 'sum'],
        'transaction_id': 'count',
        'pages_viewed': 'mean',
        'rating': 'mean',
        'discount': 'mean'
    }).round(2)
    
    session_stats.columns = [
        'Avg_Order_Value', 'Total_Revenue', 'Transactions',
        'Avg_Pages_Viewed', 'Avg_Rating', 'Avg_Discount'
    ]
    
    return session_stats.reset_index()

def rating_analysis(df):
    """
    Analyze customer ratings and satisfaction
    """
    rating_stats = df.groupby('rating').agg({
        'transaction_id': 'count',
        'total_amount': 'sum',
        'discount': 'mean',
        'delivery_days': 'mean',
        'session_duration': 'mean'
    }).round(2)
    
    rating_stats.columns = [
        'Transaction_Count', 'Total_Revenue', 'Avg_Discount',
        'Avg_Delivery_Days', 'Avg_Session_Minutes'
    ]
    
    rating_stats['Percentage'] = (
        rating_stats['Transaction_Count'] / rating_stats['Transaction_Count'].sum() * 100
    ).round(2)
    
    return rating_stats. reset_index()

def city_performance_analysis(df):
    """
    Analyze performance by city
    """
    city_stats = df.groupby('city').agg({
        'total_amount': 'sum',
        'transaction_id': 'count',
        'customer_id': 'nunique',
        'rating': 'mean',
        'delivery_days': 'mean',
        'discount': 'mean'
    }). round(2)
    
    city_stats.columns = [
        'Total_Revenue', 'Transactions', 'Unique_Customers',
        'Avg_Rating', 'Avg_Delivery_Days', 'Avg_Discount'
    ]
    
    city_stats['Revenue_Per_Customer'] = (
        city_stats['Total_Revenue'] / city_stats['Unique_Customers']
    ).round(2)
    
    return city_stats.sort_values('Total_Revenue', ascending=False).reset_index()

def returning_customer_analysis(df):
    """
    Compare returning vs new customers
    """
    customer_type_stats = df.groupby('is_returning').agg({
        'total_amount': ['mean', 'sum'],
        'transaction_id': 'count',
        'rating': 'mean',
        'session_duration': 'mean',
        'pages_viewed': 'mean',
        'discount': 'mean'
    }). round(2)
    
    customer_type_stats.columns = [
        'Avg_Order_Value', 'Total_Revenue', 'Transactions',
        'Avg_Rating', 'Avg_Session_Minutes', 'Avg_Pages_Viewed',
        'Avg_Discount'
    ]
    
    customer_type_stats. index = ['New Customer', 'Returning Customer']
    
    return customer_type_stats.reset_index()

def delivery_satisfaction_analysis(df):
    """
    Analyze relationship between delivery time and satisfaction
    """
    df['delivery_segment'] = pd.cut(
        df['delivery_days'],
        bins=[0, 3, 7, 14, 30],
        labels=['Fast (1-3d)', 'Standard (4-7d)', 'Slow (8-14d)', 'Very Slow (15d+)']
    )
    
    delivery_stats = df.groupby('delivery_segment').agg({
        'rating': ['mean', 'count'],
        'total_amount': 'mean',
        'is_returning': lambda x: (x == True).mean() * 100
    }).round(2)
    
    delivery_stats.columns = [
        'Avg_Rating', 'Transaction_Count', 'Avg_Order_Value',
        'Returning_Customer_Rate_%'
    ]
    
    return delivery_stats.reset_index()

if __name__ == "__main__":
    # Load the processed dataset (could be from real_data_processing.py or generate_sample_data.py)
    df = pd.read_csv('data/processed/master_dataset.csv', parse_dates=['transaction_date'])
    
    print(f"Loaded {len(df):,} records")
    print(f"Columns available: {list(df.columns)}\n")
    
    # Check if this is real data (has device_type column) or sample data
    if 'device_type' not in df.columns:
        print("⚠️  Warning: This appears to be sample data without device_type column.")
        print("Run 'python src/real_data_processing.py' first to process the real DATASET.csv")
        print("Skipping analyses that require real dataset columns...\n")
        has_real_cols = False
    else:
        has_real_cols = True
    
    if has_real_cols:
        print("=" * 60)
        print("DEVICE PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(device_performance_analysis(df))
    if has_real_cols:
        print("=" * 60)
        print("DEVICE PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(device_performance_analysis(df))
    
    if has_real_cols:
        print("\n" + "=" * 60)
        print("SESSION BEHAVIOR ANALYSIS")
        print("=" * 60)
        print(session_behavior_analysis(df))
    
    if has_real_cols:
        print("\n" + "=" * 60)
        print("RATING ANALYSIS")
        print("=" * 60)
        print(rating_analysis(df))
    
    if has_real_cols:
        print("\n" + "=" * 60)
        print("CITY PERFORMANCE ANALYSIS")
        print("=" * 60)
        print(city_performance_analysis(df))
    
    if has_real_cols:
        print("\n" + "=" * 60)
        print("RETURNING CUSTOMER ANALYSIS")
        print("=" * 60)
        print(returning_customer_analysis(df))
    
    if has_real_cols:
        print("\n" + "=" * 60)
        print("DELIVERY SATISFACTION ANALYSIS")
        print("=" * 60)
        print(delivery_satisfaction_analysis(df))