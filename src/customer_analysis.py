#!/usr/bin/env python3
"""
Comprehensive customer analysis utilities:
- RFM Analysis
- K-Means Clustering
- Customer Lifetime Value (CLV)
- Cohort Analysis
- Churn Analysis
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def compute_rfm(df, snapshot_date=None):
    """
    Compute RFM scores and segments for customers.
    
    Args:
        df: DataFrame with transaction data
        snapshot_date: Reference date for recency calculation
    
    Returns:
        DataFrame with RFM metrics and segments
    """
    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max()
    
    rfm = df.groupby("customer_id").agg(
        Recency=("transaction_date", lambda x: (snapshot_date - x.max()).days),
        Frequency=("transaction_id", "nunique"),
        Monetary=("total_amount", "sum")
    )
    
    # Create RFM scores (1-5 scale)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates='drop'). astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"]. rank(method="first"), 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates='drop'). astype(int)
    
    # Combined RFM score
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    
    # Segment customers
    def segment_customer(score):
        if score >= 12:
            return "Champions"
        elif score >= 9:
            return "Loyal"
        elif score >= 6:
            return "Potential"
        else:
            return "At Risk"
    
    rfm["Segment"] = rfm["RFM_Score"].apply(segment_customer)
    
    return rfm.reset_index()


def perform_kmeans_clustering(df, n_clusters=4, features=None):
    """
    Perform K-means clustering on customer behavior. 
    
    Args:
        df: DataFrame with transaction data
        n_clusters: Number of clusters
        features: List of features to use (default: total_amount, frequency, quantity)
    
    Returns:
        DataFrame with cluster assignments and features
    """
    if features is None:
        features_data = df. groupby("customer_id").agg({
            "total_amount": "sum",
            "transaction_id": "count",
            "quantity": "sum"
        })
    else:
        features_data = df.groupby("customer_id")[features].agg("sum")
    
    features_data.columns = ["Total_Spent", "Transaction_Count", "Total_Quantity"]
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_data)
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_data["Cluster"] = kmeans. fit_predict(features_scaled)
    
    # Add cluster centers (unstandardized)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    cluster_info = pd.DataFrame(
        centers,
        columns=features_data.columns[:-1],
        index=[f"Cluster_{i}" for i in range(n_clusters)]
    )
    
    return features_data. reset_index(), cluster_info


def calculate_clv(df):
    """
    Calculate Customer Lifetime Value (CLV) metrics.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        DataFrame with CLV metrics per customer
    """
    clv = df.groupby("customer_id").agg({
        "total_amount": "sum",
        "transaction_id": "count",
        "profit": "sum",
        "transaction_date": ["min", "max"]
    })
    
    clv.columns = ["Total_Spent", "Transaction_Count", "Total_Profit", "First_Purchase", "Last_Purchase"]
    
    # Calculate customer lifespan in days
    clv["Customer_Lifespan_Days"] = (clv["Last_Purchase"] - clv["First_Purchase"]).dt.days
    
    # Average order value
    clv["Avg_Order_Value"] = (clv["Total_Spent"] / clv["Transaction_Count"]).round(2)
    
    # Profit margin
    clv["Profit_Margin_Pct"] = ((clv["Total_Profit"] / clv["Total_Spent"]) * 100).round(2)
    
    return clv.reset_index()


def cohort_analysis(df):
    """
    Perform cohort analysis to track customer retention.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        Cohort retention pivot table
    """
    # Create cohort month (first purchase month)
    df_cohort = df.copy()
    df_cohort["Order_Month"] = df_cohort["transaction_date"].dt.to_period("M")
    df_cohort["Cohort_Month"] = df_cohort. groupby("customer_id")["transaction_date"].transform("min"). dt.to_period("M")
    
    # Calculate cohort index (months since first purchase)
    def get_month_diff(row):
        return (row["Order_Month"] - row["Cohort_Month"]).n
    
    df_cohort["Cohort_Index"] = df_cohort.apply(get_month_diff, axis=1)
    
    # Create cohort table
    cohort_data = df_cohort.groupby(["Cohort_Month", "Cohort_Index"])["customer_id"].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index="Cohort_Month", columns="Cohort_Index", values="customer_id")
    
    # Calculate retention rates
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0) * 100
    
    return retention


def identify_churned_customers(df, churn_threshold_days=90):
    """
    Identify churned customers based on inactivity.
    
    Args:
        df: DataFrame with transaction data
        churn_threshold_days: Days of inactivity to consider churned
    
    Returns:
        DataFrame with churn status per customer
    """
    max_date = df["transaction_date"].max()
    
    last_purchase = df.groupby("customer_id")["transaction_date"].max(). reset_index()
    last_purchase.columns = ["customer_id", "Last_Purchase_Date"]
    
    last_purchase["Days_Since_Purchase"] = (max_date - last_purchase["Last_Purchase_Date"]).dt.days
    last_purchase["Is_Churned"] = last_purchase["Days_Since_Purchase"] > churn_threshold_days
    last_purchase["Churn_Risk"] = pd.cut(
        last_purchase["Days_Since_Purchase"],
        bins=[0, 30, 60, 90, np.inf],
        labels=["Low", "Medium", "High", "Churned"]
    )
    
    return last_purchase


def customer_purchase_patterns(df):
    """
    Analyze customer purchase patterns.
    
    Args:
        df: DataFrame with transaction data
    
    Returns:
        DataFrame with purchase pattern metrics
    """
    patterns = df.groupby("customer_id").agg({
        "transaction_date": lambda x: (x.max() - x.min()).days / len(x) if len(x) > 1 else 0,
        "category": lambda x: x.mode()[0] if len(x) > 0 else None,
        "day_of_week": lambda x: x.mode()[0] if len(x) > 0 else None,
        "total_amount": ["mean", "std"],
        "quantity": "sum"
    })
    
    patterns.columns = [
        "Avg_Days_Between_Purchases",
        "Favorite_Category",
        "Favorite_Day",
        "Avg_Transaction_Amount",
        "Transaction_Amount_StdDev",
        "Total_Items_Purchased"
    ]
    
    return patterns.reset_index()


if __name__ == "__main__":
    # Test functions
    from src.analysis import load_master
    
    df = load_master()
    
    print("=" * 60)
    print("RFM ANALYSIS")
    print("=" * 60)
    rfm = compute_rfm(df)
    print(rfm.head())
    print("\nSegment Distribution:")
    print(rfm["Segment"].value_counts())
    
    print("\n" + "=" * 60)
    print("K-MEANS CLUSTERING")
    print("=" * 60)
    clusters, centers = perform_kmeans_clustering(df, n_clusters=4)
    print("Cluster Centers:")
    print(centers)
    
    print("\n" + "=" * 60)
    print("CUSTOMER LIFETIME VALUE")
    print("=" * 60)
    clv = calculate_clv(df)
    print(clv.nlargest(10, "Total_Spent"))
    
    print("\n" + "=" * 60)
    print("CHURN ANALYSIS")
    print("=" * 60)
    churn = identify_churned_customers(df, churn_threshold_days=90)
    print(f"Churn Rate: {churn['Is_Churned'].mean() * 100:.2f}%")
    print("\nChurn Risk Distribution:")
    print(churn["Churn_Risk"].value_counts())