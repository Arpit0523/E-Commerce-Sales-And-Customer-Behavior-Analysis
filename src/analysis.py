#!/usr/bin/env python3
"""
Sales analysis utilities (examples).
This module provides functions you can import into notebooks or call from scripts.
"""
import pandas as pd
import plotly.express as px

def load_master(path="data/processed/master_dataset.csv"):
    df = pd.read_csv(path, parse_dates=["transaction_date", "registration_date"])
    return df

def monthly_revenue(df):
    monthly = df.groupby(df["transaction_date"].dt.to_period("M"))["total_amount"].sum().rename("revenue").reset_index()
    monthly["transaction_date"] = monthly["transaction_date"].dt.to_timestamp()
    return monthly

def top_products(df, n=10):
    p = df.groupby("product_name").agg(revenue=("total_amount","sum"), units=("quantity","sum")).sort_values("revenue", ascending=False).head(n)
    return p

if __name__ == "__main__":
    df = load_master()
    mr = monthly_revenue(df)
    print(mr.tail())