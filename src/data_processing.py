#!/usr/bin/env python3
"""
Data cleaning and master dataset creation.

Usage:
    python src/data_processing.py
"""
import pandas as pd
import os

RAW_DIR = "data/raw"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def load_data():
    customers = pd.read_csv(os.path.join(RAW_DIR, "customers.csv"))
    products = pd.read_csv(os.path.join(RAW_DIR, "products.csv"))
    transactions = pd.read_csv(os.path.join(RAW_DIR, "transactions.csv"))
    return customers, products, transactions

def clean_and_merge(customers, products, transactions):
    # Basic cleaning
    customers = customers.drop_duplicates(subset=["customer_id"]).copy()
    customers["registration_date"] = pd.to_datetime(customers["registration_date"])

    products = products.drop_duplicates(subset=["product_id"]).copy()
    products["profit_margin_pct"] = ((products["price"] - products["cost"]) / products["price"] * 100).round(2)

    transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"])
    transactions_filtered = transactions[transactions["status"] == "Completed"].copy()

    # time features
    transactions_filtered["year"] = transactions_filtered["transaction_date"].dt.year
    transactions_filtered["month"] = transactions_filtered["transaction_date"].dt.month
    transactions_filtered["quarter"] = transactions_filtered["transaction_date"].dt.quarter
    transactions_filtered["day_of_week"] = transactions_filtered["transaction_date"].dt.day_name()

    # Merge
    master = transactions_filtered.merge(products, on="product_id", how="left") \
                                   .merge(customers, on="customer_id", how="left")

    # profit calculation
    master["profit"] = (master["price"] - master["cost"]) * master["quantity"] - master["discount"]

    return master

def save_master(master_df):
    out_path = os.path.join(OUT_DIR, "master_dataset.csv")
    master_df.to_csv(out_path, index=False)
    print("Saved master dataset to", out_path)

def main():
    customers, products, transactions = load_data()
    master = clean_and_merge(customers, products, transactions)
    save_master(master)
    print("Summary:")
    print("Total revenue:", master["total_amount"].sum())
    print("Total transactions:", len(master))

if __name__ == "__main__":
    main()