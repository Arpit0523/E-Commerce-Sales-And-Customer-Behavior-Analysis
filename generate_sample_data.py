#!/usr/bin/env python3
"""
Generate synthetic e-commerce data:
- customers.csv
- products.csv
- transactions.csv

Run:
    python generate_sample_data.py
"""
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
import os

fake = Faker()
random.seed(42)
np.random.seed(42)

OUT_DIR_RAW = "data/raw"
os.makedirs(OUT_DIR_RAW, exist_ok=True)

NUM_CUSTOMERS = 1000
NUM_PRODUCTS = 100
NUM_TRANSACTIONS = 5000
START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)

def generate_customers(n):
    rows = []
    for i in range(n):
        reg = fake.date_between(start_date=START_DATE, end_date=END_DATE)
        rows.append({
            "customer_id": f"CUST{i+1:05d}",
            "name": fake.name(),
            "email": fake.email(),
            "country": fake.country(),
            "city": fake.city(),
            "registration_date": reg.isoformat(),
            "age": random.randint(18, 70),
            "gender": random.choice(["Male", "Female", "Other"])
        })
    return pd.DataFrame(rows)

def generate_products(n):
    categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty', 'Toys']
    rows = []
    for i in range(n):
        category = random.choice(categories)
        price = round(random.uniform(10, 500), 2)
        cost = round(price * random.uniform(0.4, 0.85), 2)
        rows.append({
            "product_id": f"PROD{i+1:04d}",
            "product_name": fake.catch_phrase(),
            "category": category,
            "price": price,
            "cost": cost
        })
    return pd.DataFrame(rows)

def generate_transactions(customers_df, products_df, n):
    rows = []
    for i in range(n):
        cust = customers_df.sample(1).iloc[0]
        prod = products_df.sample(1).iloc[0]
        quantity = random.randint(1, 5)
        reg_date = pd.to_datetime(cust["registration_date"])
        tx_date = fake.date_between(start_date=reg_date, end_date=END_DATE)
        unit_price = prod["price"]
        total_amount = unit_price * quantity
        discount = round(random.uniform(0, 0.3) * total_amount, 2) if random.random() > 0.7 else 0.0
        status = random.choices(["Completed", "Cancelled", "Returned"], weights=[0.85, 0.10, 0.05])[0]
        rows.append({
            "transaction_id": f"TXN{i+1:06d}",
            "customer_id": cust["customer_id"],
            "product_id": prod["product_id"],
            "transaction_date": pd.to_datetime(tx_date).isoformat(),
            "quantity": quantity,
            "unit_price": unit_price,
            "discount": discount,
            "total_amount": round(total_amount - discount, 2),
            "payment_method": random.choice(["Credit Card", "PayPal", "Debit Card", "Bank Transfer"]),
            "shipping_method": random.choice(["Standard", "Express", "Overnight"]),
            "status": status
        })
    return pd.DataFrame(rows)

def main():
    print("Generating customers...")
    customers = generate_customers(NUM_CUSTOMERS)
    customers.to_csv(os.path.join(OUT_DIR_RAW, "customers.csv"), index=False)

    print("Generating products...")
    products = generate_products(NUM_PRODUCTS)
    products.to_csv(os.path.join(OUT_DIR_RAW, "products.csv"), index=False)

    print("Generating transactions...")
    transactions = generate_transactions(customers, products, NUM_TRANSACTIONS)
    transactions.to_csv(os.path.join(OUT_DIR_RAW, "transactions.csv"), index=False)

    print("Saved CSVs to", OUT_DIR_RAW)
    print(f"Customers: {len(customers)}, Products: {len(products)}, Transactions: {len(transactions)}")

if __name__ == "__main__":
    main()