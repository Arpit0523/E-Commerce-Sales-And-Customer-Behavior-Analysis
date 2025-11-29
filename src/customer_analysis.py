#!/usr/bin/env python3
"""
Customer analysis utilities: RFM, CLV, simple churn heuristics.

Use in notebooks or import functions.
"""
import pandas as pd
import numpy as np

def compute_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["transaction_date"].max()
    rfm = df.groupby("customer_id").agg(
        Recency = ("transaction_date", lambda x: (snapshot_date - x.max()).days),
        Frequency = ("transaction_id", "nunique"),
        Monetary = ("total_amount", "sum")
    )
    # Score (1-5)
    rfm["R_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["M_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5]).astype(int)
    rfm["RFM_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]
    return rfm.reset_index()