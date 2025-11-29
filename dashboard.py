#!/usr/bin/env python3
"""
Simple Streamlit dashboard starter.
Run:
    streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import plotly.express as px
from src.analysis import monthly_revenue, top_products, load_master

st.set_page_config(page_title="E-commerce Analytics", layout="wide")

st.title("E-commerce Analytics Dashboard - Starter")

@st.cache_data
def load_data():
    try:
        df = load_master()
        return df
    except FileNotFoundError:
        st.error("Master dataset not found. Run src/data_processing.py first.")
        st.stop()

df = load_data()

# Filters
with st.sidebar:
    st.header("Filters")
    min_date = df["transaction_date"].min().date()
    max_date = df["transaction_date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    categories = ["All"] + sorted(df["category"].dropna().unique().tolist())
    category = st.selectbox("Category", categories)

# Apply filters
if len(date_range) == 2:
    start, end = date_range
    mask = (df["transaction_date"].dt.date >= start) & (df["transaction_date"].dt.date <= end)
    df = df[mask]
if category != "All":
    df = df[df["category"] == category]

# KPI row
col1, col2, col3 = st.columns(3)
col1.metric("Total Revenue", f"${df['total_amount'].sum():,.0f}")
col2.metric("Transactions", f"{len(df):,}")
col3.metric("Customers", f"{df['customer_id'].nunique():,}")

# Revenue chart
st.subheader("Revenue Over Time")
mr = monthly_revenue(df)
fig = px.line(mr, x="transaction_date", y="revenue", title="Monthly Revenue")
st.plotly_chart(fig, use_container_width=True)

# Top products
st.subheader("Top Products")
tp = top_products(df, n=10).reset_index()
st.dataframe(tp)