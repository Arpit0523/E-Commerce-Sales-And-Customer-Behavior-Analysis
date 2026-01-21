#!/usr/bin/env python3
"""
Comprehensive E-commerce Analytics Dashboard
Multi-page Streamlit application with real dataset features

Run:
    streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

# Import custom modules
from src.analysis import load_master, monthly_revenue, top_products
from src.customer_analysis import (
    compute_rfm, 
    perform_kmeans_clustering, 
    calculate_clv,
    cohort_analysis,
    identify_churned_customers
)
from src. sales_analysis import (
    sales_trends_analysis,
    category_performance_analysis,
    product_profitability_analysis,
    market_basket_analysis,
    payment_shipping_analysis
)
from src. real_data_analysis import (
    device_performance_analysis,
    session_behavior_analysis,
    rating_analysis,
    city_performance_analysis,
    returning_customer_analysis,
    delivery_satisfaction_analysis
)

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Pro",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "E-commerce Analytics Dashboard - Built with Streamlit"
    }
)


# Custom CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #000000 !important;
    }
    .main {
        padding: 0rem 1rem;
        background-color: #000000 !important;
    }
    [data-testid="stSidebar"] {
        background-color: #0A0A0A !important;
        z-index: 999 !important;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] label {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stRadio > label {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stSelectbox label {
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stDateInput label {
        color: #FFFFFF !important;
    }
    /* All text elements - specific to main content area */
    .main p, .main span, .main label,
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: #FFFFFF !important;
    }
    /* Input fields */
    input, select, textarea {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
        color: white;
    }
    .stMetric {
        background-color: rgba(20, 20, 20, 0.95) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
        border: 2px solid rgba(102, 126, 234, 0.4) !important;
    }
    .stMetric label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 28px !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #B8B8B8 !important;
        font-weight: 500 !important;
    }
    div[data-testid="stDataFrame"],
    div[data-testid="stDataFrame"] * {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        padding: 10px !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
        min-height: 100px !important;
    }
    /* DataFrame container - all levels */
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stDataFrame"] > div > div,
    div[data-testid="stDataFrame"] > div > div > div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    /* DataFrame canvas/viewer */
    div[data-testid="stDataFrame"] canvas {
        background-color: #1A1A1A !important;
        filter: invert(0) !important;
    }
    div[data-testid="stDataFrame"] .stDataFrameGlideDataEditor {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    /* All text elements in dataframe */
    div[data-testid="stDataFrame"] span,
    div[data-testid="stDataFrame"] p,
    div[data-testid="stDataFrame"] div {
        color: #FFFFFF !important;
    }
    /* DataFrame headers */
    div[data-testid="stDataFrame"] thead tr th,
    div[data-testid="stDataFrame"] .col_heading,
    div[data-testid="stDataFrame"] th {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        padding: 12px !important;
        border: 1px solid #764ba2 !important;
    }
    /* DataFrame cells - multiple selectors */
    div[data-testid="stDataFrame"] tbody tr td,
    div[data-testid="stDataFrame"] .data,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] [role="gridcell"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        padding: 10px !important;
        border: 1px solid #333333 !important;
        font-size: 13px !important;
    }
    /* DataFrame row hover */
    div[data-testid="stDataFrame"] tbody tr:hover td {
        background-color: #2A2A2A !important;
    }
    /* Styled dataframes */
    div[data-testid="stDataFrame"] table {
        color: #FFFFFF !important;
        background-color: #1A1A1A !important;
        border-collapse: collapse !important;
    }
    div[data-testid="stDataFrame"] table * {
        color: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] table tbody {
        background-color: #1A1A1A !important;
    }
    div[data-testid="stDataFrame"] table tbody td {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] table thead {
        background-color: #667eea !important;
    }
    div[data-testid="stDataFrame"] table thead th {
        background-color: #667eea !important;
        color: #FFFFFF !important;
    }
    /* Row and column headings */
    div[data-testid="stDataFrame"] .row_heading {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDataFrame"] .blank {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    /* Glide data grid - Streamlit's modern dataframe viewer */
    div[data-testid="stDataFrame"] [class*="dvn-scroller"],
    div[data-testid="stDataFrame"] [class*="dvn"],
    [data-testid="stDataFrame"] [style*="grid"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    [class*="cell"],
    [class*="gdg-cell"],
    [role="gridcell"] {
        color: #FFFFFF !important;
        background-color: #1A1A1A !important;
    }
    [class*="header-cell"],
    [class*="gdg-header"],
    [role="columnheader"] {
        color: #FFFFFF !important;
        background-color: #667eea !important;
    }
    /* React data grid elements */
    div[data-testid="stDataFrame"] [class*="rdg"],
    div[data-testid="stDataFrame"] [class*="rdg-cell"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
    }
    div[data-testid="stDataFrame"] [class*="rdg-header"] {
        background-color: #667eea !important;
        color: #FFFFFF !important;
    }
    h1 {
        color: #667eea !important;
        font-weight: 700 !important;
        padding-bottom: 10px;
        border-bottom: 3px solid #667eea;
    }
    h2 {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        margin-top: 20px;
    }
    h3 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 26px !important;
        padding: 15px 20px !important;
        margin: 25px 0 20px 0 !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        color: #FFFFFF !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1A1A1A !important;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
        border: 1px solid #667eea;
    }
    .highlight-box {
        background-color: rgba(20, 20, 20, 0.95) !important;
        padding: 20px !important;
        border-radius: 12px !important;
        border-left: 5px solid #667eea !important;
        margin: 10px 0 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
    }
    .highlight-box * {
        color: #FFFFFF !important;
    }
    .highlight-box h4 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        margin-bottom: 12px !important;
        font-size: 18px !important;
    }
    .highlight-box p {
        color: #FFFFFF !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
        margin-bottom: 8px !important;
    }
    .highlight-box strong {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    .highlight-box ul, .highlight-box li {
        color: #FFFFFF !important;
        font-size: 15px !important;
        line-height: 1.8 !important;
    }
    .stAlert, .stAlert * {
        background-color: rgba(20, 20, 20, 0.95) !important;
        color: #FFFFFF !important;
    }
    /* Fix expander text */
    .streamlit-expanderHeader {
        color: #FFFFFF !important;
    }
    /* Fix caption text */
    .css-1v0mbdj, .css-16huue1 {
        color: #B8B8B8 !important;
    }
    /* Info, success, warning, error boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        background-color: rgba(20, 20, 20, 0.95) !important;
        color: #FFFFFF !important;
    }
    /* Buttons */
    .stButton > button {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s !important;
    }
    .stButton > button:hover {
        background-color: #764ba2 !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.5) !important;
    }
    /* Selectbox and multiselect */
    .stSelectbox > div > div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #667eea !important;
    }
    .stSelectbox label {
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }
    .stSelectbox [data-baseweb="select"] {
        background-color: #1A1A1A !important;
    }
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #667eea !important;
    }
    .stSelectbox input {
        color: #FFFFFF !important;
    }
    .stMultiSelect > div > div {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #667eea !important;
    }
    /* Dropdown options - enhanced visibility */
    [data-baseweb="popover"] {
        background-color: #0A0A0A !important;
        border: 2px solid #667eea !important;
        z-index: 9999 !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.8) !important;
    }
    [data-baseweb="menu"] {
        background-color: #0A0A0A !important;
        z-index: 9999 !important;
    }
    [role="listbox"] {
        background-color: #0A0A0A !important;
        border: 2px solid #667eea !important;
        z-index: 9999 !important;
    }
    [data-baseweb="menu"] li {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
        border-bottom: 1px solid #333333 !important;
    }
    [role="option"] {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
    }
    [data-baseweb="menu"] li:hover,
    [role="option"]:hover {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
    }
    [aria-selected="true"] {
        background-color: #764ba2 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
    /* Slider */
    .stSlider > div > div > div {
        background-color: #1A1A1A !important;
    }
    .stSlider [role="slider"] {
        background-color: #667eea !important;
    }
    .stSlider [data-baseweb="slider"] {
        background-color: #333333 !important;
    }
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent !important;
    }
    .stRadio label {
        color: #FFFFFF !important;
    }
    /* Checkbox */
    .stCheckbox label {
        color: #FFFFFF !important;
    }
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 12px !important;
        z-index: 1 !important;
    }
    .streamlit-expanderHeader:hover {
        background-color: #2A2A2A !important;
        border-color: #764ba2 !important;
    }
    .streamlit-expanderContent {
        background-color: #1A1A1A !important;
        border: 2px solid #667eea !important;
        border-top: none !important;
        border-radius: 0 0 8px 8px !important;
        padding: 15px !important;
    }
    /* Download button */
    .stDownloadButton > button {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
    }
    /* File uploader */
    .stFileUploader {
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
    }
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    /* Date input */
    .stDateInput > div > div > input {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    /* Time input */
    .stTimeInput > div > div > input {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    /* Text area */
    .stTextArea > div > div > textarea {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
    }
    /* Code block */
    .stCodeBlock {
        background-color: #1A1A1A !important;
        border: 1px solid #333333 !important;
    }
    /* Markdown code */
    code {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    /* Table */
    table {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 2px solid #667eea !important;
        border-collapse: separate !important;
        border-spacing: 0 !important;
        border-radius: 8px !important;
        width: 100% !important;
    }
    table * {
        color: #FFFFFF !important;
    }
    th {
        background-color: #667eea !important;
        color: #FFFFFF !important;
        padding: 12px !important;
        font-weight: 700 !important;
        border: 1px solid #764ba2 !important;
        text-align: left !important;
    }
    td {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        padding: 10px !important;
        border: 1px solid #333333 !important;
    }
    tbody {
        background-color: #1A1A1A !important;
    }
    thead {
        background-color: #667eea !important;
    }
    tr {
        background-color: #1A1A1A !important;
    }
    tr:hover {
        background-color: #2A2A2A !important;
    }
    tr:hover td {
        background-color: #2A2A2A !important;
    }
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    /* Progress bar */
    .stProgress > div > div {
        background-color: #667eea !important;
    }
    /* Columns */
    [data-testid="column"] {
        background-color: transparent !important;
    }
    /* Container */
    [data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }
    /* Block container - prevent dark patches */
    [data-testid="stHorizontalBlock"] {
        background-color: transparent !important;
    }
    /* Ensure content blocks are visible */
    .element-container {
        background-color: transparent !important;
    }
    /* Make empty dataframes visible */
    .dataframe {
        border: 2px solid #667eea !important;
        border-radius: 8px !important;
        min-height: 50px !important;
    }
    /* Plotly chart containers */
    .js-plotly-plot {
        border: 1px solid #333333 !important;
        border-radius: 8px !important;
        background-color: #000000 !important;
    }
    /* Header */
    header {
        background-color: #000000 !important;
    }
    /* Footer */
    footer {
        background-color: #000000 !important;
        color: #666666 !important;
    }
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #1A1A1A;
    }
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    
    /* Responsive breakpoints */
    @media (max-width: 768px) {
        .main {
            padding: 0rem 0.5rem;
        }
        .stMetric {
            padding: 15px !important;
        }
        h3 {
            font-size: 20px !important;
            padding: 12px 15px !important;
        }
    }
    
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            height: auto;
            padding: 8px 12px;
            font-size: 12px;
        }
    }
    
    /* Ensure modals and tooltips appear above everything */
    [role="dialog"],
    [role="tooltip"],
    .stTooltipIcon {
        z-index: 10000 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Set default plotly template to dark
import plotly.io as pio
pio.templates.default = "plotly_dark"

# Custom plotly layout for better visibility
def update_chart_layout(fig):
    """Update plotly chart with dark theme settings for maximum visibility"""
    fig.update_layout(
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        font=dict(color='#FFFFFF', size=12),
        title_font=dict(color='#FFFFFF', size=16, family='Arial'),
        legend=dict(
            bgcolor='rgba(0, 0, 0, 0.8)',
            bordercolor='#333333',
            borderwidth=1,
            font=dict(color='#FFFFFF')
        ),
        xaxis=dict(
            gridcolor='#333333',
            color='#FFFFFF',
            linecolor='#333333',
            tickfont=dict(color='#FFFFFF'),
            title_font=dict(color='#FFFFFF')
        ),
        yaxis=dict(
            gridcolor='#333333',
            color='#FFFFFF',
            linecolor='#333333',
            tickfont=dict(color='#FFFFFF'),
            title_font=dict(color='#FFFFFF')
        ),
        hoverlabel=dict(
            bgcolor='#1A1A1A',
            font_color='#FFFFFF',
            bordercolor='#667eea'
        ),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    # Update trace colors for better visibility on black background
    fig.update_traces(
        marker=dict(line=dict(color='#FFFFFF', width=0.5)),
        textfont=dict(color='#FFFFFF')
    )
    return fig

# Load data with caching
@st.cache_data
def load_data():
    """Load the master dataset"""
    try:
        df = load_master()
        return df
    except FileNotFoundError:
        st. error("⚠️ Master dataset not found!  Please run: python src/real_data_processing.py")
        st.stop()

# Load data
df = load_data()

# Sidebar Navigation
with st.sidebar:
    st. image("https://img.icons8.com/clouds/100/000000/shop.png", width=100)
    st.title("🛍️ E-commerce Analytics Pro")
    
    # Page selection
    page = st.radio(
        "Navigation",
        ["📊 Overview", "📈 Sales Analysis", "👥 Customer Insights", 
         "📦 Product Analysis", "📱 Device & Session", "⭐ Satisfaction",
         "🚚 Delivery Performance", "🔄 Loyalty Analysis", "🔮 Advanced Analytics"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Global Filters
    st.subheader("🔍 Global Filters")
    
    # Date range filter
    min_date = df['transaction_date'].min(). date()
    max_date = df['transaction_date'].max().date()
    
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Category filter
    categories = ['All'] + sorted(df['category'].dropna().unique(). tolist())
    selected_category = st.selectbox("Category", categories)
    
    # City filter
    cities = ['All'] + sorted(df['city']. dropna().unique().tolist())
    selected_city = st. selectbox("City", cities, index=0)
    
    # Gender filter
    genders = ['All'] + sorted(df['gender'].dropna().unique().tolist())
    selected_gender = st.selectbox("Gender", genders)
    
    # Device filter (NEW)
    devices = ['All'] + sorted(df['device_type'].dropna().unique().tolist())
    selected_device = st.selectbox("Device Type", devices)
    
    st.markdown("---")
    st.info("📊 **Dashboard by Arpit0523**")
    st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# Apply filters
filtered_df = df. copy()

if len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df['transaction_date']. dt.date >= start_date) &
        (filtered_df['transaction_date'].dt.date <= end_date)
    ]

if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['category'] == selected_category]

if selected_city != 'All':
    filtered_df = filtered_df[filtered_df['city'] == selected_city]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

if selected_device != 'All':
    filtered_df = filtered_df[filtered_df['device_type'] == selected_device]

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.title("📊 Executive Dashboard Overview")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    total_revenue = filtered_df['total_amount'].sum()
    total_transactions = len(filtered_df)
    total_customers = filtered_df['customer_id'].nunique()
    avg_order_value = filtered_df['total_amount'].mean()
    avg_rating = filtered_df['rating'].mean()
    returning_rate = (filtered_df['is_returning'] == True).sum() / len(filtered_df) * 100
    
    with col1:
        st. metric(
            label="💰 Total Revenue",
            value=f"${total_revenue:,.0f}",
            delta=f"{(total_revenue / df['total_amount'].sum() * 100):.1f}% of total"
        )
    
    with col2:
        st.metric(
            label="🛒 Transactions",
            value=f"{total_transactions:,}",
            delta=f"{(total_transactions / len(df) * 100):.1f}% of total"
        )
    
    with col3:
        st.metric(
            label="👥 Customers",
            value=f"{total_customers:,}",
            delta=f"{(total_customers / df['customer_id'].nunique() * 100):.1f}% of total"
        )
    
    with col4:
        st.metric(
            label="📦 Avg Order Value",
            value=f"${avg_order_value:.2f}",
            delta=f"{((avg_order_value - df['total_amount'].mean()) / df['total_amount'].mean() * 100):.1f}%"
        )
    
    with col5:
        st. metric(
            label="⭐ Avg Rating",
            value=f"{avg_rating:.2f}/5.0",
            delta=f"{((avg_rating - df['rating'].mean()) / df['rating'].mean() * 100):.1f}%"
        )
    
    with col6:
        st. metric(
            label="🔄 Returning Rate",
            value=f"{returning_rate:.1f}%",
            delta=f"{(returning_rate - (df['is_returning'] == True). sum() / len(df) * 100):.1f}%"
        )
    
    st.markdown("---")
    
    # Three column layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Revenue Trend")
        monthly_rev = filtered_df.groupby(filtered_df['transaction_date']. dt.to_period('M'))['total_amount'].sum().reset_index()
        monthly_rev['transaction_date'] = monthly_rev['transaction_date'].dt.to_timestamp()
        
        fig = px.area(
            monthly_rev,
            x='transaction_date',
            y='total_amount',
            title='Monthly Revenue Trend',
            labels={'total_amount': 'Revenue ($)', 'transaction_date': 'Date'}
        )
        fig.update_traces(line_color='#1f77b4', fillcolor='rgba(31, 119, 180, 0.3)')
        fig.update_layout(height=350, hovermode='x unified')
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Category Distribution")
        category_revenue = filtered_df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
        
        fig = px. pie(
            values=category_revenue.values,
            names=category_revenue.index,
            title='Revenue by Category',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig. update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=350)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st. subheader("📱 Device Usage")
        device_dist = filtered_df['device_type'].value_counts()
        
        fig = px. bar(
            x=device_dist.index,
            y=device_dist.values,
            title='Transactions by Device',
            labels={'x': 'Device Type', 'y': 'Count'},
            color=device_dist.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=350, showlegend=False)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col1, col2 = st. columns(2)
    
    with col1:
        st. subheader("🌍 Top Cities by Revenue")
        city_revenue = filtered_df.groupby('city')['total_amount'].sum(). sort_values(ascending=False). head(10)
        
        fig = px.bar(
            x=city_revenue.values,
            y=city_revenue.index,
            orientation='h',
            title='Top 10 Cities',
            labels={'x': 'Revenue ($)', 'y': 'City'},
            color=city_revenue.values,
            color_continuous_scale='Greens'
        )
        fig.update_layout(height=400, showlegend=False)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⭐ Rating Distribution")
        rating_dist = filtered_df['rating'].value_counts(). sort_index()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=rating_dist.index,
            y=rating_dist.values,
            marker_color=['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71'],
            text=rating_dist. values,
            textposition='auto'
        ))
        fig.update_layout(
            title='Customer Ratings Distribution',
            xaxis_title='Rating (Stars)',
            yaxis_title='Number of Orders',
            height=400
        )
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick Insights
    st.markdown("---")
    st.subheader("💡 Quick Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_category = filtered_df.groupby('category')['total_amount'].sum().idxmax()
        st.metric("🏆 Top Category", top_category)
    
    with col2:
        most_used_device = filtered_df['device_type'].mode()[0]
        st.metric("📱 Most Used Device", most_used_device)
    
    with col3:
        avg_session = filtered_df['session_duration'].mean()
        st.metric("⏱️ Avg Session", f"{avg_session:.1f} min")
    
    with col4:
        avg_delivery = filtered_df['delivery_days'].mean()
        st.metric("🚚 Avg Delivery", f"{avg_delivery:.1f} days")

# ============================================================================
# PAGE 2: SALES ANALYSIS
# ============================================================================
elif page == "📈 Sales Analysis":
    st.title("📈 Sales Analysis Deep Dive")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Trends", "📅 Categories", "💰 Products", "🎯 Payments & Shipping"])
    
    with tab1:
        st.subheader("Sales Trends Analysis")
        
        # Time period selector
        time_period = st.radio("Select Aggregation", ["Daily", "Weekly", "Monthly"], horizontal=True)
        
        freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "M"}
        trends = sales_trends_analysis(filtered_df, freq=freq_map[time_period])
        
        # Revenue and Profit Chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Revenue Over Time', 'Profit Over Time'),
            vertical_spacing=0.12
        )
        
        fig.add_trace(
            go.Scatter(
                x=trends['transaction_date'], 
                y=trends['Revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#1f77b4', width=2),
                fill='tozeroy'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trends['transaction_date'], 
                y=trends['Profit'],
                mode='lines+markers',
                name='Profit',
                line=dict(color='#2ecc71', width=2),
                fill='tozeroy'
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=2, col=1)
        fig.update_layout(height=700, showlegend=False, hovermode='x unified')
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_rev = trends['Revenue'].sum()
            st.metric("Total Revenue", f"${total_rev:,.0f}")
        
        with col2:
            avg_rev = trends['Revenue'].mean()
            st.metric(f"Avg {time_period} Revenue", f"${avg_rev:,.0f}")
        
        with col3:
            total_profit = trends['Profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.0f}")
        
        with col4:
            avg_transactions = trends['Transactions'].mean()
            st.metric(f"Avg {time_period} Transactions", f"{avg_transactions:.0f}")
        
        # Show data table
        with st.expander("📋 View Detailed Data"):
            st.dataframe(trends, use_container_width=True)
    
    with tab2:
        st.subheader("Category Performance Analysis")
        
        category_perf = category_performance_analysis(filtered_df)
        
        # Category metrics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                category_perf,
                x='category',
                y='Revenue',
                title='Revenue by Category',
                color='Revenue',
                color_continuous_scale='Blues'
            )
            fig. update_layout(xaxis_tickangle=-45)
            fig = update_chart_layout(fig)
            st. plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                category_perf,
                x='category',
                y='Profit_Margin_Pct',
                title='Profit Margin by Category (%)',
                color='Profit_Margin_Pct',
                color_continuous_scale='Greens'
            )
            fig.update_layout(xaxis_tickangle=-45)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📊 Category Performance Table")
        st.dataframe(
            category_perf.style. format({
                'Revenue': '${:,.2f}',
                'Profit': '${:,.2f}',
                'Avg_Transaction_Value': '${:,.2f}',
                'Profit_Margin_Pct': '{:.2f}%',
                'Discount_Rate_Pct': '{:.2f}%'
            }). background_gradient(subset=['Revenue', 'Profit'], cmap='Greens'),
            use_container_width=True
        )
    
    with tab3:
        st. subheader("Product Profitability Analysis")
        
        product_perf = product_profitability_analysis(filtered_df)
        
        # Top N selector
        top_n = st. slider("Number of products to display", 5, 50, 10)
        
        top_products_df = product_perf.head(top_n)
        
        # Product visualization
        fig = px.bar(
            top_products_df,
            x='Revenue',
            y='product_name',
            orientation='h',
            title=f'Top {top_n} Products by Revenue',
            color='Profit_Margin_Pct',
            color_continuous_scale='RdYlGn',
            labels={'Profit_Margin_Pct': 'Profit Margin (%)'}
        )
        fig.update_layout(height=max(400, top_n * 30), yaxis={'categoryorder': 'total ascending'})
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Product scatter
        st.subheader("Product Performance Matrix")
        fig = px.scatter(
            product_perf. head(50),
            x='Revenue',
            y='Profit',
            size='Units_Sold',
            color='Profit_Margin_Pct',
            hover_data=['product_name'],
            title='Product Revenue vs Profit (Top 50)',
            color_continuous_scale='Viridis'
        )
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        with st.expander("📋 View All Products"):
            st.dataframe(
                product_perf. style.format({
                    'Revenue': '${:,.2f}',
                    'Profit': '${:,.2f}',
                    'Avg_Unit_Price': '${:,.2f}',
                    'Profit_Margin_Pct': '{:.2f}%'
                }),
                use_container_width=True,
                height=400
            )
    
    with tab4:
        st.subheader("Payment & Shipping Analysis")
        
        payment_df, shipping_df = payment_shipping_analysis(filtered_df)
        
        col1, col2 = st. columns(2)
        
        with col1:
            st.markdown("#### 💳 Payment Methods")
            fig = px.pie(
                payment_df,
                values='Total_Revenue',
                names='payment_method',
                title='Revenue by Payment Method',
                hole=0.3
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                payment_df.style.format({
                    'Total_Revenue': '${:,.2f}',
                    'Avg_Transaction': '${:,.2f}',
                    'Total_Profit': '${:,.2f}',
                    'Total_Discounts': '${:,.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🚚 Shipping Methods")
            fig = px.pie(
                shipping_df,
                values='Total_Revenue',
                names='shipping_method',
                title='Revenue by Shipping Method',
                hole=0.3
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                shipping_df.style.format({
                    'Total_Revenue': '${:,.2f}',
                    'Avg_Transaction': '${:,.2f}',
                    'Total_Profit': '${:,.2f}'
                }),
                use_container_width=True
            )

# ============================================================================
# PAGE 3: CUSTOMER INSIGHTS
# ============================================================================
elif page == "👥 Customer Insights":
    st.title("👥 Customer Behavior & Insights")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🎯 RFM Segmentation", "💎 Lifetime Value", "📈 Cohort & Churn"])
    
    with tab1:
        st.subheader("Customer Overview")
        
        # Customer metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_customers = filtered_df['customer_id'].nunique()
        avg_purchases = filtered_df. groupby('customer_id'). size().mean()
        avg_customer_value = filtered_df.groupby('customer_id')['total_amount'].sum(). mean()
        repeat_customers = (filtered_df. groupby('customer_id'). size() > 1).sum()
        
        with col1:
            st.metric("Total Customers", f"{total_customers:,}")
        with col2:
            st.metric("Avg Purchases/Customer", f"{avg_purchases:.2f}")
        with col3:
            st.metric("Avg Customer Value", f"${avg_customer_value:,.2f}")
        with col4:
            st.metric("Repeat Customers", f"{repeat_customers:,}")
        
        st.markdown("---")
        
        col1, col2 = st. columns(2)
        
        with col1:
            # Age distribution
            fig = px.histogram(
                filtered_df,
                x='age',
                nbins=20,
                title='Customer Age Distribution',
                labels={'age': 'Age', 'count': 'Number of Customers'},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(showlegend=False)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gender distribution
            gender_revenue = filtered_df.groupby('gender')['total_amount'].sum()
            
            fig = px.bar(
                x=gender_revenue.index,
                y=gender_revenue.values,
                title='Revenue by Gender',
                labels={'x': 'Gender', 'y': 'Revenue ($)'},
                color=gender_revenue.values,
                color_continuous_scale='Viridis'
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # City analysis
        st.subheader("🌍 Customer Distribution by City")
        customer_city = filtered_df.groupby('city')['customer_id'].nunique(). sort_values(ascending=False). head(15)
        
        fig = px.bar(
            x=customer_city.values,
            y=customer_city.index,
            orientation='h',
            title='Top 15 Cities by Customer Count',
            labels={'x': 'Number of Customers', 'y': 'City'},
            color=customer_city.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(height=500)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 RFM Customer Segmentation")
        
        rfm_df = compute_rfm(filtered_df)
        
        # Segment distribution
        col1, col2 = st. columns(2)
        
        with col1:
            segment_counts = rfm_df['Segment'].value_counts()
            
            colors = {'Champions': '#2ecc71', 'Loyal': '#3498db', 'Potential': '#f39c12', 'At Risk': '#e74c3c'}
            
            fig = px.bar(
                x=segment_counts.index,
                y=segment_counts.values,
                title='Customer Segments Distribution',
                labels={'x': 'Segment', 'y': 'Number of Customers'},
                color=segment_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(showlegend=False)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_revenue = rfm_df. merge(
                filtered_df.groupby('customer_id')['total_amount']. sum(). reset_index(),
                on='customer_id'
            ). groupby('Segment')['total_amount'].sum()
            
            fig = px. pie(
                values=segment_revenue.values,
                names=segment_revenue.index,
                title='Revenue by Customer Segment',
                color=segment_revenue.index,
                color_discrete_map=colors,
                hole=0.3
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM scatter plot
        st.subheader("📊 RFM Analysis Scatter Plot")
        
        fig = px.scatter(
            rfm_df,
            x='Frequency',
            y='Monetary',
            size='Recency',
            color='Segment',
            hover_data=['customer_id'],
            title='Customer Segmentation - RFM Analysis',
            color_discrete_map=colors
        )
        fig.update_layout(height=500)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment statistics
        st.subheader("📈 Segment Statistics")
        
        segment_stats = rfm_df.groupby('Segment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        }). round(2)
        
        segment_stats.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Total Revenue ($)']
        segment_stats['Customer Count'] = rfm_df. groupby('Segment').size()
        
        st.dataframe(
            segment_stats.style.format({
                'Avg Recency (days)': '{:.0f}',
                'Avg Frequency': '{:.2f}',
                'Avg Monetary ($)': '${:,.2f}',
                'Total Revenue ($)': '${:,.2f}'
            }).background_gradient(cmap='RdYlGn_r', subset=['Avg Recency (days)']). background_gradient(cmap='Greens', subset=['Total Revenue ($)']),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("💎 Customer Lifetime Value Analysis")
        
        clv_df = calculate_clv(filtered_df)
        
        # CLV distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                clv_df,
                x='Total_Spent',
                nbins=30,
                title='Customer Lifetime Value Distribution',
                labels={'Total_Spent': 'Total Spent ($)'},
                color_discrete_sequence=['#9b59b6']
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                clv_df,
                x='Transaction_Count',
                y='Total_Spent',
                size='Avg_Order_Value',
                color='Profit_Margin_Pct',
                title='Transaction Count vs Lifetime Value',
                labels={'Transaction_Count': 'Number of Transactions', 'Total_Spent': 'Total Spent ($)'},
                color_continuous_scale='Viridis'
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Top customers
        st.subheader("🏆 Top 20 Customers by Lifetime Value")
        
        top_customers = clv_df. nlargest(20, 'Total_Spent')
        
        fig = px.bar(
            top_customers,
            x='customer_id',
            y='Total_Spent',
            color='Total_Profit',
            title='Top 20 Customers',
            labels={'Total_Spent': 'Total Spent ($)', 'customer_id': 'Customer ID'},
            color_continuous_scale='Blues'
        )
        fig.update_layout(xaxis_tickangle=-45)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # CLV summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg CLV", f"${clv_df['Total_Spent'].mean():,.2f}")
        with col2:
            st.metric("Median CLV", f"${clv_df['Total_Spent']. median():,.2f}")
        with col3:
            st. metric("Max CLV", f"${clv_df['Total_Spent'].max():,.2f}")
        with col4:
            st.metric("Avg Lifespan", f"{clv_df['Customer_Lifespan_Days'].mean():.0f} days")
    
    with tab4:
        st.subheader("📈 Cohort & Churn Analysis")
        
        # Churn analysis
        churn_threshold = st.slider("Churn Threshold (days)", 30, 180, 90)
        churn_df = identify_churned_customers(filtered_df, churn_threshold_days=churn_threshold)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            churn_rate = churn_df['Is_Churned'].mean() * 100
            st.metric("Churn Rate", f"{churn_rate:.2f}%")
        
        with col2:
            active_customers = (~churn_df['Is_Churned']).sum()
            st.metric("Active Customers", f"{active_customers:,}")
        
        with col3:
            churned_customers = churn_df['Is_Churned'].sum()
            st.metric("Churned Customers", f"{churned_customers:,}")
        
        # Churn risk distribution
        col1, col2 = st. columns(2)
        
        with col1:
            churn_risk_dist = churn_df['Churn_Risk'].value_counts()
            
            fig = px.pie(
                values=churn_risk_dist. values,
                names=churn_risk_dist.index,
                title='Customer Churn Risk Distribution',
                color_discrete_sequence=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                churn_df,
                x='Days_Since_Purchase',
                nbins=30,
                title='Days Since Last Purchase Distribution',
                labels={'Days_Since_Purchase': 'Days Since Last Purchase'},
                color_discrete_sequence=['#3498db']
            )
            fig.add_vline(x=churn_threshold, line_dash="dash", line_color="red", 
                         annotation_text="Churn Threshold")
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Cohort analysis
        st.subheader("📊 Cohort Retention Analysis")
        
        try:
            retention = cohort_analysis(filtered_df)
            
            fig = px.imshow(
                retention. values,
                labels=dict(x="Cohort Index (Months)", y="Cohort Month", color="Retention Rate (%)"),
                x=[f"Month {i}" for i in range(retention.shape[1])],
                y=[str(idx) for idx in retention.index],
                color_continuous_scale='RdYlGn',
                aspect='auto'
            )
            fig.update_layout(height=500)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning("Not enough data for cohort analysis.  Need multiple months of data.")

# ============================================================================
# PAGE 4: PRODUCT ANALYSIS
# ============================================================================
elif page == "📦 Product Analysis":
    st. title("📦 Product Performance Analysis")
    
    # Reuse product analysis from sales tab
    product_perf = product_profitability_analysis(filtered_df)
    category_perf = category_performance_analysis(filtered_df)
    
    col1, col2 = st. columns(2)
    
    with col1:
        st. metric("Total Products", f"{filtered_df['product_name'].nunique():,}")
        st.metric("Total Categories", f"{filtered_df['category'].nunique():,}")
    
    with col2:
        st.metric("Total Units Sold", f"{filtered_df['quantity']. sum():,}")
        st. metric("Avg Units/Transaction", f"{filtered_df['quantity'].mean():.2f}")
    
    st.markdown("---")
    
    # Product performance visualization
    st.subheader("Top Products Performance")
    
    top_n = st.selectbox("Number of products", [10, 20, 50, 100], index=0)
    
    top_products_df = product_perf.head(top_n)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=top_products_df['product_name'],
        y=top_products_df['Revenue'],
        name='Revenue',
        marker_color='lightblue'
    ))
    fig.add_trace(go. Bar(
        x=top_products_df['product_name'],
        y=top_products_df['Profit'],
        name='Profit',
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title=f'Top {top_n} Products: Revenue vs Profit',
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
    fig = update_chart_layout(fig)
    st.plotly_chart(fig, use_container_width=True)
    
    # Full product table
    st.subheader("📋 Complete Product Catalog Performance")
    st.dataframe(
        product_perf.style. format({
            'Revenue': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Avg_Unit_Price': '${:,.2f}',
            'Profit_Margin_Pct': '{:.2f}%'
        }).background_gradient(subset=['Revenue', 'Profit'], cmap='Greens'),
        use_container_width=True,
        height=500
    )

# ============================================================================
# PAGE 5: DEVICE & SESSION ANALYSIS
# ============================================================================
elif page == "📱 Device & Session":
    st.title("📱 Device & Session Analytics")
    
    tab1, tab2, tab3 = st.tabs(["📊 Device Performance", "⏱️ Session Behavior", "📄 Pages Viewed"])
    
    with tab1:
        st.subheader("Device Performance Analysis")
        
        device_stats = device_performance_analysis(filtered_df)
        
        # Device metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            mobile_revenue = device_stats[device_stats['device_type'] == 'Mobile']['Total_Revenue'].values[0] if 'Mobile' in device_stats['device_type'].values else 0
            st.metric("📱 Mobile Revenue", f"${mobile_revenue:,.0f}")
        
        with col2:
            desktop_revenue = device_stats[device_stats['device_type'] == 'Desktop']['Total_Revenue'].values[0] if 'Desktop' in device_stats['device_type'].values else 0
            st.metric("💻 Desktop Revenue", f"${desktop_revenue:,.0f}")
        
        with col3:
            tablet_revenue = device_stats[device_stats['device_type'] == 'Tablet']['Total_Revenue'].values[0] if 'Tablet' in device_stats['device_type'].values else 0
            st.metric("📲 Tablet Revenue", f"${tablet_revenue:,.0f}")
        
        # Device comparison charts
        col1, col2 = st. columns(2)
        
        with col1:
            fig = px.bar(
                device_stats,
                x='device_type',
                y='Total_Revenue',
                title='Revenue by Device Type',
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                text='Total_Revenue'
            )
            fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                device_stats,
                x='Avg_Session_Minutes',
                y='Avg_Order_Value',
                size='Transactions',
                color='device_type',
                title='Session Duration vs Order Value',
                labels={
                    'Avg_Session_Minutes': 'Avg Session (minutes)',
                    'Avg_Order_Value': 'Avg Order Value ($)'
                }
            )
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Device performance table
        st.subheader("📊 Detailed Device Metrics")
        st.dataframe(
            device_stats.style.format({
                'Total_Revenue': '${:,.2f}',
                'Avg_Order_Value': '${:,.2f}',
                'Avg_Session_Minutes': '{:.1f}',
                'Avg_Pages_Viewed': '{:.1f}',
                'Avg_Rating': '{:.2f}',
                'Conversion_Quality': '{:.2f}'
            }).background_gradient(subset=['Total_Revenue', 'Avg_Rating'], cmap='Greens'),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("Session Behavior Analysis")
        
        session_stats = session_behavior_analysis(filtered_df)
        
        # Session metrics
        col1, col2, col3 = st.columns(3)
        
        avg_session = filtered_df['session_duration'].mean()
        median_session = filtered_df['session_duration'].median()
        max_session = filtered_df['session_duration'].max()
        
        with col1:
            st.metric("⏱️ Avg Session", f"{avg_session:.1f} min")
        with col2:
            st.metric("📊 Median Session", f"{median_session:.1f} min")
        with col3:
            st.metric("⏰ Max Session", f"{max_session:.0f} min")
        
        # Session distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                filtered_df,
                x='session_duration',
                nbins=30,
                title='Session Duration Distribution',
                labels={'session_duration': 'Session Duration (minutes)'},
                color_discrete_sequence=['#3498db']
            )
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px. bar(
                session_stats,
                x='session_segment',
                y='Avg_Order_Value',
                title='Order Value by Session Length',
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                text='Avg_Order_Value'
            )
            fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st. plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.dataframe(
            session_stats. style.format({
                'Avg_Order_Value': '${:,.2f}',
                'Total_Revenue': '${:,.2f}',
                'Avg_Pages_Viewed': '{:.1f}',
                'Avg_Rating': '{:.2f}',
                'Avg_Discount': '${:.2f}'
            }),
            use_container_width=True
        )
    
    with tab3:
        st. subheader("Pages Viewed Analysis")
        
        # Pages viewed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_pages = filtered_df['pages_viewed'].mean()
        median_pages = filtered_df['pages_viewed'].median()
        max_pages = filtered_df['pages_viewed'].max()
        pages_revenue_corr = filtered_df['pages_viewed'].corr(filtered_df['total_amount'])
        
        with col1:
            st.metric("📄 Avg Pages", f"{avg_pages:.1f}")
        with col2:
            st. metric("📊 Median Pages", f"{median_pages:.0f}")
        with col3:
            st.metric("📚 Max Pages", f"{max_pages:.0f}")
        with col4:
            st.metric("🔗 Revenue Correlation", f"{pages_revenue_corr:.3f}")
        
        # Pages viewed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                filtered_df,
                x='pages_viewed',
                y='total_amount',
                trendline='ols',
                title='Pages Viewed vs Order Value',
                labels={'pages_viewed': 'Pages Viewed', 'total_amount': 'Order Value ($)'},
                opacity=0.5
            )
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            pages_groups = filtered_df.groupby(pd.cut(filtered_df['pages_viewed'], bins=[0, 5, 10, 15, 50])).agg({
                'total_amount': 'mean',
                'rating': 'mean',
                'transaction_id': 'count'
            }).reset_index()
            pages_groups.columns = ['Pages_Range', 'Avg_Order_Value', 'Avg_Rating', 'Count']
            pages_groups['Pages_Range'] = pages_groups['Pages_Range'].astype(str)
            
            fig = px.bar(
                pages_groups,
                x='Pages_Range',
                y='Avg_Order_Value',
                title='Order Value by Pages Viewed Range',
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                text='Avg_Order_Value'
            )
            fig.update_traces(texttemplate='$%{text:.0f}', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 6: SATISFACTION ANALYSIS
# ============================================================================
elif page == "⭐ Satisfaction":
    st.title("⭐ Customer Satisfaction Analysis")
    
    tab1, tab2, tab3 = st.tabs(["📊 Rating Overview", "🔍 Factors Analysis", "📈 Trends"])
    
    with tab1:
        st.subheader("Customer Rating Overview")
        
        rating_stats = rating_analysis(filtered_df)
        
        # Rating metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_rating = filtered_df['rating'].mean()
        rating_5_pct = (filtered_df['rating'] == 5).sum() / len(filtered_df) * 100
        rating_low_pct = (filtered_df['rating'] <= 2).sum() / len(filtered_df) * 100
        nps_score = rating_5_pct - rating_low_pct
        
        with col1:
            st.metric("⭐ Average Rating", f"{avg_rating:.2f}/5. 0")
        with col2:
            st.metric("🌟 5-Star Rate", f"{rating_5_pct:.1f}%")
        with col3:
            st.metric("⚠️ Low Rating Rate", f"{rating_low_pct:.1f}%")
        with col4:
            st.metric("📊 NPS Score", f"{nps_score:.1f}")
        
        # Rating distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            colors = ['#e74c3c', '#e67e22', '#f39c12', '#3498db', '#2ecc71']
            fig.add_trace(go.Bar(
                x=rating_stats['rating'],
                y=rating_stats['Transaction_Count'],
                marker_color=colors,
                text=rating_stats['Percentage']. apply(lambda x: f"{x:.1f}%"),
                textposition='auto',
                name='Count'
            ))
            fig. update_layout(
                title='Rating Distribution',
                xaxis_title='Rating (Stars)',
                yaxis_title='Number of Transactions',
                height=400
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                rating_stats,
                values='Transaction_Count',
                names='rating',
                title='Rating Proportion',
                color='rating',
                color_discrete_map={1: '#e74c3c', 2: '#e67e22', 3: '#f39c12', 4: '#3498db', 5: '#2ecc71'},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st. plotly_chart(fig, use_container_width=True)
        
        # Rating by category
        st.subheader("📊 Rating by Category")
        
        category_ratings = filtered_df.groupby('category'). agg({
            'rating': 'mean',
            'transaction_id': 'count'
        }).sort_values('rating', ascending=False). reset_index()
        category_ratings.columns = ['category', 'avg_rating', 'count']
        
        fig = px.bar(
            category_ratings,
            x='category',
            y='avg_rating',
            title='Average Rating by Category',
            color='avg_rating',
            color_continuous_scale='RdYlGn',
            range_color=[1, 5],
            text='avg_rating'
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig. update_layout(height=400, xaxis_tickangle=-45)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Rating statistics table
        st.dataframe(
            rating_stats. style.format({
                'Total_Revenue': '${:,.2f}',
                'Avg_Discount': '${:,.2f}',
                'Avg_Delivery_Days': '{:.1f}',
                'Avg_Session_Minutes': '{:.1f}',
                'Percentage': '{:.2f}%'
            }).background_gradient(subset=['Total_Revenue'], cmap='Greens'),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("🔍 Factors Affecting Satisfaction")
        
        # Delivery time vs rating
        delivery_satisfaction = delivery_satisfaction_analysis(filtered_df)
        
        col1, col2 = st. columns(2)
        
        with col1:
            fig = px.bar(
                delivery_satisfaction,
                x='delivery_segment',
                y='Avg_Rating',
                title='Rating by Delivery Speed',
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                text='Avg_Rating'
            )
            fig. update_traces(texttemplate='%{text:.2f}', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                filtered_df,
                x='delivery_days',
                y='rating',
                trendline='ols',
                title='Delivery Time vs Rating',
                labels={'delivery_days': 'Delivery Days', 'rating': 'Rating'},
                opacity=0.5
            )
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Device vs rating
        st.subheader("📱 Device Impact on Satisfaction")
        
        device_rating = filtered_df.groupby('device_type')['rating'].mean().sort_values(ascending=False)
        
        fig = px. bar(
            x=device_rating.index,
            y=device_rating.values,
            title='Average Rating by Device Type',
            labels={'x': 'Device', 'y': 'Avg Rating'},
            color=device_rating.values,
            color_continuous_scale='RdYlGn',
            text=device_rating.values
        )
        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(height=400)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔗 Correlation Analysis")
        
        corr_data = filtered_df[['rating', 'total_amount', 'discount', 'delivery_days', 'session_duration', 'pages_viewed']].corr()
        
        fig = px.imshow(
            corr_data,
            title='Satisfaction Factors Correlation',
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto='. 2f'
        )
        fig.update_layout(height=500)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st. subheader("📈 Rating Trends Over Time")
        
        # Monthly rating trend
        monthly_rating = filtered_df.groupby(filtered_df['transaction_date'].dt.to_period('M')).agg({
            'rating': 'mean',
            'transaction_id': 'count'
        }).reset_index()
        monthly_rating['transaction_date'] = monthly_rating['transaction_date'].dt.to_timestamp()
        monthly_rating. columns = ['date', 'avg_rating', 'count']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_rating['date'],
            y=monthly_rating['avg_rating'],
            mode='lines+markers',
            name='Avg Rating',
            line=dict(color='#2ecc71', width=3),
            marker=dict(size=8)
        ))
        
        fig. add_hline(y=filtered_df['rating'].mean(), line_dash="dash", 
                     line_color="red", annotation_text="Overall Average")
        
        fig. update_layout(
            title='Monthly Average Rating Trend',
            xaxis_title='Month',
            yaxis_title='Average Rating',
            height=400,
            hovermode='x unified'
        )
        fig = update_chart_layout(fig)
        st. plotly_chart(fig, use_container_width=True)
        
        # Low rating alerts
        st.subheader("⚠️ Low Rating Alerts")
        
        low_ratings = filtered_df[filtered_df['rating'] <= 2]. copy()
        
        if len(low_ratings) > 0:
            st.warning(f"⚠️ Found {len(low_ratings)} transactions with ratings ≤ 2 stars")
            
            col1, col2 = st. columns(2)
            
            with col1:
                low_by_category = low_ratings['category'].value_counts()
                st.write("**Low Ratings by Category:**")
                st. dataframe(low_by_category, use_container_width=True)
            
            with col2:
                low_by_city = low_ratings['city'].value_counts()
                st.write("**Low Ratings by City:**")
                st.dataframe(low_by_city, use_container_width=True)
        else:
            st.success("✅ No low ratings in filtered data!")

# ============================================================================
# PAGE 7: DELIVERY PERFORMANCE
# ============================================================================
elif page == "🚚 Delivery Performance":
    st.title("🚚 Delivery Performance Analysis")
    
    tab1, tab2, tab3 = st. tabs(["📊 Overview", "🌍 City Analysis", "⏱️ Speed Analysis"])
    
    with tab1:
        st.subheader("Delivery Performance Overview")
        
        # Delivery metrics
        col1, col2, col3, col4 = st.columns(4)
        
        avg_delivery = filtered_df['delivery_days']. mean()
        median_delivery = filtered_df['delivery_days']. median()
        fast_delivery_pct = (filtered_df['delivery_days'] <= 3).sum() / len(filtered_df) * 100
        slow_delivery_pct = (filtered_df['delivery_days'] > 7).sum() / len(filtered_df) * 100
        
        with col1:
            st.metric("📦 Avg Delivery", f"{avg_delivery:.1f} days")
        with col2:
            st.metric("📊 Median Delivery", f"{median_delivery:.0f} days")
        with col3:
            st.metric("⚡ Fast Delivery Rate", f"{fast_delivery_pct:.1f}%")
        with col4:
            st.metric("🐌 Slow Delivery Rate", f"{slow_delivery_pct:.1f}%")
        
        # Delivery distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                filtered_df,
                x='delivery_days',
                nbins=30,
                title='Delivery Time Distribution',
                labels={'delivery_days': 'Delivery Days'},
                color_discrete_sequence=['#3498db']
            )
            fig.add_vline(x=avg_delivery, line_dash="dash", line_color="red",
                         annotation_text=f"Avg: {avg_delivery:.1f}d")
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            delivery_satisfaction = delivery_satisfaction_analysis(filtered_df)
            
            fig = px.bar(
                delivery_satisfaction,
                x='delivery_segment',
                y='Avg_Rating',
                title='Customer Satisfaction by Delivery Speed',
                color='Avg_Rating',
                color_continuous_scale='RdYlGn',
                range_color=[1, 5],
                text='Avg_Rating'
            )
            fig. update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st. plotly_chart(fig, use_container_width=True)
        
        # Delivery by shipping method
        st.subheader("📦 Delivery by Shipping Method")
        
        shipping_perf = filtered_df.groupby('shipping_method').agg({
            'delivery_days': ['mean', 'min', 'max'],
            'rating': 'mean',
            'transaction_id': 'count'
        }).round(2)
        shipping_perf.columns = ['Avg_Days', 'Min_Days', 'Max_Days', 'Avg_Rating', 'Count']
        
        st.dataframe(
            shipping_perf.style. format({
                'Avg_Days': '{:.1f}',
                'Avg_Rating': '{:.2f}'
            }).background_gradient(subset=['Avg_Rating'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab2:
        st.subheader("🌍 Delivery Performance by City")
        
        city_delivery = filtered_df.groupby('city').agg({
            'delivery_days': ['mean', 'median'],
            'rating': 'mean',
            'transaction_id': 'count'
        }).round(2)
        city_delivery.columns = ['Avg_Delivery_Days', 'Median_Delivery_Days', 'Avg_Rating', 'Orders']
        city_delivery = city_delivery.sort_values('Avg_Delivery_Days'). reset_index()
        
        # City delivery comparison
        fig = px.bar(
            city_delivery,
            x='city',
            y='Avg_Delivery_Days',
            title='Average Delivery Time by City',
            color='Avg_Rating',
            color_continuous_scale='RdYlGn',
            text='Avg_Delivery_Days'
        )
        fig.update_traces(texttemplate='%{text:.1f}d', textposition='outside')
        fig.update_layout(height=400, xaxis_tickangle=-45)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # City performance table
        st.subheader("📊 Detailed City Metrics")
        st.dataframe(
            city_delivery.style.format({
                'Avg_Delivery_Days': '{:.1f}',
                'Median_Delivery_Days': '{:.1f}',
                'Avg_Rating': '{:.2f}'
            }).background_gradient(subset=['Avg_Rating'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    with tab3:
        st.subheader("⏱️ Delivery Speed Analysis")
        
        delivery_satisfaction = delivery_satisfaction_analysis(filtered_df)
        
        col1, col2 = st. columns(2)
        
        with col1:
            fig = px.pie(
                delivery_satisfaction,
                values='Transaction_Count',
                names='delivery_segment',
                title='Orders by Delivery Speed',
                color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c'],
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                delivery_satisfaction,
                x='delivery_segment',
                y='Returning_Customer_Rate_%',
                title='Returning Customer Rate by Delivery Speed',
                color='Returning_Customer_Rate_%',
                color_continuous_scale='Greens',
                text='Returning_Customer_Rate_%'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Delivery speed table
        st.dataframe(
            delivery_satisfaction.style.format({
                'Avg_Rating': '{:.2f}',
                'Avg_Order_Value': '${:,.2f}',
                'Returning_Customer_Rate_%': '{:.1f}%'
            }).background_gradient(subset=['Avg_Rating'], cmap='RdYlGn'),
            use_container_width=True
        )
# ============================================================================
# PAGE 8: LOYALTY ANALYSIS
# ============================================================================
elif page == "🔄 Loyalty Analysis":
    st.title("🔄 Customer Loyalty & Retention Analysis")
    
    tab1, tab2, tab3 = st. tabs(["📊 Returning vs New", "💎 Loyalty Drivers", "🎯 Recommendations"])
    
    with tab1:
        st.subheader("Returning vs New Customer Analysis")
        
        returning_stats = returning_customer_analysis(filtered_df)
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        returning_rate = (filtered_df['is_returning'] == True).sum() / len(filtered_df) * 100
        new_customers = (filtered_df['is_returning'] == False).sum()
        returning_customers = (filtered_df['is_returning'] == True).sum()
        
        with col1:
            st.metric("🔄 Returning Rate", f"{returning_rate:.1f}%")
        with col2:
            st. metric("🆕 New Customers", f"{new_customers:,}")
        with col3:
            st.metric("💚 Returning Customers", f"{returning_customers:,}")
        
        # Comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                returning_stats,
                x='index',
                y='Avg_Order_Value',
                title='Average Order Value Comparison',
                color='index',
                color_discrete_map={'New Customer': '#e74c3c', 'Returning Customer': '#2ecc71'},
                text='Avg_Order_Value'
            )
            fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
            fig. update_layout(height=400, showlegend=False)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                returning_stats,
                x='index',
                y='Avg_Rating',
                title='Average Rating Comparison',
                color='index',
                color_discrete_map={'New Customer': '#e74c3c', 'Returning Customer': '#2ecc71'},
                text='Avg_Rating'
            )
            fig.update_traces(texttemplate='%{text:.2f}⭐', textposition='outside')
            fig.update_layout(height=400, showlegend=False)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed comparison
        st.subheader("📊 Detailed Comparison")
        
        st.dataframe(
            returning_stats.style.format({
                'Avg_Order_Value': '${:,.2f}',
                'Total_Revenue': '${:,.2f}',
                'Avg_Rating': '{:.2f}',
                'Avg_Session_Minutes': '{:.1f}',
                'Avg_Pages_Viewed': '{:.1f}',
                'Avg_Discount': '${:.2f}'
            }).background_gradient(subset=['Total_Revenue', 'Avg_Rating'], cmap='Greens'),
            use_container_width=True
        )
        
        # Revenue distribution
        st.subheader("💰 Revenue Distribution")
        
        fig = px.pie(
            returning_stats,
            values='Total_Revenue',
            names='index',
            title='Revenue Share: New vs Returning Customers',
            color='index',
            color_discrete_map={'New Customer': '#e74c3c', 'Returning Customer': '#2ecc71'},
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💎 What Drives Customer Loyalty? ")
        
        # Analyze factors that correlate with returning
        returning_by_category = filtered_df.groupby('category'). agg({
            'is_returning': lambda x: (x == True).sum() / len(x) * 100,
            'transaction_id': 'count'
        }).round(2)
        returning_by_category.columns = ['Returning_Rate_%', 'Total_Orders']
        returning_by_category = returning_by_category.sort_values('Returning_Rate_%', ascending=False). reset_index()
        
        col1, col2 = st. columns(2)
        
        with col1:
            fig = px.bar(
                returning_by_category,
                x='category',
                y='Returning_Rate_%',
                title='Returning Customer Rate by Category',
                color='Returning_Rate_%',
                color_continuous_scale='Greens',
                text='Returning_Rate_%'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400, xaxis_tickangle=-45)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Device loyalty
            returning_by_device = filtered_df.groupby('device_type').agg({
                'is_returning': lambda x: (x == True).sum() / len(x) * 100
            }).round(2)
            returning_by_device.columns = ['Returning_Rate_%']
            returning_by_device = returning_by_device.sort_values('Returning_Rate_%', ascending=False).reset_index()
            
            fig = px. bar(
                returning_by_device,
                x='device_type',
                y='Returning_Rate_%',
                title='Returning Customer Rate by Device',
                color='Returning_Rate_%',
                color_continuous_scale='Greens',
                text='Returning_Rate_%'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # Rating impact
        st.subheader("⭐ Rating Impact on Loyalty")
        
        loyalty_by_rating = filtered_df.groupby('rating').agg({
            'is_returning': lambda x: (x == True).sum() / len(x) * 100,
            'transaction_id': 'count'
        }).round(2)
        loyalty_by_rating.columns = ['Returning_Rate_%', 'Count']
        loyalty_by_rating = loyalty_by_rating.reset_index()
        
        fig = px.line(
            loyalty_by_rating,
            x='rating',
            y='Returning_Rate_%',
            title='Loyalty Rate by Customer Rating',
            markers=True,
            text='Returning_Rate_%'
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='top center',
                         line_color='#2ecc71', line_width=3, marker=dict(size=12))
        fig.update_layout(height=400)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
        
        # Discount impact
        st.subheader("💰 Discount Impact on Loyalty")
        
        # Compare discount usage
        discount_comparison = filtered_df.groupby('is_returning').agg({
            'discount': 'mean',
            'total_amount': 'mean'
        }). round(2)
        discount_comparison.index = ['New Customer', 'Returning Customer']
        discount_comparison = discount_comparison.reset_index()
        discount_comparison. columns = ['Customer_Type', 'Avg_Discount', 'Avg_Order_Value']
        
        fig = px.bar(
            discount_comparison,
            x='Customer_Type',
            y='Avg_Discount',
            title='Average Discount: New vs Returning',
            color='Customer_Type',
            color_discrete_map={'New Customer': '#e74c3c', 'Returning Customer': '#2ecc71'},
            text='Avg_Discount'
        )
        fig.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        fig.update_layout(height=400, showlegend=False)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Loyalty Improvement Recommendations")
        
        # Calculate key insights
        new_customer_avg = returning_stats[returning_stats['index'] == 'New Customer']['Avg_Order_Value'].values[0]
        returning_customer_avg = returning_stats[returning_stats['index'] == 'Returning Customer']['Avg_Order_Value'].values[0]
        value_difference = returning_customer_avg - new_customer_avg
        
        new_rating = returning_stats[returning_stats['index'] == 'New Customer']['Avg_Rating'].values[0]
        returning_rating = returning_stats[returning_stats['index'] == 'Returning Customer']['Avg_Rating'].values[0]
        
        col1, col2 = st. columns(2)
        
        with col1:
            st. markdown(f"""
            <div class="highlight-box">
                <h4>📈 Value Opportunity:</h4>
                <p>Returning customers spend <strong>${value_difference:.2f} more</strong> per order</p>
                <p>Improving retention by 10% could generate:</p>
                <p><strong>${(returning_customers * 0.1 * returning_customer_avg):,.2f}</strong> additional revenue</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="highlight-box">
                <h4>⭐ Quality Opportunity:</h4>
                <p>Returning customers rate <strong>{returning_rating - new_rating:.2f} stars higher</strong></p>
                <p>Focus on first-time experience to boost loyalty</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("💡 Actionable Recommendations")
        
        col1, col2 = st. columns(2)
        
        with col1:
            st. markdown("""
            <div class="highlight-box">
                <h4>🎯 Retention Strategies:</h4>
                <ol>
                    <li><strong>Welcome Program:</strong> Special discount for first purchase</li>
                    <li><strong>Loyalty Points:</strong> Reward returning customers</li>
                    <li><strong>Email Campaigns:</strong> Re-engage after 30 days</li>
                    <li><strong>Personalization:</strong> Recommend based on past purchases</li>
                    <li><strong>VIP Program:</strong> Benefits for top customers</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="highlight-box">
                <h4>🚀 Quick Wins:</h4>
                <ol>
                    <li><strong>Improve Delivery:</strong> Fast shipping increases loyalty</li>
                    <li><strong>Follow-up:</strong> Ask for feedback after first purchase</li>
                    <li><strong>Quality Focus:</strong> High ratings correlate with loyalty</li>
                    <li><strong>Mobile Experience:</strong> Optimize for mobile users</li>
                    <li><strong>Customer Support:</strong> Resolve issues quickly</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
        
        # Category-specific recommendations
        st.subheader("📦 Category-Specific Insights")
        
        top_loyalty_category = returning_by_category.iloc[0]
        low_loyalty_category = returning_by_category.iloc[-1]
        
        st.markdown(f"""
        <div class="highlight-box">
            <h4>📊 Category Performance:</h4>
            <p><strong>Best Loyalty:</strong> {top_loyalty_category['category']} ({top_loyalty_category['Returning_Rate_%']:.1f}% returning rate)</p>
            <p><strong>Needs Improvement:</strong> {low_loyalty_category['category']} ({low_loyalty_category['Returning_Rate_%']:.1f}% returning rate)</p>
            <p><strong>Action:</strong> Apply successful strategies from {top_loyalty_category['category']} to {low_loyalty_category['category']}</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# PAGE 9: ADVANCED ANALYTICS
# ============================================================================
elif page == "🔮 Advanced Analytics":
    st.title("🔮 Advanced Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["🛒 Market Basket", "📊 K-Means Clustering", "📈 Forecasting"])
    
    with tab1:
        st.subheader("🛒 Market Basket Analysis")
        st.info("Discovering which product categories are frequently purchased together")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            min_support = st.slider("Minimum Support", 0.001, 0.1, 0.01, 0.001)
        with col2:
            min_confidence = st.slider("Minimum Confidence", 0.1, 0.9, 0.3, 0.1)
        
        mba_df = market_basket_analysis(filtered_df, min_support=min_support, min_confidence=min_confidence)
        
        if not mba_df.empty:
            st.subheader("🔝 Top Product Associations")
            
            top_mba = mba_df.head(10)
            
            fig = px.bar(
                top_mba,
                x='Lift',
                y=[f"{row['Item_1']} + {row['Item_2']}" for _, row in top_mba.iterrows()],
                orientation='h',
                title='Top 10 Product Associations by Lift',
                labels={'y': 'Product Pair', 'Lift': 'Lift Score'},
                color='Lift',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=500)
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed table
            st.subheader("📋 Association Rules Table")
            st.dataframe(
                mba_df.style.format({
                    'Support': '{:.4f}',
                    'Confidence_1_to_2': '{:.4f}',
                    'Confidence_2_to_1': '{:.4f}',
                    'Lift': '{:.4f}'
                }).background_gradient(subset=['Lift'], cmap='Greens'),
                use_container_width=True
            )
        else:
            st.warning("No associations found with current parameters.  Try lowering the thresholds.")
    
    with tab2:
        st.subheader("📊 K-Means Customer Clustering")
        
        n_clusters = st.slider("Number of Clusters", 2, 8, 4)
        
        clusters_df, cluster_centers = perform_kmeans_clustering(filtered_df, n_clusters=n_clusters)
        
        # Cluster centers
        st.subheader("🎯 Cluster Centers")
        st.dataframe(
            cluster_centers.style.format('${:,.2f}').background_gradient(cmap='Blues'),
            use_container_width=True
        )
        
        # Cluster distribution
        col1, col2 = st. columns(2)
        
        with col1:
            cluster_dist = clusters_df['Cluster'].value_counts().sort_index()
            
            fig = px.bar(
                x=[f"Cluster {i}" for i in cluster_dist.index],
                y=cluster_dist.values,
                title='Customer Distribution by Cluster',
                labels={'x': 'Cluster', 'y': 'Number of Customers'},
                color=cluster_dist.values,
                color_continuous_scale='Viridis'
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster pie
            fig = px.pie(
                values=cluster_dist.values,
                names=[f"Cluster {i}" for i in cluster_dist.index],
                title='Cluster Proportion',
                hole=0.3
            )
            fig = update_chart_layout(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        # 3D scatter
        st.subheader("📊 3D Cluster Visualization")
        
        fig = px.scatter_3d(
            clusters_df,
            x='Total_Spent',
            y='Transaction_Count',
            z='Total_Quantity',
            color='Cluster',
            title='Customer Clusters - 3D View',
            labels={
                'Total_Spent': 'Total Spent ($)',
                'Transaction_Count': 'Transaction Count',
                'Total_Quantity': 'Total Quantity'
            },
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=600)
        fig = update_chart_layout(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Advanced Sales Forecasting")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_method = st.selectbox("Forecasting Method", ["ARIMA", "Moving Average", "Exponential Smoothing"])
        
        with col2:
            forecast_period = st.selectbox("Forecast Period", ["Monthly", "Weekly"])
        
        with col3:
            periods_ahead = st.slider("Periods to Forecast", 3, 12, 6)
        
        freq = 'M' if forecast_period == "Monthly" else 'W'
        
        if forecast_method == "ARIMA":
            st.info("🤖 Using ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                p = st.number_input("AR Order (p)", 0, 5, 1)
            with col2:
                d = st. number_input("Differencing (d)", 0, 2, 1)
            with col3:
                q = st.number_input("MA Order (q)", 0, 5, 1)
            
            try:
                from src.sales_analysis import arima_sales_forecast
                
                with st.spinner("Training ARIMA model..."):
                    historical, forecast_df, summary = arima_sales_forecast(
                        filtered_df,
                        freq=freq,
                        periods=periods_ahead,
                        order=(p, d, q)
                    )
                
                if historical is not None and forecast_df is not None: 
                    fig = go.Figure()
                    
                    fig.add_trace(go. Scatter(
                        x=historical['Date'],
                        y=historical['Revenue'],
                        mode='lines',
                        name='Historical Revenue',
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Revenue'],
                        mode='lines+markers',
                        name='ARIMA Forecast',
                        line=dict(color='#e74c3c', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=pd.concat([forecast_df['Date'], forecast_df['Date'][: :-1]]),
                        y=pd. concat([forecast_df['Upper_CI'], forecast_df['Lower_CI'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(231, 76, 60, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='95% Confidence Interval',
                        showlegend=True
                    ))
                    
                    fig.update_layout(
                        title=f'ARIMA({p},{d},{q}) Sales Forecast - {forecast_period}',
                        xaxis_title='Date',
                        yaxis_title='Revenue ($)',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    fig = update_chart_layout(fig)
                    st. plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Latest Actual", f"${historical['Revenue']. iloc[-1]:,.0f}")
                    with col2:
                        st.metric("First Forecast", f"${forecast_df['Revenue'].iloc[0]: ,.0f}")
                    with col3:
                        st. metric("Avg Forecast", f"${forecast_df['Revenue'].mean():,.0f}")
                    with col4:
                        growth = ((forecast_df['Revenue'].iloc[0] - historical['Revenue'].iloc[-1]) / historical['Revenue']. iloc[-1] * 100)
                        st.metric("Forecast Growth", f"{growth:+.1f}%")
                    
                    st.subheader("📊 Model Performance")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("AIC Score", f"{summary['AIC']:.2f}")
                    with col2:
                        st. metric("BIC Score", f"{summary['BIC']:.2f}")
                    with col3:
                        st.metric("RMSE", f"${summary['RMSE']: ,.2f}")
                    
                    st.subheader("📋 Forecast Values")
                    display_forecast = forecast_df. copy()
                    display_forecast['Date'] = display_forecast['Date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(
                        display_forecast.style. format({
                            'Revenue':  '${:,.2f}',
                            'Lower_CI': '${:,.2f}',
                            'Upper_CI': '${:,.2f}'
                        }),
                        use_container_width=True
                    )
                else:
                    st.error("ARIMA modeling failed. Try different parameters or use Moving Average method.")
                    
            except Exception as e:
                st.error(f"Error running ARIMA:  {str(e)}")
                st.info("Try adjusting parameters or switching to Moving Average method.")
        
        elif forecast_method == "Exponential Smoothing":
            st.info("📊 Using Exponential Smoothing for trend-based forecasting")
            
            try:
                from src.sales_analysis import exponential_smoothing_forecast
                
                with st.spinner("Calculating exponential smoothing forecast..."):
                    historical, forecast_df = exponential_smoothing_forecast(
                        filtered_df,
                        freq=freq,
                        periods=periods_ahead
                    )
                
                if historical is not None and forecast_df is not None:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical['Date'],
                        y=historical['Revenue'],
                        mode='lines',
                        name='Historical Revenue',
                        line=dict(color='#3498db', width=2)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['Revenue'],
                        mode='lines+markers',
                        name='ES Forecast',
                        line=dict(color='#2ecc71', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f'Exponential Smoothing Forecast - {forecast_period}',
                        xaxis_title='Date',
                        yaxis_title='Revenue ($)',
                        height=500,
                        hovermode='x unified'
                    )
                    
                    fig = update_chart_layout(fig)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st. metric("Latest Revenue", f"${historical['Revenue']. iloc[-1]:,.0f}")
                    with col2:
                        st.metric("Avg Forecast", f"${forecast_df['Revenue'].mean():,.0f}")
                    with col3:
                        growth = ((forecast_df['Revenue']. iloc[0] - historical['Revenue'].iloc[-1]) / historical['Revenue'].iloc[-1] * 100)
                        st.metric("Forecast Growth", f"{growth:+.1f}%")
                else:
                    st.error("Exponential Smoothing failed.")
            except Exception as e:
                st. error(f"Error:  {str(e)}")
        
        else:  # Moving Average
            st.info("📈 Simple moving average forecast for revenue trends")
            
            window = st.slider("Moving Average Window", 3, 12, 6)
            
            trends = sales_trends_analysis(filtered_df, freq=freq)
            trends['MA'] = trends['Revenue'].rolling(window=window).mean()
            last_ma = trends['MA'].iloc[-1]
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=trends['transaction_date'],
                y=trends['Revenue'],
                mode='lines',
                name='Actual Revenue',
                line=dict(color='#3498db', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=trends['transaction_date'],
                y=trends['MA'],
                mode='lines',
                name=f'{window}-Period Moving Average',
                line=dict(color='#e74c3c', width=2, dash='dash')
            ))
            
            fig. update_layout(
                title=f'{forecast_period} Revenue with Moving Average Forecast',
                xaxis_title='Date',
                yaxis_title='Revenue ($)',
                height=500,
                hovermode='x unified'
            )
            
            fig = update_chart_layout(fig)
            st. plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Latest Revenue", f"${trends['Revenue']. iloc[-1]:,.0f}")
            with col2:
                st.metric("Moving Avg Forecast", f"${last_ma:,.0f}")
            with col3:
                diff_pct = ((last_ma - trends['Revenue'].iloc[-1]) / trends['Revenue'].iloc[-1] * 100)
                st. metric("Forecast vs Actual", f"{diff_pct:+.1f}%")

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: #7f8c8d; padding: 20px;'>
        <p><strong>E-commerce Analytics Pro Dashboard</strong> | Built with Streamlit by <strong>Arpit0523</strong></p>
        <p>📊 Data Analytics • 🤖 Machine Learning • 📈 Business Intelligence</p>
        <p>Analyzing {len(df):,} transactions from {df['transaction_date'].min(). date()} to {df['transaction_date'].max().date()}</p>
    </div>
    """,
    unsafe_allow_html=True
)         