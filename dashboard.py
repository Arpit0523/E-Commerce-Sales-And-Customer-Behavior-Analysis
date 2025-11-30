#!/usr/bin/env python3
"""
Comprehensive E-commerce Analytics Dashboard
Multi-page Streamlit application with advanced analytics

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
    identify_churned_customers,
    customer_purchase_patterns
)
from src.sales_analysis import (
    sales_trends_analysis,
    category_performance_analysis,
    product_profitability_analysis,
    market_basket_analysis,
    payment_shipping_analysis
)

# Page configuration
st.set_page_config(
    page_title="E-commerce Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with modern color scheme
st.markdown("""
    <style>
    :root {
        --primary-color: #0066CC;
        --secondary-color: #00A3E0;
        --accent-color: #00C9A7;
        --success-color: #00D9A3;
        --warning-color: #FFA500;
        --danger-color: #E74C3C;
        --dark-text: #1A202C;
        --light-bg: #F7FAFC;
    }
    .main {
        padding: 0rem 1rem;
        background-color: var(--light-bg);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,102,204,0.1);
        border-left: 4px solid var(--primary-color);
    }
    .stMetric label {
        color: #1A202C !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #000000 !important;
        font-weight: 700 !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        color: #2D3748 !important;
    }
    h1 {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    h2 {
        color: #F7FAFC !important;
        font-weight: 600 !important;
    }
    h3 {
        color: #E2E8F0 !important;
        font-weight: 600 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: #ffffff;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #EDF2F7;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        color: var(--dark-text);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #E2E8F0;
        transform: translateY(-2px);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0066CC 0%, #00A3E0 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0,102,204,0.3);
    }
    </style>
    """, unsafe_allow_html=True)

# Load data with caching
@st.cache_data
def load_data():
    """Load the master dataset"""
    try:
        df = load_master()
        return df
    except FileNotFoundError:
        st. error("⚠️ Master dataset not found!  Please run: python src/data_processing. py")
        st.stop()

# Load data
df = load_data()

# Sidebar Navigation
with st.sidebar:
    st. image("https://img.icons8.com/clouds/100/000000/shop.png", width=100)
    st.title("🛍️ E-commerce Analytics")
    
    # Page selection
    page = st.radio(
        "Navigation",
        ["📊 Overview", "📈 Sales Analysis", "👥 Customer Insights", 
         "📦 Product Analysis", "🔮 Advanced Analytics"],
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
    
    # Country filter
    countries = ['All'] + sorted(df['country'].dropna().unique().tolist())
    selected_country = st.selectbox("Country", countries, index=0)
    
    # Gender filter
    genders = ['All'] + sorted(df['gender']. dropna().unique().tolist())
    selected_gender = st. selectbox("Gender", genders)
    
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

if selected_country != 'All':
    filtered_df = filtered_df[filtered_df['country'] == selected_country]

if selected_gender != 'All':
    filtered_df = filtered_df[filtered_df['gender'] == selected_gender]

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "📊 Overview":
    st.title("📊 Executive Dashboard Overview")
    st.markdown("### Welcome to the E-commerce Analytics Dashboard")
    
    # Key Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_revenue = filtered_df['total_amount'].sum()
    total_transactions = len(filtered_df)
    total_customers = filtered_df['customer_id'].nunique()
    avg_order_value = filtered_df['total_amount'].mean()
    total_profit = filtered_df['profit'].sum()
    
    with col1:
        st.metric(
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
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        st.metric(
            label="💵 Profit Margin",
            value=f"{profit_margin:.1f}%",
            delta=f"${total_profit:,.0f} profit"
        )
    
    st.markdown("---")
    
    # Two column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Trend Over Time")
        
        # Monthly revenue
        monthly_rev = filtered_df.groupby(filtered_df['transaction_date']. dt.to_period('M'))['total_amount'].sum(). reset_index()
        monthly_rev['transaction_date'] = monthly_rev['transaction_date'].dt.to_timestamp()
        
        fig = px.line(
            monthly_rev,
            x='transaction_date',
            y='total_amount',
            title='Monthly Revenue',
            labels={'total_amount': 'Revenue ($)', 'transaction_date': 'Date'}
        )
        fig.update_traces(line_color='#0066CC', line_width=3, fill='tozeroy', fillcolor='rgba(0,102,204,0.1)')
        fig.update_layout(
            hovermode='x unified',
            plot_bgcolor='#F7FAFC',
            paper_bgcolor='white',
            height=400,
            font=dict(color='#1A202C')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Revenue by Category")
        
        category_revenue = filtered_df.groupby('category')['total_amount'].sum().sort_values(ascending=False)
        
        fig = px.pie(
            values=category_revenue.values,
            names=category_revenue.index,
            title='Category Distribution',
            hole=0.4,
            color_discrete_sequence=['#0066CC', '#00A3E0', '#00C9A7', '#7B68EE', '#FF6B9D', '#FFA500', '#FFD93D']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col1, col2 = st. columns(2)
    
    with col1:
        st. subheader("🌍 Top 10 Countries by Revenue")
        
        country_revenue = filtered_df.groupby('country')['total_amount'].sum().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=country_revenue.values,
            y=country_revenue.index,
            orientation='h',
            title='Revenue by Country',
            labels={'x': 'Revenue ($)', 'y': 'Country'},
            color=country_revenue.values,
            color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#0066CC'], [1, '#003D7A']]
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💳 Payment Method Distribution")
        
        payment_dist = filtered_df['payment_method'].value_counts()
        
        fig = px. bar(
            x=payment_dist.index,
            y=payment_dist.values,
            title='Transactions by Payment Method',
            labels={'x': 'Payment Method', 'y': 'Count'},
            color=payment_dist.values,
            color_continuous_scale=[[0, '#B3E5FC'], [0.5, '#00A3E0'], [1, '#0066CC']]
        )
        fig. update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick Stats
    st.markdown("---")
    st.subheader("📊 Quick Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        top_category = filtered_df.groupby('category')['total_amount'].sum().idxmax()
        st.metric("🏆 Top Category", top_category)
    
    with col2:
        popular_payment = filtered_df['payment_method'].mode()[0]
        st.metric("🌟 Most Popular Payment", popular_payment)
    
    with col3:
        repeat_customers = filtered_df.groupby('customer_id').size()
        repeat_rate = (repeat_customers > 1).sum() / len(repeat_customers) * 100
        st.metric("🔁 Repeat Customer Rate", f"{repeat_rate:.1f}%")
    
    with col4:
        avg_items = filtered_df['quantity'].mean()
        st.metric("📦 Avg Items/Order", f"{avg_items:.1f}")

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
                line=dict(color='#0066CC', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,102,204,0.2)',
                marker=dict(size=6, color='#0066CC')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=trends['transaction_date'], 
                y=trends['Profit'],
                mode='lines+markers',
                name='Profit',
                line=dict(color='#00C9A7', width=3),
                fill='tozeroy',
                fillcolor='rgba(0,201,167,0.2)',
                marker=dict(size=6, color='#00C9A7')
            ),
            row=2, col=1
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Revenue ($)", row=1, col=1)
        fig.update_yaxes(title_text="Profit ($)", row=2, col=1)
        fig.update_layout(height=700, showlegend=False, hovermode='x unified')
        
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
            total_profit = trends['Profit']. sum()
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
                color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#00A3E0'], [1, '#0066CC']]
            )
            fig. update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                category_perf,
                x='category',
                y='Profit_Margin_Pct',
                title='Profit Margin by Category (%)',
                color='Profit_Margin_Pct',
                color_continuous_scale=[[0, '#E0F7F4'], [0.5, '#00C9A7'], [1, '#00856F']]
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed table
        st.subheader("📊 Category Performance Table")
        st.dataframe(
            category_perf. style.format({
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
            color_continuous_scale=[[0, '#FFE5E5'], [0.5, '#FFA500'], [1, '#00C9A7']],
            labels={'Profit_Margin_Pct': 'Profit Margin (%)'}
        )
        fig.update_layout(height=max(400, top_n * 30), yaxis={'categoryorder': 'total ascending'})
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
            color_continuous_scale=[[0, '#FF6B9D'], [0.5, '#00A3E0'], [1, '#00C9A7']]
        )
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
                values='Revenue',
                names='payment_method',
                title='Revenue by Payment Method',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                payment_df.style.format({
                    'Revenue': '${:,.2f}',
                    'Avg_Transaction_Value': '${:,.2f}'
                }),
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🚚 Shipping Methods")
            fig = px. pie(
                shipping_df,
                values='Revenue',
                names='shipping_method',
                title='Revenue by Shipping Method',
                hole=0.3
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(
                shipping_df.style.format({
                    'Revenue': '${:,.2f}',
                    'Avg_Transaction_Value': '${:,.2f}'
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
        avg_purchases = filtered_df.groupby('customer_id').size().mean()
        avg_customer_value = filtered_df.groupby('customer_id')['total_amount'].sum().mean()
        repeat_customers = (filtered_df.groupby('customer_id').size() > 1).sum()
        
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
                color_discrete_sequence=['#0066CC']
            )
            fig.update_layout(showlegend=False)
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
                color_continuous_scale=[[0, '#00A3E0'], [1, '#0066CC']]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Country analysis
        st.subheader("🌍 Customer Distribution by Country")
        customer_country = filtered_df.groupby('country')['customer_id'].nunique(). sort_values(ascending=False). head(15)
        
        fig = px.bar(
            x=customer_country.values,
            y=customer_country.index,
            orientation='h',
            title='Top 15 Countries by Customer Count',
            labels={'x': 'Number of Customers', 'y': 'Country'},
            color=customer_country.values,
            color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#00A3E0'], [1, '#0066CC']]
        )
        fig. update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🎯 RFM Customer Segmentation")
        
        rfm_df = compute_rfm(filtered_df)
        
        # Segment distribution
        col1, col2 = st. columns(2)
        
        with col1:
            segment_counts = rfm_df['Segment'].value_counts()
            
            colors = {'Champions': '#00C9A7', 'Loyal': '#0066CC', 'Potential': '#FFA500', 'At Risk': '#E74C3C'}
            color_list = [colors. get(seg, '#95a5a6') for seg in segment_counts.index]
            
            fig = px.bar(
                x=segment_counts. index,
                y=segment_counts.values,
                title='Customer Segments Distribution',
                labels={'x': 'Segment', 'y': 'Number of Customers'},
                color=segment_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            segment_revenue = rfm_df.merge(
                filtered_df.groupby('customer_id')['total_amount'].sum().reset_index(),
                on='customer_id'
            ).groupby('Segment')['total_amount'].sum()
            
            fig = px. pie(
                values=segment_revenue.values,
                names=segment_revenue.index,
                title='Revenue by Customer Segment',
                color=segment_revenue.index,
                color_discrete_map=colors,
                hole=0.3
            )
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
        st.plotly_chart(fig, use_container_width=True)
        
        # Segment statistics
        st.subheader("📈 Segment Statistics")
        
        segment_stats = rfm_df.groupby('Segment'). agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': ['mean', 'sum']
        }). round(2)
        
        segment_stats.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary ($)', 'Total Revenue ($)']
        segment_stats['Customer Count'] = rfm_df.groupby('Segment').size()
        
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
                color_discrete_sequence=['#7B68EE']
            )
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
                color_continuous_scale=[[0, '#FF6B9D'], [0.5, '#00A3E0'], [1, '#00C9A7']]
            )
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
            color_continuous_scale=[[0, '#B3E5FC'], [0.5, '#00A3E0'], [1, '#0066CC']]
        )
        fig. update_layout(xaxis_tickangle=-45)
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
            churn_risk_dist = churn_df['Churn_Risk']. value_counts()
            
            fig = px.pie(
                values=churn_risk_dist.values,
                names=churn_risk_dist.index,
                title='Customer Churn Risk Distribution',
                color_discrete_sequence=['#00C9A7', '#FFA500', '#FF8C42', '#E74C3C']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                churn_df,
                x='Days_Since_Purchase',
                nbins=30,
                title='Days Since Last Purchase Distribution',
                labels={'Days_Since_Purchase': 'Days Since Last Purchase'},
                color_discrete_sequence=['#00A3E0']
            )
            fig.add_vline(x=churn_threshold, line_dash="dash", line_color="red", 
                         annotation_text="Churn Threshold")
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
                color_continuous_scale=[[0, '#FFE5E5'], [0.5, '#00A3E0'], [1, '#00C9A7']],
                aspect='auto'
            )
            fig.update_layout(height=500)
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
        st.metric("Total Categories", f"{filtered_df['category']. nunique():,}")
    
    with col2:
        st.metric("Total Units Sold", f"{filtered_df['quantity'].sum():,}")
        st.metric("Avg Units/Transaction", f"{filtered_df['quantity'].mean():.2f}")
    
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
        marker_color='#00A3E0',
        marker_line=dict(color='#0066CC', width=1)
    ))
    fig.add_trace(go.Bar(
        x=top_products_df['product_name'],
        y=top_products_df['Profit'],
        name='Profit',
        marker_color='#00C9A7',
        marker_line=dict(color='#00856F', width=1)
    ))
    fig.update_layout(
        title=f'Top {top_n} Products: Revenue vs Profit',
        barmode='group',
        xaxis_tickangle=-45,
        height=500
    )
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
# PAGE 5: ADVANCED ANALYTICS
# ============================================================================
elif page == "🔮 Advanced Analytics":
    st. title("🔮 Advanced Analytics & Insights")
    
    tab1, tab2, tab3 = st.tabs(["🛒 Market Basket", "📊 K-Means Clustering", "📈 Forecasting"])
    
    with tab1:
        st.subheader("🛒 Market Basket Analysis")
        st.info("Discovering which product categories are frequently purchased together")
        
        # Parameters
        col1, col2 = st. columns(2)
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
                color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#00A3E0'], [1, '#0066CC']]
            )
            fig.update_layout(height=500)
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
            cluster_dist = clusters_df['Cluster'].value_counts(). sort_index()
            
            fig = px.bar(
                x=[f"Cluster {i}" for i in cluster_dist.index],
                y=cluster_dist.values,
                title='Customer Distribution by Cluster',
                labels={'x': 'Cluster', 'y': 'Number of Customers'},
                color=cluster_dist.values,
                color_continuous_scale=[[0, '#E3F2FD'], [0.5, '#00A3E0'], [1, '#0066CC']]
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Cluster pie
            fig = px.pie(
                values=cluster_dist.values,
                names=[f"Cluster {i}" for i in cluster_dist.index],
                title='Cluster Proportion',
                hole=0.3
            )
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
            color_continuous_scale=[[0, '#0066CC'], [0.5, '#00A3E0'], [1, '#00C9A7']]
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Sales Forecasting")
        
        st.info("Simple moving average forecast for revenue trends")
        
        forecast_period = st.selectbox("Forecast Period", ["Monthly", "Weekly"])
        window = st.slider("Moving Average Window", 3, 12, 6)
        
        freq = 'M' if forecast_period == "Monthly" else 'W'
        trends = sales_trends_analysis(filtered_df, freq=freq)
        
        # Calculate moving average
        trends['MA'] = trends['Revenue'].rolling(window=window).mean()
        
        # Simple forecast (last MA value)
        last_ma = trends['MA'].iloc[-1]
        
        fig = go.Figure()
        
        fig.add_trace(go. Scatter(
            x=trends['transaction_date'],
            y=trends['Revenue'],
            mode='lines',
            name='Actual Revenue',
            line=dict(color='#0066CC', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=trends['transaction_date'],
            y=trends['MA'],
            mode='lines',
            name=f'{window}-Period Moving Average',
            line=dict(color='#FFA500', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title=f'{forecast_period} Revenue with Moving Average Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Latest Revenue", f"${trends['Revenue'].iloc[-1]:,.0f}")
        with col2:
            st.metric("Moving Avg Forecast", f"${last_ma:,.0f}")
        with col3:
            diff_pct = ((last_ma - trends['Revenue']. iloc[-1]) / trends['Revenue'].iloc[-1] * 100)
            st. metric("Forecast vs Actual", f"{diff_pct:+.1f}%")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-top: 20px;'>
        <p style='color: white; font-size: 18px; margin-bottom: 10px; font-weight: 600;'>E-commerce Analytics Dashboard</p>
        <p style='color: rgba(255,255,255,0.9); font-size: 14px; margin-bottom: 5px;'>Built with Streamlit by <strong>Arpit0523</strong></p>
        <p style='color: rgba(255,255,255,0.8); font-size: 12px;'>Data Analytics • Machine Learning • Business Intelligence</p>
    </div>
    """,
    unsafe_allow_html=True
)