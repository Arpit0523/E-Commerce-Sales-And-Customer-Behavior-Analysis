#!/usr/bin/env python3
"""
Data processing for real e-commerce dataset (DATASET. csv)
Maps real dataset columns to our analysis framework
"""
import pandas as pd
import os

RAW_FILE = "DATASET.csv"
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

def load_and_process_real_data():
    """
    Load real dataset and transform to match our analysis structure
    """
    print("Loading real e-commerce dataset...")
    df = pd.read_csv(RAW_FILE)
    
    print(f"Loaded {len(df):,} records")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert date column (try ISO8601 format first, then mixed)
    try:
        df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    except:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    
    # Rename columns to match our analysis framework
    df_processed = df.rename(columns={
        'Order_ID': 'transaction_id',
        'Customer_ID': 'customer_id',
        'Date': 'transaction_date',
        'Age': 'age',
        'Gender': 'gender',
        'City': 'city',
        'Product_Category': 'category',
        'Unit_Price': 'unit_price',
        'Quantity': 'quantity',
        'Discount_Amount': 'discount',
        'Total_Amount': 'total_amount',
        'Payment_Method': 'payment_method',
        'Device_Type': 'device_type',
        'Session_Duration_Minutes': 'session_duration',
        'Pages_Viewed': 'pages_viewed',
        'Is_Returning_Customer': 'is_returning',
        'Delivery_Time_Days': 'delivery_days',
        'Customer_Rating': 'rating'
    })
    
    # Add derived columns needed for analysis
    
    # Extract product name (use category for now since we don't have individual product names)
    df_processed['product_name'] = df_processed['category'] + '_' + df_processed['transaction_id'].str[-4:]
    
    # Extract product_id (derived from category)
    category_codes = {cat: f"PROD{i:04d}" for i, cat in enumerate(df_processed['category'].unique(), 1)}
    df_processed['product_id'] = df_processed['category'].map(category_codes)
    
    # Calculate cost (assuming 60% of unit price as cost for profit calculation)
    df_processed['cost'] = (df_processed['unit_price'] * 0.6).round(2)
    
    # Calculate price (unit_price is the selling price)
    df_processed['price'] = df_processed['unit_price']
    
    # Calculate profit
    df_processed['profit'] = ((df_processed['unit_price'] - df_processed['cost']) * df_processed['quantity'] - df_processed['discount']).round(2)
    
    # Add time-based features
    df_processed['year'] = df_processed['transaction_date'].dt.year
    df_processed['month'] = df_processed['transaction_date'].dt. month
    df_processed['quarter'] = df_processed['transaction_date'].dt.quarter
    df_processed['day_of_week'] = df_processed['transaction_date'].dt. day_name()
    
    # Add country column (all from Turkey)
    df_processed['country'] = 'Turkey'
    
    # Create registration_date (approximate as 30-180 days before first purchase)
    import numpy as np
    np.random.seed(42)
    first_purchases = df_processed. groupby('customer_id')['transaction_date'].min()
    registration_dates = {}
    for cust_id, first_date in first_purchases.items():
        days_before = np.random.randint(30, 180)
        registration_dates[cust_id] = first_date - pd.Timedelta(days=days_before)
    
    df_processed['registration_date'] = df_processed['customer_id'].map(registration_dates)
    
    # Add name column (anonymized)
    df_processed['name'] = df_processed['customer_id'].apply(lambda x: f"Customer_{x}")
    
    # Add email column (anonymized)
    df_processed['email'] = df_processed['customer_id'].apply(lambda x: f"{x. lower()}@example.com")
    
    # Add shipping method (derived from delivery time)
    def get_shipping_method(days):
        if days <= 2:
            return 'Express'
        elif days <= 5:
            return 'Standard'
        else:
            return 'Economy'
    
    df_processed['shipping_method'] = df_processed['delivery_days'].apply(get_shipping_method)
    
    # Add status (assume all completed since we have ratings)
    df_processed['status'] = 'Completed'
    
    return df_processed

def save_processed_data(df):
    """Save processed dataset"""
    output_path = os.path.join(OUT_DIR, "master_dataset.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Processed dataset saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total Transactions: {len(df):,}")
    print(f"Total Customers: {df['customer_id'].nunique():,}")
    print(f"Date Range: {df['transaction_date'].min(). date()} to {df['transaction_date'].max().date()}")
    print(f"Total Revenue: ${df['total_amount']. sum():,.2f}")
    print(f"Total Profit: ${df['profit']. sum():,.2f}")
    print(f"Average Order Value: ${df['total_amount'].mean():.2f}")
    print(f"Average Rating: {df['rating'].mean():.2f}/5. 0")
    print(f"\nCategories: {', '.join(df['category'].unique())}")
    print(f"Cities: {', '.join(df['city'].unique())}")
    print(f"Payment Methods: {', '.join(df['payment_method'].unique())}")
    print("=" * 60)

def main():
    """Main execution function"""
    try:
        df = load_and_process_real_data()
        save_processed_data(df)
        
        # Display sample
        print("\n📋 Sample of processed data:")
        print(df[['transaction_id', 'customer_id', 'transaction_date', 'category', 
                  'total_amount', 'profit', 'rating']].head(10))
        
        print("\n✅ Data processing complete!")
        print("You can now run: streamlit run dashboard.py")
        
    except FileNotFoundError:
        print(f"\n❌ Error: {RAW_FILE} not found!")
        print(f"Please ensure {RAW_FILE} is in the project root directory.")
    except Exception as e:
        print(f"\n❌ Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()