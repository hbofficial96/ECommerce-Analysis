"""
Quick Setup Script for E-commerce Analysis Project
===================================================
This script automates the database creation and data loading process.

Usage:
    python quick_setup.py
"""

import pandas as pd
import sqlite3
import os
from datetime import datetime

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = ['data', 'images', 'notebooks', 'scripts']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")

def load_and_clean_data(csv_path):
    """Load and perform initial cleaning of the dataset"""
    print("\n" + "="*60)
    print("LOADING AND CLEANING DATA")
    print("="*60)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"\n❌ Error: File not found at {csv_path}")
        print("Please download the dataset from:")
        print("https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci")
        return None
    
    # Read CSV
    print(f"\nReading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    print(f"✓ Loaded {len(df):,} rows")
    
    # Rename columns to match schema
    column_mapping = {
        'Invoice': 'invoice_no',
        'StockCode': 'stock_code',
        'Description': 'description',
        'Quantity': 'quantity',
        'InvoiceDate': 'invoice_date',
        'Price': 'unit_price',
        'Customer ID': 'customer_id',
        'Country': 'country'
    }
    
    # Try to rename columns (handle different column name formats)
    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
    
    print("✓ Column names standardized")
    
    # Initial cleaning
    original_size = len(df)
    
    # Remove rows with missing CustomerID
    df = df[df['customer_id'].notna()]
    print(f"✓ Removed {original_size - len(df):,} rows with missing CustomerID")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"✓ Removed duplicate rows")
    
    # Convert data types
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['customer_id'] = df['customer_id'].astype(str)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
    print("✓ Converted data types")
    
    # Remove invalid transactions
    df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)]
    print("✓ Removed invalid transactions")
    
    # Remove outliers
    quantity_99 = df['quantity'].quantile(0.99)
    price_99 = df['unit_price'].quantile(0.99)
    df = df[(df['quantity'] <= quantity_99) & (df['unit_price'] <= price_99)]
    print("✓ Removed extreme outliers")
    
    # Feature engineering
    df['total_price'] = df['quantity'] * df['unit_price']
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    print("✓ Created derived features")
    
    print(f"\nFinal dataset size: {len(df):,} rows")
    print(f"Date range: {df['invoice_date'].min().date()} to {df['invoice_date'].max().date()}")
    
    return df

def create_database(df, db_path):
    """Create SQLite database and load data"""
    print("\n" + "="*60)
    print("CREATING DATABASE")
    print("="*60)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    print(f"✓ Connected to database: {db_path}")
    
    # Load data to database
    df.to_sql('transactions', conn, if_exists='replace', index=False)
    print(f"✓ Loaded {len(df):,} transactions to database")
    
    # Execute setup.sql if it exists
    sql_script_path = 'scripts/setup.sql'
    if os.path.exists(sql_script_path):
        print(f"\n✓ Executing SQL setup script: {sql_script_path}")
        with open(sql_script_path, 'r') as f:
            sql_script = f.read()
            # Execute each statement separately (SQLite limitation)
            for statement in sql_script.split(';'):
                if statement.strip():
                    try:
                        conn.execute(statement)
                    except Exception as e:
                        # Skip errors for statements that already exist
                        pass
        print("✓ SQL setup complete")
    
    # Create summary statistics
    cursor = conn.cursor()
    
    # Business metrics
    cursor.execute("""
        SELECT 
            COUNT(DISTINCT invoice_no) as total_transactions,
            COUNT(DISTINCT customer_id) as total_customers,
            COUNT(DISTINCT stock_code) as total_products,
            COUNT(DISTINCT country) as total_countries,
            ROUND(SUM(total_price), 2) as total_revenue,
            ROUND(AVG(total_price), 2) as avg_transaction
        FROM transactions
    """)
    
    metrics = cursor.fetchone()
    
    print("\n" + "="*60)
    print("DATABASE SUMMARY")
    print("="*60)
    print(f"Total Transactions:   {metrics[0]:,}")
    print(f"Unique Customers:     {metrics[1]:,}")
    print(f"Unique Products:      {metrics[2]:,}")
    print(f"Countries:            {metrics[3]}")
    print(f"Total Revenue:        £{metrics[4]:,.2f}")
    print(f"Avg Transaction:      £{metrics[5]:.2f}")
    
    conn.commit()
    conn.close()
    print("\n✓ Database connection closed")

def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("E-COMMERCE ANALYSIS - QUICK SETUP")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create directories
    create_directories()
    
    # Define paths
    csv_path = 'data/online_retail_II.csv'
    db_path = 'data/ecommerce.db'
    
    # Load and clean data
    df = load_and_clean_data(csv_path)
    
    if df is not None:
        # Create database
        create_database(df, db_path)
        
        print("\n" + "="*60)
        print("✅ SETUP COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("1. Open Jupyter Notebook: jupyter notebook")
        print("2. Navigate to: notebooks/E-commerce_Analysis.ipynb")
        print("3. Run all cells to perform the complete analysis")
        print("\nDatabase location: " + db_path)
        print("="*60)
    else:
        print("\n❌ Setup failed. Please check the error messages above.")

if __name__ == "__main__":
    main()