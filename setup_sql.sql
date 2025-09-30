-- ========================================
-- E-commerce Database Setup Script
-- ========================================
-- This script creates a normalized relational schema for e-commerce data
-- and prepares it for analysis
-- ========================================

-- Drop existing tables if they exist
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS countries;

-- ========================================
-- Table: countries
-- Purpose: Store unique country information
-- ========================================
CREATE TABLE IF NOT EXISTS countries (
    country_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country_name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================
-- Table: customers
-- Purpose: Store customer information
-- ========================================
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(20) PRIMARY KEY,
    country_id INTEGER,
    first_transaction_date DATE,
    last_transaction_date DATE,
    total_transactions INTEGER DEFAULT 0,
    total_spent DECIMAL(10, 2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (country_id) REFERENCES countries(country_id)
);

-- ========================================
-- Table: products
-- Purpose: Store product catalog information
-- ========================================
CREATE TABLE IF NOT EXISTS products (
    stock_code VARCHAR(20) PRIMARY KEY,
    description TEXT,
    average_price DECIMAL(10, 2),
    total_quantity_sold INTEGER DEFAULT 0,
    total_revenue DECIMAL(12, 2) DEFAULT 0.00,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================
-- Table: transactions
-- Purpose: Store all transaction records (fact table)
-- ========================================
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no VARCHAR(20) NOT NULL,
    stock_code VARCHAR(20),
    description TEXT,
    quantity INTEGER,
    invoice_date DATETIME,
    unit_price DECIMAL(10, 2),
    customer_id VARCHAR(20),
    country VARCHAR(100),
    total_price DECIMAL(10, 2),
    year INTEGER,
    month INTEGER,
    day_of_week INTEGER,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
    FOREIGN KEY (stock_code) REFERENCES products(stock_code)
);

-- ========================================
-- Create Indexes for Performance
-- ========================================

-- Index on invoice_date for temporal queries
CREATE INDEX idx_invoice_date ON transactions(invoice_date);

-- Index on customer_id for customer analysis
CREATE INDEX idx_customer_id ON transactions(customer_id);

-- Index on stock_code for product analysis
CREATE INDEX idx_stock_code ON transactions(stock_code);

-- Index on country for geographical analysis
CREATE INDEX idx_country ON transactions(country);

-- Composite index for date-based aggregations
CREATE INDEX idx_year_month ON transactions(year, month);

-- Index on invoice_no for grouping transactions
CREATE INDEX idx_invoice_no ON transactions(invoice_no);

-- ========================================
-- Views for Common Queries
-- ========================================

-- View: Monthly Sales Summary
CREATE VIEW IF NOT EXISTS monthly_sales_summary AS
SELECT 
    year,
    month,
    COUNT(DISTINCT invoice_no) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    SUM(quantity) AS total_quantity,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(total_price), 2) AS avg_transaction_value
FROM transactions
WHERE total_price > 0
GROUP BY year, month
ORDER BY year, month;

-- View: Product Performance
CREATE VIEW IF NOT EXISTS product_performance AS
SELECT 
    stock_code,
    description,
    SUM(quantity) AS total_quantity_sold,
    COUNT(DISTINCT invoice_no) AS times_purchased,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(unit_price), 2) AS avg_unit_price
FROM transactions
WHERE quantity > 0 AND unit_price > 0
GROUP BY stock_code, description
ORDER BY total_revenue DESC;

-- View: Customer Summary
CREATE VIEW IF NOT EXISTS customer_summary AS
SELECT 
    customer_id,
    country,
    COUNT(DISTINCT invoice_no) AS total_orders,
    SUM(quantity) AS total_items_purchased,
    ROUND(SUM(total_price), 2) AS total_spent,
    ROUND(AVG(total_price), 2) AS avg_order_value,
    MIN(invoice_date) AS first_purchase,
    MAX(invoice_date) AS last_purchase,
    ROUND(JULIANDAY(MAX(invoice_date)) - JULIANDAY(MIN(invoice_date)), 0) AS customer_lifetime_days
FROM transactions
WHERE customer_id IS NOT NULL AND total_price > 0
GROUP BY customer_id, country
ORDER BY total_spent DESC;

-- View: Country Performance
CREATE VIEW IF NOT EXISTS country_performance AS
SELECT 
    country,
    COUNT(DISTINCT customer_id) AS unique_customers,
    COUNT(DISTINCT invoice_no) AS total_orders,
    SUM(quantity) AS total_quantity,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(total_price), 2) AS avg_transaction_value
FROM transactions
WHERE total_price > 0
GROUP BY country
ORDER BY total_revenue DESC;

-- ========================================
-- Analytical Queries (Examples)
-- ========================================

-- Top 10 Products by Revenue
CREATE VIEW IF NOT EXISTS top_products_revenue AS
SELECT 
    stock_code,
    description,
    ROUND(SUM(total_price), 2) AS revenue,
    SUM(quantity) AS units_sold
FROM transactions
WHERE quantity > 0 AND unit_price > 0
GROUP BY stock_code, description
ORDER BY revenue DESC
LIMIT 10;

-- Top 10 Customers by Spending
CREATE VIEW IF NOT EXISTS top_customers AS
SELECT 
    customer_id,
    country,
    COUNT(DISTINCT invoice_no) AS orders,
    ROUND(SUM(total_price), 2) AS total_spent
FROM transactions
WHERE customer_id IS NOT NULL AND total_price > 0
GROUP BY customer_id, country
ORDER BY total_spent DESC
LIMIT 10;

-- Daily Sales Pattern (Day of Week)
CREATE VIEW IF NOT EXISTS daily_sales_pattern AS
SELECT 
    CASE day_of_week
        WHEN 0 THEN 'Sunday'
        WHEN 1 THEN 'Monday'
        WHEN 2 THEN 'Tuesday'
        WHEN 3 THEN 'Wednesday'
        WHEN 4 THEN 'Thursday'
        WHEN 5 THEN 'Friday'
        WHEN 6 THEN 'Saturday'
    END AS day_name,
    day_of_week,
    COUNT(DISTINCT invoice_no) AS total_orders,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(total_price), 2) AS avg_order_value
FROM transactions
WHERE total_price > 0
GROUP BY day_of_week
ORDER BY day_of_week;

-- ========================================
-- Data Quality Checks
-- ========================================

-- Check for missing customer IDs
CREATE VIEW IF NOT EXISTS missing_customer_data AS
SELECT 
    COUNT(*) AS records_with_missing_customer_id,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM transactions), 2) AS percentage
FROM transactions
WHERE customer_id IS NULL OR customer_id = '';

-- Check for negative quantities or prices (potential returns/errors)
CREATE VIEW IF NOT EXISTS data_quality_issues AS
SELECT 
    'Negative Quantities' AS issue_type,
    COUNT(*) AS count
FROM transactions
WHERE quantity < 0
UNION ALL
SELECT 
    'Negative Prices' AS issue_type,
    COUNT(*) AS count
FROM transactions
WHERE unit_price < 0
UNION ALL
SELECT 
    'Zero Prices' AS issue_type,
    COUNT(*) AS count
FROM transactions
WHERE unit_price = 0;

-- ========================================
-- Instructions for Data Import
-- ========================================

-- After creating tables, load data using Python:
/*
import pandas as pd
import sqlite3

# Read CSV
df = pd.read_csv('online_retail_II.csv', encoding='ISO-8859-1')

# Connect to database
conn = sqlite3.connect('ecommerce.db')

# Load to staging table
df.to_sql('transactions_staging', conn, if_exists='replace', index=False)

# Then clean and insert into main tables using Python/SQL
*/

-- ========================================
-- Useful Aggregate Queries
-- ========================================

-- Overall Business Metrics
CREATE VIEW IF NOT EXISTS business_metrics AS
SELECT 
    COUNT(DISTINCT invoice_no) AS total_invoices,
    COUNT(DISTINCT customer_id) AS total_customers,
    COUNT(DISTINCT stock_code) AS total_products,
    COUNT(DISTINCT country) AS total_countries,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(total_price), 2) AS avg_transaction_value,
    MIN(invoice_date) AS first_transaction,
    MAX(invoice_date) AS last_transaction
FROM transactions
WHERE total_price > 0 AND customer_id IS NOT NULL;

-- ========================================
-- RFM Analysis Base Query
-- ========================================

CREATE VIEW IF NOT EXISTS rfm_base AS
SELECT 
    customer_id,
    MAX(invoice_date) AS last_purchase_date,
    COUNT(DISTINCT invoice_no) AS frequency,
    ROUND(SUM(total_price), 2) AS monetary
FROM transactions
WHERE customer_id IS NOT NULL AND total_price > 0
GROUP BY customer_id;

-- ========================================
-- End of Setup Script
-- ========================================

-- Verify table creation
SELECT 'Tables created successfully!' AS status;
SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;
SELECT name FROM sqlite_master WHERE type='view' ORDER BY name;