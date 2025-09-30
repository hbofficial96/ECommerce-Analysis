# E-commerce Analysis Project - Complete Implementation Guide

## 📚 Table of Contents
1. [Project Architecture](#project-architecture)
2. [Data Schema Design](#data-schema-design)
3. [Analysis Methodology](#analysis-methodology)
4. [Technical Implementation Details](#technical-implementation-details)
5. [Key SQL Queries](#key-sql-queries)
6. [RFM Segmentation Deep Dive](#rfm-segmentation-deep-dive)
7. [Troubleshooting](#troubleshooting)
8. [Portfolio Presentation Tips](#portfolio-presentation-tips)

---

## 🏗️ Project Architecture

### Directory Structure
```
ecommerce-analysis/
│
├── data/
│   ├── online_retail_II.csv      # Raw dataset (download from Kaggle)
│   └── ecommerce.db               # SQLite database (generated)
│
├── notebooks/
│   └── E-commerce_Analysis.ipynb  # Main analysis notebook
│
├── scripts/
│   ├── setup.sql                  # Database schema and views
│   └── quick_setup.py             # Automated setup script
│
├── images/                        # Generated visualizations
│   ├── temporal_analysis.png
│   ├── top_products.png
│   ├── geographic_analysis.png
│   ├── rfm_distributions.png
│   ├── optimal_clusters.png
│   └── customer_segments.png
│
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
├── PROJECT_GUIDE.md              # This file
└── .gitignore                    # Git ignore file
```

### Technology Stack

**Database Layer:**
- SQLite 3.x - Lightweight, serverless database
- Benefits: Portable, no installation, perfect for portfolio projects

**Analysis Layer:**
- Python 3.8+ - Core programming language
- Pandas - Data manipulation and analysis
- NumPy - Numerical computing
- Scikit-learn - Machine learning (K-Means clustering)

**Visualization Layer:**
- Matplotlib - Core plotting library
- Seaborn - Statistical visualizations
- 3D plotting for multi-dimensional analysis

---

## 📊 Data Schema Design

### Entity-Relationship Diagram

```
┌─────────────────┐
│   Countries     │
├─────────────────┤
│ country_id (PK) │
│ country_name    │
└────────┬────────┘
         │
         │ 1:N
         │
┌────────▼────────┐        ┌──────────────────┐
│   Customers     │        │    Products      │
├─────────────────┤        ├──────────────────┤
│ customer_id(PK) │        │ stock_code (PK)  │
│ country_id (FK) │        │ description      │
│ first_txn_date  │        │ average_price    │
│ last_txn_date   │        │ total_qty_sold   │
│ total_txns      │        │ total_revenue    │
│ total_spent     │        └────────┬─────────┘
└────────┬────────┘                 │
         │                          │
         │ 1:N                  N:1 │
         │                          │
    ┌────▼──────────────────────────▼────┐
    │        Transactions (Fact)         │
    ├────────────────────────────────────┤
    │ transaction_id (PK)                │
    │ invoice_no                         │
    │ stock_code (FK)                    │
    │ customer_id (FK)                   │
    │ quantity                           │
    │ unit_price                         │
    │ total_price (calculated)           │
    │ invoice_date                       │
    │ year, month, day_of_week           │
    └────────────────────────────────────┘
```

### Key Design Decisions

1. **Normalized Structure**: Separates entities (customers, products, countries) to reduce redundancy
2. **Fact Table Pattern**: `transactions` serves as the central fact table
3. **Derived Columns**: Pre-calculated fields like `total_price` for performance
4. **Temporal Fields**: Extracted date components for efficient time-based queries
5. **Indexes**: Strategic indexing on frequently queried columns

---

## 🔍 Analysis Methodology

### Phase 1: Data Quality Assessment
- **Missing Values**: Identify and quantify missing data
- **Duplicates**: Remove exact duplicate transactions
- **Outliers**: Detect and handle extreme values using percentile-based filtering
- **Data Types**: Ensure correct typing (dates, numerics, strings)

### Phase 2: Exploratory Data Analysis (EDA)

#### Temporal Analysis
- Monthly revenue trends
- Day-of-week patterns
- Hourly transaction distribution
- Seasonal decomposition

#### Product Analysis
- Best-sellers by revenue
- Best-sellers by volume
- Product performance metrics
- Revenue concentration (Pareto analysis)

#### Geographic Analysis
- Country-level revenue distribution
- Customer concentration by region
- Market penetration analysis

### Phase 3: Customer Segmentation (RFM)

#### RFM Framework
**Recency (R):** How recently did the customer purchase?
- Lower = Better
- Calculation: Days since last purchase

**Frequency (F):** How often does the customer purchase?
- Higher = Better
- Calculation: Total number of distinct purchases

**Monetary (M):** How much does the customer spend?
- Higher = Better
- Calculation: Total spend across all purchases

#### Scoring Method
- Divide each metric into quintiles (5 equal groups)
- Assign scores 1-5 (5 = best)
- Combine into RFM score (e.g., "555" = top customer)

#### Segmentation Strategy
Use K-Means clustering on standardized RFM values:
1. **Standardization**: Scale features to mean=0, std=1
2. **Optimal K**: Elbow method + Silhouette analysis
3. **Cluster Assignment**: K-Means algorithm
4. **Segment Naming**: Based on RFM characteristics

---

## 💻 Technical Implementation Details

### Data Cleaning Pipeline

```python
def clean_ecommerce_data(df):
    """
    Comprehensive data cleaning pipeline
    
    Steps:
    1. Remove missing CustomerID
    2. Drop duplicates
    3. Convert data types
    4. Filter invalid transactions
    5. Remove outliers
    6. Create derived features
    """
    # Original size
    original_size = len(df)
    
    # Step 1: Remove missing CustomerID
    df = df[df['customer_id'].notna()].copy()
    
    # Step 2: Remove duplicates
    df = df.drop_duplicates()
    
    # Step 3: Convert data types
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['customer_id'] = df['customer_id'].astype(str)
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
    
    # Step 4: Filter invalid transactions
    df = df[(df['quantity'] > 0) & (df['unit_price'] > 0)]
    
    # Step 5: Remove outliers (>99th percentile)
    q99_quantity = df['quantity'].quantile(0.99)
    q99_price = df['unit_price'].quantile(0.99)
    df = df[(df['quantity'] <= q99_quantity) & 
            (df['unit_price'] <= q99_price)]
    
    # Step 6: Create derived features
    df['total_price'] = df['quantity'] * df['unit_price']
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    
    print(f"Cleaned: {original_size:,} → {len(df):,} rows")
    return df
```

### RFM Calculation

```python
def calculate_rfm(df, reference_date=None):
    """
    Calculate RFM metrics for customer segmentation
    
    Args:
        df: DataFrame with transaction data
        reference_date: Date to calculate recency from (default: max date + 1 day)
    
    Returns:
        DataFrame with RFM scores
    """
    if reference_date is None:
        reference_date = df['invoice_date'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('customer_id').agg({
        'invoice_date': lambda x: (reference_date - x.max()).days,
        'invoice_no': 'nunique',
        'total_price': 'sum'
    }).reset_index()
    
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    
    # Calculate scores (quintiles)
    rfm['r_score'] = pd.qcut(rfm['recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
    rfm['f_score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
    rfm['m_score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1,2,3,4,5], duplicates='drop')
    
    rfm['rfm_score'] = (rfm['r_score'].astype(str) + 
                        rfm['f_score'].astype(str) + 
                        rfm['m_score'].astype(str))
    
    return rfm
```

### K-Means Clustering

```python
def perform_kmeans_clustering(rfm, optimal_k=4):
    """
    Perform K-Means clustering on RFM data
    
    Args:
        rfm: DataFrame with RFM metrics
        optimal_k: Number of clusters
    
    Returns:
        rfm DataFrame with cluster assignments
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    # Select features for clustering
    features = rfm[['recency', 'frequency', 'monetary']].copy()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    rfm['cluster'] = kmeans.fit_predict(features_scaled)
    
    return rfm
```

---

## 🔑 Key SQL Queries

### 1. Monthly Sales Trend

```sql
SELECT 
    strftime('%Y-%m', invoice_date) AS year_month,
    COUNT(DISTINCT invoice_no) AS total_orders,
    COUNT(DISTINCT customer_id) AS unique_customers,
    ROUND(SUM(total_price), 2) AS revenue,
    ROUND(AVG(total_price), 2) AS avg_order_value
FROM transactions
WHERE total_price > 0
GROUP BY year_month
ORDER BY year_month;
```

### 2. Top Products by Revenue

```sql
SELECT 
    stock_code,
    description,
    SUM(quantity) AS units_sold,
    COUNT(DISTINCT invoice_no) AS times_purchased,
    ROUND(SUM(total_price), 2) AS total_revenue,
    ROUND(AVG(unit_price), 2) AS avg_price
FROM transactions
WHERE quantity > 0 AND unit_price > 0
GROUP BY stock_code, description
ORDER BY total_revenue DESC
LIMIT 20;
```

### 3. Customer Lifetime Value

```sql
SELECT 
    customer_id,
    country,
    MIN(invoice_date) AS first_purchase,
    MAX(invoice_date) AS last_purchase,
    ROUND(JULIANDAY(MAX(invoice_date)) - JULIANDAY(MIN(invoice_date))) AS lifetime_days,
    COUNT(DISTINCT invoice_no) AS total_orders,
    SUM(quantity) AS total_items,
    ROUND(SUM(total_price), 2) AS lifetime_value,
    ROUND(AVG(total_price), 2) AS avg_order_value
FROM transactions
WHERE customer_id IS NOT NULL AND total_price > 0
GROUP BY customer_id, country
ORDER BY lifetime_value DESC
LIMIT 100;
```

### 4. Country Performance

```sql
SELECT 
    country,
    COUNT(DISTINCT customer_id) AS customers,
    COUNT(DISTINCT invoice_no) AS orders,
    SUM(quantity) AS items_sold,
    ROUND(SUM(total_price), 2) AS revenue,
    ROUND(AVG(total_price), 2) AS avg_transaction,
    ROUND(SUM(total_price) * 100.0 / 
        (SELECT SUM(total_price) FROM transactions WHERE total_price > 0), 2) AS revenue_share_pct
FROM transactions
WHERE total_price > 0
GROUP BY country
ORDER BY revenue DESC;
```

### 5. Cohort Analysis (Monthly Cohorts)

```sql
WITH first_purchase AS (
    SELECT 
        customer_id,
        DATE(MIN(invoice_date), 'start of month') AS cohort_month
    FROM transactions
    WHERE customer_id IS NOT NULL
    GROUP BY customer_id
),
purchase_activity AS (
    SELECT 
        t.customer_id,
        fp.cohort_month,
        DATE(t.invoice_date, 'start of month') AS purchase_month,
        SUM(t.total_price) AS revenue
    FROM transactions t
    JOIN first_purchase fp ON t.customer_id = fp.customer_id
    WHERE t.total_price > 0
    GROUP BY t.customer_id, fp.cohort_month, purchase_month
)
SELECT 
    cohort_month,
    COUNT(DISTINCT customer_id) AS cohort_size,
    SUM(revenue) AS cohort_revenue
FROM purchase_activity
GROUP BY cohort_month
ORDER BY cohort_month;
```

---

## 🎯 RFM Segmentation Deep Dive

### Understanding RFM Scores

| RFM Score | Recency | Frequency | Monetary | Interpretation |
|-----------|---------|-----------|----------|----------------|
| 555 | Recent | Very Often | High Spend | **Champions** - Best customers |
| 544 | Recent | Often | High Spend | **Loyal** - Regular high-value |
| 454 | Moderate | Very Often | High Spend | **At Risk** - Was great, fading |
| 345 | Old | Moderate | Moderate | **Hibernating** - Need reactivation |
| 111 | Very Old | Rarely | Low Spend | **Lost** - Lowest value |

### Segment Action Matrix

| Segment | Characteristics | Marketing Strategy | Budget Allocation |
|---------|----------------|-------------------|-------------------|
| **Champions** (R:5, F:5, M:5) | Recent, frequent, high-spend | VIP rewards, early access, referral program | 30% |
| **Loyal Customers** (R:4-5, F:3-4, M:3-4) | Regular buyers | Upsell, cross-sell, loyalty bonuses | 25% |
| **At Risk** (R:2-3, F:4-5, M:4-5) | Haven't purchased recently | Win-back campaigns, special offers | 25% |
| **Lost Customers** (R:1-2, F:1-2, M:1-2) | Long inactive, low value | Re-engagement or removal | 10% |
| **New Customers** (F:1, M: varies) | First-time buyers | Onboarding, welcome offers | 10% |

### Custom Segment Definitions

```python
def assign_customer_segments(rfm):
    """
    Assign descriptive segment names based on RFM scores
    """
    def segment_logic(row):
        r, f, m = row['r_score'], row['f_score'], row['m_score']
        
        # Champions: Recent + Frequent + High Spend
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        
        # Loyal: Recent + Moderate-High Frequency
        elif r >= 3 and f >= 3:
            return 'Loyal Customers'
        
        # At Risk: Not recent but was valuable
        elif r <= 2 and f >= 3 and m >= 3:
            return 'At Risk'
        
        # Hibernating: Moderate on all
        elif r == 3 and f <= 3:
            return 'Hibernating'
        
        # Lost: Low on all metrics
        elif r <= 2 and f <= 2:
            return 'Lost Customers'
        
        # New: First-time or recent low-frequency
        elif f == 1:
            return 'New Customers'
        
        # Promising: Recent but moderate F/M
        elif r >= 4 and f <= 3:
            return 'Promising'
        
        # Default
        else:
            return 'Need Attention'
    
    rfm['segment'] = rfm.apply(segment_logic, axis=1)
    return rfm
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Database Lock Error
**Error:** `database is locked`
**Solution:**
```python
# Use timeout parameter
conn = sqlite3.connect('ecommerce.db', timeout=10)
```

#### 2. Memory Error with Large Dataset
**Error:** `MemoryError`
**Solution:**
```python
# Use chunking for large files
chunks = []
for chunk in pd.read_csv('data.csv', chunksize=50000):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

#### 3. K-Means Convergence Warning
**Warning:** `ConvergenceWarning: Number of distinct clusters found smaller than n_clusters`
**Solution:**
```python
# Increase n_init and max_iter
kmeans = KMeans(n_clusters=4, random_state=42, n_init=20, max_iter=500)
```

#### 4. Visualization Not Displaying
**Issue:** Plots not showing in Jupyter
**Solution:**
```python
# Add at the beginning of notebook
%matplotlib inline
```

#### 5. CSV Encoding Issues
**Error:** `UnicodeDecodeError`
**Solution:**
```python
# Try different encodings
df = pd.read_csv('data.csv', encoding='ISO-8859-1')
# or
df = pd.read_csv('data.csv', encoding='latin1')
```

---

## 🎤 Portfolio Presentation Tips

### Structure Your Presentation

1. **Problem Statement** (30 seconds)
   - "E-commerce businesses struggle with customer retention"
   - "Need data-driven approach to identify high-value customers"

2. **Approach** (1 minute)
   - Data pipeline: SQL + Python
   - RFM methodology for segmentation
   - Machine learning for clustering

3. **Key Findings** (2 minutes)
   - Show top visualizations
   - Highlight surprising insights
   - Quantify business impact

4. **Recommendations** (1 minute)
   - Actionable strategies per segment
   - Expected ROI

5. **Technical Skills** (30 seconds)
   - SQL database design
   - Python data analysis
   - ML implementation
   - Business intelligence

### Talking Points for Each Section

**SQL Skills:**
- "Designed normalized schema with fact and dimension tables"
- "Created indexed views for optimized query performance"
- "Implemented complex analytical queries for business metrics"

**Python/Pandas:**
- "Built robust ETL pipeline with data quality checks"
- "Handled 500K+ transactions with memory-efficient processing"
- "Created reusable functions for reproducibility"

**Machine Learning:**
- "Applied K-Means clustering after proper feature scaling"
- "Used silhouette score and elbow method for optimal clusters"
- "Achieved clear customer segments with business interpretation"

**Business Impact:**
- "Identified 31% of customers at risk of churn"
- "Champions segment drives 45% of revenue with targeted retention"
- "Projected 20% increase in CLV with segment-specific campaigns"

### GitHub Repository Best Practices

1. **README.md** - Clear, visual, compelling
2. **Clean Code** - Well-commented, PEP 8 compliant
3. **Documentation** - This guide as reference
4. **Visualizations** - High-quality, professional charts
5. **Reproducibility** - requirements.txt, clear instructions

### Questions to Prepare For

1. **Why RFM over other segmentation methods?**
   - Simple, interpretable, actionable
   - Proven framework in retail/e-commerce
   - No need for complex feature engineering

2. **How did you handle missing data?**
   - Removed 25% with missing CustomerID (can't segment without ID)
   - Documented decision and impact on analysis

3. **Why K-Means instead of other clustering algorithms?**
   - Efficient, scalable, well-understood
   - Works well with continuous numeric features
   - Easy to interpret centroids

4. **How would you deploy this in production?**
   - Automated ETL pipeline (Airflow)
   - Daily RFM score updates
   - Dashboard integration (Tableau/Power BI)
   - Marketing automation triggers

5. **What would you do differently with more time?**
   - Product recommendation system
   - Churn prediction model
   - Time series forecasting
   - A/B testing framework

---

## 📈 Success Metrics

This project demonstrates:

✅ **Database Design** - Normalized schema, efficient queries
✅ **Data Engineering** - ETL pipeline, data quality checks
✅ **Statistical Analysis** - EDA, trend analysis, distributions
✅ **Machine Learning** - Clustering, model evaluation
✅ **Business Intelligence** - Actionable insights, ROI focus
✅ **Visualization** - Professional charts, clear communication
✅ **Documentation** - Comprehensive, reproducible

---

## 📚 Additional Resources

- [RFM Analysis Guide](https://www.putler.com/rfm-analysis)
- [K-Means Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Pandas Best Practices](https://pandas.pydata.org/docs/user_guide/enhancingperf.html)
- [SQL Performance Tuning](https://www.sqlite.org/optoverview.html)

---

**Good luck with your portfolio project! 🚀**
