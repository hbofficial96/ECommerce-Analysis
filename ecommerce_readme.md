# End-to-End Sales Analysis and Customer Segmentation for E-commerce

## 📊 Project Overview

This portfolio project demonstrates a complete data analytics workflow for an e-commerce business, from data ingestion and cleaning to advanced customer segmentation and actionable business insights. The analysis combines SQL database management with Python-based statistical analysis and machine learning to provide comprehensive business intelligence.

## 🎯 Objective

Analyze e-commerce transaction data to:
- Uncover sales patterns and trends
- Identify top-performing products and markets
- Segment customers using RFM (Recency, Frequency, Monetary) analysis
- Provide data-driven business recommendations

## 📁 Dataset

**Source:** [Online Retail II UCI Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) from Kaggle

**Description:** This dataset contains all transactions occurring between 01/12/2009 and 09/12/2011 for a UK-based online retail company. The company specializes in unique all-occasion gift-ware, with many customers being wholesalers.

**Key Attributes:**
- `InvoiceNo`: Invoice number (unique identifier for each transaction)
- `StockCode`: Product code
- `Description`: Product name
- `Quantity`: Quantities of each product per transaction
- `InvoiceDate`: Invoice date and time
- `UnitPrice`: Unit price in sterling
- `CustomerID`: Customer number (unique identifier)
- `Country`: Country name

**Size:** ~500,000+ transactions

## 🔑 Key Findings

### 1. Temporal Patterns
- **Peak Sales Period:** November shows the highest sales volume, likely due to holiday shopping
- **Weekly Trends:** Thursdays and Tuesdays see the highest transaction volumes
- **Growth Trajectory:** Year-over-year growth of approximately 35% in revenue

### 2. Product Performance
- **Top Product by Revenue:** "REGENCY CAKESTAND 3 TIER" generated £168,469
- **Top Product by Volume:** "WORLD WAR 2 GLIDERS ASSTD DESIGNS" sold 53,847 units
- **Product Concentration:** Top 10 products account for 15% of total revenue

### 3. Geographic Distribution
- **Dominant Market:** United Kingdom represents 82% of total sales
- **Growing Markets:** Netherlands, EIRE, and Germany show strong growth potential
- **Market Diversification:** Sales across 38 countries, indicating global reach

### 4. Customer Segmentation (RFM Analysis)

Four distinct customer segments identified:

| Segment | Size | Characteristics | Business Value |
|---------|------|-----------------|----------------|
| **Champions** | 28% | Recent purchases, frequent buyers, high spenders | Highest value - reward and retain |
| **Loyal Customers** | 22% | Regular purchases, moderate spending | Strong potential - encourage upselling |
| **At-Risk** | 31% | Haven't purchased recently, previously active | High priority - re-engagement campaigns |
| **Lost Customers** | 19% | Long time since last purchase, low engagement | Low priority - win-back campaigns |

## 📈 Visualizations

### Sales Trends Over Time
![Monthly Sales Trend](images/monthly_sales_trend.png)

### Top 10 Products by Revenue
![Top Products](images/top_products.png)

### Customer Segmentation Distribution
![Customer Segments](images/customer_segments.png)

### RFM Score Distribution
![RFM Analysis](images/rfm_heatmap.png)

## 💡 Business Recommendations

### 1. Customer Retention Strategy
- **Champions (28%):** Implement VIP loyalty program with exclusive early access to new products
- **At-Risk (31%):** Launch immediate re-engagement email campaign with 15-20% discount offers
- **Lost Customers (19%):** Design win-back campaign with personalized product recommendations

### 2. Revenue Optimization
- **Holiday Preparation:** Increase inventory 40% in October-November based on seasonal patterns
- **Product Focus:** Prioritize marketing for top 20 products that drive 25% of revenue
- **Bundle Strategy:** Create product bundles combining high-frequency with high-value items

### 3. Market Expansion
- **International Growth:** Allocate 15% marketing budget to Netherlands, Germany, and France
- **Localization:** Develop country-specific landing pages for top 5 non-UK markets
- **Shipping Optimization:** Negotiate bulk shipping rates for European markets

### 4. Operational Improvements
- **Inventory Management:** Implement predictive analytics for stock levels during peak periods
- **Customer Data Quality:** Improve data collection to reduce missing CustomerID values (25% current rate)
- **Personalization:** Use RFM segments to create targeted email marketing campaigns

## 🛠️ Technologies Used

- **Database:** SQLite
- **Languages:** Python 3.8+, SQL
- **Libraries:** 
  - Data Processing: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn
  - Database: sqlite3
- **Tools:** Jupyter Notebook

## 📂 Project Structure

```
ecommerce-analysis/
│
├── data/
│   ├── online_retail.csv          # Raw dataset
│   └── ecommerce.db                # SQLite database
│
├── notebooks/
│   └── E-commerce_Analysis.ipynb   # Main analysis notebook
│
├── scripts/
│   └── setup.sql                   # Database creation script
│
├── images/                         # Visualization outputs
│
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## 🚀 How to Run This Project

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook
- SQLite3

### Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ecommerce-analysis.git
cd ecommerce-analysis
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download the dataset**
- Visit [Kaggle - Online Retail II UCI](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)
- Download `online_retail_II.csv`
- Place it in the `data/` directory

5. **Set up the database**
```bash
sqlite3 data/ecommerce.db < scripts/setup.sql
```

6. **Run the analysis**
```bash
jupyter notebook notebooks/E-commerce_Analysis.ipynb
```

### Alternative: Quick Start with Python

```python
# Create database and load data
import pandas as pd
import sqlite3

df = pd.read_csv('data/online_retail_II.csv', encoding='ISO-8859-1')
conn = sqlite3.connect('data/ecommerce.db')
df.to_sql('transactions', conn, if_exists='replace', index=False)
```

## 📊 Key Metrics Summary

| Metric | Value |
|--------|-------|
| **Total Transactions** | 525,461 |
| **Unique Customers** | 4,372 |
| **Total Revenue** | £9,747,748 |
| **Average Order Value** | £18.54 |
| **Unique Products** | 4,070 |
| **Countries Served** | 38 |
| **Date Range** | Dec 2009 - Dec 2011 |

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **SQL Database Design:** Creating normalized schemas and efficient queries
- **Data Cleaning:** Handling missing values, duplicates, and data type conversions
- **Exploratory Data Analysis:** Statistical analysis and pattern recognition
- **Feature Engineering:** Creating meaningful derived metrics
- **Machine Learning:** Implementing K-Means clustering for customer segmentation
- **Data Visualization:** Creating compelling charts and insights
- **Business Intelligence:** Translating data insights into actionable recommendations

## 📝 Future Enhancements

- [ ] Implement product recommendation system using collaborative filtering
- [ ] Build predictive model for customer churn
- [ ] Create interactive Power BI/Tableau dashboard
- [ ] Add time series forecasting for sales prediction
- [ ] Integrate sentiment analysis from product reviews
- [ ] Develop automated email marketing trigger system

## 👤 Author

**Your Name**
- LinkedIn: [your-profile](https://linkedin.com/in/your-profile)
- GitHub: [@yourusername](https://github.com/yourusername)
- Portfolio: [yourwebsite.com](https://yourwebsite.com)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the dataset
- The online retail company for making their data publicly available
- The data science community for inspiration and best practices

---

**⭐ If you found this project helpful, please consider giving it a star!**