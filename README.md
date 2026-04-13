# Customer Purchase Behavior and Sentiment Analysis

An end-to-end data analysis project that analyzes customer purchase behavior and sentiment using **SQL**, **Python**, and **Power BI**. Combines structured transactional data with customer reviews to deliver actionable business insights, predictive ML models, and interactive dashboards.

<div align="center">
  <img src="dashboard/Ecommerce Dashboard.gif" height="300" alt="Ecommerce Dashboard Preview" />
</div>

---

## Project Overview

### Objective
Identify patterns in customer purchase behavior and analyze customer sentiment from product reviews. These insights empower businesses to make informed decisions about marketing, product development, and customer engagement strategies.

### Datasets

1. **Customer Purchase Data** — Transactions including Customer ID, Product details, Purchase Quantity, Price, Date, and Country.
2. **Customer Reviews Data** — Review text linked to Customer ID and Product ID, with sentiment analysis.

---

## Key Features

### 1. Data Extraction & Transformation (SQL)
- Database schema creation and data ingestion via SQL scripts
- Data cleaning, normalization, and referential integrity
- Advanced aggregation queries: revenue by customer, sales by product, purchase trends over time

### 2. Data Analysis (Python)
- **Sentiment Analysis** — Classified reviews into Positive, Neutral, and Negative using `TextBlob`
- **Exploratory Data Analysis (EDA)** — Purchase distributions, revenue by category, seasonal spend trends, correlation heatmaps
- **Predictive ML Model** — `RandomForestClassifier` for subscription churn prediction using product category, quantity, price, and sentiment as features
- **Visualization** — Publication-quality plots with `matplotlib` and `seaborn`

### 3. Data Visualization & Reporting (Power BI)
- **Interactive Dashboard** (`Ecommerce Analytics Dashboard.pbix`) showcasing purchase trends, top products, customer segmentation, and sentiment insights
- **Dynamic Filtering** — Slice data by date, product category, or customer segment
- **[Live Dashboard Link](https://app.powerbi.com/view?r=eyJrIjoiZmVlNTUwMzItYjYzOC00ZjQ5LTkwZDYtMmZjOTBkZDU0NmY0IiwidCI6IjZjZTcwOTA0LTUwOWMtNGI0Zi1iNjc2LTJiMGRlZjA3M2U2YyJ9)** — View the dashboard online without Power BI Desktop

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| **Database** | MySQL / SQLite (local) via `sqlalchemy` |
| **Programming** | Python 3 (`pandas`, `numpy`, `scikit-learn`, `TextBlob`, `matplotlib`, `seaborn`) |
| **ML Model** | `RandomForestClassifier` — subscription churn prediction |
| **Visualization** | Power BI Desktop |
| **Infrastructure** | Docker Compose (PostgreSQL), Jupyter Notebook |

---

## Repository Structure

```
├── data/
│   ├── customer_purchase_data.csv          # Raw purchase transactions
│   ├── customer_reviews_data.csv           # Raw customer reviews
│   ├── updated_review_data.csv             # Reviews with sentiment scores
│   └── powerbi_enriched_data.csv           # Enriched export for Power BI
├── sql/
│   └── cust_pur_details.sql                # SQL: database creation & transformations
├── notebooks/
│   ├── Customer Purchase Behavior and Sentiment Analysis.ipynb   # Original analysis notebook
│   └── Customer_Shopping_Behavior_Analysis.ipynb                 # Advanced EDA + ML pipeline
├── dashboard/
│   ├── Ecommerce Analytics Dashboard.pbix  # Power BI dashboard (interactive)
│   ├── Ecommerce Analytics Dashboard.pdf   # Dashboard export (static)
│   ├── Ecommerce Dashboard.gif             # Dashboard preview animation
│   └── Demo link of an interactive dashboard.txt
├── plots/
│   ├── eda_dashboard.png                   # EDA visualization panel
│   └── feature_importance.png              # ML feature importance chart
├── docker-compose.yml                      # PostgreSQL container setup
├── generate_notebook.py                    # Notebook generation script
├── requirements.txt                        # Python dependencies
├── .gitignore
└── README.md
```

---

## How to Use

### Prerequisites
- Python 3.8+ with `pip`
- Power BI Desktop (for `.pbix` dashboard)
- Docker (optional, for PostgreSQL)

### Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/harshrajput4343/customer-trends-data-analysis-SQL-Python-PowerBI.git
   cd customer-trends-data-analysis-SQL-Python-PowerBI
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ```

3. **Run the analysis notebook:**
   ```bash
   jupyter notebook notebooks/
   ```
   Open either notebook to explore EDA, sentiment analysis, and ML models.

4. **Open the Power BI dashboard:**
   - Open `dashboard/Ecommerce Analytics Dashboard.pbix` in Power BI Desktop
   - Or use the [published web link](https://app.powerbi.com/view?r=eyJrIjoiZmVlNTUwMzItYjYzOC00ZjQ5LTkwZDYtMmZjOTBkZDU0NmY0IiwidCI6IjZjZTcwOTA0LTUwOWMtNGI0Zi1iNjc2LTJiMGRlZjA3M2U2YyJ9)

5. **(Optional) Start PostgreSQL via Docker:**
   ```bash
   docker-compose up -d
   ```

---

## Results & Insights

- **Purchase Trends** — Monthly and yearly trends revealing seasonal purchase patterns
- **Top Customers** — High-value customer identification and behavioral analysis
- **Product Performance** — Top-performing categories by revenue and volume
- **Sentiment Analysis** — Positive sentiments correlate strongly with higher lifetime margins
- **ML Model** — Feature importance analysis reveals key predictors of subscription status

---

## Acknowledgments

Special thanks to the creators of `TextBlob`, `pandas`, `scikit-learn`, and other open-source libraries used in this project.
