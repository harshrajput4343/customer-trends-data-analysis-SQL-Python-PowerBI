# 📊 Customer Purchase Behavior & Sentiment Analysis
![Premium Dashboard Preview](file:///C:/Users/ASUS/.gemini/antigravity/brain/76f57204-8ed0-4391-8cc6-2402ee80c1cb/modern_ecommerce_bi_dashboard_2026_1776118634882.png)

## 🌟 Project Overview
This project is an **industry-grade data science portfolio piece** that transforms raw e-commerce data into actionable business intelligence. We implement an advanced analytics pipeline to analyze customer lifetime value (LTV), sentiment impact on revenue, and predictive churn/subscription drivers.

---

## 🏗️ Project Structure
```text
├── data/                       # Enriched datasets (Source + Export)
├── notebooks/                  # Advanced Jupyter Analysis (15+ Plots)
├── plots/                      # Publication-quality visualizations
├── sql/                        # Schema & Preprocessing queries
├── dashboard/                  # Power BI Design System & Guide
├── generate_advanced_notebook.py # Automated Pipeline Generator
└── README.md                   # Project Documentation
```

---

## 📈 Visualization Gallery (High-Impact Only)
We've distilled the analysis into the **15 most important visualizations** for executive-level reporting.

| # | Visual Category | Key Feature |
|---|---|---|
| **01** | Data Quality | [Missing Data Audit](plots/01_missing_data_audit.png) |
| **02** | Revenue Analysis | [Revenue Distribution](plots/02_revenue_distribution.png) |
| **03** | Segmentation | [Revenue by Category](plots/03_revenue_by_category.png) |
| **04** | Time Series | [Monthly Revenue Trend](plots/04_monthly_revenue_trend.png) |
| **05** | Performance | [Top 10 Products by Revenue](plots/05_top_products.png) |
| **06** | Loyalty | [Top 10 Customers (LTV)](plots/06_top_customers.png) |
| **07** | Multivariable | [Feature Correlation Matrix](plots/07_correlation_heatmap.png) |
| **08** | Sentiment | [Customer Sentiment Distribution](plots/08_sentiment_distribution.png) |
| **09** | Correlation | [Revenue Variance by Sentiment](plots/09_sentiment_vs_revenue.png) |
| **10** | Detail | [Sentiment Analysis by Product](plots/10_product_sentiment.png) |
| **11** | Feedback | [Feedback Word Cloud](plots/11_wordcloud.png) |
| **12** | Clustering | [RFM Customer Segments](plots/12_rfm_clusters.png) |
| **13** | Retention | [Monthly Cohort Retention Matrix](plots/13_retention_matrix.png) |
| **14** | Predictives | [ML Prediction Performance (ROC)](plots/14_roc_curve.png) |
| **15** | Drivers | [Key Drivers of Subscription](plots/15_feature_importance.png) |

---

## 🛠️ Advanced Technical Implementation

### 1. Robust Feature Engineering
- **RFM Analysis**: Automated scoring across Recency, Frequency, and Monetary dimensions to classify customers.
- **Sentiment Normalization**: Enriched purchase data with NLP-derived sentiment scores.
- **Cohort Analysis**: Dynamic retention tracking calculated by acquisition month.

### 2. Machine Learning Pipeline
- Trained and compared **Logistic Regression**, **Random Forest**, and **XGBoost**.
- Evaluated models using **Stratified 5-Fold Cross-Validation**.
- Optimized for **ROC-AUC** and **F1-Score** to ensure balanced predictive power.

### 3. Enterprise Power BI Integration
- Exported `data/powerbi_advanced_export.csv`, a perfectly structured 35-column dataset.
- Includes pre-calculated measures for rapid dashboard creation.

---

## 🚀 Efficiency & Performance
- **Automated Generation**: Entire notebook and plots generated in < 60 seconds.
- **Scalable SQL**: Core data cleaning offloaded to SQLite/PostgreSQL logic.
- **Premium Aesthetics**: Standardized high-DPI plots with a modern dark-theme design system.
