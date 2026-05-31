# 📣 MarketMind – Marketing Campaign Analytics Platform

MarketMind is a full-stack Marketing Analytics Platform built with Python and Flask that helps analyze campaign performance, marketing funnels, ROI, ROAS, forecasting, and A/B testing through an interactive dashboard and REST APIs.

The project demonstrates real-world Data Analytics, Business Intelligence, SQL, Statistical Analysis, Data Visualization, ETL Pipelines, and Machine Learning concepts commonly used in marketing analytics.

---

## 🚀 Features

### 📊 Marketing Analytics
- Campaign Performance Analysis
- ROI (Return on Investment)
- ROAS (Return on Ad Spend)
- CTR (Click Through Rate)
- CVR (Conversion Rate)
- CPC (Cost Per Click)
- CPA (Cost Per Acquisition)

### 🔽 Funnel Analytics
- Impressions → Clicks → Conversions Funnel
- Funnel Performance Visualization
- Conversion Analysis

### 🔮 Revenue Forecasting
- Polynomial Regression Forecasting
- Future Revenue Prediction
- Forecast Accuracy Measurement (R² Score)

### 🧪 A/B Testing
- Google Ads vs Meta Ads Comparison
- Statistical Significance Testing
- T-Test Analysis using SciPy

### 🗄️ SQL Analytics
- Channel Performance Analysis
- Objective-Level Analysis
- Industry Analysis
- Revenue Share Calculations
- Advanced SQL Queries using GROUP BY and CTEs

### 📈 Interactive Dashboard
- Executive KPI Dashboard
- ROI Analysis
- ROAS Heatmaps
- Budget vs Revenue Analysis
- Funnel Visualization
- Revenue Forecast Charts

### ⚡ REST APIs
- Marketing KPI APIs
- Campaign Analytics APIs
- Forecasting APIs
- Industry Analytics APIs
- A/B Testing APIs

---

## 🛠️ Tech Stack

### Backend
- Python
- Flask

### Data Processing
- Pandas
- NumPy

### Database
- SQLite

### Machine Learning
- Scikit-Learn
- Linear Regression
- Polynomial Regression

### Statistical Analysis
- SciPy

### Visualization
- Matplotlib
- Seaborn
- Chart.js

### Deployment
- Gunicorn
- Render

---

## 📂 Project Structure

```text
MarketMind
│
├── app.py
├── requirements.txt
├── marketmind.db
├── render.yaml
├── README.md
│
├── ETL Pipeline
├── SQLite Database
├── Marketing Analytics
├── Funnel Analysis
├── Revenue Forecasting
├── A/B Testing
├── SQL Analytics
├── REST APIs
└── Dashboard
```

---

## ⚙️ How to Run Locally

### Prerequisites

- Python 3.11+
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/geniusInCode/MarketMind.git
cd MarketMind
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Mac/Linux

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Application

```bash
python app.py
```

### 6. Open in Browser

```text
http://localhost:5002
```

or

```text
http://127.0.0.1:5002
```

---

## 📡 API Endpoints

| Endpoint | Description |
|-----------|------------|
| `/api/kpis` | Marketing KPIs |
| `/api/channel-performance` | Channel Performance Analysis |
| `/api/objective-analysis` | Objective-Level Analysis |
| `/api/monthly-trend` | Monthly Marketing Trends |
| `/api/top-campaigns` | Top Campaigns by ROAS |
| `/api/forecast` | Revenue Forecast |
| `/api/ab-test` | A/B Testing Results |
| `/api/industry-analysis` | Industry Analytics |

---

## 📊 Dashboard Modules

### 📈 Overview
- Marketing KPIs
- ROI Analysis
- ROAS Heatmaps

### 📊 Channel Analytics
- Channel Performance
- Budget vs Revenue Analysis
- Monthly Revenue Trends

### 🔽 Funnel Analytics
- Conversion Funnel
- Top Campaign Analysis

### 🔮 Forecasting
- Revenue Forecasting
- Industry Analysis

### 🧪 A/B Testing
- Google Ads vs Meta Ads Comparison
- Statistical Significance Testing

### ⚡ REST API Explorer
- API Documentation
- JSON Response Preview

---

## 🎯 Skills Demonstrated

- Marketing Analytics
- Data Analytics
- Business Intelligence
- ETL Pipelines
- SQL
- Statistical Analysis
- Data Visualization
- Machine Learning
- Forecasting
- A/B Testing
- REST API Development
- Flask Development
- Render Deployment

---

## ☁️ Deployment

### Render

Build Command

```bash
pip install -r requirements.txt
```

Start Command

```bash
gunicorn app:app
```

---

## 📝 Future Improvements

- Customer Lifetime Value (CLV) Analysis
- Marketing Attribution Modeling
- Real Dataset Integration
- Campaign Recommendation Engine
- Advanced Time-Series Forecasting
- User Authentication
- Export Reports to PDF/Excel

---

## 👨‍💻 Author

**GeniusInCode**

Marketing Analytics • Data Science • Machine Learning • Business Intelligence

---

⭐ If you found this project useful, consider giving it a star on GitHub.
