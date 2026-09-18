# India Electricity Consumption Analysis & Forecasting

---

## 1. Project Overview

This project performs a comprehensive analysis and forecasting of India's daily electricity consumption from 2013 to 2024. It covers national-level trends, state and entity-level comparisons, IQR-based anomaly detection, and a 30-day Holt-Winters forecast — all presented through an interactive Streamlit dashboard built with Plotly visualisations.

The project was built as a Data Analyst internship portfolio submission. All analysis, cleaning, visualisation, anomaly detection, and forecasting code is contained in a single Python file.

---

## 2. Dataset

| Field | Detail |
|-------|--------|
| **Name** | India's Electricity Consumption Dataset |
| **Source** | [Kaggle — State-wise Electricity Consumption in India](https://www.kaggle.com/datasets/aryankhurana1701/state-wise-electricity-consumption-in-india) |
| **Underlying data source** | POSOCO (Power System Operation Corporation of India) |
| **Granularity** | Daily |
| **Coverage** | 2013-01-06 to 2024-09-29 |
| **Records** | ~3,707 daily rows |
| **Geographic scope** | 28 Indian states, major Union Territories, and related entities |
| **Unit of measurement** | MU (Million Units / GWh) |

The dataset contains one row per date with separate consumption columns for each state/entity, plus a `Total Consumption` aggregate column.

> **Note:** 2013, 2023, and 2024 are partial/incomplete years in the dataset.

---

## 3. Objectives

- Load, inspect, and clean the raw dataset.
- Reshape wide-format data into a long-format tidy structure for state-level analysis.
- Compute national and entity-level KPIs, trends, and patterns.
- Detect anomalous consumption days using the IQR statistical method.
- Forecast national total consumption 30 days beyond the last recorded date.
- Present all results through a fully interactive, filterable Streamlit dashboard.

---

## 4. Data Cleaning

Steps applied to the raw dataset:

- Removed the unnamed auto-generated index column.
- Converted the `Dates` column from string to `datetime`.
- Converted all entity and total consumption columns to `float64`.
- Recomputed 6 missing `Total Consumption` values (April 2023) as the row-wise sum of entity columns.
- Rounded `Total Consumption` to 2 decimal places to remove floating-point precision artefacts.
- Sorted all rows chronologically.
- Missing values for entities with sparse data (`Pondy` 56.1%, `Tripura` 47.0%, `DD`, `DNH`, `Essar steel`) were intentionally preserved — not zero-filled.

---

## 5. Exploratory Data Analysis

- **National KPIs:** Total consumption, average daily consumption, highest and lowest single-day consumption.
- **Yearly totals and year-over-year growth rates** (2013–2024).
- **Monthly consumption patterns** (January–December, aggregated across all years).
- **Top states and entities** ranked by total consumption.
- **Data-quality summary:** missing values by column, duplicate checks, date-gap analysis.

---

## 6. Entity / State Analysis

For each selected state or entity the dashboard provides:

- Total consumption, average daily consumption, peak day, lowest day.
- Year-by-year consumption bar chart.
- Month-by-month consumption bar chart (January–December order).

---

## 7. Anomaly Detection

**Method: Interquartile Range (IQR)**

```
Q1  = 25th percentile
Q3  = 75th percentile
IQR = Q3 - Q1
Lower fence = Q1 - 1.5 × IQR
Upper fence = Q3 + 1.5 × IQR
```

Any day whose consumption falls outside `[Lower fence, Upper fence]` is labelled **Anomaly**.

The IQR method is:
- Distribution-free (no normality assumption required).
- Robust to the very outliers it detects.
- Fully transparent — every threshold is derived from the data.

**Entity-specific anomaly detection** is applied separately for each selected state or entity, using that entity's own IQR thresholds. This ensures small states are not penalised by national-scale magnitudes.

**National results:**
- Anomaly days: **37**
- Anomaly rate: **~1.0%**
- IQR Lower fence: 1,926.27 MU
- IQR Upper fence: 5,064.08 MU
- All 37 anomalies are on the **high side** (above the upper fence), consistent with India's rising peak summer demand in 2024.

---

## 8. Forecasting

**Method: Holt-Winters Exponential Smoothing (Triple Exponential Smoothing)**

The model simultaneously captures three components of the time series:
1. **Level** — current baseline.
2. **Trend** — direction and rate of change.
3. **Seasonality** — repeating annual pattern (period = 365 days).

Configuration:
- Additive trend, additive seasonality.
- Seasonal period: **365 days**.
- Optimised via grid search + numerical optimisation (`use_brute=True`).
- Implemented using `statsmodels.tsa.holtwinters.ExponentialSmoothing`.

**Train / Test split (chronological — no shuffling):**
- Training: 2013-01-06 → 2024-07-01
- Holdout test: 2024-07-02 → 2024-09-29 (90 days)
- Forecast: 2024-09-30 → 2024-10-29 (30 days)

**Holdout validation metrics (90-day test period):**

| Metric | Value |
|--------|-------|
| MAE | **183.01 MU** |
| RMSE | **219.93 MU** |
| MAPE | **3.84%** |

> These metrics are computed on the held-out test period and reflect how well the model would have predicted the most recent 90 days of historical data. They do not guarantee accuracy for future predictions.

Missing calendar dates in the dataset are linearly interpolated before model fitting; original values are used for all display purposes.

---

## 9. Interactive Dashboard

The dashboard is built with **Streamlit** and **Plotly** and launches in the default web browser.

### Sidebar Controls

| Control | Description |
|---------|-------------|
| **Select State / Entity** | Filters the Entity Analysis and Anomaly Detection sections |
| **Start Date** | Sets the start of the date range for the Daily Trend chart |
| **End Date** | Sets the end of the date range for the Daily Trend chart |

Each control operates **independently** — changing the entity does not reset the date range, and vice versa.

### Dashboard Sections

| Section | Content |
|---------|---------|
| **KPI Cards** | Total Consumption · Avg Daily Consumption · Peak Day · Anomaly Days |
| **National Trends** | Four tabs: Daily Trend · Monthly Pattern · Yearly Total · By State/Entity |
| **Entity Analysis** | KPI cards + yearly bar chart + monthly bar chart for the selected entity |
| **Anomaly Detection** | Entity-specific IQR anomaly count, rate, upper fence · Annotated time-series chart · Top 10 highest anomaly days table |
| **30-Day Forecast** | Holt-Winters forecast chart (historical + test actual/predicted + forecast + ±RMSE band) · MAE / RMSE / MAPE · Expandable 30-day forecast table |

---

## 10. Key Findings

| Finding | Value |
|---------|-------|
| **Total Consumption (2013–2024)** | 13,185,119.20 MU |
| **Average Daily Consumption** | 3,556.82 MU |
| **Highest Daily Consumption** | 5,460.70 MU — 2024-05-30 |
| **Lowest Daily Consumption** | 2,460.50 MU — 2014-03-01 |
| **National Anomaly Days** | 37 (~1.0% of days) |
| **All anomalies** | High-side (above upper fence) — peak summer demand |
| **Top consuming states** | Maharashtra · UP · Gujarat · Tamil Nadu · Rajasthan |
| **COVID-19 impact (2020)** | Year-over-year consumption dropped ~16.7% |
| **Peak month (all years combined)** | August |
| **Lowest months (all years combined)** | January and December |

> **Partial years note:** 2013 (data from Jan 6), 2023, and 2024 (data to Sep 29) are incomplete calendar years. Year-over-year comparisons involving these years should be interpreted with care.

---

## 11. Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.x | Core language |
| Pandas | ≥ 2.0.0 | Data loading, cleaning, analysis |
| NumPy | ≥ 1.24.0 | Numerical operations, metrics |
| Plotly | ≥ 5.18.0 | Interactive visualisations |
| Streamlit | ≥ 1.28.0 | Dashboard UI |
| Statsmodels | ≥ 0.14.0 | Holt-Winters forecasting |

---

## 12. Project Structure

```
Project/
├── india_electricity_analysis.py   ← All project code (single file)
├── requirements.txt                ← Python dependencies
├── README.md                       ← This file
└── Indias_Electricity_Consumption_Dataset.csv   ← Dataset
```

**All Python project code is contained in `india_electricity_analysis.py`.**

There are no other Python files in this project.

---

## 13. Installation / Setup

### Step 1 — Install Python

Download and install **Python 3.9 or later** from [python.org](https://www.python.org/downloads/).

### Step 2 — Open a terminal in the project directory

Navigate to the folder containing `india_electricity_analysis.py`.

### Step 3 — (Optional) Create a virtual environment

```bash
python -m venv venv
```

Activate it:

- **Windows:** `venv\Scripts\activate`
- **macOS/Linux:** `source venv/bin/activate`

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs all five required packages: `pandas`, `numpy`, `plotly`, `streamlit`, and `statsmodels`.

---

## 14. How to Run

```bash
streamlit run india_electricity_analysis.py
```

Streamlit will start a local web server and automatically open the dashboard in your default browser. If it does not open automatically, navigate to the URL shown in the terminal (typically `http://localhost:8501`).

---

## 15. Dashboard Usage Guide

1. **Select an entity** from the sidebar dropdown ("Select State / Entity") to update the Entity Analysis and Anomaly Detection sections.
2. **Set Start Date** in the sidebar to define the beginning of the trend chart date range.
3. **Set End Date** in the sidebar to define the end of the trend chart date range.
4. **Explore national trends** using the four tabs: Daily Trend, Monthly Pattern, Yearly Total, By State/Entity.
5. **Review the Entity Analysis section** to see yearly and monthly consumption patterns for the selected state.
6. **Check the Anomaly Detection section** for IQR-based anomaly count, rate, upper fence, annotated chart, and top anomaly days — all entity-specific.
7. **Review the Forecast section** to see the 30-day Holt-Winters forecast, validation metrics, and the expandable forecast table.

---

## 16. Limitations

| Limitation | Detail |
|------------|--------|
| **Missing calendar dates** | 578 calendar days are absent from the dataset; gaps are not filled in analysis. |
| **Missing entity values** | `Pondy` (56.1% missing), `Tripura` (47.0%), `DD` (19.1%), `DNH` (19.2%), `Essar steel` (14.4%) have significant gaps. |
| **Partial years** | 2013, 2023, and 2024 do not contain full calendar-year data. Year-level totals are not directly comparable. |
| **Forecast validation** | MAE, RMSE, and MAPE are computed on a held-out historical test period, not on genuinely future data. |
| **Seasonality assumption** | A 365-day seasonal period is assumed. The actual seasonal cycle may vary. |
| **Anomaly causality** | IQR flags statistically unusual days. The method does not explain the cause (e.g., extreme weather, data reporting issues). |
| **Interpolation for forecasting** | Missing dates in the series are linearly interpolated before model fitting. |

---

## 17. Future Improvements

- Add SARIMA or Prophet models for comparison with Holt-Winters.
- Incorporate weather data (temperature, monsoon) to enrich anomaly explanations.
- Add per-capita consumption analysis using population data.
- Fill the 578 missing calendar dates using forward-fill or spline interpolation.
- Add a year-selector filter to allow single-year deep-dives.
- Explore renewable vs. conventional energy breakdown if data becomes available.

---

## 18. Dataset Attribution / Source

**Dataset:** India's Electricity Consumption Dataset

**Kaggle page:** [https://www.kaggle.com/datasets/aryankhurana1701/state-wise-electricity-consumption-in-india](https://www.kaggle.com/datasets/aryankhurana1701/state-wise-electricity-consumption-in-india)

**Underlying data source:** POSOCO (Power System Operation Corporation of India)

---

*Project developed as part of a Data Analyst internship portfolio.*
