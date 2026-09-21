"""
India Electricity Consumption Analysis & Forecasting
Single project file — all code lives here.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ===========================================================================
# Constants
# ===========================================================================

CSV_PATH = "Indias_Electricity_Consumption_Dataset.csv"

# All state/entity consumption columns (everything except Dates and Total)
ENTITY_COLUMNS = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chandigarh",
    "Chhattisgarh", "DD", "Delhi", "DNH", "DVC", "Essar steel", "Goa",
    "Gujarat", "Haryana", "HP", "J&K", "Jharkhand", "Karnataka", "Kerala",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "MP", "Nagaland",
    "Odisha", "Pondy", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "UP", "Uttarakhand", "West Bengal",
]


# ===========================================================================
# SECTION 1 — Data Loading
# ===========================================================================

def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    """
    Load the raw CSV into a DataFrame.

    The first column in the file is an unnamed pandas-generated row index.
    Setting index_col=0 absorbs it so it never appears as a data column.
    No other modifications are made; the caller receives the raw data.
    """
    df = pd.read_csv(path, index_col=0)
    return df


# ===========================================================================
# SECTION 2 — Data Cleaning
# ===========================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw DataFrame.

    Steps (in order):
    1. Convert `Dates` to datetime (errors='coerce' handles any bad rows).
    2. Sort chronologically and reset the integer index.
    3. Convert all entity columns and `Total Consumption` to float
       (errors='coerce' turns non-numeric strings into NaN).
    4. Recompute the 6 missing `Total Consumption` values as the row-wise
       sum of entity columns (skipna=True, matching original calculation).
    5. Round `Total Consumption` to 2 decimal places to remove floating-point
       precision artifacts (e.g. 2845.6999999994 becomes 2845.70).
    6. `Pondy` and `Tripura` NaN values are intentionally left untouched.

    Returns the cleaned DataFrame.
    """
    df = df.copy()

    # ------------------------------------------------------------------
    # Step 1 — Convert Dates to datetime
    # ------------------------------------------------------------------
    df["Dates"] = pd.to_datetime(df["Dates"], errors="coerce")

    # ------------------------------------------------------------------
    # Step 2 — Sort chronologically and reset index
    # ------------------------------------------------------------------
    df = df.sort_values("Dates").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Step 3 — Convert all numeric columns to float
    # ------------------------------------------------------------------
    numeric_cols = ENTITY_COLUMNS + ["Total Consumption"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------------------------------------
    # Step 4 — Recompute the 6 missing Total Consumption values
    # ------------------------------------------------------------------
    entity_cols_present = [c for c in ENTITY_COLUMNS if c in df.columns]
    missing_total_mask = df["Total Consumption"].isna()
    if missing_total_mask.sum() > 0:
        df.loc[missing_total_mask, "Total Consumption"] = (
            df.loc[missing_total_mask, entity_cols_present].sum(axis=1, skipna=True)
        )

    # ------------------------------------------------------------------
    # Step 5 — Round Total Consumption to 2 decimal places
    # ------------------------------------------------------------------
    df["Total Consumption"] = df["Total Consumption"].round(2)

    return df


# ===========================================================================
# Cleaning summary helper
# ===========================================================================

def _print_cleaning_summary(raw: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    print("=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"  Raw shape      : {raw.shape[0]} rows x {raw.shape[1]} cols")
    print(f"  Cleaned shape  : {cleaned.shape[0]} rows x {cleaned.shape[1]} cols")
    print()

    print("  Date column:")
    print(f"    Type         : {cleaned['Dates'].dtype}")
    print(f"    Range        : {cleaned['Dates'].min().date()} to {cleaned['Dates'].max().date()}")
    print(f"    NaT values   : {cleaned['Dates'].isna().sum()}")
    print()

    print("  Missing values by column (only columns with NaN shown):")
    missing = cleaned.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("    None")
    else:
        for col, count in missing.items():
            pct = count / len(cleaned) * 100
            print(f"    {col:<25}: {count:>4} ({pct:.1f}%)")
    print()

    print("  Total Consumption:")
    raw_tc_nan = raw["Total Consumption"].isna().sum() if "Total Consumption" in raw.columns else "N/A"
    print(f"    NaN before fix : {raw_tc_nan}")
    print(f"    NaN after fix  : {cleaned['Total Consumption'].isna().sum()}")
    print(f"    Min            : {cleaned['Total Consumption'].min():.2f}")
    print(f"    Max            : {cleaned['Total Consumption'].max():.2f}")
    print(f"    Mean           : {cleaned['Total Consumption'].mean():.2f}")
    print()
    print("  Pondy / Tripura NaN counts (intentionally untouched):")
    for col in ["Pondy", "Tripura"]:
        n = cleaned[col].isna().sum()
        pct = n / len(cleaned) * 100
        print(f"    {col:<10}: {n} NaN ({pct:.1f}%)")
    print("=" * 60)


# ===========================================================================
# SECTION 3 — Reshape for State Analysis
# ===========================================================================

# Columns excluded from the long-format melt:
#   - "Total Consumption" (aggregate, not a state)
#   - "DVC"              (power utility, not a geographic entity)
#   - "Essar steel"      (industrial consumer, not a state)
EXCLUDE_FROM_MELT = {"Total Consumption", "DVC", "Essar steel"}

# State columns actually used in the long-format table
STATE_COLUMNS = [c for c in ENTITY_COLUMNS if c not in EXCLUDE_FROM_MELT]


def reshape_for_state_analysis(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot the wide-format cleaned DataFrame into a long-format table:

        Dates | State | Consumption

    Rules:
    - Only the columns listed in STATE_COLUMNS are melted.
    - Total Consumption, DVC, and Essar steel are excluded.
    - Missing (NaN) consumption values are kept as-is — not filled with zero.
    - The original cleaned_df is not modified.

    Returns a new DataFrame with columns: Dates, State, Consumption.
    """
    # Keep only Dates + state columns for the melt
    cols_to_use = ["Dates"] + [c for c in STATE_COLUMNS if c in cleaned_df.columns]
    wide = cleaned_df[cols_to_use]

    long_df = wide.melt(
        id_vars="Dates",
        var_name="State",
        value_name="Consumption",
    )

    # Restore chronological order and reset index
    long_df = long_df.sort_values(["Dates", "State"]).reset_index(drop=True)

    return long_df


# ===========================================================================
# SECTION 4 — Basic Data Analysis
# ===========================================================================

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def calculate_kpis(cleaned_df: pd.DataFrame) -> dict:
    """
    Return top-level KPIs computed from the Total Consumption column.

    - total_consumption      : sum of all daily totals (MU)
    - avg_daily_consumption  : mean daily total (MU)
    - highest_daily          : dict with date and value of the peak day
    - lowest_daily           : dict with date and value of the lowest day

    NaN rows in Total Consumption are excluded from all calculations.
    """
    tc = cleaned_df[["Dates", "Total Consumption"]].dropna(subset=["Total Consumption"])

    peak_idx = tc["Total Consumption"].idxmax()
    low_idx  = tc["Total Consumption"].idxmin()

    return {
        "total_consumption":     round(tc["Total Consumption"].sum(), 2),
        "avg_daily_consumption": round(tc["Total Consumption"].mean(), 2),
        "highest_daily": {
            "date":  tc.loc[peak_idx, "Dates"].date(),
            "value": tc.loc[peak_idx, "Total Consumption"],
        },
        "lowest_daily": {
            "date":  tc.loc[low_idx, "Dates"].date(),
            "value": tc.loc[low_idx, "Total Consumption"],
        },
    }


def calculate_yearly_consumption(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with total electricity consumption by calendar year.

    Columns: Year | Total Consumption (MU)
    NaN days are excluded from each year's sum.
    """
    df = cleaned_df[["Dates", "Total Consumption"]].dropna(subset=["Total Consumption"]).copy()
    df["Year"] = df["Dates"].dt.year
    yearly = (
        df.groupby("Year", sort=True)["Total Consumption"]
        .sum()
        .round(2)
        .reset_index()
    )
    yearly.columns = ["Year", "Total Consumption (MU)"]
    return yearly


def calculate_yearly_growth(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with year-over-year growth percentage.

    Columns: Year | Total Consumption (MU) | YoY Growth (%)
    The first year has no previous year, so its growth is NaN.
    """
    yearly = calculate_yearly_consumption(cleaned_df)
    yearly["YoY Growth (%)"] = yearly["Total Consumption (MU)"].pct_change() * 100
    yearly["YoY Growth (%)"] = yearly["YoY Growth (%)"].round(2)
    return yearly


def calculate_monthly_consumption(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with total electricity consumption by calendar month,
    ordered January through December (aggregated across all years).

    Columns: Month | Total Consumption (MU)
    NaN days are excluded.
    """
    df = cleaned_df[["Dates", "Total Consumption"]].dropna(subset=["Total Consumption"]).copy()
    df["Month_Num"]  = df["Dates"].dt.month
    df["Month_Name"] = df["Month_Num"].apply(lambda m: MONTH_NAMES[m - 1])
    monthly = (
        df.groupby(["Month_Num", "Month_Name"], sort=True)["Total Consumption"]
        .sum()
        .round(2)
        .reset_index()
    )
    monthly = monthly[["Month_Name", "Total Consumption (MU)".replace("Total Consumption (MU)", "Total Consumption")]].copy()
    # rename cleanly
    monthly.columns = ["Month", "Total Consumption (MU)"]
    return monthly


def calculate_state_consumption(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame with total consumption by state/entity,
    sorted descending.  NaN values are excluded (not treated as zero).

    Columns: State | Total Consumption (MU)
    """
    state_total = (
        long_df.groupby("State")["Consumption"]
        .sum(min_count=1)   # returns NaN if ALL values for a state are NaN
        .round(2)
        .reset_index()
        .sort_values("Consumption", ascending=False)
        .reset_index(drop=True)
    )
    state_total.columns = ["State", "Total Consumption (MU)"]
    return state_total


def calculate_state_statistics(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-state descriptive statistics.

    Columns: State | Total (MU) | Mean (MU) | Max (MU) | Min (MU)
    All aggregations skip NaN.
    """
    stats = (
        long_df.groupby("State")["Consumption"]
        .agg(
            Total=lambda x: round(x.sum(min_count=1), 2),
            Mean=lambda x:  round(x.mean(), 2),
            Max=lambda x:   round(x.max(), 2),
            Min=lambda x:   round(x.min(), 2),
        )
        .sort_values("Total", ascending=False)
        .reset_index()
    )
    stats.columns = ["State", "Total (MU)", "Mean (MU)", "Max (MU)", "Min (MU)"]
    return stats


def calculate_data_quality_summary(raw_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> dict:
    """
    Return a dictionary describing data quality metrics of the cleaned dataset.

    Keys:
    - num_rows, num_columns
    - missing_by_column   : {col: count} for columns with any NaN
    - duplicate_rows      : int
    - duplicate_dates     : int
    - date_range          : {"start": date, "end": date, "calendar_days": int, "data_rows": int, "missing_dates": int}
    """
    missing_by_col = (
        cleaned_df.isna().sum()
        .pipe(lambda s: s[s > 0])
        .sort_values(ascending=False)
        .to_dict()
    )

    dup_rows  = int(cleaned_df.duplicated().sum())
    dup_dates = int(cleaned_df["Dates"].duplicated().sum())

    start = cleaned_df["Dates"].min().date()
    end   = cleaned_df["Dates"].max().date()
    calendar_days = (cleaned_df["Dates"].max() - cleaned_df["Dates"].min()).days + 1
    missing_dates = calendar_days - len(cleaned_df)

    return {
        "num_rows":        len(cleaned_df),
        "num_columns":     len(cleaned_df.columns),
        "missing_by_column": missing_by_col,
        "duplicate_rows":  dup_rows,
        "duplicate_dates": dup_dates,
        "date_range": {
            "start":         start,
            "end":           end,
            "calendar_days": calendar_days,
            "data_rows":     len(cleaned_df),
            "missing_dates": missing_dates,
        },
    }


# ===========================================================================
# SECTION 5 — State / Entity Analysis
# ===========================================================================

def calculate_state_kpis(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-state/entity KPI table with four metrics.

    Columns: State | Total (MU) | Avg Daily (MU) | Max Day (MU) | Min Day (MU)

    NaN consumption values are excluded from all aggregations.
    States where every value is NaN will have NaN in every metric column.
    """
    stats = (
        long_df.groupby("State")["Consumption"]
        .agg(
            Total   = lambda x: round(x.sum(min_count=1), 2),
            AvgDay  = lambda x: round(x.mean(), 2),
            MaxDay  = lambda x: round(x.max(), 2),
            MinDay  = lambda x: round(x.min(), 2),
        )
        .sort_values("Total", ascending=False)
        .reset_index()
    )
    stats.columns = ["State", "Total (MU)", "Avg Daily (MU)", "Max Day (MU)", "Min Day (MU)"]
    return stats


def calculate_top_entities(long_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Return the top-N entities ranked by total consumption.

    Rank is determined dynamically from the actual data — no hard-coded order.
    States with all-NaN consumption are excluded before ranking.

    Columns: Rank | State | Total Consumption (MU)
    """
    state_totals = (
        long_df.groupby("State")["Consumption"]
        .sum(min_count=1)
        .dropna()
        .sort_values(ascending=False)
        .head(n)
        .round(2)
        .reset_index()
    )
    state_totals.columns = ["State", "Total Consumption (MU)"]
    state_totals.insert(0, "Rank", range(1, len(state_totals) + 1))
    return state_totals


def calculate_entity_yearly_trend(long_df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Return the total electricity consumption by year for a single entity.

    Parameters
    ----------
    long_df : long-format DataFrame (Dates, State, Consumption)
    entity  : exact State name string, e.g. "Maharashtra"

    Columns: Year | Total Consumption (MU)
    NaN values for that entity are excluded from each year's sum.
    Raises ValueError if entity is not found in the data.
    """
    available = long_df["State"].unique().tolist()
    if entity not in available:
        raise ValueError(
            f"Entity '{entity}' not found. Available: {sorted(available)}"
        )

    df = long_df[long_df["State"] == entity].copy()
    df["Year"] = df["Dates"].dt.year

    yearly = (
        df.groupby("Year")["Consumption"]
        .sum(min_count=1)
        .round(2)
        .reset_index()
    )
    yearly.columns = ["Year", "Total Consumption (MU)"]
    return yearly


def calculate_entity_monthly_trend(long_df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Return total electricity consumption by calendar month for a single entity,
    ordered January through December (aggregated across all years).

    Parameters
    ----------
    long_df : long-format DataFrame (Dates, State, Consumption)
    entity  : exact State name string, e.g. "Maharashtra"

    Columns: Month | Total Consumption (MU)
    NaN values are excluded from each month's sum.
    Raises ValueError if entity is not found.
    """
    available = long_df["State"].unique().tolist()
    if entity not in available:
        raise ValueError(
            f"Entity '{entity}' not found. Available: {sorted(available)}"
        )

    df = long_df[long_df["State"] == entity].copy()
    df["Month_Num"]  = df["Dates"].dt.month
    df["Month_Name"] = df["Month_Num"].apply(lambda m: MONTH_NAMES[m - 1])

    monthly = (
        df.groupby(["Month_Num", "Month_Name"])["Consumption"]
        .sum(min_count=1)
        .round(2)
        .reset_index()
        .sort_values("Month_Num")
    )
    monthly = monthly[["Month_Name", "Consumption"]].copy()
    monthly.columns = ["Month", "Total Consumption (MU)"]
    return monthly


# ===========================================================================
# SECTION 6 — Anomaly / Unusual Consumption Detection
# ===========================================================================
#
# Method: Interquartile Range (IQR)
# ─────────────────────────────────
# For a given series of consumption values:
#   Q1  = 25th percentile
#   Q3  = 75th percentile
#   IQR = Q3 - Q1
#   Lower fence = Q1 - 1.5 × IQR
#   Upper fence = Q3 + 1.5 × IQR
#
# Any day whose consumption falls outside [Lower fence, Upper fence]
# is labelled "Anomaly"; all others are labelled "Normal".
#
# Why IQR?
#   - Distribution-free: makes no assumption that consumption is normal.
#   - Robust: not distorted by the very outliers it is trying to detect.
#   - Transparent: every threshold is derived from the data itself,
#     making the result easy to explain in a portfolio context.
# ===========================================================================

_ANOMALY_LABEL = "Anomaly"
_NORMAL_LABEL  = "Normal"


def _iqr_bounds(series: pd.Series):
    """
    Compute IQR-based lower and upper fences for a numeric Series.
    NaN values are ignored during percentile calculation.
    Returns (lower_fence, upper_fence).
    """
    q1  = series.quantile(0.25)
    q3  = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def detect_daily_anomalies(cleaned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect unusually high or low national daily consumption using the IQR method.

    A copy of the cleaned DataFrame is returned with one additional column:
        Anomaly : "Anomaly" | "Normal" | NaN  (NaN if Total Consumption is NaN)

    The original DataFrame is not modified.
    Thresholds are derived solely from the data — no hard-coded values.
    """
    df = cleaned_df[["Dates", "Total Consumption"]].copy()

    series = df["Total Consumption"].dropna()
    lower, upper = _iqr_bounds(series)

    def _label(val):
        if pd.isna(val):
            return np.nan
        return _ANOMALY_LABEL if (val < lower or val > upper) else _NORMAL_LABEL

    df["Anomaly"] = df["Total Consumption"].apply(_label)
    df["Lower Fence"] = round(lower, 2)
    df["Upper Fence"] = round(upper, 2)
    return df


def get_anomaly_summary(anomaly_df: pd.DataFrame) -> dict:
    """
    Summarise anomaly detection results from the output of detect_daily_anomalies().

    Returns a dict with:
        total_days       : int   — total rows with a non-NaN Anomaly label
        anomaly_days     : int   — rows labelled "Anomaly"
        anomaly_pct      : float — percentage of labelled days that are anomalies
        lower_fence      : float — IQR lower threshold
        upper_fence      : float — IQR upper threshold
        top_high         : DataFrame — highest anomaly days (Total Consumption desc)
        top_low          : DataFrame — lowest anomaly days (Total Consumption asc)
    """
    labelled   = anomaly_df.dropna(subset=["Anomaly"])
    anomalies  = labelled[labelled["Anomaly"] == _ANOMALY_LABEL].copy()

    total_days   = len(labelled)
    anomaly_days = len(anomalies)
    anomaly_pct  = round(anomaly_days / total_days * 100, 2) if total_days > 0 else 0.0

    top_high = (
        anomalies.sort_values("Total Consumption", ascending=False)
        [["Dates", "Total Consumption"]]
        .head(10)
        .reset_index(drop=True)
    )
    top_low = (
        anomalies.sort_values("Total Consumption", ascending=True)
        [["Dates", "Total Consumption"]]
        .head(10)
        .reset_index(drop=True)
    )

    return {
        "total_days":   total_days,
        "anomaly_days": anomaly_days,
        "anomaly_pct":  anomaly_pct,
        "lower_fence":  round(anomaly_df["Lower Fence"].iloc[0], 2),
        "upper_fence":  round(anomaly_df["Upper Fence"].iloc[0], 2),
        "top_high":     top_high,
        "top_low":      top_low,
    }


def detect_entity_anomalies(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect unusual consumption for each geographic entity independently
    using the IQR method.

    Each entity's thresholds are computed from that entity's own data,
    so small states are not penalised by large-state magnitudes.

    Returns a copy of long_df with two extra columns:
        Anomaly      : "Anomaly" | "Normal" | NaN
        Lower Fence  : entity-specific lower threshold
        Upper Fence  : entity-specific upper threshold

    Missing (NaN) Consumption values remain NaN in the Anomaly column.
    """
    result_parts = []

    for entity, group in long_df.groupby("State"):
        grp = group.copy()
        series = grp["Consumption"].dropna()

        if len(series) < 4:
            # Too few non-NaN points to compute meaningful IQR fences
            grp["Anomaly"]     = np.nan
            grp["Lower Fence"] = np.nan
            grp["Upper Fence"] = np.nan
        else:
            lower, upper = _iqr_bounds(series)

            def _label(val):
                if pd.isna(val):
                    return np.nan
                return _ANOMALY_LABEL if (val < lower or val > upper) else _NORMAL_LABEL

            grp["Anomaly"]     = grp["Consumption"].apply(_label)
            grp["Lower Fence"] = round(lower, 2)
            grp["Upper Fence"] = round(upper, 2)

        result_parts.append(grp)

    return pd.concat(result_parts).sort_values(["Dates", "State"]).reset_index(drop=True)


def create_anomaly_chart(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Chart — Daily Total Consumption with anomaly days highlighted.

    Normal days: thin blue line.
    Anomaly days: red scatter markers overlaid on the line.
    IQR fence lines: dashed grey reference lines.
    """
    anomaly_df = detect_daily_anomalies(cleaned_df)
    normal     = anomaly_df[anomaly_df["Anomaly"] == _NORMAL_LABEL]
    anomalies  = anomaly_df[anomaly_df["Anomaly"] == _ANOMALY_LABEL]

    lower_fence = anomaly_df["Lower Fence"].iloc[0]
    upper_fence = anomaly_df["Upper Fence"].iloc[0]

    fig = go.Figure()

    # Base line — all valid consumption points
    fig.add_trace(go.Scatter(
        x=anomaly_df["Dates"],
        y=anomaly_df["Total Consumption"],
        mode="lines",
        name="Daily Total",
        line=dict(color="#1f77b4", width=1),
        connectgaps=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Consumption: %{y:,.1f} MU<extra></extra>",
    ))

    # Anomaly markers on top
    fig.add_trace(go.Scatter(
        x=anomalies["Dates"],
        y=anomalies["Total Consumption"],
        mode="markers",
        name="Anomaly",
        marker=dict(color="red", size=6, symbol="circle"),
        hovertemplate="<b>%{x|%d %b %Y}</b><br><b>ANOMALY</b>: %{y:,.1f} MU<extra></extra>",
    ))

    # IQR fence lines
    for fence_val, label, dash in [
        (upper_fence, f"Upper fence ({upper_fence:,.0f})", "dash"),
        (lower_fence, f"Lower fence ({lower_fence:,.0f})", "dot"),
    ]:
        fig.add_hline(
            y=fence_val,
            line=dict(color="grey", width=1, dash=dash),
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=11, color="grey"),
        )

    fig.update_layout(**_base_layout(
        title="India Daily Electricity Consumption — Anomaly Detection (IQR Method)",
        xaxis_title="Date",
        yaxis_title="Total Consumption (MU)",
    ))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    return fig


def detect_single_entity_anomalies(long_df: pd.DataFrame, entity: str) -> pd.DataFrame:
    """
    Run IQR anomaly detection on a single entity's daily consumption.

    Uses the same IQR method as detect_entity_anomalies() but returns a
    DataFrame shaped like detect_daily_anomalies() output so that the
    existing get_anomaly_summary() helper can consume it directly:

        Dates | Consumption | Anomaly | Lower Fence | Upper Fence

    The column is named "Total Consumption" to match get_anomaly_summary().

    Parameters
    ----------
    long_df : long-format DataFrame (Dates, State, Consumption)
    entity  : exact State name string

    Missing Consumption values remain NaN in the Anomaly column.
    """
    grp = long_df[long_df["State"] == entity][["Dates", "Consumption"]].copy()
    grp = grp.sort_values("Dates").reset_index(drop=True)

    series = grp["Consumption"].dropna()
    if len(series) < 4:
        grp["Anomaly"]     = np.nan
        grp["Lower Fence"] = np.nan
        grp["Upper Fence"] = np.nan
    else:
        lower, upper = _iqr_bounds(series)

        def _label(val):
            if pd.isna(val):
                return np.nan
            return _ANOMALY_LABEL if (val < lower or val > upper) else _NORMAL_LABEL

        grp["Anomaly"]     = grp["Consumption"].apply(_label)
        grp["Lower Fence"] = round(lower, 2)
        grp["Upper Fence"] = round(upper, 2)

    # Rename so get_anomaly_summary() can read "Total Consumption" column
    grp = grp.rename(columns={"Consumption": "Total Consumption"})
    return grp


def create_entity_anomaly_chart(long_df: pd.DataFrame, entity: str) -> go.Figure:
    """
    Plotly chart showing one entity's daily consumption with IQR anomalies.

    Same visual style as create_anomaly_chart() but:
    - Uses entity-specific consumption values and IQR fences.
    - Dynamic title: "<Entity> Daily Electricity Consumption — Anomaly Detection"
    """
    anomaly_df = detect_single_entity_anomalies(long_df, entity)
    anomalies  = anomaly_df[anomaly_df["Anomaly"] == _ANOMALY_LABEL]

    has_fences = anomaly_df["Lower Fence"].notna().any()
    lower_fence = anomaly_df["Lower Fence"].iloc[0] if has_fences else None
    upper_fence = anomaly_df["Upper Fence"].iloc[0] if has_fences else None

    fig = go.Figure()

    # Base consumption line
    fig.add_trace(go.Scatter(
        x=anomaly_df["Dates"],
        y=anomaly_df["Total Consumption"],
        mode="lines",
        name="Daily Consumption",
        line=dict(color="#1f77b4", width=1),
        connectgaps=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Consumption: %{y:,.1f} MU<extra></extra>",
    ))

    # Anomaly markers
    if not anomalies.empty:
        fig.add_trace(go.Scatter(
            x=anomalies["Dates"],
            y=anomalies["Total Consumption"],
            mode="markers",
            name="Anomaly",
            marker=dict(color="red", size=6, symbol="circle"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br><b>ANOMALY</b>: %{y:,.1f} MU<extra></extra>",
        ))

    # IQR fence lines (only if fences were computable)
    if has_fences:
        for fence_val, label, dash in [
            (upper_fence, f"Upper fence ({upper_fence:,.0f})", "dash"),
            (lower_fence, f"Lower fence ({lower_fence:,.0f})", "dot"),
        ]:
            fig.add_hline(
                y=fence_val,
                line=dict(color="grey", width=1, dash=dash),
                annotation_text=label,
                annotation_position="top right",
                annotation_font=dict(size=11, color="grey"),
            )

    fig.update_layout(**_base_layout(
        title=f"{entity} Daily Electricity Consumption — Anomaly Detection (IQR Method)",
        xaxis_title="Date",
        yaxis_title="Consumption (MU)",
    ))
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.15))
    return fig


# ===========================================================================
# SECTION 7 — Forecasting
# ===========================================================================
#
# Method: Holt-Winters Exponential Smoothing (Triple Exponential Smoothing)
# ──────────────────────────────────────────────────────────────────────────
# Holt-Winters models three components of a time series simultaneously:
#   1. Level   — the current baseline value (weighted recent average)
#   2. Trend   — the direction and speed of change over time
#   3. Seasonality — repeating patterns within a fixed period
#
# This implementation uses an additive seasonality with a period of 365 days
# (annual cycle), which is appropriate for electricity demand that follows
# summer/winter patterns.
#
# Why this method?
#   - No deep learning required; fully statistical and interpretable.
#   - Well-suited for data with both trend and seasonality.
#   - Widely taught in data analytics courses and internship projects.
#   - Implemented in statsmodels (ExponentialSmoothing), a standard library.
#
# Train / Test split:
#   - All data up to the last 90 days forms the training set.
#   - The final 90 days form the held-out test set.
#   - The model is fitted on training data only, then used to forecast
#     the test period + an additional `forecast_days` into the future.
#   - Time order is strictly preserved — no shuffling.
#
# Important caveat:
#   Because the dataset has 578 missing calendar dates, the series is first
#   reindexed to a complete daily date range and missing Total Consumption
#   values are linearly interpolated before fitting. This avoids errors
#   from irregular date spacing while staying close to actual values.
#   The raw (non-interpolated) values are preserved for display.
# ===========================================================================

from statsmodels.tsa.holtwinters import ExponentialSmoothing

_TEST_DAYS = 90        # held-out test period length in days
_SEASON_PERIODS = 365  # annual seasonality


def _prepare_forecast_series(cleaned_df: pd.DataFrame) -> pd.Series:
    """
    Build a complete daily date-indexed Series of Total Consumption.

    Steps:
    1. Extract Dates and Total Consumption; drop rows with NaN consumption.
    2. Set Dates as index and sort.
    3. Reindex to a complete calendar day range (fills gaps with NaN).
    4. Linearly interpolate the NaN gaps so the series has no holes.
       (Interpolation is only for fitting; original values are used elsewhere.)

    Returns a pandas Series with a DatetimeIndex at daily frequency.
    """
    ts = (
        cleaned_df[["Dates", "Total Consumption"]]
        .dropna(subset=["Total Consumption"])
        .set_index("Dates")["Total Consumption"]
        .sort_index()
    )
    full_idx = pd.date_range(start=ts.index.min(), end=ts.index.max(), freq="D")
    ts = ts.reindex(full_idx)
    ts = ts.interpolate(method="linear")
    ts.index.freq = "D"
    return ts


def forecast_total_consumption(
    cleaned_df: pd.DataFrame,
    forecast_days: int = 30,
) -> dict:
    """
    Forecast national Total Consumption using Holt-Winters Exponential Smoothing.

    Parameters
    ----------
    cleaned_df    : cleaned wide-format DataFrame
    forecast_days : number of future days to forecast beyond the last data date

    Returns a dict with keys:
        method          : str   — name of the forecasting method
        train_start     : date
        train_end       : date
        test_start      : date
        test_end        : date
        forecast_start  : date  — first day after last data date
        forecast_end    : date  — last forecasted day
        history_dates   : list[date]
        history_values  : list[float]
        test_dates      : list[date]
        test_actual     : list[float]
        test_predicted  : list[float]
        forecast_dates  : list[date]
        forecast_values : list[float]
        metrics         : dict{MAE, RMSE, MAPE}
    """
    ts = _prepare_forecast_series(cleaned_df)

    # ── Train / test split ──────────────────────────────────────────────────
    train_ts = ts.iloc[:-_TEST_DAYS]
    test_ts  = ts.iloc[-_TEST_DAYS:]

    # ── Fit Holt-Winters on training data ───────────────────────────────────
    model = ExponentialSmoothing(
        train_ts,
        trend="add",
        seasonal="add",
        seasonal_periods=_SEASON_PERIODS,
        initialization_method="estimated",
    )
    # use_brute=True does a grid search to find starting values — more
    # robust for long series and avoids optimizer convergence warnings.
    fitted = model.fit(optimized=True, use_brute=True)

    # ── Forecast test period + future days ──────────────────────────────────
    total_steps = _TEST_DAYS + forecast_days
    fcast_all   = fitted.forecast(total_steps)

    test_predicted = fcast_all.iloc[:_TEST_DAYS]
    future_fcast   = fcast_all.iloc[_TEST_DAYS:]

    # ── Evaluation metrics (on test period) ─────────────────────────────────
    actual = test_ts.values
    pred   = test_predicted.values

    mae  = float(np.mean(np.abs(actual - pred)))
    rmse = float(np.sqrt(np.mean((actual - pred) ** 2)))
    # MAPE: only where actual != 0 (all values here are positive)
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)

    return {
        "method":         "Holt-Winters Exponential Smoothing (additive trend + additive seasonality, period=365)",
        "train_start":    train_ts.index[0].date(),
        "train_end":      train_ts.index[-1].date(),
        "test_start":     test_ts.index[0].date(),
        "test_end":       test_ts.index[-1].date(),
        "forecast_start": future_fcast.index[0].date(),
        "forecast_end":   future_fcast.index[-1].date(),
        "history_dates":  [d.date() for d in ts.index],
        "history_values": [round(v, 2) for v in ts.values],
        "test_dates":     [d.date() for d in test_ts.index],
        "test_actual":    [round(v, 2) for v in actual],
        "test_predicted": [round(v, 2) for v in pred],
        "forecast_dates": [d.date() for d in future_fcast.index],
        "forecast_values":[round(float(v), 2) for v in future_fcast.values],
        "metrics": {
            "MAE":  round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE": round(mape, 2),
        },
    }


def create_forecast_chart(forecast_result: dict) -> go.Figure:
    """
    Plotly chart showing:
    - Historical Total Consumption (blue line, last 365 days for readability)
    - Test period actual values (grey line)
    - Test period predicted values (orange dashed line)
    - Future forecast (red dashed line with shaded uncertainty band ±RMSE)
    """
    res = forecast_result
    rmse = res["metrics"]["RMSE"]

    # Last 365 days of history for context
    hist_dates  = res["history_dates"][-365:]
    hist_values = res["history_values"][-365:]

    test_dates  = res["test_dates"]
    test_actual = res["test_actual"]
    test_pred   = res["test_predicted"]

    fc_dates  = res["forecast_dates"]
    fc_values = res["forecast_values"]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_values,
        mode="lines", name="Historical",
        line=dict(color="#1f77b4", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:,.1f} MU<extra></extra>",
    ))

    # Test actual
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_actual,
        mode="lines", name="Test Actual",
        line=dict(color="#555555", width=1.5),
        hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:,.1f} MU<extra></extra>",
    ))

    # Test predicted
    fig.add_trace(go.Scatter(
        x=test_dates, y=test_pred,
        mode="lines", name="Test Predicted",
        line=dict(color="#ff7f0e", width=1.5, dash="dash"),
        hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:,.1f} MU<extra></extra>",
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=fc_values,
        mode="lines", name=f"Forecast (+{len(fc_dates)}d)",
        line=dict(color="#d62728", width=2, dash="dot"),
        hovertemplate="%{x|%d %b %Y}<br>Forecast: %{y:,.1f} MU<extra></extra>",
    ))

    # Uncertainty band (±1 RMSE around forecast)
    upper = [v + rmse for v in fc_values]
    lower = [v - rmse for v in fc_values]
    fig.add_trace(go.Scatter(
        x=fc_dates + fc_dates[::-1],
        y=upper + lower[::-1],
        fill="toself",
        fillcolor="rgba(214,39,40,0.10)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=False,
        name="Forecast band",
    ))

    fig.update_layout(**_base_layout(
        title="India Total Electricity Consumption — Holt-Winters Forecast",
        xaxis_title="Date",
        yaxis_title="Total Consumption (MU)",
    ))
    fig.update_layout(
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18),
    )
    return fig


# ===========================================================================
# SECTION 8 — Core Visualizations
# ===========================================================================

# Years that do not have a full 365/366 days of data in the dataset
PARTIAL_YEARS = {2013, 2023, 2024}

# Shared colour palette
_BLUE   = "#1f77b4"
_ORANGE = "#ff7f0e"
_GREEN  = "#2ca02c"
_PARTIAL_BAR_COLOUR = "#aec7e8"   # lighter blue for partial years


def _base_layout(title: str, xaxis_title: str, yaxis_title: str) -> dict:
    """Return a dict of common layout kwargs for all charts."""
    return dict(
        title=dict(text=title, font=dict(size=18)),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", size=13, color="#333333"),
        xaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
        yaxis=dict(showgrid=True, gridcolor="#eeeeee", linecolor="#cccccc"),
        margin=dict(l=60, r=30, t=70, b=60),
    )


def create_time_series_chart(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Chart 1 — Electricity Use Over Time.

    Line chart of daily Total Consumption across the full date range.
    NaN values produce natural gaps (connectgaps=False).
    """
    tc = cleaned_df[["Dates", "Total Consumption"]].dropna(subset=["Total Consumption"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tc["Dates"],
        y=tc["Total Consumption"],
        mode="lines",
        name="Daily Total",
        line=dict(color=_BLUE, width=1),
        connectgaps=False,
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Consumption: %{y:,.1f} MU<extra></extra>",
    ))

    fig.update_layout(**_base_layout(
        title="India Daily Electricity Consumption (2013–2024)",
        xaxis_title="Date",
        yaxis_title="Total Consumption (MU)",
    ))
    fig.update_layout(hovermode="x unified")
    return fig


def create_monthly_chart(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Chart 2 — Electricity Use by Month.

    Bar chart showing total consumption for each calendar month
    (January–December, aggregated across all years).
    """
    monthly = calculate_monthly_consumption(cleaned_df)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["Month"],
        y=monthly["Total Consumption (MU)"],
        marker_color=_GREEN,
        text=monthly["Total Consumption (MU)"].apply(lambda v: f"{v/1e6:.3f}M"),
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f} MU<extra></extra>",
        name="Monthly Total",
    ))

    fig.update_layout(**_base_layout(
        title="Total Electricity Consumption by Month (All Years Combined)",
        xaxis_title="Month",
        yaxis_title="Total Consumption (MU)",
    ))
    fig.update_layout(showlegend=False, uniformtext_minsize=9, uniformtext_mode="hide")
    return fig


def create_yearly_chart(cleaned_df: pd.DataFrame) -> go.Figure:
    """
    Chart 3 — Yearly Electricity Use.

    Bar chart of total annual consumption.
    Partial years (2013, 2023, 2024) are rendered in a lighter colour and
    annotated with a '*' marker so users know they are incomplete.
    """
    yearly = calculate_yearly_consumption(cleaned_df)

    colours = [
        _PARTIAL_BAR_COLOUR if yr in PARTIAL_YEARS else _BLUE
        for yr in yearly["Year"]
    ]
    labels = [
        f"{yr}*" if yr in PARTIAL_YEARS else str(yr)
        for yr in yearly["Year"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=yearly["Total Consumption (MU)"],
        marker_color=colours,
        hovertemplate="<b>%{x}</b><br>Total: %{y:,.0f} MU<extra></extra>",
        name="Annual Total",
    ))

    # Annotation explaining the * symbol
    fig.add_annotation(
        text="* Partial year (incomplete data)",
        xref="paper", yref="paper",
        x=0.01, y=1.06,
        showarrow=False,
        font=dict(size=11, color="#888888"),
    )

    fig.update_layout(**_base_layout(
        title="Total Electricity Consumption by Year",
        xaxis_title="Year",
        yaxis_title="Total Consumption (MU)",
    ))
    fig.update_layout(showlegend=False)
    return fig


def create_state_chart(long_df: pd.DataFrame) -> go.Figure:
    """
    Chart 4 — Electricity Use by State.

    Horizontal bar chart of total consumption per state/entity,
    sorted dynamically by actual total (descending → top at top of chart).
    """
    state_totals = calculate_state_consumption(long_df)
    # Drop states where total is NaN (all values were missing)
    state_totals = state_totals.dropna(subset=["Total Consumption (MU)"])
    # Sort ascending so the highest bar appears at the top of the horizontal chart
    state_totals = state_totals.sort_values("Total Consumption (MU)", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=state_totals["Total Consumption (MU)"],
        y=state_totals["State"],
        orientation="h",
        marker_color=_ORANGE,
        hovertemplate="<b>%{y}</b><br>Total: %{x:,.0f} MU<extra></extra>",
        name="Total Consumption",
    ))

    fig.update_layout(**_base_layout(
        title="Total Electricity Consumption by State / Entity (2013–2024)",
        xaxis_title="Total Consumption (MU)",
        yaxis_title="",
    ))
    fig.update_layout(
        showlegend=False,
        height=750,
        yaxis=dict(tickfont=dict(size=12)),
    )
    return fig


# ===========================================================================
# SECTION 9 — Streamlit Dashboard
# ===========================================================================

@st.cache_data
def _load_all_data():
    """Load, clean, and reshape data once; cached for the whole session."""
    raw     = load_data()
    cleaned = clean_data(raw)
    long    = reshape_for_state_analysis(cleaned)
    return raw, cleaned, long


def _kpi_card(col, label: str, value: str, sub: str = ""):
    """Render a single KPI metric inside a Streamlit column."""
    with col:
        st.metric(label=label, value=value, delta=sub if sub else None,
                  delta_color="off")


def run_dashboard():
    """
    Streamlit dashboard entry point.

    Sections:
        0  Page config
        1  Sidebar filters
        2  KPI cards
        3  National trend charts
        4  Selected entity section
        5  Anomaly section
        6  Forecast section
    """
    # ── 0. Page config ──────────────────────────────────────────────────────
    st.set_page_config(
        page_title="India Electricity Analysis",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Load data (cached) ───────────────────────────────────────────────────
    raw_df, cleaned_df, long_df = _load_all_data()

    # ── 1. Sidebar ───────────────────────────────────────────────────────────
    #
    # SESSION STATE STRATEGY
    # ─────────────────────────────────────────────────────────────────────────
    # Both widgets are given a stable key= so Streamlit stores user input in
    # st.session_state automatically and never loses it across reruns.
    #
    # The date range is additionally initialised via setdefault so the very
    # first run picks up the full dataset range, but subsequent reruns (caused
    # by any widget interaction — including changing the entity) leave the
    # user's existing selection completely untouched.
    # ─────────────────────────────────────────────────────────────────────────

    min_date = cleaned_df["Dates"].min().date()
    max_date = cleaned_df["Dates"].max().date()

    # ── TWO INDEPENDENT DATE INPUTS ───────────────────────────────────────────
    #
    # We deliberately avoid the single two-date range picker (st.date_input
    # with value=(start, end)).  That widget fires a Streamlit rerun after the
    # user clicks the *first* date, returning a 1-element tuple or bare date.
    # Any attempt to interpret that intermediate value reliably across all
    # Streamlit versions causes the first date to appear to jump or reset.
    #
    # The definitive fix is TWO SEPARATE st.date_input widgets, each with its
    # own stable key.  Every interaction only affects the widget the user
    # actually touched.  The other widget's session_state key is untouched.
    #
    # Keys:
    #   "trend_start_date"  — Start Date widget
    #   "trend_end_date"    — End Date widget
    #
    # Initialise defaults ONCE via `not in st.session_state` guard.
    # ─────────────────────────────────────────────────────────────────────────
    if "trend_start_date" not in st.session_state:
        st.session_state["trend_start_date"] = min_date
    if "trend_end_date" not in st.session_state:
        st.session_state["trend_end_date"] = max_date

    with st.sidebar:
        st.title("Filters")
        st.markdown("---")

        # Entity selector — stable key, unaffected by date changes
        all_states = sorted(long_df["State"].unique().tolist())
        default_entity_idx = (
            all_states.index("Maharashtra") if "Maharashtra" in all_states else 0
        )
        selected_entity = st.selectbox(
            "Select State / Entity",
            options=all_states,
            index=default_entity_idx,
            key="entity_selector",
        )

        # ── Two independent date inputs ───────────────────────────────────────
        # Each widget is bound to its own session_state key.
        # Changing one widget triggers a rerun; the other widget reads from its
        # own unchanged session_state key — so it stays exactly as the user set it.
        st.markdown("**Date Range (Trend Chart)**")

        st.date_input(
            "Start Date",
            min_value=min_date,
            max_value=max_date,
            key="trend_start_date",
        )

        st.date_input(
            "End Date",
            min_value=min_date,
            max_value=max_date,
            key="trend_end_date",
        )

        # Read the authoritative values directly from session state
        start_date = st.session_state["trend_start_date"]
        end_date   = st.session_state["trend_end_date"]

        st.markdown("---")
        st.caption(
            "Data: India Electricity Consumption Dataset  \n"
            "Period: 2013-01-06 to 2024-09-29"
        )

    # ── 2. Title & KPI cards ─────────────────────────────────────────────────
    st.title("India Electricity Consumption Analysis")
    st.markdown("National and state-level analysis of daily electricity consumption (2013–2024).")
    st.markdown("---")

    kpis       = calculate_kpis(cleaned_df)
    anomaly_df = detect_daily_anomalies(cleaned_df)
    a_summary  = get_anomaly_summary(anomaly_df)

    c1, c2, c3, c4 = st.columns(4)
    _kpi_card(c1, "Total Consumption",
              f"{kpis['total_consumption']/1e6:,.2f} M MU")
    _kpi_card(c2, "Avg Daily Consumption",
              f"{kpis['avg_daily_consumption']:,.1f} MU")
    _kpi_card(c3, "Peak Day",
              f"{kpis['highest_daily']['value']:,.1f} MU",
              str(kpis['highest_daily']['date']))
    _kpi_card(c4, "Anomaly Days",
              str(a_summary["anomaly_days"]),
              f"{a_summary['anomaly_pct']}% of days")

    # ── 3. National trend charts ─────────────────────────────────────────────
    st.markdown("---")
    st.header("National Consumption Trends")

    # Validate that Start Date <= End Date before filtering
    if start_date > end_date:
        st.warning(
            f"Start Date ({start_date}) is after End Date ({end_date}). "
            "Please correct the date range. Showing full dataset instead."
        )
        filtered_df = cleaned_df
    else:
        # Convert to pd.Timestamp — works with datetime64[ns] and datetime64[us]
        ts_start = pd.Timestamp(start_date)
        ts_end   = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask        = (cleaned_df["Dates"] >= ts_start) & (cleaned_df["Dates"] <= ts_end)
        filtered_df = cleaned_df[mask]

        if filtered_df.empty:
            st.warning(
                f"No data found between {start_date} and {end_date}. "
                "Showing full dataset instead."
            )
            filtered_df = cleaned_df

    tab_trend, tab_monthly, tab_yearly, tab_state = st.tabs([
        "Daily Trend", "Monthly Pattern", "Yearly Total", "By State / Entity"
    ])

    with tab_trend:
        st.plotly_chart(create_time_series_chart(filtered_df), use_container_width=True)

    with tab_monthly:
        st.plotly_chart(create_monthly_chart(cleaned_df), use_container_width=True)
        st.caption("Aggregated across all years. January and December show the lowest consumption.")

    with tab_yearly:
        st.plotly_chart(create_yearly_chart(cleaned_df), use_container_width=True)
        st.caption("Years marked * are partial (incomplete calendar year data).")

    with tab_state:
        st.plotly_chart(create_state_chart(long_df), use_container_width=True)

    # ── 4. Selected entity section ───────────────────────────────────────────
    st.markdown("---")
    st.header(f"Entity Analysis: {selected_entity}")

    entity_long = long_df[long_df["State"] == selected_entity]
    e_total = entity_long["Consumption"].sum(min_count=1)
    e_mean  = entity_long["Consumption"].mean()
    e_max   = entity_long["Consumption"].max()
    e_min   = entity_long["Consumption"].min()

    ea, eb, ec, ed = st.columns(4)
    _kpi_card(ea, "Total Consumption",   f"{e_total:,.1f} MU" if not pd.isna(e_total) else "N/A")
    _kpi_card(eb, "Avg Daily",           f"{e_mean:,.1f} MU"  if not pd.isna(e_mean)  else "N/A")
    _kpi_card(ec, "Peak Day",            f"{e_max:,.1f} MU"   if not pd.isna(e_max)   else "N/A")
    _kpi_card(ed, "Lowest Day",          f"{e_min:,.1f} MU"   if not pd.isna(e_min)   else "N/A")

    col_yr, col_mo = st.columns(2)

    with col_yr:
        e_yearly = calculate_entity_yearly_trend(long_df, selected_entity)
        fig_ey = go.Figure(go.Bar(
            x=e_yearly["Year"],
            y=e_yearly["Total Consumption (MU)"],
            marker_color=_BLUE,
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} MU<extra></extra>",
        ))
        fig_ey.update_layout(**_base_layout(
            f"{selected_entity} — Yearly Consumption", "Year", "Total (MU)"
        ))
        fig_ey.update_layout(showlegend=False)
        st.plotly_chart(fig_ey, use_container_width=True)

    with col_mo:
        e_monthly = calculate_entity_monthly_trend(long_df, selected_entity)
        fig_em = go.Figure(go.Bar(
            x=e_monthly["Month"],
            y=e_monthly["Total Consumption (MU)"],
            marker_color=_GREEN,
            hovertemplate="<b>%{x}</b><br>%{y:,.1f} MU<extra></extra>",
        ))
        fig_em.update_layout(**_base_layout(
            f"{selected_entity} — Monthly Pattern", "Month", "Total (MU)"
        ))
        fig_em.update_layout(showlegend=False)
        st.plotly_chart(fig_em, use_container_width=True)

    # ── 5. Anomaly section (entity-specific) ────────────────────────────────
    st.markdown("---")
    st.header(f"Anomaly Detection — {selected_entity} (IQR Method)")

    # Compute IQR anomalies for the selected entity using its own thresholds
    entity_anomaly_df = detect_single_entity_anomalies(long_df, selected_entity)
    entity_a_summary  = get_anomaly_summary(entity_anomaly_df)

    aa, ab, ac = st.columns(3)
    _kpi_card(aa, "Anomaly Days",     str(entity_a_summary["anomaly_days"]))
    _kpi_card(ab, "Anomaly Rate",     f"{entity_a_summary['anomaly_pct']}%")
    uf = entity_a_summary["upper_fence"]
    _kpi_card(ac, "IQR Upper Fence",
              f"{uf:,.2f} MU" if not pd.isna(uf) else "N/A")

    st.plotly_chart(create_entity_anomaly_chart(long_df, selected_entity),
                    use_container_width=True)

    with st.expander(f"Top 10 Highest Anomaly Days — {selected_entity}"):
        top_high = entity_a_summary["top_high"].copy()
        top_high["Dates"] = top_high["Dates"].astype(str)
        st.dataframe(top_high, use_container_width=True, hide_index=True)

    # ── 6. Forecast section ──────────────────────────────────────────────────
    st.markdown("---")
    st.header("30-Day Forecast (Holt-Winters Exponential Smoothing)")

    st.info(
        "The forecast uses Holt-Winters Triple Exponential Smoothing "
        "(additive trend + additive seasonality, period = 365 days) fitted on "
        "all data up to the last 90-day test window. "
        "Forecast values are estimates — not actual recorded data.",
        icon="ℹ️",
    )

    with st.spinner("Fitting forecast model…"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = forecast_total_consumption(cleaned_df, forecast_days=30)

    fa, fb, fc_mape = st.columns(3)
    _kpi_card(fa, "MAE",  f"{fc['metrics']['MAE']:,.2f} MU")
    _kpi_card(fb, "RMSE", f"{fc['metrics']['RMSE']:,.2f} MU")
    _kpi_card(fc_mape, "MAPE", f"{fc['metrics']['MAPE']:.2f}%")

    st.plotly_chart(create_forecast_chart(fc), use_container_width=True)

    with st.expander("Forecasted Values (next 30 days)"):
        fc_table = pd.DataFrame({
            "Date":            [str(d) for d in fc["forecast_dates"]],
            "Forecast (MU)":   fc["forecast_values"],
        })
        st.dataframe(fc_table, use_container_width=True, hide_index=True)

    st.caption(
        f"Train: {fc['train_start']} to {fc['train_end']}  |  "
        f"Test: {fc['test_start']} to {fc['test_end']}  |  "
        f"Forecast: {fc['forecast_start']} to {fc['forecast_end']}"
    )


# ===========================================================================
# Entry point
# ===========================================================================
# When launched with `streamlit run india_electricity_analysis.py` the
# Streamlit runtime executes the module at the top level, so we call
# run_dashboard() outside of __main__.  The __main__ guard keeps the
# original CLI analysis output available when run with plain `python`.
# ===========================================================================

# Detect Streamlit runtime and launch the dashboard automatically
try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is not None:
        run_dashboard()
except Exception:
    pass

if __name__ == "__main__":
    import sys
    # If invoked as `streamlit run ...`, Streamlit handles execution above.
    # If invoked as `python india_electricity_analysis.py`, run CLI output.
    raw_df = load_data()
    cleaned_df = clean_data(raw_df)
    _print_cleaning_summary(raw_df, cleaned_df)
    print()
    long_df = reshape_for_state_analysis(cleaned_df)
    SEP = "=" * 60
    kpis = calculate_kpis(cleaned_df)
    print(SEP)
    print("KPIs")
    print(SEP)
    print(f"  Total consumption     : {kpis['total_consumption']:>12,.2f} MU")
    print(f"  Avg daily consumption : {kpis['avg_daily_consumption']:>12,.2f} MU")
    print(f"  Highest daily         : {kpis['highest_daily']['value']:>12,.2f} MU  ({kpis['highest_daily']['date']})")
    print(f"  Lowest daily          : {kpis['lowest_daily']['value']:>12,.2f} MU  ({kpis['lowest_daily']['date']})")
    print(SEP)
    print("Run  `streamlit run india_electricity_analysis.py`  to open the dashboard.")
    print(SEP)
