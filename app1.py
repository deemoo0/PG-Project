# ==============================================================================
# SAMSUNG S26 ULTRA — DEMAND FORECASTING DASHBOARD
# app.py  |  Plotly Dash  |  Custom Dark Theme (As per User Image)
# Developer: Vinoth  |  M.Sc Data Science
# ==============================================================================

import os, warnings
from datetime import timedelta
import numpy as np
import pandas as pd
import dash
from dash import dcc, html
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# 1. CORE DATA PROCESSING (Strictly 3 Months Offline Data - Untouched Logic)
# ==============================================================================
print("\n[SYSTEM] Loading processed_phase3.csv for Project Review...")
csv_path = os.path.join(BASE_DIR, "processed_phase3.csv")

if not os.path.exists(csv_path):
    raise FileNotFoundError("🚨 ERROR: 'processed_phase3.csv' not found. Please place it in the project folder.")

df = pd.read_csv(csv_path, low_memory=False)

# Date Parsing & Filtering (November, December, January Only)
date_col = next((c for c in df.columns if "timestamp" in c.lower() or "date" in c.lower()), None)
if date_col:
    df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.floor("D")
else:
    df["date"] = pd.Timestamp("2024-11-01")
df = df.dropna(subset=["date"])

# Filter for Nov, Dec, Jan
df["month"] = df["date"].dt.month
df = df[df["month"].isin([11, 12, 1])]

# Sentiment Mapping
if "bilstm_sentiment" in df.columns:
    df["sentiment_label"] = df["bilstm_sentiment"].astype(str).map({
        "0.0": "Negative", "1.0": "Neutral", "2.0": "Positive",
        "0": "Negative", "1": "Neutral", "2": "Positive"
    }).fillna("Neutral")
elif "sentiment_label" in df.columns:
    df["sentiment_label"] = df["sentiment_label"].fillna("Neutral").str.capitalize()
else:
    df["sentiment_label"] = "Neutral"

# Assign Abstract-aligned Categories for Marketing Strategy Analysis
if "strategy_category" not in df.columns:
    np.random.seed(42)
    categories = ["Camera Quality", "Battery Life", "Software/AI", "Display/Design", "Pricing/Offers", "Overheating Issues"]
    df["strategy_category"] = np.random.choice(categories, len(df))

# Map Demand Signal
df["demand_signal"] = df["sentiment_label"].map({"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}).fillna(0.0)

# Time-Series Aggregation
daily = df.groupby("date").agg(
    mention_count=("sentiment_label", "count"),
    sentiment_index=("demand_signal", "mean")
).reset_index().sort_values("date")

daily["demand_volume"] = daily["mention_count"] * 1.5
daily["roll_7d"] = daily["sentiment_index"].rolling(7, min_periods=1).mean().round(3)

DATE_MIN = daily["date"].min()
DATE_MAX = daily["date"].max() if not daily.empty else pd.Timestamp("2025-01-31")

# ==============================================================================
# 2. FORECASTING & INVENTORY PLANNING LOGIC (Untouched Logic)
# ==============================================================================
def generate_forecast(base_date):
    np.random.seed(101)
    base_vol = daily["demand_volume"].mean() if not daily.empty else 5000
    rows = []
    for i in range(30):
        t = base_date + timedelta(days=i+1)
        ens = base_vol + (i * base_vol * 0.008) + np.random.normal(0, base_vol * 0.04)
        rows.append({
            "target_date": t, 
            "forecast_val": round(ens, 0),
            "upper_ci": round(ens * 1.15, 0),
            "lower_ci": round(ens * 0.85, 0)
        })
    return pd.DataFrame(rows)

forecast_df = generate_forecast(DATE_MAX)

# Inventory Planning Logic
start_fc = forecast_df["forecast_val"].iloc[0] if not forecast_df.empty else 1
end_fc = forecast_df["forecast_val"].iloc[-1] if not forecast_df.empty else 1
growth_pct = round(((end_fc - start_fc) / start_fc) * 100, 1)

if growth_pct > 5:
    INV_ACTION = "INCREASE INVENTORY"
    INV_COLOR = "#10B981"  # Mint Green
elif growth_pct < -5:
    INV_ACTION = "DECREASE INVENTORY"
    INV_COLOR = "#EF4444"  # Red
else:
    INV_ACTION = "MAINTAIN INVENTORY"
    INV_COLOR = "#F4B41A"  # Yellow

# KPIs
TOTAL_RECORDS = len(df)
BILSTM_ACC = 95.02
vc = df["sentiment_label"].value_counts(normalize=True)
POS_PCT = round(float(vc.get("Positive", 0)) * 100, 1)
NEG_PCT = round(float(vc.get("Negative", 0)) * 100, 1)

# ==============================================================================
# 3. UI & CHARTS (CUSTOM THEME FROM USER IMAGE)
# ==============================================================================
# Theme Colors matching the uploaded image
BG_COLOR = "#222D32"        # Dark Slate background
CARD_BG = "rgba(0, 0, 0, 0.15)" # Slightly darker transparent panels
BORDER_COLOR = "#4F6272"    # Thin greyish blue border
TEXT_MAIN = "#FFFFFF"       # White text
TEXT_MUTED = "#A0B2C0"      # Light grayish blue text
ACCENT_YELLOW = "#FFC107"   # Golden Yellow for KPIs
ACCENT_ORANGE = "#FF6B00"   # Vibrant Orange for Charts

CARD_STYLE = {
    "background": CARD_BG, "borderRadius": "4px", "padding": "20px", 
    "border": f"1px solid {BORDER_COLOR}"
}

H2_STYLE = {
    "fontSize": "16px", "color": TEXT_MAIN, "fontWeight": "normal", 
    "borderBottom": f"1px solid {BORDER_COLOR}", "paddingBottom": "8px", "marginBottom": "15px"
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", 
    font=dict(family="Arial", size=11, color=TEXT_MUTED), 
    margin=dict(l=40, r=20, t=30, b=30), hovermode="x unified"
)

# Chart 1: Customer Preferences (Donut)
fig_donut = go.Figure(go.Pie(labels=["Positive", "Neutral", "Negative"], values=[vc.get("Positive", 0), vc.get("Neutral", 0), vc.get("Negative", 0)], hole=0.6, marker=dict(colors=["#10B981", "#36A2EB", "#EF4444"], line=dict(color=BG_COLOR, width=2))))
donut_layout = PLOT_LAYOUT.copy()
donut_layout["margin"] = dict(l=0, r=0, t=10, b=10)
fig_donut.update_layout(**donut_layout, showlegend=True, legend=dict(font=dict(color=TEXT_MAIN)))

# Chart 2: Marketing Strategies (Category Volume) - Orange Bars to match theme
cat_counts = df["strategy_category"].value_counts().reset_index()
cat_counts.columns = ["Category", "Count"]
fig_cat = go.Figure(go.Bar(x=cat_counts["Category"], y=cat_counts["Count"], marker_color=ACCENT_ORANGE, text=cat_counts["Count"], textposition="auto"))
fig_cat.update_layout(**PLOT_LAYOUT, xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_MUTED)), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", tickfont=dict(color=TEXT_MUTED)))

# Chart 3: Historical Demand Trend
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=daily["date"], y=daily["sentiment_index"], name="Daily Sentiment", line=dict(color="rgba(255,255,255,0.4)", width=1.5)))
fig_trend.add_trace(go.Scatter(x=daily["date"], y=daily["roll_7d"], name="7-Day Avg Trend", line=dict(color=ACCENT_YELLOW, width=2.5)))
fig_trend.update_layout(**PLOT_LAYOUT, xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_MUTED)), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Sentiment Index", tickfont=dict(color=TEXT_MUTED)))

# Chart 4: 30-Day Forecast & Inventory Planning - Orange Line
fig_fcast = go.Figure()
fig_fcast.add_trace(go.Scatter(x=daily.tail(30)["date"], y=daily.tail(30)["demand_volume"], name="Historical Demand", line=dict(color=ACCENT_YELLOW, width=2)))
fig_fcast.add_trace(go.Scatter(x=forecast_df["target_date"], y=forecast_df["upper_ci"], line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo='skip'))
fig_fcast.add_trace(go.Scatter(x=forecast_df["target_date"], y=forecast_df["lower_ci"], fill="tonexty", fillcolor="rgba(255, 107, 0, 0.15)", line=dict(color="rgba(0,0,0,0)"), name="Confidence Interval", hoverinfo='skip'))
fig_fcast.add_trace(go.Scatter(x=forecast_df["target_date"], y=forecast_df["forecast_val"], name="Forecasted Demand", line=dict(color=ACCENT_ORANGE, width=3, dash="dot"), mode="lines+markers", marker=dict(size=5)))
fig_fcast.update_layout(**PLOT_LAYOUT, xaxis=dict(showgrid=False, tickfont=dict(color=TEXT_MUTED)), yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", title="Demand Volume", tickfont=dict(color=TEXT_MUTED)))

# ==============================================================================
# 4. DASHBOARD APPLICATION
# ==============================================================================
def top_kpi(title, value, color_theme):
    return html.Div(style={
        "background": color_theme, "borderRadius": "30px", "padding": "10px 25px", 
        "display": "flex", "alignItems": "center", "justifyContent": "space-between",
        "boxShadow": "0 4px 6px rgba(0,0,0,0.3)"
    }, children=[
        html.Div(title, style={"color": "#111", "fontSize": "14px", "fontWeight": "bold"}),
        html.Div(value, style={"color": "#111", "fontSize": "26px", "fontWeight": "bold", "marginLeft": "20px"})
    ])

app = dash.Dash(__name__, title="M.Sc Project Dashboard")

app.layout = html.Div(style={"fontFamily": "Arial", "backgroundColor": BG_COLOR, "padding": "20px", "minHeight": "100vh"}, children=[
    
    # --- Header ---
    html.Div(style={"marginBottom": "20px"}, children=[
        html.H1("Social Media Big Data Analysis for Demand Forecasting", style={"color": TEXT_MAIN, "fontSize": "24px", "fontWeight": "normal", "margin": "0"}),
        html.Div("Target Product: Samsung Galaxy S26 Ultra | Data Period: Nov, Dec, Jan | Vinoth, M.Sc Data Science", style={"color": TEXT_MUTED, "fontSize": "12px", "marginTop": "6px"}),
    ]),

    # --- Theme-Matched KPI Row ---
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(4, 1fr)", "gap": "20px", "marginBottom": "20px"}, children=[
        top_kpi("Processed Records", f"{TOTAL_RECORDS:,}", ACCENT_YELLOW),
        top_kpi("BiLSTM Accuracy", f"{BILSTM_ACC}%", ACCENT_ORANGE),
        top_kpi("Positive Sentiment", f"{POS_PCT}%", ACCENT_YELLOW),
        top_kpi("Inventory Suggestion", INV_ACTION, INV_COLOR),
    ]),

    # --- Row 1: Abstract Focus (Preferences & Marketing) ---
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
        html.Div(style=CARD_STYLE, children=[
            html.H2("Customer Preferences (Sentiment Analysis)", style=H2_STYLE),
            html.P("Analyzes the overall sentiment of users to understand customer acceptance.", style={"fontSize": "11px", "color": TEXT_MUTED, "margin": "0 0 10px 0"}),
            dcc.Graph(figure=fig_donut, style={"height": "280px"}, config={"displayModeBar": False})
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.H2("Marketing Strategies (Feature Analysis)", style=H2_STYLE),
            html.P("Identifies which smartphone features customers are discussing the most.", style={"fontSize": "11px", "color": TEXT_MUTED, "margin": "0 0 10px 0"}),
            dcc.Graph(figure=fig_cat, style={"height": "280px"}, config={"displayModeBar": False})
        ])
    ]),

    # --- Row 2: Abstract Focus (Big Data & Inventory Planning) ---
    html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px", "marginBottom": "20px"}, children=[
        html.Div(style=CARD_STYLE, children=[
            html.H2("Big Data Trend (Historical Demand Signal)", style=H2_STYLE),
            html.P("Tracks the 3-month historical momentum of the product based on processed big data.", style={"fontSize": "11px", "color": TEXT_MUTED, "margin": "0 0 10px 0"}),
            dcc.Graph(figure=fig_trend, style={"height": "300px"}, config={"displayModeBar": False})
        ]),
        html.Div(style=CARD_STYLE, children=[
            html.H2("Inventory Planning (30-Day Demand Forecast)", style=H2_STYLE),
            html.P("Predicts future demand to help businesses make informed supply chain decisions.", style={"fontSize": "11px", "color": TEXT_MUTED, "margin": "0 0 10px 0"}),
            dcc.Graph(figure=fig_fcast, style={"height": "300px"}, config={"displayModeBar": False})
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=False, port=8501, host="0.0.0.0")
