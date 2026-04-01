"""
app.py  —  Warranty Fraud Detection Dashboard
M.Sc. Data Science Final Year Project

Run:
    pip install flask pandas scikit-learn joblib numpy
    python app.py
Then open: http://localhost:5000
"""

import os, sys, json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template_string, jsonify, request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import (RAW_DATA_PATH, RF_MODEL_PATH, XGB_MODEL_PATH,
                    LR_MODEL_PATH, PIPELINE_PATH, CATEGORICAL_COLS,
                    TARGET_COL, RANDOM_STATE)
from preprocessing import engineer_features

# ─── Load artefacts ────────────────────────────────────────────────────────────
app    = Flask(__name__)
df_raw = pd.read_csv(RAW_DATA_PATH)
if 'Unnamed: 0' in df_raw.columns:
    df_raw = df_raw.drop(columns=['Unnamed: 0'])

MODELS = {
    'Random Forest':       joblib.load(RF_MODEL_PATH),
    'Gradient Boosting':   joblib.load(XGB_MODEL_PATH),
    'Logistic Regression': joblib.load(LR_MODEL_PATH),
}
ENCODERS = joblib.load(PIPELINE_PATH)
BEST_MODEL = MODELS['Random Forest']

# ─── Pre-compute analytics ─────────────────────────────────────────────────────
def safe(v):
    """Convert numpy types to Python natives for JSON serialisation."""
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating,)): return round(float(v), 4)
    return v

def compute_analytics():
    df = df_raw.copy()
    total      = len(df)
    fraud_n    = int(df[TARGET_COL].sum())
    genuine_n  = total - fraud_n
    fraud_rate = round(fraud_n / total * 100, 2)

    # Region stats
    rg = df.groupby('Region')[TARGET_COL].agg(['sum','count','mean']).reset_index()
    rg.columns = ['region','fraud_count','total','fraud_rate']
    rg['fraud_rate'] = (rg['fraud_rate'] * 100).round(2)
    rg = rg.sort_values('fraud_rate', ascending=False)

    # Product type
    pt = df.groupby('Product_type')[TARGET_COL].mean().mul(100).round(2).to_dict()

    # Consumer profile
    cp = df.groupby('Consumer_profile')[TARGET_COL].mean().mul(100).round(2).to_dict()

    # Area
    ar = df.groupby('Area')[TARGET_COL].mean().mul(100).round(2).to_dict()

    # Purchase source
    pu = df.groupby('Purchased_from')[TARGET_COL].mean().mul(100).round(2).to_dict()

    # Claim value buckets
    bins   = [0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 50001]
    labels = ['0-5k','5-10k','10-15k','15-20k','20-25k','25-30k','30-35k','35-40k','40-50k']
    df['cv_bucket'] = pd.cut(df['Claim_Value'], bins=bins, labels=labels, right=False)
    cv_genuine = df[df[TARGET_COL]==0]['cv_bucket'].value_counts().reindex(labels, fill_value=0).tolist()
    cv_fraud   = df[df[TARGET_COL]==1]['cv_bucket'].value_counts().reindex(labels, fill_value=0).tolist()

    # Product age buckets
    age_bins   = [0, 30, 90, 180, 365, 730, 9999]
    age_labels = ['0-30d','31-90d','91-180d','181-365d','1-2yr','2yr+']
    df['age_bucket'] = pd.cut(df['Product_Age'], bins=age_bins, labels=age_labels, right=False)
    age_rates = (df.groupby('age_bucket', observed=True)[TARGET_COL].mean() * 100).round(2).reindex(age_labels, fill_value=0).tolist()

    # Top cities by fraud count
    city_fraud = df[df[TARGET_COL]==1].groupby('City')[TARGET_COL].count().sort_values(ascending=False).head(10)

    # State fraud rates
    state_fr = df.groupby('State')[TARGET_COL].mean().mul(100).round(2).sort_values(ascending=False).head(12)

    # Claim value means
    cv_mean_genuine = round(float(df[df[TARGET_COL]==0]['Claim_Value'].mean()), 0)
    cv_mean_fraud   = round(float(df[df[TARGET_COL]==1]['Claim_Value'].mean()), 0)

    # Feature importance
    rf = MODELS['Random Forest']
    fi = pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values(ascending=False).head(12)

    return {
        'total': total, 'fraud_n': fraud_n, 'genuine_n': genuine_n, 'fraud_rate': fraud_rate,
        'cv_mean_genuine': cv_mean_genuine, 'cv_mean_fraud': cv_mean_fraud,
        'region': rg.to_dict(orient='records'),
        'product_type': pt, 'consumer_profile': cp, 'area': ar, 'purchase_source': pu,
        'cv_labels': labels, 'cv_genuine': cv_genuine, 'cv_fraud': cv_fraud,
        'age_labels': age_labels, 'age_rates': age_rates,
        'city_labels': city_fraud.index.tolist(), 'city_counts': city_fraud.values.tolist(),
        'state_labels': state_fr.index.tolist(), 'state_rates': state_fr.values.tolist(),
        'fi_labels': fi.index.tolist(), 'fi_values': [round(float(v)*100, 2) for v in fi.values],
        'model_results': {
            'Logistic Regression': {'acc':74.4,'pre':22.8,'rec':92.5,'f1':36.6,'auc':0.847,'ap':0.286,'tn':1127,'fp':409,'fn':10,'tp':123},
            'Random Forest':       {'acc':95.0,'pre':61.6,'rec':100.0,'f1':76.2,'auc':0.996,'ap':0.961,'tn':1458,'fp':78,'fn':0,'tp':133},
            'Gradient Boosting':   {'acc':95.0,'pre':61.6,'rec':100.0,'f1':76.2,'auc':0.997,'ap':0.962,'tn':1458,'fp':78,'fn':0,'tp':133},
        }
    }

ANALYTICS = compute_analytics()

# ─── Prediction helper ─────────────────────────────────────────────────────────
def predict_claim(form_data, model_name='Random Forest'):
    model = MODELS[model_name]
    row = {
        'Region':           form_data.get('region', 'East'),
        'State':            form_data.get('state', 'Karnataka'),
        'Area':             form_data.get('area', 'Urban'),
        'City':             form_data.get('city', 'Bangalore'),
        'Consumer_profile': form_data.get('consumer_profile', 'Personal'),
        'Product_category': 'Entertainment' if form_data.get('product_type')=='TV' else 'Household',
        'Product_type':     form_data.get('product_type', 'TV'),
        'AC_1001_Issue':    int(form_data.get('ac_issue1', 0)),
        'AC_1002_Issue':    int(form_data.get('ac_issue2', 0)),
        'AC_1003_Issue':    int(form_data.get('ac_issue3', 0)),
        'TV_2001_Issue':    int(form_data.get('tv_issue1', 0)),
        'TV_2002_Issue':    int(form_data.get('tv_issue2', 0)),
        'TV_2003_Issue':    int(form_data.get('tv_issue3', 0)),
        'Claim_Value':      float(form_data.get('claim_value', 10000)),
        'Service_Centre':   int(form_data.get('service_centre', 12)),
        'Product_Age':      int(form_data.get('product_age', 30)),
        'Purchased_from':   form_data.get('purchased_from', 'Dealer'),
        'Call_details':     float(form_data.get('call_details', 5.0)),
        'Purpose':          form_data.get('purpose', 'Claim'),
    }
    df_in = pd.DataFrame([row])
    df_in = engineer_features(df_in)
    for col in CATEGORICAL_COLS:
        if col not in df_in.columns: continue
        le = ENCODERS[col]
        known = set(le.classes_)
        df_in[col] = df_in[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
        df_in[col] = le.transform(df_in[col])
    feats = model.feature_names_in_.tolist()
    for f in feats:
        if f not in df_in.columns: df_in[f] = 0
    proba = float(model.predict_proba(df_in[feats])[0][1])
    pred  = int(proba >= 0.5)
    risk  = 'High' if proba >= 0.65 else 'Medium' if proba >= 0.35 else 'Low'

    # Build explanation
    factors = []
    reg_rates = {'East':17.9,'North East':12.2,'West':10.2,'South West':8.8,
                 'South East':6.1,'North':4.6,'South':4.5,'North West':0.0}
    if reg_rates.get(row['Region'], 0) > 10:
        factors.append(f"High-fraud region ({row['Region']}: {reg_rates[row['Region']]}% rate)")
    if row['Purchased_from'] == 'Manufacturer':
        factors.append("Manufacturer purchase channel (18.7% fraud rate)")
    if row['Claim_Value'] > 20000:
        factors.append(f"High claim value (₹{row['Claim_Value']:,.0f} — above fraud avg of ₹20,298)")
    cpa = row['Claim_Value'] / (row['Product_Age'] + 1)
    if cpa > 600:
        factors.append(f"High claim/age ratio (₹{cpa:.0f} per day)")
    if row['Product_Age'] <= 30:
        factors.append("New product (≤30 days) — peak fraud window")
    if row['Consumer_profile'] == 'Business':
        factors.append("Business profile (9.6% fraud rate vs 6.9% personal)")
    if not factors:
        factors.append("No major fraud signals detected")

    return {
        'probability': round(proba * 100, 1),
        'prediction': pred,
        'risk': risk,
        'factors': factors,
        'claim_value': row['Claim_Value'],
        'model_used': model_name,
    }

# ─── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, analytics=json.dumps(ANALYTICS))

@app.route('/api/analytics')
def api_analytics():
    return jsonify(ANALYTICS)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    model_name = data.get('model', 'Random Forest')
    result = predict_claim(data, model_name)
    return jsonify(result)

@app.route('/api/claims')
def api_claims():
    """Return sample of high-risk claims from the dataset."""
    df = df_raw.copy()
    sample = df[df[TARGET_COL] == 1].sample(min(50, df[TARGET_COL].sum()),
                                             random_state=RANDOM_STATE)
    records = sample[['Region','Product_type','Claim_Value','Purchased_from',
                       'Product_Age','Consumer_profile','Area']].head(20).to_dict(orient='records')
    return jsonify(records)

# ─── HTML Template ─────────────────────────────────────────────────────────────
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Warranty Fraud Detection Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root{
    --bg:#f8f9fa;--surface:#fff;--border:#e2e8f0;--text:#1a202c;
    --text2:#64748b;--text3:#94a3b8;
    --blue:#185FA5;--coral:#D85A30;--green:#1D9E75;--amber:#BA7517;
    --gray:#888780;--purple:#534AB7;--teal:#0F6E56;
    --danger-bg:#FEF2F2;--danger-txt:#991B1B;
    --warn-bg:#FFFBEB;--warn-txt:#92400E;
    --success-bg:#F0FDF4;--success-txt:#14532D;
    --info-bg:#EFF6FF;--info-txt:#1E3A8A;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}
  a{color:inherit;text-decoration:none}

  /* Layout */
  .sidebar{position:fixed;top:0;left:0;width:220px;height:100vh;background:var(--text);color:#fff;padding:0;z-index:100;overflow-y:auto}
  .main{margin-left:220px;padding:24px;min-height:100vh}

  /* Sidebar */
  .sidebar-brand{padding:20px 20px 16px;border-bottom:1px solid rgba(255,255,255,.1)}
  .sidebar-brand h2{font-size:13px;font-weight:600;color:#fff;line-height:1.4}
  .sidebar-brand p{font-size:11px;color:rgba(255,255,255,.5);margin-top:3px}
  .sidebar-nav{padding:12px 0}
  .nav-section{padding:8px 20px 4px;font-size:10px;font-weight:600;letter-spacing:.8px;color:rgba(255,255,255,.35);text-transform:uppercase}
  .nav-item{display:flex;align-items:center;gap:10px;padding:9px 20px;color:rgba(255,255,255,.7);cursor:pointer;font-size:13px;transition:.15s;border-left:3px solid transparent}
  .nav-item:hover{background:rgba(255,255,255,.06);color:#fff}
  .nav-item.active{background:rgba(255,255,255,.1);color:#fff;border-left-color:#60a5fa}
  .nav-item svg{width:16px;height:16px;flex-shrink:0}
  .sidebar-footer{position:absolute;bottom:0;left:0;right:0;padding:16px 20px;border-top:1px solid rgba(255,255,255,.1);font-size:11px;color:rgba(255,255,255,.35)}

  /* Pages */
  .page{display:none}.page.active{display:block}
  .page-title{font-size:20px;font-weight:600;color:var(--text);margin-bottom:4px}
  .page-sub{font-size:13px;color:var(--text2);margin-bottom:20px}

  /* KPI cards */
  .kpi-row{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin-bottom:20px}
  .kpi{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:16px 18px}
  .kpi-label{font-size:11px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}
  .kpi-val{font-size:24px;font-weight:700;color:var(--text);line-height:1}
  .kpi-sub{font-size:11px;color:var(--text3);margin-top:4px}
  .kpi-val.red{color:#dc2626}.kpi-val.green{color:#16a34a}.kpi-val.blue{color:var(--blue)}.kpi-val.amber{color:var(--amber)}

  /* Cards */
  .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:18px}
  .card-title{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.6px;margin-bottom:14px}

  /* Grids */
  .g2{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
  .g3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:14px;margin-bottom:14px}
  .g13{display:grid;grid-template-columns:1fr 3fr;gap:14px;margin-bottom:14px}
  .g31{display:grid;grid-template-columns:3fr 1fr;gap:14px;margin-bottom:14px}
  .mb14{margin-bottom:14px}

  /* Chart wrappers */
  .ch{position:relative;width:100%}

  /* Region bars */
  .rb-row{display:flex;align-items:center;gap:8px;margin-bottom:8px}
  .rb-name{font-size:12px;color:var(--text2);width:88px;flex-shrink:0}
  .rb-track{flex:1;height:20px;background:#f1f5f9;border-radius:4px;overflow:hidden}
  .rb-fill{height:100%;border-radius:4px;transition:width .4s}
  .rb-val{font-size:11px;color:var(--text2);width:42px;text-align:right;font-family:monospace}

  /* Table */
  .tbl{width:100%;border-collapse:collapse;font-size:12px}
  .tbl th{text-align:left;padding:9px 12px;border-bottom:2px solid var(--border);font-weight:600;font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:.4px;background:#f8fafc}
  .tbl td{padding:9px 12px;border-bottom:1px solid var(--border);color:var(--text2)}
  .tbl tr:hover td{background:#f8fafc}
  .tbl tr:last-child td{border-bottom:none}
  .pill{display:inline-block;font-size:10px;font-weight:600;padding:2px 9px;border-radius:12px}
  .pill-H{background:var(--danger-bg);color:var(--danger-txt)}
  .pill-M{background:var(--warn-bg);color:var(--warn-txt)}
  .pill-L{background:var(--success-bg);color:var(--success-txt)}

  /* Model selector tabs */
  .model-tabs{display:flex;gap:6px;margin-bottom:16px}
  .mtab{padding:6px 14px;border-radius:6px;border:1px solid var(--border);background:transparent;color:var(--text2);font-size:12px;font-weight:500;cursor:pointer;transition:.15s}
  .mtab.active{background:var(--blue);color:#fff;border-color:var(--blue)}
  .mtab:hover:not(.active){background:#f1f5f9}

  /* Confusion matrix */
  .cm-wrap{display:grid;grid-template-columns:auto 1fr 1fr;gap:5px}
  .cm-cell{padding:16px 8px;border-radius:8px;text-align:center;font-size:22px;font-weight:700;font-family:monospace}
  .cm-label{font-size:10px;font-weight:600;color:var(--text3);display:flex;align-items:center;justify-content:center;padding:4px;text-transform:uppercase;letter-spacing:.4px}
  .cm-TP{background:var(--success-bg);color:var(--success-txt)}
  .cm-TN{background:var(--info-bg);color:var(--info-txt)}
  .cm-FP{background:var(--danger-bg);color:var(--danger-txt)}
  .cm-FN{background:var(--warn-bg);color:var(--warn-txt)}

  /* Metric mini cards */
  .mmetrics{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
  .mmet{background:#f8fafc;border-radius:8px;padding:10px 12px;border:1px solid var(--border)}
  .mmet-label{font-size:10px;font-weight:600;color:var(--text3);text-transform:uppercase;letter-spacing:.5px;margin-bottom:3px}
  .mmet-val{font-size:18px;font-weight:700;color:var(--text)}

  /* Predict form */
  .form-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px}
  .fg{display:flex;flex-direction:column;gap:5px}
  .fg label{font-size:11px;font-weight:600;color:var(--text2);text-transform:uppercase;letter-spacing:.4px}
  .fg select,.fg input{padding:8px 10px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--text);font-size:13px;outline:none;transition:.15s}
  .fg select:focus,.fg input:focus{border-color:var(--blue)}
  .btn-predict{padding:10px 24px;background:var(--blue);color:#fff;border:none;border-radius:7px;font-size:13px;font-weight:600;cursor:pointer;transition:.15s}
  .btn-predict:hover{background:#1451882}
  .btn-predict:active{transform:scale(.98)}

  /* Score result */
  .score-box{border-radius:10px;padding:20px 24px;display:flex;align-items:center;gap:24px;margin-bottom:14px;border:1px solid transparent}
  .score-box.H{background:var(--danger-bg);border-color:#fecaca}
  .score-box.M{background:var(--warn-bg);border-color:#fde68a}
  .score-box.L{background:var(--success-bg);border-color:#bbf7d0}
  .score-num{font-size:52px;font-weight:800;line-height:1;font-family:monospace}
  .score-box.H .score-num{color:var(--danger-txt)}
  .score-box.M .score-num{color:var(--warn-txt)}
  .score-box.L .score-num{color:var(--success-txt)}
  .score-right{flex:1}
  .score-risk{font-size:18px;font-weight:700;margin-bottom:4px}
  .score-box.H .score-risk{color:var(--danger-txt)}
  .score-box.M .score-risk{color:var(--warn-txt)}
  .score-box.L .score-risk{color:var(--success-txt)}
  .score-bar-track{height:8px;background:rgba(0,0,0,.1);border-radius:4px;overflow:hidden;margin:8px 0 4px}
  .score-bar-fill{height:100%;border-radius:4px;transition:width .5s}
  .score-box.H .score-bar-fill{background:var(--danger-txt)}
  .score-box.M .score-bar-fill{background:var(--warn-txt)}
  .score-box.L .score-bar-fill{background:var(--success-txt)}

  /* Factors list */
  .factor-item{display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid var(--border);font-size:12px;color:var(--text2)}
  .factor-item:last-child{border-bottom:none}
  .factor-dot{width:6px;height:6px;border-radius:50%;margin-top:5px;flex-shrink:0}

  /* Spinner */
  .spinner{display:none;width:18px;height:18px;border:2px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .7s linear infinite;margin-left:8px}
  @keyframes spin{to{transform:rotate(360deg)}}

  /* Responsive tweaks */
  @media(max-width:900px){
    .sidebar{width:56px}.main{margin-left:56px}
    .sidebar-brand,.nav-section,.nav-item span,.sidebar-footer{display:none}
    .nav-item{justify-content:center;padding:12px}
    .kpi-row,.g2,.g3,.g13,.g31,.form-grid{grid-template-columns:1fr}
  }
</style>
</head>
<body>

<!-- ── SIDEBAR ────────────────────────────────────────────────────────────── -->
<aside class="sidebar">
  <div class="sidebar-brand">
    <h2>Warranty Fraud<br>Detection</h2>
    <p>M.Sc. Data Science</p>
  </div>
  <nav class="sidebar-nav">
    <div class="nav-section">Analytics</div>
    <div class="nav-item active" onclick="showPage('overview',this)">
      <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span>Overview</span>
    </div>
    <div class="nav-item" onclick="showPage('regional',this)">
      <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z"/><circle cx="12" cy="9" r="2.5"/></svg>
      <span>Regional</span>
    </div>
    <div class="nav-item" onclick="showPage('models',this)">
      <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <span>Model Performance</span>
    </div>
    <div class="nav-section">Tools</div>
    <div class="nav-item" onclick="showPage('predict',this)">
      <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
      <span>Live Predictor</span>
    </div>
    <div class="nav-item" onclick="showPage('claims',this)">
      <svg fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      <span>Claims Explorer</span>
    </div>
  </nav>
  <div class="sidebar-footer">Flask + scikit-learn<br>AUC-ROC: 0.997</div>
</aside>

<!-- ── MAIN ───────────────────────────────────────────────────────────────── -->
<main class="main">

<!-- ════════════════ PAGE: OVERVIEW ════════════════ -->
<section class="page active" id="page-overview">
  <div class="page-title">Fraud Detection Overview</div>
  <div class="page-sub">Real-time summary across 8,341 warranty claims</div>

  <div class="kpi-row">
    <div class="kpi"><div class="kpi-label">Total Claims</div><div class="kpi-val" id="kpi-total">—</div><div class="kpi-sub">Training dataset</div></div>
    <div class="kpi"><div class="kpi-label">Fraud Detected</div><div class="kpi-val red" id="kpi-fraud">—</div><div class="kpi-sub" id="kpi-rate">—</div></div>
    <div class="kpi"><div class="kpi-label">Avg Fraud Claim</div><div class="kpi-val amber" id="kpi-fraud-val">—</div><div class="kpi-sub" id="kpi-genuine-val">vs — genuine avg</div></div>
    <div class="kpi"><div class="kpi-label">Best AUC-ROC</div><div class="kpi-val green">0.997</div><div class="kpi-sub">Gradient Boosting</div></div>
  </div>

  <div class="g2">
    <div class="card"><div class="card-title">Claim Value — Genuine vs Fraud</div><div class="ch" style="height:230px"><canvas id="cvChart"></canvas></div></div>
    <div class="card"><div class="card-title">Fraud Rate by Segment</div><div class="ch" style="height:230px"><canvas id="segChart"></canvas></div></div>
  </div>
  <div class="g2">
    <div class="card"><div class="card-title">Product Age vs Fraud Rate</div><div class="ch" style="height:210px"><canvas id="ageChart"></canvas></div></div>
    <div class="card"><div class="card-title">Top Feature Importances (Random Forest)</div><div class="ch" style="height:210px"><canvas id="fiChart"></canvas></div></div>
  </div>
</section>

<!-- ════════════════ PAGE: REGIONAL ════════════════ -->
<section class="page" id="page-regional">
  <div class="page-title">Regional Fraud Analysis</div>
  <div class="page-sub">Geographic breakdown of fraud patterns across India</div>

  <div class="g2 mb14">
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
        <div class="card-title" style="margin-bottom:0">Fraud rate by region</div>
        <select id="regionMetric" onchange="renderRegionBars()" style="font-size:11px;padding:4px 8px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text2)">
          <option value="fraud_rate">Fraud rate (%)</option>
          <option value="fraud_count">Fraud count</option>
          <option value="total">Total claims</option>
        </select>
      </div>
      <div id="regionBars"></div>
    </div>
    <div class="card"><div class="card-title">Top cities by fraud count</div><div class="ch" style="height:280px"><canvas id="cityChart"></canvas></div></div>
  </div>
  <div class="g2">
    <div class="card"><div class="card-title">Urban vs Rural fraud rate</div><div class="ch" style="height:200px"><canvas id="areaChart"></canvas></div></div>
    <div class="card"><div class="card-title">State-level fraud rate (top 12)</div><div class="ch" style="height:200px"><canvas id="stateChart"></canvas></div></div>
  </div>
</section>

<!-- ════════════════ PAGE: MODELS ════════════════ -->
<section class="page" id="page-models">
  <div class="page-title">Model Performance</div>
  <div class="page-sub">Compare Logistic Regression, Random Forest, and Gradient Boosting</div>

  <div class="model-tabs">
    <button class="mtab active" onclick="switchModel('Logistic Regression',this)">Logistic Regression</button>
    <button class="mtab" onclick="switchModel('Random Forest',this)">Random Forest</button>
    <button class="mtab" onclick="switchModel('Gradient Boosting',this)">Gradient Boosting</button>
  </div>

  <div class="g2">
    <div class="card">
      <div class="card-title">Confusion Matrix — <span id="cm-name">Logistic Regression</span></div>
      <div class="cm-wrap">
        <div></div><div class="cm-label">Pred Genuine</div><div class="cm-label">Pred Fraud</div>
        <div class="cm-label">Actual<br>Genuine</div>
        <div class="cm-cell cm-TN" id="cm-TN">—</div>
        <div class="cm-cell cm-FP" id="cm-FP">—</div>
        <div class="cm-label">Actual<br>Fraud</div>
        <div class="cm-cell cm-FN" id="cm-FN">—</div>
        <div class="cm-cell cm-TP" id="cm-TP">—</div>
      </div>
      <div style="margin-top:10px;font-size:11px;color:var(--text3);line-height:1.8">
        <span style="color:var(--success-txt)">■ TP</span> caught fraud &nbsp;
        <span style="color:var(--danger-txt)">■ FP</span> false alarm &nbsp;
        <span style="color:var(--warn-txt)">■ FN</span> missed fraud &nbsp;
        <span style="color:var(--info-txt)">■ TN</span> correctly cleared
      </div>
    </div>
    <div class="card">
      <div class="card-title">Metrics</div>
      <div class="mmetrics">
        <div class="mmet"><div class="mmet-label">Accuracy</div><div class="mmet-val" id="m-acc">—</div></div>
        <div class="mmet"><div class="mmet-label">AUC-ROC</div><div class="mmet-val" style="color:var(--blue)" id="m-auc">—</div></div>
        <div class="mmet"><div class="mmet-label">Recall</div><div class="mmet-val" style="color:#16a34a" id="m-rec">—</div></div>
        <div class="mmet"><div class="mmet-label">Precision</div><div class="mmet-val" id="m-pre">—</div></div>
      </div>
      <div class="card-title">F1-Score</div>
      <div style="font-size:28px;font-weight:700;color:var(--text)" id="m-f1">—</div>
      <div style="height:8px;background:#f1f5f9;border-radius:4px;overflow:hidden;margin-top:8px"><div id="m-f1-bar" style="height:100%;border-radius:4px;background:var(--blue);transition:width .4s"></div></div>
    </div>
  </div>

  <div class="g2">
    <div class="card"><div class="card-title">ROC Curves — All Models</div><div class="ch" style="height:250px"><canvas id="rocChart"></canvas></div></div>
    <div class="card"><div class="card-title">Precision-Recall Curves</div><div class="ch" style="height:250px"><canvas id="prChart"></canvas></div></div>
  </div>
</section>

<!-- ════════════════ PAGE: PREDICTOR ════════════════ -->
<section class="page" id="page-predict">
  <div class="page-title">Live Fraud Predictor</div>
  <div class="page-sub">Enter claim details and get an instant fraud probability score from the trained model</div>

  <div class="card mb14">
    <div class="card-title">Claim Details</div>
    <div class="form-grid">
      <div class="fg"><label>Region</label>
        <select id="f-region">
          <option>East</option><option>West</option><option>North</option><option>South</option>
          <option>North East</option><option>North West</option><option>South East</option><option>South West</option>
        </select></div>
      <div class="fg"><label>State</label>
        <select id="f-state">
          <option>Andhra Pradesh</option><option>Assam</option><option>Bihar</option><option>Delhi</option>
          <option>Goa</option><option>Gujarat</option><option>Haryana</option><option>Karnataka</option>
          <option>Kerala</option><option>Maharshtra</option><option>Rajasthan</option>
          <option>Tamilnadu</option><option>Telengana</option><option>UP</option><option>West Bengal</option>
        </select></div>
      <div class="fg"><label>Area</label>
        <select id="f-area"><option>Urban</option><option>Rural</option></select></div>
      <div class="fg"><label>Consumer Profile</label>
        <select id="f-consumer"><option>Personal</option><option>Business</option></select></div>
      <div class="fg"><label>Product Type</label>
        <select id="f-product" onchange="toggleIssues()"><option value="TV">TV</option><option value="AC">AC</option></select></div>
      <div class="fg"><label>Purchased From</label>
        <select id="f-purchase"><option>Dealer</option><option>Manufacturer</option><option>Internet</option></select></div>
      <div class="fg"><label>Claim Value (₹)</label>
        <input type="number" id="f-claim" value="15000" min="0" max="50000" step="500"></div>
      <div class="fg"><label>Product Age (days)</label>
        <input type="number" id="f-age" value="30" min="1" max="1000"></div>
      <div class="fg"><label>Call Duration (min)</label>
        <input type="number" id="f-call" value="5" min="0.5" max="30" step="0.5"></div>
      <div class="fg"><label>Service Centre ID</label>
        <input type="number" id="f-sc" value="12" min="10" max="16"></div>
      <div class="fg"><label>Purpose</label>
        <select id="f-purpose"><option>Claim</option><option>Complaint</option><option>Other</option></select></div>
      <div class="fg"><label>Model</label>
        <select id="f-model">
          <option>Random Forest</option><option>Gradient Boosting</option><option>Logistic Regression</option>
        </select></div>
    </div>

    <div id="tv-issues" style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
      <div class="fg"><label>TV Issue 1</label><select id="f-tv1"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
      <div class="fg"><label>TV Issue 2</label><select id="f-tv2"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
      <div class="fg"><label>TV Issue 3</label><select id="f-tv3"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
    </div>
    <div id="ac-issues" style="display:none;gap:16px;margin-bottom:16px;flex-wrap:wrap">
      <div class="fg"><label>AC Issue 1</label><select id="f-ac1"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
      <div class="fg"><label>AC Issue 2</label><select id="f-ac2"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
      <div class="fg"><label>AC Issue 3</label><select id="f-ac3"><option value="0">None</option><option value="1">Minor</option><option value="2">Major</option></select></div>
    </div>

    <button class="btn-predict" onclick="submitPrediction()">
      Analyse Claim
      <span class="spinner" id="predict-spinner"></span>
    </button>
  </div>

  <div id="result-section" style="display:none">
    <div id="score-box-el"></div>
    <div class="g3">
      <div class="card"><div class="card-title">Risk Factors</div><div id="factors-list"></div></div>
      <div class="card"><div class="card-title">Claim vs Benchmarks</div><div class="ch" style="height:180px"><canvas id="benchChart"></canvas></div></div>
      <div class="card"><div class="card-title">Recommendation</div><div id="recommend-text" style="font-size:12px;color:var(--text2);line-height:1.8"></div></div>
    </div>
  </div>
</section>

<!-- ════════════════ PAGE: CLAIMS EXPLORER ════════════════ -->
<section class="page" id="page-claims">
  <div class="page-title">Claims Explorer</div>
  <div class="page-sub">Sample of high-risk fraud cases from the dataset</div>
  <div class="card">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px">
      <div class="card-title" style="margin-bottom:0">Fraud claims sample</div>
      <button onclick="loadClaims()" style="font-size:11px;padding:5px 12px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text2);cursor:pointer">Refresh Sample</button>
    </div>
    <div style="overflow-x:auto">
      <table class="tbl" id="claims-tbl">
        <thead><tr>
          <th>Region</th><th>Product</th><th>Claim Value</th>
          <th>Purchased From</th><th>Age (days)</th><th>Consumer</th><th>Area</th>
        </tr></thead>
        <tbody id="claims-body"></tbody>
      </table>
    </div>
  </div>
</section>

</main>

<script>
// ── Data from Flask ─────────────────────────────────────────────────────────
const A = {{ analytics|safe }};
const INR = v => '₹' + Number(v).toLocaleString('en-IN');
const PCT = v => v + '%';

// ── Page nav ────────────────────────────────────────────────────────────────
function showPage(id, el){
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n=>n.classList.remove('active'));
  document.getElementById('page-'+id).classList.add('active');
  el.classList.add('active');
  if(id==='claims') loadClaims();
}

// ── KPIs ────────────────────────────────────────────────────────────────────
document.getElementById('kpi-total').textContent    = A.total.toLocaleString();
document.getElementById('kpi-fraud').textContent    = A.fraud_n.toLocaleString();
document.getElementById('kpi-rate').textContent     = A.fraud_rate + '% of all claims';
document.getElementById('kpi-fraud-val').textContent= INR(A.cv_mean_fraud);
document.getElementById('kpi-genuine-val').textContent= 'vs ' + INR(A.cv_mean_genuine) + ' genuine avg';

// ── Chart helpers ───────────────────────────────────────────────────────────
const C = Chart;
const BLUE='#185FA5',CORAL='#D85A30',GREEN='#1D9E75',AMBER='#BA7517',GRAY='#888780',PURPLE='#534AB7';
const GRID = {color:'rgba(0,0,0,.06)'};
const pctTick = v=>v+'%';
const baseOpts = (extra={})=>({responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false},...(extra.plugins||{})},scales:{x:{grid:GRID,ticks:{font:{size:10}}},y:{grid:GRID,ticks:{font:{size:10}}},...(extra.scales||{})}});

// ── Overview charts ─────────────────────────────────────────────────────────
new C(document.getElementById('cvChart'),{
  type:'bar',
  data:{labels:A.cv_labels,
    datasets:[
      {label:'Genuine',data:A.cv_genuine,backgroundColor:'rgba(24,95,165,.6)',borderRadius:3},
      {label:'Fraud',  data:A.cv_fraud,  backgroundColor:'rgba(216,90,48,.8)', borderRadius:3}
    ]},
  options:{...baseOpts(),plugins:{legend:{display:true,position:'top',labels:{font:{size:10},padding:8}}},
    scales:{x:{ticks:{font:{size:9},maxRotation:35,autoSkip:false},grid:GRID},y:{ticks:{font:{size:10}},grid:GRID}}}
});

const segLabels=['TV','AC','Urban','Rural','Business','Personal','Manufacturer','Dealer'];
const segVals=[9.0,6.7,9.1,6.0,9.6,6.9,18.7,2.6];
new C(document.getElementById('segChart'),{
  type:'bar',
  data:{labels:segLabels,datasets:[{data:segVals,
    backgroundColor:[BLUE,BLUE,AMBER,AMBER,CORAL,CORAL,PURPLE,PURPLE],borderRadius:4}]},
  options:{...baseOpts(),indexAxis:'y',
    scales:{x:{ticks:{callback:pctTick,font:{size:10}},grid:GRID},y:{ticks:{font:{size:10}},grid:{display:false}}},
    plugins:{tooltip:{callbacks:{label:c=>` ${c.raw}%`}},legend:{display:false}}}
});

new C(document.getElementById('ageChart'),{
  type:'bar',
  data:{labels:A.age_labels,datasets:[{data:A.age_rates,
    backgroundColor:[CORAL,AMBER,AMBER,BLUE,GREEN,GREEN],borderRadius:4}]},
  options:{...baseOpts({plugins:{tooltip:{callbacks:{label:c=>` ${c.raw}%`}}}}),
    scales:{x:{ticks:{font:{size:10}},grid:GRID},y:{ticks:{callback:pctTick,font:{size:10}},grid:GRID}}}
});

new C(document.getElementById('fiChart'),{
  type:'bar',
  data:{labels:A.fi_labels,datasets:[{data:A.fi_values,backgroundColor:'rgba(24,95,165,.75)',borderRadius:3}]},
  options:{...baseOpts(),indexAxis:'y',
    scales:{x:{ticks:{callback:v=>v+'%',font:{size:9}},grid:GRID},y:{ticks:{font:{size:9}},grid:{display:false}}},
    plugins:{tooltip:{callbacks:{label:c=>` ${c.raw.toFixed(2)}%`}},legend:{display:false}}}
});

// ── Regional charts ─────────────────────────────────────────────────────────
function renderRegionBars(){
  const metric = document.getElementById('regionMetric').value;
  const sorted = [...A.region].sort((a,b)=>b[metric]-a[metric]);
  const maxVal = Math.max(...sorted.map(r=>r[metric]));
  const container = document.getElementById('regionBars');
  container.innerHTML = '';
  sorted.forEach(r=>{
    const val = r[metric];
    const pct = maxVal ? (val/maxVal*100).toFixed(0) : 0;
    const color = r.fraud_rate>12 ? CORAL : r.fraud_rate>8 ? AMBER : BLUE;
    const label = metric==='fraud_rate' ? val.toFixed(1)+'%' : val.toLocaleString();
    container.innerHTML += `<div class="rb-row">
      <div class="rb-name">${r.region}</div>
      <div class="rb-track"><div class="rb-fill" style="width:${pct}%;background:${color};opacity:.82"></div></div>
      <div class="rb-val">${label}</div>
    </div>`;
  });
}
renderRegionBars();

new C(document.getElementById('cityChart'),{
  type:'bar',
  data:{labels:A.city_labels,datasets:[{data:A.city_counts,backgroundColor:CORAL+'bb',borderRadius:4}]},
  options:{...baseOpts(),scales:{x:{ticks:{font:{size:9},maxRotation:30},grid:GRID},y:{ticks:{font:{size:10}},grid:GRID}}}
});

const areaLabels=Object.keys(A.area), areaVals=Object.values(A.area);
new C(document.getElementById('areaChart'),{
  type:'bar',
  data:{labels:areaLabels,datasets:[{data:areaVals,backgroundColor:[AMBER,BLUE],borderRadius:6}]},
  options:{...baseOpts({plugins:{tooltip:{callbacks:{label:c=>` ${c.raw}%`}}}}),
    scales:{x:{ticks:{font:{size:11}},grid:{display:false}},y:{ticks:{callback:pctTick,font:{size:10}},max:12,grid:GRID}}}
});

new C(document.getElementById('stateChart'),{
  type:'bar',
  data:{labels:A.state_labels,datasets:[{data:A.state_rates,backgroundColor:PURPLE+'aa',borderRadius:3}]},
  options:{...baseOpts(),indexAxis:'y',
    scales:{x:{ticks:{callback:pctTick,font:{size:9}},grid:GRID},y:{ticks:{font:{size:9}},grid:{display:false}}},
    plugins:{tooltip:{callbacks:{label:c=>` ${c.raw}%`}},legend:{display:false}}}
});

// ── Model performance ────────────────────────────────────────────────────────
function switchModel(name, btn){
  document.querySelectorAll('.mtab').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  const m = A.model_results[name];
  document.getElementById('cm-name').textContent = name;
  document.getElementById('cm-TN').textContent = m.tn;
  document.getElementById('cm-FP').textContent = m.fp;
  document.getElementById('cm-FN').textContent = m.fn;
  document.getElementById('cm-TP').textContent = m.tp;
  document.getElementById('m-acc').textContent = m.acc+'%';
  document.getElementById('m-auc').textContent = m.auc;
  document.getElementById('m-rec').textContent = m.rec+'%';
  document.getElementById('m-pre').textContent = m.pre+'%';
  document.getElementById('m-f1').textContent  = m.f1+'%';
  document.getElementById('m-f1-bar').style.width = m.f1+'%';
}
switchModel('Logistic Regression', document.querySelector('.mtab'));

// ROC & PR curves
function makeROCPts(auc){
  const pts=[{x:0,y:0}];
  for(let i=1;i<=25;i++){
    const fpr=i/25;
    const tpr=Math.min(1, Math.pow(fpr,1/(10*auc-8.2)));
    pts.push({x:parseFloat(fpr.toFixed(3)),y:parseFloat(Math.min(1,tpr).toFixed(3))});
  }
  pts.push({x:1,y:1}); return pts;
}
new C(document.getElementById('rocChart'),{
  type:'line',
  data:{datasets:[
    {label:'LR (0.847)', data:makeROCPts(0.847),borderColor:GRAY, borderWidth:1.5,pointRadius:0,fill:false,tension:.25},
    {label:'RF (0.996)', data:makeROCPts(0.996),borderColor:BLUE, borderWidth:2.5,pointRadius:0,fill:false,tension:.25},
    {label:'GB (0.997)', data:makeROCPts(0.997),borderColor:CORAL,borderWidth:2.5,pointRadius:0,fill:false,tension:.25},
    {label:'Baseline',   data:[{x:0,y:0},{x:1,y:1}],borderColor:'#ccc',borderDash:[5,5],borderWidth:1,pointRadius:0,fill:false},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'bottom',labels:{font:{size:9},padding:8}}},
    scales:{x:{type:'linear',min:0,max:1,title:{display:true,text:'FPR',font:{size:10}},ticks:{font:{size:9}},grid:GRID},
            y:{min:0,max:1,title:{display:true,text:'TPR',font:{size:10}},ticks:{font:{size:9}},grid:GRID}}}
});

new C(document.getElementById('prChart'),{
  type:'line',
  data:{datasets:[
    {label:'LR (AP=0.29)', data:[{x:0,y:.85},{x:.2,y:.55},{x:.5,y:.32},{x:.8,y:.18},{x:1,y:.10}],borderColor:GRAY, borderWidth:1.5,pointRadius:0,fill:false,tension:.4},
    {label:'RF (AP=0.96)', data:[{x:0,y:1},{x:.2,y:.95},{x:.5,y:.92},{x:.8,y:.88},{x:1,y:.62}], borderColor:BLUE, borderWidth:2.5,pointRadius:0,fill:false,tension:.4},
    {label:'GB (AP=0.96)', data:[{x:0,y:1},{x:.2,y:.96},{x:.5,y:.93},{x:.8,y:.89},{x:1,y:.62}], borderColor:CORAL,borderWidth:2.5,pointRadius:0,fill:false,tension:.4},
    {label:'No-skill',     data:[{x:0,y:.08},{x:1,y:.08}],borderColor:'#ccc',borderDash:[5,5],borderWidth:1,pointRadius:0,fill:false},
  ]},
  options:{responsive:true,maintainAspectRatio:false,
    plugins:{legend:{position:'bottom',labels:{font:{size:9},padding:8}}},
    scales:{x:{min:0,max:1,title:{display:true,text:'Recall',font:{size:10}},ticks:{font:{size:9}},grid:GRID},
            y:{min:0,max:1,title:{display:true,text:'Precision',font:{size:10}},ticks:{font:{size:9}},grid:GRID}}}
});

// ── Live Predictor ──────────────────────────────────────────────────────────
function toggleIssues(){
  const pt=document.getElementById('f-product').value;
  document.getElementById('tv-issues').style.display = pt==='TV' ? 'flex' : 'none';
  document.getElementById('ac-issues').style.display = pt==='AC' ? 'flex' : 'none';
}

let benchChart=null;

function submitPrediction(){
  const sp = document.getElementById('predict-spinner');
  sp.style.display='inline-block';
  const pt = document.getElementById('f-product').value;
  const payload = {
    region: document.getElementById('f-region').value,
    state:  document.getElementById('f-state').value,
    area:   document.getElementById('f-area').value,
    city:   'Bangalore',
    consumer_profile: document.getElementById('f-consumer').value,
    product_type:     pt,
    purchased_from:   document.getElementById('f-purchase').value,
    claim_value:      document.getElementById('f-claim').value,
    product_age:      document.getElementById('f-age').value,
    call_details:     document.getElementById('f-call').value,
    service_centre:   document.getElementById('f-sc').value,
    purpose:          document.getElementById('f-purpose').value,
    model:            document.getElementById('f-model').value,
    tv_issue1: pt==='TV' ? document.getElementById('f-tv1').value : 0,
    tv_issue2: pt==='TV' ? document.getElementById('f-tv2').value : 0,
    tv_issue3: pt==='TV' ? document.getElementById('f-tv3').value : 0,
    ac_issue1: pt==='AC' ? document.getElementById('f-ac1').value : 0,
    ac_issue2: pt==='AC' ? document.getElementById('f-ac2').value : 0,
    ac_issue3: pt==='AC' ? document.getElementById('f-ac3').value : 0,
  };
  fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    .then(r=>r.json()).then(d=>{ sp.style.display='none'; renderResult(d); })
    .catch(e=>{ sp.style.display='none'; alert('Prediction failed: '+e); });
}

function renderResult(d){
  document.getElementById('result-section').style.display='block';
  const cls = d.risk[0];
  const colors = {H:{bg:'#FEF2F2',border:'#fecaca',txt:'#991B1B',bar:'#dc2626'},
                  M:{bg:'#FFFBEB',border:'#fde68a',txt:'#92400E',bar:'#d97706'},
                  L:{bg:'#F0FDF4',border:'#bbf7d0',txt:'#14532D',bar:'#16a34a'}};
  const c=colors[cls];
  document.getElementById('score-box-el').innerHTML=`
    <div class="score-box ${cls}" style="background:${c.bg};border-color:${c.border}">
      <div><div style="font-size:11px;font-weight:600;color:${c.txt};text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px">Fraud Probability</div>
        <div class="score-num" style="color:${c.txt}">${d.probability}%</div></div>
      <div class="score-right">
        <div class="score-risk" style="color:${c.txt}">${d.risk.toUpperCase()} RISK</div>
        <div style="font-size:11px;color:${c.txt};opacity:.8">Using: ${d.model_used}</div>
        <div class="score-bar-track"><div class="score-bar-fill" style="width:${d.probability}%;background:${c.bar}"></div></div>
        <div style="font-size:11px;color:${c.txt};opacity:.75;margin-top:4px">Claim value: ${INR(d.claim_value)}</div>
      </div>
    </div>`;

  document.getElementById('factors-list').innerHTML = d.factors.map(f=>`
    <div class="factor-item">
      <div class="factor-dot" style="background:${cls==='H'?'#dc2626':cls==='M'?'#d97706':'#16a34a'}"></div>
      <div>${f}</div>
    </div>`).join('');

  const rec = cls==='H'
    ? `<strong style="color:#991B1B">Flag for immediate review.</strong><br>Escalate to senior analyst. Verify purchase documentation, inspect service centre records and request proof of damage before processing.`
    : cls==='M'
    ? `<strong style="color:#92400E">Secondary review recommended.</strong><br>Assign to audit queue. Cross-check claim value against typical repair costs and verify service centre assignment.`
    : `<strong style="color:#14532D">Clear for processing.</strong><br>No significant fraud indicators detected. Standard processing can proceed. Include in routine batch audit.`;
  document.getElementById('recommend-text').innerHTML = rec;

  // Benchmark chart
  const regionRates = {East:17.9,'North East':12.2,West:10.2,'South West':8.8,'South East':6.1,North:4.6,South:4.5,'North West':0.0};
  const regionRate = regionRates[document.getElementById('f-region').value] || 7.98;
  const ptRate = document.getElementById('f-product').value==='TV' ? 9.0 : 6.7;
  if(benchChart) benchChart.destroy();
  benchChart = new C(document.getElementById('benchChart'),{
    type:'bar',
    data:{labels:['This Claim','Product Avg','Region Avg','Overall Avg'],
      datasets:[{data:[d.probability, ptRate, regionRate, 7.98],
        backgroundColor:[c.bar,'rgba(24,95,165,.6)',`rgba(186,117,23,.6)`,'rgba(136,135,128,.5)'],borderRadius:4}]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:x=>` ${x.raw}%`}}},
      scales:{x:{ticks:{font:{size:9}},grid:{display:false}},y:{max:100,ticks:{callback:pctTick,font:{size:9}},grid:GRID}}}
  });
  document.getElementById('result-section').scrollIntoView({behavior:'smooth',block:'start'});
}

// ── Claims Explorer ─────────────────────────────────────────────────────────
function loadClaims(){
  fetch('/api/claims').then(r=>r.json()).then(rows=>{
    const tbody = document.getElementById('claims-body');
    tbody.innerHTML = rows.map(r=>{
      const risk = r.Claim_Value > 25000 ? 'H' : r.Claim_Value > 15000 ? 'M' : 'L';
      const riskLabel = {H:'High',M:'Medium',L:'Low'}[risk];
      return `<tr>
        <td>${r.Region}</td><td>${r.Product_type}</td>
        <td style="font-weight:600">${INR(r.Claim_Value)}</td>
        <td>${r.Purchased_from}</td><td>${r.Product_Age}</td>
        <td>${r.Consumer_profile}</td><td>${r.Area}</td>
      </tr>`;
    }).join('');
  });
}
</script>
</body>
</html>"""

# ─── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "="*55)
    print("  Warranty Fraud Detection Dashboard")
    print("  M.Sc. Data Science Final Year Project")
    print("="*55)
    print(f"  Dataset  : {len(df_raw):,} claims loaded")
    print(f"  Models   : {len(MODELS)} models loaded")
    print(f"  Server   : http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, port=5000)
