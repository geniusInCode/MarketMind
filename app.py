"""
MarketMind – Marketing Campaign Analytics Platform
Tools: Python · Flask · Pandas · NumPy · SQLite/SQL · Matplotlib · Seaborn
       Scikit-Learn · SciPy · Chart.js · REST API · ETL · Funnel · Forecasting
Deployable on Render
"""

from flask import Flask, jsonify, send_file
import pandas as pd
import numpy as np
from scipy import stats
import sqlite3, io, base64, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

app = Flask(__name__)
BG, GRID = '#1e293b', '#334155'

# ═══════════════════════════════════════════════════════════════════════════════
#  ETL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_raw():
    np.random.seed(77)
    n = 500
    channels   = ['Google Ads','Meta Ads','Email','SEO','Influencer','Affiliate']
    objectives = ['Awareness','Lead Gen','Conversion','Retention']
    industries = ['E-Commerce','FinTech','EdTech','HealthTech','SaaS']

    df = pd.DataFrame({
        'campaign_id'   : [f"CAM{str(i).zfill(4)}" for i in range(1, n+1)],
        'channel'       : np.random.choice(channels, n, p=[0.25,0.25,0.20,0.15,0.10,0.05]),
        'objective'     : np.random.choice(objectives, n, p=[0.20,0.30,0.35,0.15]),
        'industry'      : np.random.choice(industries, n),
        'start_date'    : pd.date_range('2023-01-01','2024-12-01', periods=n),
        'budget'        : np.random.uniform(10000, 500000, n),
        'impressions'   : np.random.randint(1000, 5000000, n),
        'clicks'        : np.random.randint(50, 200000, n),
        'conversions'   : np.random.randint(0, 5000, n),
        'revenue'       : np.random.uniform(5000, 800000, n),
        'duration_days' : np.random.randint(7, 90, n),
    })

    # Inject dirty data
    df.loc[np.random.choice(n,20),'budget']  = np.nan
    df.loc[np.random.choice(n,15),'revenue'] = -1
    return df


def clean_data(raw):
    df = raw.copy()
    df['budget']  = df['budget'].fillna(df['budget'].median())
    df = df[df['revenue'] > 0].copy()
    df['budget']  = df['budget'].abs()

    # Funnel metrics
    df['ctr']          = (df['clicks'] / df['impressions'] * 100).round(4)
    df['cvr']          = (df['conversions'] / df['clicks'].replace(0,1) * 100).round(4)
    df['cpc']          = (df['budget'] / df['clicks'].replace(0,1)).round(2)
    df['cpa']          = (df['budget'] / df['conversions'].replace(0,1)).round(2)
    df['roi_pct']      = ((df['revenue'] - df['budget']) / df['budget'] * 100).round(2)
    df['roas']         = (df['revenue'] / df['budget']).round(3)
    df['month']        = df['start_date'].dt.month
    df['quarter']      = df['start_date'].dt.quarter
    df['year']         = df['start_date'].dt.year
    df['month_name']   = df['start_date'].dt.strftime('%b %Y')
    return df


def load_db(df):
    conn = sqlite3.connect('marketmind.db')
    df.to_sql('campaigns', conn, if_exists='replace', index=False)
    conn.commit(); conn.close()


def run_etl():
    print("▶ MarketMind ETL running …")
    raw = generate_raw()
    df  = clean_data(raw)
    load_db(df)
    print(f"✔ ETL done — {len(df)} campaigns loaded")
    return df

DF = run_etl()


# ═══════════════════════════════════════════════════════════════════════════════
#  SQL HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def sql(q):
    conn = sqlite3.connect('marketmind.db')
    df   = pd.read_sql_query(q, conn); conn.close()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  FORECASTING (Polynomial Regression on monthly revenue)
# ═══════════════════════════════════════════════════════════════════════════════

def build_forecast():
    monthly = DF.groupby(['year','month'])['revenue'].sum().reset_index()
    monthly['t'] = range(len(monthly))
    X, y   = monthly[['t']], monthly['revenue']
    model  = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(X, y)
    future_t = pd.DataFrame({'t': range(len(monthly), len(monthly)+6)})
    forecast = model.predict(future_t).clip(0)
    r2 = r2_score(y, model.predict(X))
    return {
        'r2'      : round(r2, 4),
        'forecast': [round(v,2) for v in forecast],
        'periods' : [f"Month +{i+1}" for i in range(6)]
    }

FORECAST = build_forecast()


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=BG)
    buf.seek(0); data = base64.b64encode(buf.read()).decode()
    plt.close(fig); return data

def make_charts():
    plt.rcParams.update({'text.color':'#e2e8f0','axes.labelcolor':'#94a3b8',
                         'xtick.color':'#94a3b8','ytick.color':'#94a3b8'})
    charts = {}

    # 1. ROI by Channel — Seaborn barplot
    ch_roi = DF.groupby('channel')['roi_pct'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(9,4), facecolor=BG); ax.set_facecolor(BG)
    colors = ['#f87171' if v<0 else '#4ade80' for v in ch_roi.values]
    ax.barh(ch_roi.index, ch_roi.values, color=colors)
    ax.axvline(0, color='white', lw=0.8, linestyle='--')
    ax.set_title('Avg ROI % by Channel', color='white', fontsize=13)
    ax.set_xlabel('ROI (%)')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['roi_channel'] = to_b64(fig)

    # 2. Funnel — Conversion stages
    funnel = {
        'Impressions': DF['impressions'].sum(),
        'Clicks'     : DF['clicks'].sum(),
        'Conversions': DF['conversions'].sum(),
    }
    fig, ax = plt.subplots(figsize=(8,4), facecolor=BG); ax.set_facecolor(BG)
    pal = ['#38bdf8','#a78bfa','#4ade80']
    bars = ax.bar(funnel.keys(), funnel.values(), color=pal, width=0.5)
    ax.set_title('Marketing Funnel — Volume by Stage', color='white', fontsize=13)
    ax.set_yscale('log')
    for b, v in zip(bars, funnel.values()):
        ax.text(b.get_x()+b.get_width()/2, v*1.3, f'{v:,.0f}',
                ha='center', va='bottom', color='white', fontsize=10)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['funnel'] = to_b64(fig)

    # 3. ROAS Heatmap — Channel × Objective — Seaborn
    heat = DF.pivot_table(index='channel', columns='objective', values='roas', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(9,5), facecolor=BG); ax.set_facecolor(BG)
    sns.heatmap(heat, annot=True, fmt='.2f', cmap='RdYlGn', ax=ax,
                linewidths=0.5, annot_kws={'color':'white','size':9})
    ax.set_title('Avg ROAS — Channel × Objective', color='white', fontsize=13)
    charts['roas_heat'] = to_b64(fig)

    # 4. Budget vs Revenue Scatter — Matplotlib
    fig, ax = plt.subplots(figsize=(8,5), facecolor=BG); ax.set_facecolor(BG)
    ch_colors = {'Google Ads':'#4ade80','Meta Ads':'#38bdf8','Email':'#facc15',
                 'SEO':'#a78bfa','Influencer':'#f87171','Affiliate':'#fb923c'}
    for ch, grp in DF.groupby('channel'):
        ax.scatter(grp['budget']/1000, grp['revenue']/1000,
                   label=ch, alpha=0.55, s=25, color=ch_colors.get(ch,'#94a3b8'))
    # trend line
    z = np.polyfit(DF['budget']/1000, DF['revenue']/1000, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0, DF['budget'].max()/1000, 100)
    ax.plot(x_line, p(x_line), '--', color='white', lw=1.2, alpha=0.6, label='Trend')
    ax.set_title('Budget vs Revenue by Channel (₹K)', color='white', fontsize=13)
    ax.set_xlabel('Budget (₹K)'); ax.set_ylabel('Revenue (₹K)')
    ax.legend(labelcolor='white', facecolor=BG, edgecolor=GRID, fontsize=8)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    for sp in ['bottom','left']: ax.spines[sp].set_color(GRID)
    charts['scatter'] = to_b64(fig)

    return charts

CHARTS = make_charts()


# ═══════════════════════════════════════════════════════════════════════════════
#  REST API
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/kpis')
def api_kpis():
    return jsonify({
        'total_campaigns' : int(len(DF)),
        'total_budget'    : round(DF['budget'].sum(), 2),
        'total_revenue'   : round(DF['revenue'].sum(), 2),
        'total_conversions': int(DF['conversions'].sum()),
        'avg_roas'        : round(DF['roas'].mean(), 3),
        'avg_roi_pct'     : round(DF['roi_pct'].mean(), 2),
        'avg_ctr_pct'     : round(DF['ctr'].mean(), 4),
        'avg_cvr_pct'     : round(DF['cvr'].mean(), 4),
    })

@app.route('/api/channel-performance')
def api_channel():
    return jsonify(sql("""
        SELECT channel,
               COUNT(*)                        AS campaigns,
               ROUND(SUM(budget),2)            AS total_budget,
               ROUND(SUM(revenue),2)           AS total_revenue,
               ROUND(AVG(roas),3)              AS avg_roas,
               ROUND(AVG(roi_pct),2)           AS avg_roi,
               ROUND(AVG(ctr),4)               AS avg_ctr,
               ROUND(AVG(cvr),4)               AS avg_cvr,
               SUM(conversions)                AS conversions
        FROM   campaigns
        GROUP  BY channel
        ORDER  BY avg_roas DESC
    """).to_dict(orient='records'))

@app.route('/api/objective-analysis')
def api_objective():
    return jsonify(sql("""
        WITH obj_stats AS (
            SELECT objective,
                   COUNT(*)             AS campaigns,
                   ROUND(AVG(roas),3)   AS avg_roas,
                   ROUND(AVG(roi_pct),2)AS avg_roi,
                   ROUND(SUM(revenue),2)AS revenue,
                   SUM(conversions)     AS total_conversions
            FROM campaigns GROUP BY objective
        ),
        total AS (SELECT SUM(revenue) AS t FROM campaigns)
        SELECT o.*, ROUND(o.revenue*100.0/t.t,2) AS rev_share_pct
        FROM   obj_stats o, total t
        ORDER  BY avg_roas DESC
    """).to_dict(orient='records'))

@app.route('/api/monthly-trend')
def api_monthly():
    return jsonify(sql("""
        SELECT year, month,
               COUNT(*)                   AS campaigns,
               ROUND(SUM(budget),2)       AS budget,
               ROUND(SUM(revenue),2)      AS revenue,
               ROUND(AVG(roas),3)         AS avg_roas,
               SUM(conversions)           AS conversions
        FROM   campaigns
        GROUP  BY year, month ORDER BY year, month
    """).to_dict(orient='records'))

@app.route('/api/top-campaigns')
def api_top():
    return jsonify(sql("""
        SELECT campaign_id, channel, objective,
               ROUND(budget,0) AS budget,
               ROUND(revenue,0) AS revenue,
               ROUND(roas,3)   AS roas,
               ROUND(roi_pct,2)AS roi_pct,
               conversions
        FROM   campaigns
        ORDER  BY roas DESC LIMIT 15
    """).to_dict(orient='records'))

@app.route('/api/forecast')
def api_forecast():
    return jsonify(FORECAST)

@app.route('/api/ab-test')
def api_ab():
    ga  = DF[DF['channel']=='Google Ads']['roas']
    meta= DF[DF['channel']=='Meta Ads']['roas']
    t, p = stats.ttest_ind(ga, meta)
    return jsonify({
        'test'        : 'Google Ads vs Meta Ads ROAS',
        'google_ads'  : {'mean': round(ga.mean(),3),  'n': len(ga)},
        'meta_ads'    : {'mean': round(meta.mean(),3), 'n': len(meta)},
        't_stat'      : round(t,4), 'p_value': round(p,4),
        'significant' : bool(p<0.05),
        'conclusion'  : 'Significant ROAS difference' if p<0.05 else 'No significant difference'
    })

@app.route('/api/industry-analysis')
def api_industry():
    return jsonify(sql("""
        SELECT industry,
               COUNT(*) AS campaigns,
               ROUND(AVG(budget),2)  AS avg_budget,
               ROUND(AVG(roas),3)    AS avg_roas,
               ROUND(SUM(revenue),2) AS total_revenue
        FROM   campaigns
        GROUP  BY industry ORDER BY total_revenue DESC
    """).to_dict(orient='records'))

@app.route('/chart/<name>')
def chart(name):
    if name in CHARTS:
        return send_file(io.BytesIO(base64.b64decode(CHARTS[name])), mimetype='image/png')
    return 'Not found', 404


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

HTML = r"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MarketMind Analytics</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif}
.hdr{background:linear-gradient(135deg,#1a1a2e,#0f172a);padding:22px 32px;border-bottom:1px solid #1e293b}
.hdr h1{font-size:24px;color:#fb923c;font-weight:700}
.hdr p{color:#94a3b8;font-size:12px;margin-top:3px}
.nav{display:flex;gap:8px;padding:14px 32px;background:#0f172a;border-bottom:1px solid #1e293b;flex-wrap:wrap}
.pill{padding:6px 16px;border-radius:20px;cursor:pointer;font-size:13px;border:1px solid #334155;color:#94a3b8;background:transparent;transition:all .2s}
.pill.on,.pill:hover{background:#fb923c;color:#0f172a;border-color:#fb923c;font-weight:700}
.wrap{padding:24px 32px}
.kgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(155px,1fr));gap:14px;margin-bottom:24px}
.kcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;transition:transform .2s,border-color .2s}
.kcard:hover{transform:translateY(-3px);border-color:#fb923c}
.klbl{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:1px}
.kval{font-size:22px;font-weight:700;color:#fb923c;margin-top:6px}
.cgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px;margin-bottom:20px}
.ccard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px}
.ccard h3{font-size:13px;color:#cbd5e1;margin-bottom:14px;font-weight:600}
.ccard img{width:100%;border-radius:8px}
.tcard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:20px;margin-bottom:20px;overflow-x:auto}
.tcard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.qbox{background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;padding:9px 12px;color:#64748b;font-size:10px;text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #334155}
td{padding:9px 12px;border-bottom:1px solid #1e293b;color:#cbd5e1}
tr:hover td{background:#334155}
.badge{display:inline-block;padding:2px 9px;border-radius:10px;font-size:11px;font-weight:600}
.gn{background:#14532d;color:#4ade80}.or{background:#431407;color:#fb923c}.bl{background:#1e3a5f;color:#38bdf8}
.sgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;margin-bottom:20px}
.scard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px}
.scard h3{font-size:13px;color:#cbd5e1;margin-bottom:12px;font-weight:600}
.srow{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid #1e293b;font-size:13px}
.srow:last-child{border-bottom:none}.sk{color:#94a3b8}.sv{color:#fb923c;font-weight:600}
.sec{display:none}.sec.on{display:block}
.tag{display:inline-block;padding:2px 8px;border-radius:4px;background:#431407;color:#fb923c;font-size:11px;margin:2px}
.mth{display:inline-block;padding:2px 8px;border-radius:4px;background:#14532d;color:#4ade80;font-size:11px;font-weight:700;margin-right:8px}
.acard{background:#1e293b;border:1px solid #334155;border-radius:12px;padding:18px;margin-bottom:14px}
</style></head>
<body>
<div class="hdr">
  <h1>📣 MarketMind Analytics</h1>
  <p>Python · Flask · Pandas · NumPy · SQLite/SQL · Matplotlib · Seaborn · Scikit-Learn · SciPy · Chart.js · REST API · ETL · Funnel · A/B Testing · Forecasting</p>
</div>
<div class="nav">
  <button class="pill on" onclick="show('ov',this)">📈 Overview</button>
  <button class="pill" onclick="show('ch',this)">📊 Channels</button>
  <button class="pill" onclick="show('fn',this)">🔽 Funnel</button>
  <button class="pill" onclick="show('fc',this)">🔮 Forecast</button>
  <button class="pill" onclick="show('ab',this)">🧪 A/B Test</button>
  <button class="pill" onclick="show('ap',this)">⚡ REST API</button>
</div>
<div class="wrap">

<div id="ov" class="sec on">
  <div class="kgrid" id="kgrid"><div class="kcard"><div class="klbl">Loading…</div></div></div>
  <div class="cgrid">
    <div class="ccard"><h3>📊 Avg ROI % by Channel — Seaborn</h3><img src="/chart/roi_channel"></div>
    <div class="ccard"><h3>🔥 ROAS Heatmap — Channel × Objective — Seaborn</h3><img src="/chart/roas_heat"></div>
  </div>
</div>

<div id="ch" class="sec">
  <div class="tcard">
    <h3>📡 Channel Performance — SQL GROUP BY</h3>
    <div class="qbox">SELECT channel, COUNT(*), SUM(budget), SUM(revenue), AVG(roas), AVG(roi_pct), AVG(ctr), AVG(cvr), SUM(conversions) FROM campaigns GROUP BY channel ORDER BY avg_roas DESC</div>
    <table id="tchan"></table>
  </div>
  <div class="ccard"><h3>💰 Budget vs Revenue Scatter — Matplotlib with Trend Line</h3><img src="/chart/scatter"></div>
  <div class="ccard" style="margin-top:20px"><h3>📊 Monthly Revenue vs Budget — Chart.js</h3><canvas id="monChart" height="80"></canvas></div>
</div>

<div id="fn" class="sec">
  <div class="ccard"><h3>🔽 Conversion Funnel — Matplotlib Log Scale</h3><img src="/chart/funnel"></div>
  <div class="tcard" style="margin-top:20px">
    <h3>🏆 Top 15 Campaigns by ROAS — SQL ORDER BY</h3>
    <div class="qbox">SELECT campaign_id, channel, objective, budget, revenue, roas, roi_pct, conversions FROM campaigns ORDER BY roas DESC LIMIT 15</div>
    <table id="ttop"></table>
  </div>
</div>

<div id="fc" class="sec">
  <div class="sgrid">
    <div class="scard"><h3>🔮 Revenue Forecast (Polynomial Regression)</h3><div id="fcmet"></div></div>
    <div class="scard"><h3>🏭 Industry Analysis — SQL GROUP BY</h3><div id="indmet"></div></div>
  </div>
  <div class="ccard"><h3>📈 Forecast — Next 6 Months — Chart.js</h3><canvas id="fcChart" height="80"></canvas></div>
</div>

<div id="ab" class="sec">
  <div class="sgrid">
    <div class="scard"><h3>🧪 A/B Test — Google Ads vs Meta Ads ROAS</h3><div id="abmet"></div></div>
    <div class="scard"><h3>📊 Objective Revenue Share — SQL CTE</h3><div id="objmet"></div></div>
  </div>
</div>

<div id="ap" class="sec">
  <div class="acard"><h3>⚡ REST API Endpoints</h3>
    <div style="margin-top:12px">
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/kpis</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/channel-performance — SQL GROUP BY aggregation</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/objective-analysis — SQL CTE revenue share</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/monthly-trend — Time-series data</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/top-campaigns — Top 15 by ROAS</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/forecast — Polynomial regression forecast</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/ab-test — Google Ads vs Meta Ads ROAS</div>
      <div style="background:#0f172a;border-radius:8px;padding:10px 14px;font-family:monospace;font-size:12px;color:#4ade80;margin-bottom:8px"><span class="mth">GET</span>/api/industry-analysis — Industry breakdown</div>
    </div>
  </div>
  <div class="acard"><h3>🛠️ Tech Stack</h3><div style="margin-top:10px">
    <span class="tag">Python</span><span class="tag">Flask</span><span class="tag">Pandas</span><span class="tag">NumPy</span>
    <span class="tag">SQLite</span><span class="tag">SQL CTEs</span><span class="tag">SQL GROUP BY</span>
    <span class="tag">Matplotlib</span><span class="tag">Seaborn</span><span class="tag">Chart.js</span>
    <span class="tag">Polynomial Regression</span><span class="tag">Forecasting</span>
    <span class="tag">Funnel Analysis</span><span class="tag">A/B Testing</span>
    <span class="tag">SciPy</span><span class="tag">REST API</span><span class="tag">ETL Pipeline</span>
  </div></div>
</div>
</div>

<script>
function show(id,btn){
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('on'));
  document.querySelectorAll('.pill').forEach(p=>p.classList.remove('on'));
  document.getElementById(id).classList.add('on'); btn.classList.add('on');
}
const fmt=v=>'₹'+(v/1e6).toFixed(2)+'M';

async function loadKPIs(){
  const d=await fetch('/api/kpis').then(r=>r.json());
  document.getElementById('kgrid').innerHTML=`
    <div class="kcard"><div class="klbl">Total Campaigns</div><div class="kval">${d.total_campaigns}</div></div>
    <div class="kcard"><div class="klbl">Total Budget</div><div class="kval">${fmt(d.total_budget)}</div></div>
    <div class="kcard"><div class="klbl">Total Revenue</div><div class="kval">${fmt(d.total_revenue)}</div></div>
    <div class="kcard"><div class="klbl">Conversions</div><div class="kval">${d.total_conversions.toLocaleString()}</div></div>
    <div class="kcard"><div class="klbl">Avg ROAS</div><div class="kval">${d.avg_roas}x</div></div>
    <div class="kcard"><div class="klbl">Avg ROI</div><div class="kval">${d.avg_roi_pct}%</div></div>
    <div class="kcard"><div class="klbl">Avg CTR</div><div class="kval">${d.avg_ctr_pct}%</div></div>
    <div class="kcard"><div class="klbl">Avg CVR</div><div class="kval">${d.avg_cvr_pct}%</div></div>`;
}

async function loadChannel(){
  const d=await fetch('/api/channel-performance').then(r=>r.json());
  document.getElementById('tchan').innerHTML=`
    <tr><th>Channel</th><th>Campaigns</th><th>Budget</th><th>Revenue</th><th>Avg ROAS</th><th>Avg ROI%</th><th>Conversions</th></tr>
    ${d.map(x=>`<tr><td>${x.channel}</td><td>${x.campaigns}</td><td>${fmt(x.total_budget)}</td><td>${fmt(x.total_revenue)}</td><td><span class="badge ${x.avg_roas>1?'gn':'or'}">${x.avg_roas}x</span></td><td>${x.avg_roi}%</td><td>${x.conversions.toLocaleString()}</td></tr>`).join('')}`;
  const mon=await fetch('/api/monthly-trend').then(r=>r.json());
  new Chart(document.getElementById('monChart'),{
    type:'bar',
    data:{labels:mon.map(x=>`${x.year}-${String(x.month).padStart(2,'0')}`),datasets:[
      {label:'Revenue',data:mon.map(x=>x.revenue),backgroundColor:'#fb923c88',borderColor:'#fb923c',borderWidth:2},
      {label:'Budget', data:mon.map(x=>x.budget), backgroundColor:'#38bdf888',borderColor:'#38bdf8',borderWidth:2}
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:12},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
}

async function loadFunnel(){
  const d=await fetch('/api/top-campaigns').then(r=>r.json());
  document.getElementById('ttop').innerHTML=`
    <tr><th>Campaign</th><th>Channel</th><th>Objective</th><th>Budget</th><th>Revenue</th><th>ROAS</th><th>ROI%</th><th>Conversions</th></tr>
    ${d.map(x=>`<tr><td>${x.campaign_id}</td><td>${x.channel}</td><td>${x.objective}</td><td>₹${Number(x.budget).toLocaleString()}</td><td>₹${Number(x.revenue).toLocaleString()}</td><td><span class="badge gn">${x.roas}x</span></td><td>${x.roi_pct}%</td><td>${x.conversions}</td></tr>`).join('')}`;
}

async function loadForecast(){
  const fc=await fetch('/api/forecast').then(r=>r.json());
  const mon=await fetch('/api/monthly-trend').then(r=>r.json());
  document.getElementById('fcmet').innerHTML=`
    <div class="srow"><span class="sk">Model</span><span class="sv">Polynomial Regression (deg 2)</span></div>
    <div class="srow"><span class="sk">R² Score</span><span class="sv">${fc.r2}</span></div>
    <div class="srow"><span class="sk">Forecast Horizon</span><span class="sv">6 months</span></div>`;
  const hist=mon.map(x=>x.revenue);
  const histLbl=mon.map(x=>`${x.year}-${String(x.month).padStart(2,'0')}`);
  new Chart(document.getElementById('fcChart'),{
    type:'line',
    data:{labels:[...histLbl,...fc.periods],datasets:[
      {label:'Historical',data:[...hist,...Array(6).fill(null)],borderColor:'#fb923c',backgroundColor:'#fb923c15',fill:true,tension:0.4},
      {label:'Forecast', data:[...Array(hist.length).fill(null),...fc.forecast],borderColor:'#4ade80',backgroundColor:'#4ade8015',fill:true,tension:0.4,borderDash:[6,3]}
    ]},
    options:{responsive:true,plugins:{legend:{labels:{color:'#94a3b8'}}},scales:{x:{ticks:{color:'#94a3b8',maxTicksLimit:15},grid:{color:'#1e293b'}},y:{ticks:{color:'#94a3b8'},grid:{color:'#334155'}}}}
  });
  const ind=await fetch('/api/industry-analysis').then(r=>r.json());
  document.getElementById('indmet').innerHTML=ind.map(i=>`
    <div class="srow"><span class="sk">${i.industry}</span><span class="sv">${fmt(i.total_revenue)} (ROAS: ${i.avg_roas}x)</span></div>`).join('');
}

async function loadAB(){
  const ab=await fetch('/api/ab-test').then(r=>r.json());
  document.getElementById('abmet').innerHTML=`
    <div class="srow"><span class="sk">Google Ads ROAS</span><span class="sv">${ab.google_ads.mean}x (n=${ab.google_ads.n})</span></div>
    <div class="srow"><span class="sk">Meta Ads ROAS</span><span class="sv">${ab.meta_ads.mean}x (n=${ab.meta_ads.n})</span></div>
    <div class="srow"><span class="sk">T-Statistic</span><span class="sv">${ab.t_stat}</span></div>
    <div class="srow"><span class="sk">P-Value</span><span class="sv">${ab.p_value}</span></div>
    <div class="srow"><span class="sk">Significant?</span><span class="sv">${ab.significant?'✅ Yes':'❌ No'}</span></div>
    <div class="srow"><span class="sk">Conclusion</span><span class="sv" style="font-size:11px">${ab.conclusion}</span></div>`;
  const obj=await fetch('/api/objective-analysis').then(r=>r.json());
  document.getElementById('objmet').innerHTML=obj.map(o=>`
    <div class="srow"><span class="sk">${o.objective}</span><span class="sv">${fmt(o.revenue)} — ${o.rev_share_pct}% share</span></div>`).join('');
}

loadKPIs();loadChannel();loadFunnel();loadForecast();loadAB();
</script>
</body></html>"""

@app.route('/')
def index(): return HTML

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=False)
