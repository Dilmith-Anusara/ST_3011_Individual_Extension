import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from utils import load_and_preprocess

st.set_page_config(page_title="EDA Explorer", page_icon="📊", layout="wide")

df = load_and_preprocess()

# ============================================================
# THEME — ALL TEXT SET TO WHITE
# ============================================================
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

  :root {
    --bg:       #06090F;
    --surface:  #0C1120;
    --surface2: #111827;
    --border:   #1C2840;
    --border2:  #243050;
    --muted:    #FFFFFF; 
    --text:     #FFFFFF;
    --text-dim: #FFFFFF;
    --accent:   #5B8DEF;
    --accent2:  #8B6FEE;
    --green:    #3ECFA0;
    --amber:    #F5A843;
  }

  html, body, [class*="css"], .stText, .stMarkdown, p, h1, h2, h3, h4, span, label, li { 
    font-family: 'Plus Jakarta Sans', sans-serif !important; 
    color: #FFFFFF !important; 
  }

  .stApp { background: var(--bg); color: #FFFFFF; }

  [data-testid="stMetricValue"], 
  [data-testid="stMetricLabel"], 
  [data-testid="stCaptionContainer"],
  [data-testid="stWidgetLabel"] p {
    color: #FFFFFF !important;
  }

  /* ── Sidebar toggle fix ── */
  header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
  }
  header[data-testid="stHeader"] > div:first-child { visibility: hidden; }
  header[data-testid="stHeader"] button[data-testid="baseButton-header"] {
    visibility: visible !important;
  }

  /* ── MAKE SIDEBAR NON-TOGGLEABLE ── */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapseButton"] {
    display: none !important;
  }

  footer { visibility: hidden; }

  .block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: #FFFFFF !important; }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }

  /* ── Dropdown POPUP list — black text so it's readable on light bg ── */
  /* Targets the floating menu that appears when you open a selectbox/multiselect */
  [data-baseweb="popover"] *,
  [data-baseweb="menu"] *,
  [data-baseweb="menu"] li,
  [data-baseweb="menu"] span,
  [data-baseweb="option"],
  [data-baseweb="option"] *,
  ul[data-baseweb="menu"] li,
  ul[data-baseweb="menu"] span {
    color: #111111 !important;
  }
  /* Also cover the dropdown container itself */
  [data-baseweb="popover"] {
    background: #ffffff !important;
  }

  /* ── Page hero ── */
  .page-hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px 36px 28px;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
  }
  .page-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    background:
      radial-gradient(ellipse 50% 60% at 90% 10%, rgba(62,207,160,0.08) 0%, transparent 65%),
      radial-gradient(ellipse 35% 40% at 5%  85%, rgba(91,141,239,0.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .page-eyebrow {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FFFFFF !important;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .page-eyebrow span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green);
  }
  .page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #FFFFFF;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin: 0 0 10px;
  }
  .page-desc {
    color: #FFFFFF;
    font-size: 0.97rem;
    font-weight: 300;
    line-height: 1.65;
    max-width: 680px;
  }

  /* ── Filter status pill ── */
  .filter-status {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255, 255, 255, 0.07);
    border: 1px solid rgba(255, 255, 255, 0.20);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 0.82rem;
    color: #FFFFFF;
    margin-bottom: 10px;
  }
  .filter-status b { color: #FFFFFF; font-weight: 600; }

  /* ── Section blocks ── */
  .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #FFFFFF;
    margin-bottom: 6px;
    margin-top: 36px;
  }
  .section-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
  }
  .section-desc {
    color: #FFFFFF;
    font-size: 0.88rem;
    font-weight: 300;
    line-height: 1.6;
    max-width: 700px;
    margin-bottom: 4px;
  }
  .section-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px 22px 20px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
  }
  .section-block::before {
    content: "";
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    border-radius: 18px 18px 0 0;
  }
  .section-block.blue::before   { background: linear-gradient(90deg, var(--accent),  transparent); }
  .section-block.green::before  { background: linear-gradient(90deg, var(--green),   transparent); }
  .section-block.purple::before { background: linear-gradient(90deg, var(--accent2), transparent); }

  /* ── Insight callout ── */
  .insight {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 12px;
    padding: 13px 16px;
    margin-top: 14px;
  }
  .insight.green-t {
    background: rgba(62,207,160,0.05);
    border-color: rgba(62,207,160,0.15);
  }
  .insight-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
  .insight-text {
    font-size: 0.86rem;
    color: #FFFFFF;
    line-height: 1.6;
    font-weight: 300;
  }
  .insight-text b { color: #FFFFFF; font-weight: 500; }

  /* ── Footer ── */
  .footer-line {
    font-size: 0.8rem;
    color: #FFFFFF;
    margin-top: 36px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
  }
</style>
""", unsafe_allow_html=True)

# ── Plotly base theme (Adjusted for white text) ────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF", size=12),
    title_font=dict(family="Plus Jakarta Sans, sans-serif", color="#FFFFFF", size=14),
    margin=dict(l=10, r=10, t=44, b=10),
    xaxis=dict(gridcolor="#1C2840", linecolor="#1C2840", zerolinecolor="#1C2840", tickfont=dict(color="#FFFFFF")),
    yaxis=dict(gridcolor="#1C2840", linecolor="#1C2840", zerolinecolor="#1C2840", tickfont=dict(color="#FFFFFF")),
    colorway=["#5B8DEF", "#3ECFA0", "#8B6FEE", "#F5A843", "#EF5B8D", "#43C4F5"],
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)", font=dict(color="#FFFFFF")),
)

def T(fig):
    fig.update_layout(**PLOT_THEME)
    return fig

def filter_pill(dff, total):
    n = len(dff)
    pct = round(n / total * 100)
    if n == total:
        label = f"Showing <b>all {n:,} listings</b> · no filter active"
    else:
        label = f"Showing <b>{n:,} of {total:,} listings</b> ({pct}% of dataset)"
    st.markdown(f'<div class="filter-status">{label}</div>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR — global filters (Sections 1 & 2)
# ============================================================
TOTAL = len(df)
comfort_features = ["air_condition", "power_steering", "power_mirror", "power_window"]

with st.sidebar:
    st.markdown("## 📊 EDA Explorer")
    st.caption("Dec 2024 – Feb 2025 · Pre-liberalization market")
    st.markdown("---")

    st.markdown("### 🎛 Global Filters")
    st.caption("Sections 1 & 2 react to these. Section 3 has its own province filter.")

    all_fuels = sorted(df["fuel_type"].dropna().unique().tolist())
    sel_fuel = st.multiselect("Fuel type", all_fuels, default=all_fuels, key="fuel")

    all_gears = sorted(df["gear"].dropna().unique().tolist())
    sel_gear = st.multiselect("Transmission", all_gears, default=all_gears, key="gear")

    all_brands = sorted(df["brand"].dropna().unique().tolist())
    sel_brand = st.multiselect("Brand", all_brands, default=all_brands, key="brand")

    st.markdown("---")
    st.markdown("**Sections**")
    st.write("① Mechanical structure")
    st.write("② Comfort features")
    st.write("③ Geo-economic patterns")

# ── Filtered frame for Sections 1 & 2 ────────────────────────
dff = df[
    df["fuel_type"].isin(sel_fuel) &
    df["gear"].isin(sel_gear) &
    df["brand"].isin(sel_brand)
]

# ============================================================
# PAGE HERO
# ============================================================
st.markdown("""
<div class="page-hero">
  <div class="page-eyebrow"><span></span>General Market Evaluation</div>
  <div class="page-title">EDA Explorer</div>
  <div class="page-desc">
    A high-level overview of the used vehicle market before statistical modelling.
    Sidebar filters (fuel type, transmission, brand) apply to Sections 1 &amp; 2 and
    update charts live. Section 3 (geography) has its own province filter.
  </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total listings", f"{TOTAL:,}")
m2.metric("Filtered listings", f"{len(dff):,}")
m3.metric("Fuel types", df["fuel_type"].nunique())
m4.metric("Provinces", df["province"].nunique())


# ============================================================
# SECTION 1 — MECHANICAL MARKET STRUCTURE
# ============================================================
st.markdown("""
<div class="section-block blue">
  <div class="label">Section 01 · reacts to sidebar filters</div>
  <div class="section-heading">Mechanical Market Structure</div>
  <div class="section-desc">
    Transmission type, fuel technology, and engine segment define the mechanical profile
    of the market. Filter by brand or fuel type in the sidebar to see how composition shifts.
  </div>
</div>
""", unsafe_allow_html=True)

filter_pill(dff, TOTAL)

if dff.empty:
    st.warning("No listings match the current filters — try relaxing a selection.")
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        fig = px.histogram(dff, x="gear", color="gear", title="Gear Type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(T(fig), use_container_width=True)
    with col2:
        fig = px.histogram(dff, x="fuel_type", color="fuel_type", title="Fuel Type")
        fig.update_layout(showlegend=False)
        st.plotly_chart(T(fig), use_container_width=True)
    with col3:
        fig = px.histogram(dff, x="engine_segment", color="engine_segment", title="Engine Segment")
        fig.update_layout(showlegend=False)
        st.plotly_chart(T(fig), use_container_width=True)

    st.markdown('<div class="label" style="margin-top:20px">Transmission × Fuel</div>',
                unsafe_allow_html=True)
    fig = px.histogram(dff, x="gear", color="fuel_type", barmode="group",
                       title="Transmission Type by Fuel Technology")
    st.plotly_chart(T(fig), use_container_width=True)

    st.markdown('<div class="label" style="margin-top:4px">Fuel × Engine segment</div>',
                unsafe_allow_html=True)
    col_chart, col_ctrl = st.columns([4, 1])
    with col_ctrl:
        st.markdown("&nbsp;")
        barmode = st.radio(
            "Bar style",
            ["Absolute", "Proportional"],
            index=0,
            key="fuel_engine_barmode",
            help="Proportional (100% bar) compares engine mix fairly when fuel-type counts differ greatly.",
        )
    with col_chart:
        fig = px.histogram(
            dff, x="fuel_type", color="engine_segment",
            barmode="group",
            barnorm="percent" if barmode == "Proportional" else None,
            title="Fuel Type by Engine Segment" + (" (proportional)" if barmode == "Proportional" else ""),
        )
        st.plotly_chart(T(fig), use_container_width=True)

    st.markdown("""
    <div class="insight">
      <div class="insight-icon">💡</div>
      <div class="insight-text">
        <b>Try this:</b> Filter to a single brand in the sidebar and compare its mechanical
        profile against the full market. Switch to <b>Proportional</b> on the Fuel × Engine
        chart when one fuel type vastly outnumbers another — raw counts hide the engine mix.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 2 — COMFORT FEATURES
# ============================================================
st.markdown("""
<div class="section-block green" style="margin-top:36px">
  <div class="label">Section 02 · reacts to sidebar filters</div>
  <div class="section-heading">Comfort Features</div>
  <div class="section-desc">
    Comfort features signal vehicle quality tier. These charts show how frequently each feature
    appears in the filtered market and how availability varies across brands.
  </div>
</div>
""", unsafe_allow_html=True)

filter_pill(dff, TOTAL)

if dff.empty:
    st.warning("No listings match the current filters.")
else:
    st.markdown('<div class="label">Choose features to display</div>', unsafe_allow_html=True)
    selected_features = st.multiselect(
        "Features",
        options=comfort_features,
        default=comfort_features,
        format_func=lambda x: x.replace("_", " ").title(),
        key="comfort_select",
        label_visibility="collapsed",
    )

    if not selected_features:
        st.info("Select at least one comfort feature above.")
    else:
        st.markdown('<div class="label">Feature availability</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(selected_features), 2))
        for i, feat in enumerate(selected_features):
            fig = px.histogram(
                dff, x=feat, color=feat,
                title=feat.replace("_", " ").title(),
                category_orders={feat: ["Available", "Not_Available"]},
            )
            fig.update_layout(showlegend=False)
            cols[i % 2].plotly_chart(T(fig), use_container_width=True)

        st.markdown('<div class="label" style="margin-top:8px">Feature penetration by brand</div>',
                    unsafe_allow_html=True)

        col_chart, col_ctrl = st.columns([4, 1])
        with col_ctrl:
            st.markdown("&nbsp;")
            view = st.radio(
                "View as",
                ["Grouped bars", "Heatmap"],
                index=0,
                key="penetration_view",
                help="Heatmap is cleaner when many brands are selected.",
            )
            top_n_brands = st.slider(
                "Top N brands",
                min_value=3, max_value=15, value=8, step=1,
                key="top_brands_slider",
                help="Show only the N most common brands to keep the chart readable.",
            )

        top_brands = dff["brand"].value_counts().head(top_n_brands).index.tolist()
        brand_df = dff[dff["brand"].isin(top_brands)]

        brand_agg = (
            brand_df[['brand'] + selected_features]
            .replace({'Available': 1, 'Not_Available': 0})
            .groupby('brand')[selected_features]
            .mean() * 100
        ).reset_index()

        with col_chart:
            if view == "Grouped bars":
                brand_long = brand_agg.melt(
                    id_vars='brand', var_name='Feature', value_name='Penetration (%)'
                )
                brand_long['Feature'] = brand_long['Feature'].str.replace("_", " ").str.title()
                fig = px.bar(
                    brand_long, x='brand', y='Penetration (%)', color='Feature',
                    barmode='group',
                    title=f"Comfort Feature Penetration — Top {top_n_brands} Brands (%)",
                )
                st.plotly_chart(T(fig), use_container_width=True)
            else:
                heat = brand_agg.set_index('brand')[selected_features].copy()
                heat.columns = [c.replace("_", " ").title() for c in heat.columns]
                fig = go.Figure(go.Heatmap(
                    z=heat.values,
                    x=heat.columns.tolist(),
                    y=heat.index.tolist(),
                    colorscale=[[0, "#0C1120"], [0.5, "#2B5499"], [1, "#3ECFA0"]],
                    text=[[f"{v:.1f}%" for v in row] for row in heat.values],
                    texttemplate="%{text}",
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="#FFFFFF"), title="% penetration"),
                ))
                fig.update_layout(
                    title=f"Feature Penetration Heatmap — Top {top_n_brands} Brands",
                    **PLOT_THEME
                )
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="insight green-t">
      <div class="insight-icon">💡</div>
      <div class="insight-text">
        <b>Try the heatmap:</b> It's easier to spot which brand–feature combinations stand out.
      </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SECTION 3 — GEO-ECONOMIC ANALYSIS
# ============================================================
st.markdown("""
<div class="section-block purple" style="margin-top:36px">
  <div class="label">Section 03 · own local filter</div>
  <div class="section-heading">Geo-Economic Analysis</div>
  <div class="section-desc">
    Where listings are concentrated tells us about market reach and supply geography.
  </div>
</div>
""", unsafe_allow_html=True)

all_provinces = sorted(df["province"].dropna().unique().tolist())
sel_province = st.multiselect(
    "Filter by province (Section 3 only)",
    options=all_provinces,
    default=all_provinces,
    key="province_filter",
)
dfg = df[df["province"].isin(sel_province)]
filter_pill(dfg, TOTAL)

if dfg.empty:
    st.warning("No listings match the selected provinces.")
else:
    col1, col2 = st.columns(2)

    with col1:
        leasing_by = st.selectbox(
            "Break leasing chart down by",
            options=["None", "fuel_type", "gear"],
            format_func=lambda x: {
                "None": "No breakdown",
                "fuel_type": "Fuel type",
                "gear": "Transmission",
            }.get(x, x),
            key="leasing_split",
        )
        if leasing_by == "None":
            fig = px.histogram(dfg, x="leasing", color="leasing",
                               title="Listings by Leasing Availability")
            fig.update_layout(showlegend=False)
        else:
            fig = px.histogram(dfg, x="leasing", color=leasing_by, barmode="group",
                               title=f"Leasing by {leasing_by.replace('_', ' ').title()}")
        st.plotly_chart(T(fig), use_container_width=True)

    with col2:
        top_n = st.slider(
            "Number of towns to show",
            min_value=5, max_value=30, value=15, step=5,
            key="top_n_towns",
        )
        top_towns = dfg["town"].value_counts().head(top_n).reset_index()
        top_towns.columns = ["town", "count"]
        fig = px.bar(top_towns, y="town", x="count", orientation="h",
                     title=f"Top {top_n} Towns by Listings")
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(T(fig), use_container_width=True)

    st.markdown('<div class="label">Provincial distribution (full dataset)</div>',
                unsafe_allow_html=True)
    prov = df["province"].value_counts().reset_index()
    prov.columns = ["province", "count"]
    prov["selected"] = prov["province"].isin(sel_province)
    fig = px.bar(
        prov, x="province", y="count",
        color="selected",
        color_discrete_map={True: "#5B8DEF", False: "#243050"},
        title="Listings by Province — highlighted = active filter",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(T(fig), use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    '<div class="footer-line">ST 3011 Statistical Programming · Individual Streamlit Application</div>',
    unsafe_allow_html=True,
)
