import streamlit as st

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Sri Lanka Used-Car Market Dashboard",
    page_icon="🚗",
    layout="wide",
)

# ----------------------------
# Styling
# ----------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

      :root {
        --bg:       #06090F;
        --surface:  #0C1120;
        --surface2: #111827;
        --border:   #1C2840;
        --border2:  #243050;
        --muted:    #7A8FAD;
        --text:     #E8EFFE;
        --text-dim: #B0BDD4;
        --accent:   #5B8DEF;
        --accent2:  #8B6FEE;
        --green:    #3ECFA0;
        --amber:    #F5A843;
      }

      /* ── Base ── */
      html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
      }
      .stApp {
        background: var(--bg);
        color: var(--text);
      }

      header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: none !important;
      }
      header[data-testid="stHeader"] > div:first-child {
        visibility: hidden;
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
        max-width: 1080px;
      }

      /* ── Sidebar ── */
      [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
      }
      [data-testid="stSidebar"] * {
        color: var(--text-dim) !important;
      }
      [data-testid="stSidebar"] h2,
      [data-testid="stSidebar"] h3 {
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
      }

      /* ── Typography ── */
      h1, h2, h3, h4 {
        font-family: 'Plus Jakarta Sans', sans-serif;
        letter-spacing: -0.02em;
        color: var(--text);
      }

      /* ── Hero ── */
      .hero {
        position: relative;
        overflow: hidden;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 42px 44px 36px;
        margin-bottom: 2rem;
      }
      .hero::before {
        content: "";
        position: absolute;
        inset: 0;
        background:
          radial-gradient(ellipse 60% 50% at 85% 20%, rgba(91,141,239,0.10) 0%, transparent 65%),
          radial-gradient(ellipse 40% 40% at 15% 80%, rgba(139,111,238,0.07) 0%, transparent 60%);
        pointer-events: none;
      }
      .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 14px;
      }
      .hero-eyebrow span {
        display: inline-block;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--accent);
      }
      .hero-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.03em;
        line-height: 1.12;
        margin: 0 0 14px;
      }
      .hero-title em {
        font-style: normal;
        color: var(--accent);
      }
      .hero-desc {
        color: var(--text-dim);
        font-size: 1.02rem;
        font-weight: 300;
        line-height: 1.65;
        max-width: 600px;
        margin-bottom: 0;
      }

      /* ── Section label ── */
      .label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 12px;
        margin-top: 36px;
      }

      /* ── Step cards ── */
      .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 2rem;
      }
      @media (max-width: 900px) { .steps { grid-template-columns: 1fr; } }

      .step-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 20px 20px;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s, transform 0.2s;
      }
      .step-card:hover {
        border-color: var(--border2);
        transform: translateY(-2px);
      }
      .step-card::before {
        content: "";
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        border-radius: 4px 0 0 4px;
      }
      .step-card.c1::before { background: var(--accent); }
      .step-card.c2::before { background: var(--green); }
      .step-card.c3::before { background: var(--accent2); }

      .step-num {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 10px;
      }
      .step-card.c1 .step-num { color: var(--accent); }
      .step-card.c2 .step-num { color: var(--green); }
      .step-card.c3 .step-num { color: var(--accent2); }

      .step-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.0rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 8px;
        line-height: 1.25;
      }
      .step-desc {
        font-size: 0.88rem;
        color: var(--muted);
        line-height: 1.55;
        font-weight: 300;
      }

      /* ── Nav buttons ── */
      [data-testid="stPageLink"] a {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-decoration: none !important;
        border-radius: 12px !important;
        padding: 0.9rem 1.2rem !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        transition: all 0.18s ease !important;
        color: #ffffff !important;
      }
      .btn-primary [data-testid="stPageLink"] a {
        background: linear-gradient(135deg, #3D6FD4, #5B8DEF) !important;
        border: 1px solid rgba(91,141,239,0.5) !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(91,141,239,0.25) !important;
      }
      .btn-primary [data-testid="stPageLink"] a:hover {
        box-shadow: 0 6px 24px rgba(91,141,239,0.40) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
      }
      .btn-secondary [data-testid="stPageLink"] a {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: #ffffff !important;
      }
      .btn-secondary [data-testid="stPageLink"] a:hover {
        border-color: var(--accent) !important;
        background: rgba(91,141,239,0.12) !important;
        color: #ffffff !important;
      }
      [data-testid="stPageLink"] a,
      [data-testid="stPageLink"] a *,
      [data-testid="stPageLink"] a p,
      [data-testid="stPageLink"] a span {
        color: #ffffff !important;
      }

      /* ── Tip bar ── */
      .tip-bar {
        margin-top: 28px;
        display: flex;
        align-items: flex-start;
        gap: 14px;
        background: rgba(91,141,239,0.05);
        border: 1px solid rgba(91,141,239,0.18);
        border-radius: 14px;
        padding: 16px 20px;
      }
      .tip-icon {
        font-size: 1.1rem;
        margin-top: 1px;
        flex-shrink: 0;
      }
      .tip-text {
        font-size: 0.88rem;
        color: var(--text-dim);
        line-height: 1.6;
        font-weight: 300;
      }
      .tip-text b { color: var(--text); font-weight: 500; }

      /* ── Footer ── */
      .footer-line {
        font-size: 0.8rem;
        color: var(--muted);
        margin-top: 36px;
        padding-top: 20px;
        border-top: 1px solid var(--border);
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.markdown("## 🚗 Navigation")
    st.caption("Dec 2024 – Feb 2025 · Pre-liberalization market")
    st.markdown("---")
    st.markdown("**Pages**")
    st.write("📊 Objectives Explorer")
    st.write("🧠 Modeling & Analysis")
    st.markdown("---")
    st.markdown("**Dataset**")
    st.write("• Period: Dec 2024 – Feb 2025")
    st.write("• Scope: Sri Lanka used vehicles")
    st.write("• Context: Pre-import liberalization")
    st.markdown("---")
    st.caption("Use sidebar page list to jump between pages anytime.")

# ----------------------------
# Hero
# ----------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow"><span></span>Statistical Programming · ST 3011</div>
      <div class="hero-title">Sri Lanka<br/><em>Used-Car Market</em><br/>Dashboard</div>
      <div class="hero-desc">
        An interactive deep-dive into the pre-import-liberalization vehicle market
        (Dec 2024 – Feb 2025). Understand what drove listing prices during a
        scarcity-heavy, volatile period — through EDA, objectives analysis, and modeling.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Step cards
# ----------------------------
st.markdown('<div class="label">What you can do</div>', unsafe_allow_html=True)

st.markdown(
    """
    <div class="steps">
      <div class="step-card c1">
        <div class="step-num">Step 01</div>
        <div class="step-title">Explore the Market</div>
        <div class="step-desc">
          Inspect price distributions, brand premiums,
          fuel type effects, and whether location mattered during this period.
        </div>
      </div>
      <div class="step-card c2">
        <div class="step-num">Step 02</div>
        <div class="step-title">Objective-Driven Insights</div>
        <div class="step-desc">
          Walk through all project objectives as an interactive story 
          — with charts and data driven insights.
        </div>
      </div>
      <div class="step-card c3">
        <div class="step-num">Step 03</div>
        <div class="step-title">Model & Interpret</div>
        <div class="step-desc">
          Apply Quantile Regression model to understand relationships in the data 
          and interpret how different factors influence outcomes across the market.
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# CTA buttons
# ----------------------------
st.markdown('<div class="label">Start exploring</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="medium")

with c1:
    st.markdown('<div class="btn-primary">', unsafe_allow_html=True)
    if hasattr(st, "page_link"):
        st.page_link(
            "pages/1_Objectives_Explorer.py",
            label="📊  Objectives Explorer",
            use_container_width=True,
        )
    else:
        st.info("Use the sidebar to open Objectives Explorer.")
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown('<div class="btn-secondary">', unsafe_allow_html=True)
    if hasattr(st, "page_link"):
        st.page_link(
            "pages/2_Modeling.py",
            label="🧠  Modeling & Analysis",
            use_container_width=True,
        )
    else:
        st.info("Use the sidebar to open Modeling.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------
# Tip bar
# ----------------------------
st.markdown(
    """
    <div class="tip-bar">
      <div class="tip-icon">💡</div>
      <div class="tip-text">
        <b>Suggested flow:</b> Start in the Objectives Explorer — apply filters (brand, fuel type,
        transmission, age range), then read the interpretation under each chart.
        Once you have a feel for the data, head to Modeling to quantify the effects.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Footer
# ----------------------------
st.markdown(
    '<div class="footer-line">ST 3011 Statistical Programming · Individual Streamlit App</div>',
    unsafe_allow_html=True,
)