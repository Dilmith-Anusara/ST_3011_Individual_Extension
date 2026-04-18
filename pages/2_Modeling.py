import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from utils import create_model_dataset

st.set_page_config(page_title="Modeling", page_icon="🧠", layout="wide")

st.markdown("""
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

  html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
  .stApp { background: var(--bg); color: var(--text); }

  header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
  }
  header[data-testid="stHeader"] > div:first-child { visibility: hidden; }

  footer { visibility: hidden; }

  /* ── Hide ALL sidebar toggle controls — sidebar is non-togglable ── */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  button[data-testid="baseButton-header"],
  button[kind="header"],
  section[data-testid="stSidebar"] > div > button,
  [data-testid="stSidebar"] button[kind="header"] {
    display: none !important;
    visibility: hidden !important;
  }

  .block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1100px;
  }

  [data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
  }
  [data-testid="stSidebar"] * { color: var(--text-dim) !important; }
  [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 {
    color: var(--text) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }

  h1, h2, h3, h4 {
    font-family: 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -0.02em;
    color: var(--text);
  }

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
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse 55% 55% at 88% 15%, rgba(139,111,238,0.10) 0%, transparent 65%),
      radial-gradient(ellipse 35% 40% at 5%  85%, rgba(91,141,239,0.06) 0%, transparent 60%);
    pointer-events: none;
  }
  .page-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--accent2); margin-bottom: 10px;
    display: flex; align-items: center; gap: 7px;
  }
  .page-eyebrow span {
    display: inline-block; width: 6px; height: 6px;
    border-radius: 50%; background: var(--accent2);
  }
  .page-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.9rem; font-weight: 800; color: var(--text);
    letter-spacing: -0.03em; line-height: 1.15; margin: 0 0 10px;
  }
  .page-desc {
    color: var(--text-dim); font-size: 0.97rem;
    font-weight: 300; line-height: 1.65; max-width: 680px;
  }

  .label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 6px; margin-top: 36px;
  }
  .section-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.2rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .section-desc {
    color: var(--muted); font-size: 0.88rem;
    font-weight: 300; line-height: 1.6;
    max-width: 700px; margin-bottom: 4px;
  }
  .section-block {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 24px 22px 20px;
    margin-bottom: 16px;
    position: relative; overflow: hidden;
  }
  .section-block::before {
    content: ""; position: absolute;
    top: 0; left: 0; width: 100%; height: 3px;
    border-radius: 18px 18px 0 0;
  }
  .section-block.purple::before { background: linear-gradient(90deg, var(--accent2), transparent); }
  .section-block.blue::before   { background: linear-gradient(90deg, var(--accent),  transparent); }
  .section-block.green::before  { background: linear-gradient(90deg, var(--green),   transparent); }
  .section-block.amber::before  { background: linear-gradient(90deg, var(--amber),   transparent); }

  .q-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-top: 14px;
  }
  @media (max-width: 900px) { .q-grid { grid-template-columns: 1fr 1fr; } }
  .q-card {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 14px; padding: 14px 14px 12px;
  }
  .q-num {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.5rem; font-weight: 800; margin-bottom: 4px;
  }
  .q-label { font-size: 0.78rem; color: var(--muted); font-weight: 300; line-height: 1.4; }
  .q25 .q-num { color: #5B8DEF; }
  .q50 .q-num { color: #3ECFA0; }
  .q75 .q-num { color: #F5A843; }
  .q90 .q-num { color: #EF5B8D; }

  .insight {
    display: flex; gap: 12px; align-items: flex-start;
    background: rgba(91,141,239,0.05);
    border: 1px solid rgba(91,141,239,0.15);
    border-radius: 12px; padding: 13px 16px; margin-top: 14px;
  }
  .insight.purple-t { background: rgba(139,111,238,0.05); border-color: rgba(139,111,238,0.15); }
  .insight.green-t  { background: rgba(62,207,160,0.05);  border-color: rgba(62,207,160,0.15);  }
  .insight.amber-t  { background: rgba(245,168,67,0.05);  border-color: rgba(245,168,67,0.18);  }
  .insight-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
  .insight-text { font-size: 0.86rem; color: var(--text-dim); line-height: 1.6; font-weight: 300; }
  .insight-text b { color: var(--text); font-weight: 500; }

  .buyer-card {
    background: var(--surface2);
    border: 1px solid var(--border2);
    border-radius: 16px;
    padding: 20px 22px 18px;
    margin-top: 16px;
  }
  .buyer-card-title {
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 10px;
  }
  .buyer-verdict {
    font-size: 1.05rem; font-weight: 700; color: var(--text);
    letter-spacing: -0.01em; margin-bottom: 8px; line-height: 1.3;
  }
  .buyer-body {
    font-size: 0.88rem; color: var(--text-dim);
    line-height: 1.65; font-weight: 300;
  }
  .buyer-body b { color: var(--text); font-weight: 500; }
  .signal-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
  .signal-pill {
    display: inline-flex; align-items: center; gap: 6px;
    border-radius: 8px; padding: 5px 11px;
    font-size: 0.78rem; font-weight: 600;
  }
  .pill-green { background: rgba(62,207,160,0.10); border: 1px solid rgba(62,207,160,0.25); color: #3ECFA0; }
  .pill-amber { background: rgba(245,168,67,0.10); border: 1px solid rgba(245,168,67,0.25); color: #F5A843; }
  .pill-red   { background: rgba(239,91,141,0.10); border: 1px solid rgba(239,91,141,0.25); color: #EF5B8D; }
  .pill-muted { background: rgba(122,143,173,0.10); border: 1px solid rgba(122,143,173,0.20); color: #7A8FAD; }

  [data-testid="stMetricLabel"] { color: var(--muted) !important; }
  [data-testid="stMetricValue"] { color: var(--text) !important; font-family: 'Plus Jakarta Sans', sans-serif !important; }
  [data-baseweb="select"] > div { background-color: var(--surface2) !important; border-color: var(--border) !important; }
  [data-baseweb="select"] * { color: var(--text) !important; }
  button[data-baseweb="tab"] { color: var(--text-dim) !important; }
  button[data-baseweb="tab"][aria-selected="true"] { color: var(--text) !important; }
  [data-testid="stDataFrame"] div { color: var(--text) !important; }
  .streamlit-expanderContent { color: var(--text) !important; }

  /* ── Dropdown popup — black text on light background ── */
  [data-baseweb="popover"] *,
  [data-baseweb="menu"] *,
  [data-baseweb="option"],
  [data-baseweb="option"] * { color: #111111 !important; }
  [data-baseweb="popover"] { background: #ffffff !important; }

  .footer-line {
    font-size: 0.8rem; color: var(--muted);
    margin-top: 36px; padding-top: 20px;
    border-top: 1px solid var(--border);
  }
</style>
""", unsafe_allow_html=True)


plt.rcParams.update({
    "figure.facecolor":  "#0C1120",
    "axes.facecolor":    "#0C1120",
    "axes.edgecolor":    "#1C2840",
    "axes.labelcolor":   "#B0BDD4",
    "axes.titlecolor":   "#E8EFFE",
    "xtick.color":       "#7A8FAD",
    "ytick.color":       "#7A8FAD",
    "text.color":        "#B0BDD4",
    "grid.color":        "#1C2840",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "600",
    "axes.labelsize":    11,
    "lines.linewidth":   2,
    "lines.markersize":  8,
})


@st.cache_data
def run_quantile_regression(df):
    formula = """
    log_price ~ age + log_engine_cc + leasing + power_steering + power_mirror + power_window
    + C(gear)
    + C(province)
    + C(brand_segment)
    + C(fuel_segment)
    """
    quantiles = [0.25, 0.50, 0.75, 0.90]
    qr_results = {}
    for q in quantiles:
        qr_results[q] = smf.quantreg(formula, df).fit(q=q)
    return qr_results, quantiles


def build_tables(qr_results, quantiles):
    coef_table = pd.DataFrame({f"Q{int(q*100)}": qr_results[q].params for q in quantiles})
    pval_table = pd.DataFrame({f"Q{int(q*100)}": qr_results[q].pvalues for q in quantiles})
    return coef_table, pval_table


def variable_label(var_name):
    mapping = {
        "Intercept":      "Intercept",
        "age":            "Vehicle age",
        "log_engine_cc":  "Engine capacity",
        "leasing":        "Leasing available",
        "power_steering": "Power steering",
        "power_mirror":   "Power mirror",
        "power_window":   "Power window",
    }
    if var_name in mapping:
        return mapping[var_name]
    for prefix, label in [
        ("C(gear)[T.",         "Transmission"),
        ("C(province)[T.",     "Province"),
        ("C(brand_segment)[T.","Brand segment"),
        ("C(fuel_segment)[T.", "Fuel segment"),
    ]:
        if var_name.startswith(prefix):
            level = var_name[len(prefix):].rstrip("]")
            return f"{label}: {level}"
    return var_name


def significant_variables(pval_table, alpha=0.05):
    sig_dict = {}
    for col in pval_table.columns:
        sig_vars = [variable_label(v) for v in pval_table.index[pval_table[col] < alpha]]
        sig_dict[col] = pd.Series(sig_vars)
    return pd.DataFrame(sig_dict)


def get_feature_presence_summary(sig_vars_df):
    all_vars = pd.Series(sig_vars_df.values.ravel()).dropna()
    counts = all_vars.value_counts().reset_index()
    counts.columns = ["Factor", "Significant in N quantiles"]
    return counts


def overall_takeaway(sig_vars_df):
    all_vars = pd.Series(sig_vars_df.values.ravel()).dropna()
    core = [v for v in all_vars.unique() if (all_vars == v).sum() >= 3 and v != "Intercept"]
    if not core:
        return ("No single factor is consistently significant across most quantiles. "
                "This suggests price drivers are highly segment-specific in this market.")
    return (f"The most consistently significant factors are: {', '.join(core)}. "
            f"These influence vehicle prices across the whole market — not just at one price level.")


def coef_to_pct(coef):
    return round((np.exp(coef) - 1) * 100, 1)


def interpret_variable(values, var_name, pvals=None, quantile_labels=None):
    label = variable_label(var_name)
    n_q   = len(values)
    ql    = quantile_labels or [f"Q{int(q*100)}" for q in [0.25, 0.50, 0.75, 0.90]]
    pcts  = [coef_to_pct(v) for v in values]

    if pvals is not None:
        sig_qs   = [ql[i] for i, p in enumerate(pvals) if p < 0.05]
        insig_qs = [ql[i] for i, p in enumerate(pvals) if p >= 0.05]
        n_sig    = len(sig_qs)
    else:
        sig_qs = ql; insig_qs = []; n_sig = n_q

    all_pos = all(v > 0 for v in values)
    all_neg = all(v < 0 for v in values)
    spread  = abs(pcts[-1] - pcts[0])

    if spread > 5 and pcts[-1] > pcts[0]:
        trend_desc = (f"The effect <b>grows stronger towards premium vehicles</b> "
                      f"(from {pcts[0]:+.1f}% at Q25 to {pcts[-1]:+.1f}% at Q90 — "
                      f"a {spread:.1f} percentage-point swing).")
    elif spread > 5 and pcts[-1] < pcts[0]:
        trend_desc = (f"The effect is <b>stronger at the budget end</b> "
                      f"(from {pcts[0]:+.1f}% at Q25 to {pcts[-1]:+.1f}% at Q90 — "
                      f"it weakens as price rises).")
    else:
        trend_desc = (f"The effect is <b>relatively stable across price levels</b> "
                      f"(ranging from {min(pcts):+.1f}% to {max(pcts):+.1f}% — "
                      f"less than a {spread:.1f} percentage-point spread).")

    if n_sig == n_q:
        sig_desc = (f"The effect is <b>statistically significant at all {n_q} quantiles</b>, "
                    f"confirming this is a genuine, market-wide price driver — not a chance pattern.")
    elif n_sig == 0:
        sig_desc = (f"<b>None of the quantiles reach significance</b> (p ≥ 0.05 at Q25, Q50, Q75 and Q90). "
                    f"The data provides no reliable evidence that this factor moves prices.")
    else:
        sig_desc = (f"Significant at <b>{', '.join(sig_qs)}</b> (p &lt; 0.05) but <b>not at {', '.join(insig_qs)}</b>. "
                    f"The price effect is real in those segments but uncertain elsewhere.")

    rows = ""
    for i, (q, v, pct) in enumerate(zip(ql, values, pcts)):
        p_str = f"{pvals[i]:.4f}" if pvals is not None else "—"
        sig_marker = "✓" if (pvals is not None and pvals[i] < 0.05) else "✗"
        color = "#3ECFA0" if sig_marker == "✓" else "#7A8FAD"
        rows += (
            f"<tr style='border-bottom:1px solid #1C2840'>"
            f"<td style='padding:5px 10px;color:#B0BDD4'>{q}</td>"
            f"<td style='padding:5px 10px;color:#E8EFFE;font-weight:600'>{v:+.4f}</td>"
            f"<td style='padding:5px 10px;color:#E8EFFE'>{pct:+.1f}%</td>"
            f"<td style='padding:5px 10px;color:{color};font-weight:600'>{sig_marker} p={p_str}</td>"
            f"</tr>"
        )
    breakdown_table = (
        f"<table style='width:100%;border-collapse:collapse;font-size:0.83rem;margin-top:10px'>"
        f"<thead><tr style='border-bottom:2px solid #243050'>"
        f"<th style='padding:5px 10px;color:#7A8FAD;font-weight:500;text-align:left'>Quantile</th>"
        f"<th style='padding:5px 10px;color:#7A8FAD;font-weight:500;text-align:left'>Coefficient</th>"
        f"<th style='padding:5px 10px;color:#7A8FAD;font-weight:500;text-align:left'>≈ % price change</th>"
        f"<th style='padding:5px 10px;color:#7A8FAD;font-weight:500;text-align:left'>Significance</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )

    if var_name == "age":
        core = (f"Each additional year of vehicle age is associated with a price reduction — "
                f"specifically <b>{abs(pcts[0]):.1f}% per year</b> at the budget end (Q25) and "
                f"<b>{abs(pcts[-1]):.1f}% per year</b> at the premium end (Q90). "
                f"This is among the strongest continuous predictors in the model.")
        bottom_line = (f"A five-year-old car at the median price point is estimated to list at roughly "
                       f"<b>{abs(pcts[1]) * 5:.0f}% less</b> than an equivalent brand-new vehicle. "
                       f"Age is never 'free' — it is always priced in.")
    elif var_name == "log_engine_cc":
        core = (f"A 10% increase in engine displacement is associated with approximately a "
                f"<b>{pcts[1] * 0.095:.1f}% price increase</b> at the median market (Q50). "
                f"At the premium end (Q90) that rises to <b>{pcts[-1] * 0.095:.1f}%</b>. "
                f"Engine size is one of the most powerful mechanical price signals in this dataset.")
        bottom_line = (f"Moving from a 1,000cc to a 2,000cc engine (a 100% log-scale increase) "
                       f"is associated with roughly a <b>{abs(pcts[1]):.0f}% price premium</b> at the median. "
                       f"Always weigh the engine premium against your actual driving needs.")
    elif var_name == "leasing":
        direction_word = "higher" if all_pos else "lower" if all_neg else "inconsistent"
        core = (f"Vehicles listed with leasing availability are priced <b>{direction_word}</b> "
                f"than those without — by <b>{pcts[1]:+.1f}%</b> at the median (Q50) and "
                f"<b>{pcts[-1]:+.1f}%</b> at the premium end (Q90). "
                f"This likely reflects that leasable vehicles tend to be newer, dealer-backed stock.")
        bottom_line = (f"If you don't need financing, filtering for non-leasable listings may expose "
                       f"vehicles priced <b>~{abs(pcts[1]):.0f}% lower</b> with similar specs.")
    elif var_name == "power_steering":
        direction_word = "premium" if all_pos else "discount"
        core = (f"Power steering commands a price <b>{direction_word}</b> — "
                f"estimated at <b>{pcts[1]:+.1f}%</b> at the median and "
                f"<b>{pcts[-1]:+.1f}%</b> at the premium end. "
                f"This feature acts as a proxy for overall vehicle modernity in this market.")
        bottom_line = (f"At the median price level, power steering alone accounts for roughly "
                       f"<b>{abs(pcts[1]):.1f}%</b> of the asking price. "
                       f"Its strongest signal is in the budget segment where its absence is still common.")
    elif var_name == "power_mirror":
        direction_word = "premium" if all_pos else "discount"
        core = (f"Power mirrors are associated with a <b>{direction_word}</b> of "
                f"<b>{pcts[1]:+.1f}%</b> at the median and <b>{pcts[-1]:+.1f}%</b> at the top end. "
                f"This comfort feature tends to co-vary with other quality signals.")
        bottom_line = (f"The price premium for power mirrors is "
                       f"{'consistent across the market' if spread < 5 else 'larger for higher-priced vehicles'}. "
                       f"Expect sellers to price this in, especially at Q25–Q50.")
    elif var_name == "power_window":
        direction_word = "premium" if all_pos else "discount"
        core = (f"Power windows carry a price <b>{direction_word}</b> — "
                f"<b>{pcts[1]:+.1f}%</b> at Q50 and <b>{pcts[-1]:+.1f}%</b> at Q90. "
                f"In premium segments this feature is near-universal, "
                f"so its marginal price contribution is likely concentrated at Q25–Q50.")
        bottom_line = (f"Budget-end vehicles without power windows may offer a "
                       f"<b>~{abs(pcts[0]):.1f}% discount</b> versus comparable vehicles that have it.")
    elif "C(brand_segment)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        direction_word = "above" if all_pos else "below"
        core = (f"The <b>{level}</b> brand segment is priced <b>{direction_word}</b> the reference group — "
                f"by <b>{pcts[1]:+.1f}%</b> at the median and <b>{pcts[-1]:+.1f}%</b> at the premium end. "
                f"Brand premiums reflect perceived reliability, resale value, and parts availability.")
        bottom_line = (f"At the median, choosing the <b>{level}</b> segment "
                       f"{'adds' if all_pos else 'saves'} roughly <b>{abs(pcts[1]):.0f}%</b> vs the baseline. "
                       f"{'Well-supported by the data.' if n_sig >= 3 else 'Only significant in some quantiles — interpret with caution.'}")
    elif "C(fuel_segment)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        direction_word = "higher" if all_pos else "lower"
        core = (f"Vehicles using <b>{level}</b> fuel are listed <b>{direction_word}</b> than the baseline — "
                f"by <b>{pcts[1]:+.1f}%</b> at Q50 and <b>{pcts[-1]:+.1f}%</b> at Q90. "
                f"This likely reflects running cost differences, technology scarcity, or demand patterns.")
        bottom_line = (f"The <b>{level}</b> fuel premium is "
                       f"{'largest in the premium segment' if pcts[-1] > pcts[0] else 'largest in the budget segment, narrowing as price rises'}. "
                       f"Factor long-run fuel costs into whether this premium is justified.")
    elif "C(province)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        direction_word = "higher" if all_pos else "lower"
        core = (f"Listings in <b>{level}</b> are priced <b>{direction_word}</b> than the reference province — "
                f"by <b>{pcts[1]:+.1f}%</b> at the median and <b>{pcts[-1]:+.1f}%</b> at Q90. "
                f"Provincial price differences arise from local demand, supply scarcity, and buyer purchasing power.")
        _prov_end = "premium" if pcts[-1] > pcts[0] else "budget"
        _prov_spread = "consistent across all budget levels" if spread < 5 else f"especially pronounced at the {_prov_end} end"
        _prov_save = (f"Buying in a lower-priced province could save ~{abs(pcts[1])}% at the median — weigh this against transport or inspection costs."
                      if not all_pos else "Location is adding real money to these listings.")
        bottom_line = f"The location premium for <b>{level}</b> is {_prov_spread}. {_prov_save}"
    elif "C(gear)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        direction_word = "higher" if all_pos else "lower"
        core = (f"<b>{level}</b> transmission is priced <b>{direction_word}</b> than the reference gearbox — "
                f"by <b>{pcts[1]:+.1f}%</b> at the median and <b>{pcts[-1]:+.1f}%</b> at Q90. "
                f"Transmission preference reflects consumer demand patterns and relative scarcity.")
        bottom_line = (f"At Q50, the transmission choice accounts for roughly <b>{abs(pcts[1]):.0f}%</b> of the price. "
                       f"{'This gap narrows at higher prices.' if pcts[-1] < pcts[1] else 'This gap grows at higher prices.'}")
    else:
        direction_word = "positive" if all_pos else "negative" if all_neg else "mixed"
        core = (f"<b>{label}</b> shows a {direction_word} association with log price. "
                f"At Q50, the estimated effect is <b>{pcts[1]:+.1f}%</b>. "
                f"Range across quantiles: <b>{min(pcts):+.1f}%</b> to <b>{max(pcts):+.1f}%</b>.")
        bottom_line = ("This is a consistent, market-wide effect." if n_sig == n_q
                       else "Interpret with caution — not significant at all price levels.")

    return (f"{core}<br/><br/>"
            f"{trend_desc}<br/><br/>"
            f"{sig_desc}<br/><br/>"
            f"<b>Per-quantile breakdown:</b>{breakdown_table}<br/>"
            f"<b>Bottom line:</b> {bottom_line}")


def interpret_variable_simple(values, var_name):
    direction = ("positive" if all(v > 0 for v in values)
                 else "negative" if all(v < 0 for v in values) else "mixed")
    trend = ("grows stronger at higher price levels" if abs(values[-1]) > abs(values[0]) * 1.1
             else "is stronger at lower price levels" if abs(values[-1]) < abs(values[0]) * 0.9
             else "is fairly stable across all price levels")
    return direction, trend


BUYER_LABELS = {
    "age":            "vehicle age",
    "log_engine_cc":  "engine size",
    "leasing":        "leasing option",
    "power_steering": "power steering",
    "power_mirror":   "power mirrors",
    "power_window":   "power windows",
}

def buyer_label(var_name):
    if var_name in BUYER_LABELS:
        return BUYER_LABELS[var_name]
    for prefix, label in [
        ("C(gear)[T.",          "transmission type"),
        ("C(province)[T.",      "province"),
        ("C(brand_segment)[T.", "brand tier"),
        ("C(fuel_segment)[T.",  "fuel type"),
    ]:
        if var_name.startswith(prefix):
            level = var_name[len(prefix):].rstrip("]")
            return f"{label} ({level})"
    return variable_label(var_name).lower()


def buyer_verdict_text(var_name, values, pvals, p_b, p_c, diff_b):
    blabel    = buyer_label(var_name)
    n_sig     = sum(p < 0.05 for p in pvals)
    max_coef  = max(abs(v) for v in values)
    max_pct   = coef_to_pct(max_coef)
    direction = "positive" if sum(v > 0 for v in values) >= 3 else "negative"
    pills     = []

    if n_sig == 0:
        headline = f"📋 {blabel.title()} has little effect on price"
        body = (f"The data shows <b>no reliable connection</b> between {blabel} and listing price. "
                f"Sellers don't appear to charge more or less based on this feature.")
        pills = [("Doesn't move price", "pill-muted")]
        return headline, body, pills

    up_or_down = "higher" if direction == "positive" else "lower"
    changes_across_market = p_b < 0.05 or p_c < 0.05

    if var_name == "age":
        headline = "📉 Older cars are cheaper — this is confirmed and consistent"
        q25_pct = coef_to_pct(values[0]); q90_pct = coef_to_pct(values[-1])
        body = (f"Every extra year on the clock is linked to a price drop — "
                f"roughly <b>{abs(q25_pct):.0f}% per year</b> at the budget end and "
                f"<b>{abs(q90_pct):.0f}% per year</b> for premium vehicles. ")
        if changes_across_market:
            body += ("Age hits <b>premium listings harder</b>." if diff_b < 0
                     else "Age hits <b>budget listings harder</b>.")
        else:
            body += "This effect is <b>consistent across all price ranges</b>."
        pills = [("Confirmed price driver", "pill-green"),
                 ("Affects all budgets", "pill-green" if not changes_across_market else "pill-amber")]

    elif var_name == "log_engine_cc":
        headline = "🔧 Bigger engines cost more — and the gap grows for premium cars"
        q25_pct = coef_to_pct(values[0]); q90_pct = coef_to_pct(values[-1])
        body = (f"Engine size is one of the strongest price signals. "
                f"Budget end: +<b>{abs(q25_pct):.0f}%</b>. Premium end: +<b>{abs(q90_pct):.0f}%</b>. ")
        body += ("<b>Effect is significantly stronger for expensive cars.</b>" if changes_across_market
                 else "Engine size affects price <b>equally at every budget level</b>.")
        pills = [("Strong price driver", "pill-red"), ("Worth negotiating on", "pill-amber")]

    elif var_name == "leasing":
        headline = f"💳 Leasing availability is linked to {'higher' if direction == 'positive' else 'lower'} prices"
        body = (f"Vehicles with a leasing option are priced "
                f"<b>{'higher' if direction == 'positive' else 'lower'}</b> by around <b>{max_pct:.0f}%</b>. ")
        body += ("Pattern varies across price bands." if changes_across_market
                 else "Holds across all price ranges.")
        pills = [("Affects listed price", "pill-amber")]

    elif var_name in ("power_steering", "power_mirror", "power_window"):
        feat_name = BUYER_LABELS[var_name]
        if n_sig >= 3:
            headline = f"✅ {feat_name.title()} is a genuine price signal"
            body = (f"Vehicles with <b>{feat_name}</b> are consistently priced "
                    f"<b>{up_or_down}</b> by around <b>{max_pct:.0f}%</b>. Sellers price it in. ")
            body += ("Premium bigger for expensive vehicles." if changes_across_market
                     else "Premium is similar at every budget level.")
            pills = [("Priced into market", "pill-green")]
        else:
            headline = f"📋 {feat_name.title()} has a small or inconsistent price effect"
            body = f"<b>{feat_name.title()}</b> doesn't reliably move price across most of the market."
            pills = [("Weak price signal", "pill-muted")]

    elif "C(brand_segment)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        if direction == "positive":
            headline = f"🏷️ Brand tier '{level}' commands a real price premium"
            body = f"<b>{level}</b> brand tier is listed around <b>{max_pct:.0f}% more</b> than the reference. "
        else:
            headline = f"🏷️ Brand tier '{level}' is priced below average"
            body = f"<b>{level}</b> brand tier is listed around <b>{max_pct:.0f}% less</b> than the reference. "
        if changes_across_market:
            body += f"Premium {'grows' if diff_b > 0 else 'shrinks'} for more expensive vehicles."
        pills = [("Brand premium confirmed", "pill-green" if direction == "positive" else "pill-amber")]

    elif "C(fuel_segment)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        if direction == "positive":
            headline = f"⛽ Fuel type '{level}' adds to the asking price"
            body = f"<b>{level}</b> fuel listed around <b>{max_pct:.0f}% higher</b> than baseline. "
        else:
            headline = f"⛽ Fuel type '{level}' is priced below average"
            body = f"<b>{level}</b> fuel listed around <b>{max_pct:.0f}% cheaper</b> than baseline. "
        pills = [("Fuel type priced in", "pill-amber")]

    elif "C(province)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        if direction == "positive":
            headline = f"📍 Vehicles in {level} are listed at a premium"
            body = f"Listings in <b>{level}</b> are roughly <b>{max_pct:.0f}% higher</b> than the reference province. "
        else:
            headline = f"📍 Vehicles in {level} tend to be cheaper"
            body = f"Listings in <b>{level}</b> are roughly <b>{max_pct:.0f}% lower</b> than the reference province. "
        pills = [("Location affects price", "pill-amber"), ("Consider nearby provinces", "pill-green")]

    elif "C(gear)" in var_name:
        level = var_name.split("[T.")[-1].rstrip("]")
        if direction == "positive":
            headline = f"⚙️ {level} transmission adds to the price"
            body = f"<b>{level}</b> gearboxes listed around <b>{max_pct:.0f}% higher</b> than the reference. "
        else:
            headline = f"⚙️ {level} transmission is priced lower"
            body = f"<b>{level}</b> gearboxes listed around <b>{max_pct:.0f}% lower</b> than the reference. "
        if changes_across_market:
            body += "Difference is <b>not the same across all budgets</b>."
        pills = [("Transmission priced in", "pill-green" if direction == "positive" else "pill-amber")]

    else:
        headline = f"📊 {blabel.title()} affects price"
        body = (f"Linked to <b>{up_or_down} prices</b> by around <b>{max_pct:.0f}%</b>. "
                f"Effect is {'consistent' if not changes_across_market else 'stronger for some budgets'}.")
        pills = [("Price driver", "pill-amber")]

    return headline, body, pills


def render_buyer_card(headline, body, pills):
    pill_html = "".join(f'<span class="signal-pill {s}">{t}</span>' for t, s in pills)
    st.markdown(f"""
    <div class="buyer-card">
      <div class="buyer-card-title">🛒 What this means if you're buying a car</div>
      <div class="buyer-verdict">{headline}</div>
      <div class="buyer-body">{body}</div>
      <div class="signal-row">{pill_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 Modeling")
    st.caption("Quantile regression — price drivers across the market")
    st.markdown("---")
    st.markdown("**Model spec**")
    st.write("• Response: log price")
    st.write("• Quantiles: Q25, Q50, Q75, Q90")
    st.write("• Significance threshold: α = 0.05")
    st.markdown("---")
    st.markdown("**On this page**")
    st.write("① What the model does")
    st.write("② Significant factors by quantile")
    st.write("③ How often each factor matters")
    st.write("④ Explore any factor interactively")
    st.write("⑤ What does this mean for buyers?")
    st.markdown("---")
    st.caption("Use the sidebar page list to move between pages.")


# ============================================================
# LOAD & RUN
# ============================================================
df = create_model_dataset()
qr_results, quantiles = run_quantile_regression(df)
coef_table, pval_table = build_tables(qr_results, quantiles)
sig_vars_df = significant_variables(pval_table)
presence_summary = get_feature_presence_summary(sig_vars_df)
quantile_labels = [f"Q{int(q*100)}" for q in quantiles]
variables = [v for v in coef_table.index.tolist() if v != "Intercept"]


st.markdown("""
<div class="page-hero">
  <div class="page-eyebrow"><span></span>Statistical Modelling</div>
  <div class="page-title">Quantile Regression</div>
  <div class="page-desc">
    This page quantifies how vehicle price drivers behave across different segments of the market —
    from budget listings to premium vehicles. Instead of one average effect, we estimate each
    factor separately at four price levels so you can see where it matters most.
  </div>
</div>
""", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Predictors in model", len(variables))
m2.metric("Quantiles estimated", len(quantiles))
m3.metric("Significance level", "α = 0.05")
n_core = len([r for r in presence_summary["Significant in N quantiles"] if r >= 3])
m4.metric("Consistent drivers (≥3 quantiles)", n_core)


st.markdown("""
<div class="section-block purple">
  <div class="label">Section 01</div>
  <div class="section-heading">How to Read This Model</div>
  <div class="section-desc">
    Ordinary regression gives one estimate per factor — the average effect across all vehicles.
    Quantile regression gives four estimates: one for each price segment below.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="q-grid">
  <div class="q-card q25"><div class="q-num">Q25</div><div class="q-label">Lower-priced vehicles — the budget end.</div></div>
  <div class="q-card q50"><div class="q-num">Q50</div><div class="q-label">Median market. The typical listing.</div></div>
  <div class="q-card q75"><div class="q-num">Q75</div><div class="q-label">Upper-mid segment. Priced above 75% of the market.</div></div>
  <div class="q-card q90"><div class="q-num">Q90</div><div class="q-label">Premium end. Top 10% of listings.</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight purple-t" style="margin-top:14px">
  <div class="insight-icon">💡</div>
  <div class="insight-text">
    <b>What to look for:</b> If a factor's coefficient is large at Q90 but small at Q25,
    it matters mainly for premium vehicles. The coefficient represents the estimated
    change in <em>log price</em> — e.g. 0.05 ≈ 5% price change.
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-block blue" style="margin-top:36px">
  <div class="label">Section 02</div>
  <div class="section-heading">Which Factors Are Statistically Significant?</div>
  <div class="section-desc">
    A factor is significant (p &lt; 0.05) when its association with price is unlikely to be chance.
    Scan across columns to see if the same factors recur at different price levels.
  </div>
</div>
""", unsafe_allow_html=True)

st.dataframe(sig_vars_df, use_container_width=True, hide_index=True)
takeaway = overall_takeaway(sig_vars_df)
st.markdown(f"""
<div class="insight">
  <div class="insight-icon">📌</div>
  <div class="insight-text"><b>Takeaway:</b> {takeaway}</div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-block green" style="margin-top:36px">
  <div class="label">Section 03</div>
  <div class="section-heading">How Often Is Each Factor Significant?</div>
  <div class="section-desc">
    A factor significant at all four quantiles has a consistent, market-wide effect.
    A factor significant at only one quantile is segment-specific.
  </div>
</div>
""", unsafe_allow_html=True)

st.dataframe(presence_summary, use_container_width=True, hide_index=True)
st.markdown("""
<div class="insight green-t">
  <div class="insight-icon">💡</div>
  <div class="insight-text">
    <b>Rule of thumb:</b> Factors in 3–4 quantiles are your most reliable price drivers.
    Factors in only 1 quantile still matter — but only within that segment.
  </div>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="section-block amber" style="margin-top:36px">
  <div class="label">Section 04 · Factor Impact Analysis</div>
  <div class="section-heading">Explore a Price Driver</div>
  <div class="section-desc">
    Select any factor to see how its effect changes across the four quantiles,
    with significance at each level and a detailed interpretation with real numbers.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight amber-t">
  <div class="insight-icon">📖</div>
  <div class="insight-text">
    <b>How to read the chart:</b> Positive = pushes price up. Negative = pushes price down.
    Flat line = consistent effect. Sloping line = effect changes across market segments.
    <b>Green dots = significant</b> (p &lt; 0.05). Dark dots = not significant.
  </div>
</div>
""", unsafe_allow_html=True)

selected_var = st.selectbox("Select a factor to explore", variables,
                             format_func=variable_label, key="var_select")

values = [coef_table.loc[selected_var, q] for q in quantile_labels]
pvals  = [pval_table.loc[selected_var, q] for q in quantile_labels]
sig_count = sum(p < 0.05 for p in pvals)

mc1, mc2, mc3, mc4 = st.columns(4)
direction_word = ("Positive ↑" if all(v > 0 for v in values)
                  else "Negative ↓" if all(v < 0 for v in values) else "Mixed ↕")
consistency = "High" if sig_count == 4 else "Partial" if sig_count >= 2 else "Low"
mc1.metric("Effect direction", direction_word)
mc2.metric("Significant at", f"{sig_count} / {len(quantiles)} quantiles")
mc3.metric("Overall consistency", consistency)
mc4.metric("Max |coefficient|", f"{max(abs(v) for v in values):.4f}")

col_chart, col_interp = st.columns([1.3, 1], gap="large")

with col_chart:
    st.markdown('<div class="label" style="margin-top:12px">Coefficient across quantiles</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0C1120"); ax.set_facecolor("#0C1120")
    ax.axhline(0, color="#243050", linewidth=1.2, linestyle="--", zorder=1)
    ax.fill_between(quantiles, values, 0, alpha=0.12, color="#5B8DEF", zorder=2)
    ax.plot(quantiles, values, color="#5B8DEF", linewidth=2.2, marker="o", markersize=9, zorder=3)
    for q_val, v, p in zip(quantiles, values, pvals):
        color = "#3ECFA0" if p < 0.05 else "#243050"
        edge  = "#3ECFA0" if p < 0.05 else "#5B8DEF"
        ax.scatter(q_val, v, color=color, edgecolors=edge, s=90, linewidths=2, zorder=4)
    for q_val, v in zip(quantiles, values):
        ax.annotate(f"{v:+.3f}", xy=(q_val, v), xytext=(0, 12),
                    textcoords="offset points", ha="center", fontsize=9.5, color="#B0BDD4")
    ax.set_xticks(quantiles)
    ax.set_xticklabels([f"Q{int(q*100)}" for q in quantiles])
    ax.set_xlabel("Price quantile", labelpad=8)
    ax.set_ylabel("Coefficient (log price scale)", labelpad=8)
    ax.set_title(f"Effect of  '{variable_label(selected_var)}'  across quantiles", pad=12, fontsize=12)
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#1C2840")
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#3ECFA0',
               markeredgecolor='#3ECFA0', markersize=8, label='Significant (p < 0.05)'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#243050',
               markeredgecolor='#5B8DEF', markersize=8, label='Not significant'),
    ]
    ax.legend(handles=legend_elements, loc="best", framealpha=0, labelcolor="#B0BDD4", fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col_interp:
    st.markdown('<div class="label" style="margin-top:12px">Detailed interpretation</div>',
                unsafe_allow_html=True)
    interp_text = interpret_variable(values, selected_var, pvals=pvals, quantile_labels=quantile_labels)
    st.markdown(f"""
    <div class="insight purple-t" style="margin-top:4px">
      <div class="insight-icon">🔍</div>
      <div class="insight-text">{interp_text}</div>
    </div>
    """, unsafe_allow_html=True)


with st.expander("Show full coefficient and p-value tables"):
    tab1, tab2 = st.tabs(["Coefficients", "P-values"])
    with tab1:
        display_coef = coef_table.copy()
        display_coef.index = [variable_label(v) for v in display_coef.index]
        st.dataframe(display_coef.style.format("{:+.4f}"), use_container_width=True)
    with tab2:
        display_pval = pval_table.copy()
        display_pval.index = [variable_label(v) for v in display_pval.index]
        st.dataframe(display_pval.style.format("{:.4f}"), use_container_width=True)


import numpy as np
from scipy import stats as _stats

se_table = pd.DataFrame({f"Q{int(q*100)}": qr_results[q].bse for q in quantiles})
ses    = [se_table.loc[selected_var, q] for q in quantile_labels]
b_q25  = values[0]; b_q90 = values[-1]
se_q25 = se_table.loc[selected_var, "Q25"]
se_q90 = se_table.loc[selected_var, "Q90"]
diff_b = b_q90 - b_q25
se_db  = np.sqrt(se_q25**2 + se_q90**2)
z_b    = diff_b / se_db if se_db > 0 else float("nan")
p_b    = 2 * (1 - _stats.norm.cdf(abs(z_b)))
ref_c  = coef_table.loc[selected_var, "Q50"]
ref_se = se_table.loc[selected_var, "Q50"]
devs   = [(coef_table.loc[selected_var, q] - ref_c,
           np.sqrt(se_table.loc[selected_var, q]**2 + ref_se**2))
          for q in quantile_labels]
chi2_c = sum((d/s)**2 for d, s in devs if s > 0)
df_c   = len(devs) - 1
p_c    = 1 - _stats.chi2.cdf(chi2_c, df=df_c)

headline, body, pills = buyer_verdict_text(selected_var, values, pvals, p_b, p_c, diff_b)

st.markdown("""
<div class="section-block green" style="margin-top:36px">
  <div class="label">Section 05 · Buyer Insight</div>
  <div class="section-heading">What Does This Mean If You're Buying a Car?</div>
  <div class="section-desc">
    Plain shopping advice — no maths required. Updates automatically when you change the factor above.
  </div>
</div>
""", unsafe_allow_html=True)

render_buyer_card(headline, body, pills)

st.markdown('<div class="label" style="margin-top:24px">Three questions answered by the data</div>',
            unsafe_allow_html=True)

blabel  = buyer_label(selected_var)
n_sig   = sum(p < 0.05 for p in pvals)
max_pct = coef_to_pct(max(abs(v) for v in values))

if n_sig == 0:
    q1_ans = f"No — no reliable price difference based on {blabel}."; q1_icon = "❌"
elif n_sig <= 2:
    q1_ans = "Sometimes — it matters in some parts of the market but not all."; q1_icon = "⚠️"
else:
    q1_ans = f"Yes — {blabel} reliably affects price across most of the market."; q1_icon = "✅"

if p_b >= 0.05:
    q2_ans = "No — it affects cheap and expensive cars about equally."; q2_icon = "⚖️"
elif diff_b > 0:
    q2_ans = "Yes — bigger impact on expensive vehicles."; q2_icon = "📈"
else:
    q2_ans = "Yes — bigger impact on cheaper vehicles."; q2_icon = "📉"

if max_pct < 3:
    q3_ans = "Probably not — under 3%, within normal negotiation range."; q3_icon = "🤏"
elif max_pct < 10:
    q3_ans = f"Possibly — around {max_pct:.0f}% difference. Worth checking."; q3_icon = "💬"
else:
    q3_ans = f"Yes — can reach {max_pct:.0f}%. Moves price significantly."; q3_icon = "💰"

col_q1, col_q2, col_q3 = st.columns(3)
for col, icon, question, answer in [
    (col_q1, q1_icon, f"Does {blabel} affect what I pay?", q1_ans),
    (col_q2, q2_icon, "Does it hit cheap and expensive cars the same way?", q2_ans),
    (col_q3, q3_icon, "Is the price gap big enough to matter?", q3_ans),
]:
    col.markdown(f"""
    <div style="background:var(--surface2);border:1px solid var(--border);
                border-radius:14px;padding:16px 16px 14px;height:100%;">
      <div style="font-size:1.4rem;margin-bottom:8px">{icon}</div>
      <div style="font-size:0.82rem;font-weight:600;color:var(--text);
                  margin-bottom:8px;line-height:1.3">{question}</div>
      <div style="font-size:0.83rem;color:var(--text-dim);font-weight:300;line-height:1.55">{answer}</div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("📐 See the statistical tests behind this verdict"):
    st.markdown("""
    <div class="insight purple-t" style="margin-top:0">
      <div class="insight-icon">🔬</div>
      <div class="insight-text">Three formal tests power the verdict above.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-top:16px">Test A — Does this factor matter at each price level?</div>', unsafe_allow_html=True)
    st.caption("Wald z-test: is the coefficient at each quantile meaningfully different from zero?")
    rows_a = []
    for ql, v, p, se in zip(quantile_labels, values, pvals, ses):
        rows_a.append({
            "Price level": {"Q25":"Budget","Q50":"Mid-range","Q75":"Upper-mid","Q90":"Premium"}[ql],
            "Effect size": f"{coef_to_pct(v):+.1f}%",
            "Reliable?": "✓ Yes" if p < 0.05 else "✗ No (could be chance)",
            "p-value": f"{p:.4f}" if p >= 0.0001 else "< 0.0001",
        })
    st.dataframe(pd.DataFrame(rows_a), use_container_width=True, hide_index=True)
    st.markdown('<div class="label" style="margin-top:16px">Test B — Is the effect different for cheap vs expensive cars?</div>', unsafe_allow_html=True)
    st.caption("Interquantile z-test: Q25 coefficient vs Q90 coefficient.")
    b_verdict = "Yes — significantly different at the two ends." if p_b < 0.05 else "No — similar at both ends."
    st.markdown(f"""
    <div style="background:var(--surface2);border:1px solid var(--border2);
                border-left:3px solid {'#3ECFA0' if p_b < 0.05 else '#F5A843'};
                border-radius:12px;padding:14px 16px;">
      <div style="font-size:0.85rem;color:var(--text-dim);">
        Budget: <b style="color:var(--text)">{coef_to_pct(b_q25):+.1f}%</b> &nbsp;|&nbsp;
        Premium: <b style="color:var(--text)">{coef_to_pct(b_q90):+.1f}%</b> &nbsp;|&nbsp;
        p: <b style="color:var(--text)">{p_b:.4f}</b><br/>
        <b style="color:{'#3ECFA0' if p_b < 0.05 else '#F5A843'}">{b_verdict}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-top:16px">Test C — Is the effect consistent across all four price levels?</div>', unsafe_allow_html=True)
    st.caption("Chi-squared joint test: are all four coefficients statistically the same?")
    c_verdict = "No — effect varies across price levels." if p_c < 0.05 else "Yes — effect is statistically flat."
    st.markdown(f"""
    <div style="background:var(--surface2);border:1px solid var(--border2);
                border-left:3px solid {'#3ECFA0' if p_c < 0.05 else '#F5A843'};
                border-radius:12px;padding:14px 16px;">
      <div style="font-size:0.85rem;color:var(--text-dim);">
        χ² = <b style="color:var(--text)">{chi2_c:.3f}</b> &nbsp;|&nbsp;
        df = <b style="color:var(--text)">{df_c}</b> &nbsp;|&nbsp;
        p = <b style="color:var(--text)">{p_c:.4f}</b><br/>
        <b style="color:{'#3ECFA0' if p_c < 0.05 else '#F5A843'}">{c_verdict}</b>
      </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="insight amber-t" style="margin-top:20px">
  <div class="insight-icon">⚠️</div>
  <div class="insight-text">
    <b>A note on using this as a buyer:</b> Results describe what is <em>typical</em>, not guaranteed.
    Always inspect a vehicle in person and check its history before buying.
    Price differences reflect <em>association</em>, not guaranteed value.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="insight amber-t" style="margin-top:28px">
  <div class="insight-icon">⚠️</div>
  <div class="insight-text">
    <b>Model notes:</b> Quantile regression measures <em>association</em>, not causation.
    Categorical variables are interpreted relative to a <em>reference category</em>.
    The response is log price — coefficients approximate % changes (0.10 ≈ 10%).
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="footer-line">ST 3011 Statistical Programming · Individual Streamlit Application</div>',
    unsafe_allow_html=True,
)
