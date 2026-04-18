# Sri Lanka Used-Car Market Dashboard
### ST3011 Statistical Programming — Individual Streamlit Application
**D.A. Yahathugoda (s16877)**

---

## Overview

This is an interactive web dashboard built as an individual extension of the [Group 07 ST3011 project](https://github.com/Dilmith-Anusara/ST_3011_Data_Analysis_Group_Project.git).

The group project analyzed used car prices in Sri Lanka during the pre- and early-2025 import liberalization period using a static notebook. This app turns those findings into an interactive experience — and extends the analysis with **Quantile Regression** as the individual contribution.

---

## Pages

**Home (`App.py`)**
Landing page explaining the dashboard context and how to navigate it.

**Objectives Explorer (`1_Objectives_Explorer.py`)**
Interactive version of the group project's four research objectives. Explore mechanical market structure, comfort feature distributions, and geo-economic patterns through filterable charts.

**Modeling & Analysis (`2_Modeling.py`) — Individual Contribution**
Quantile regression model fitted at Q25, Q50, Q75, and Q90. Instead of modeling the average price, this shows how each factor affects prices differently across cheap, mid-range, and premium vehicles. Includes:
- Significance summary table per quantile
- Interactive feature explorer — select any factor and see how its effect changes across price levels
- Plain-language interpretation of each coefficient
- Full coefficient and p-value tables

---

## Project Structure

```
├── App.py                              # Main landing page
├── pages/
│   ├── 1_Objectives_Explorer.py        # Group project EDA — interactive
│   └── 2_Modeling.py                   # Individual contribution — quantile regression
├── utils.py                            # Data loading, cleaning, feature engineering
├── data/
│   └── car_price_dataset.csv           # Dataset
├── requirements.txt
└── README.md
```

---

## How to Run

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the app**
```bash
streamlit run App.py
```

Then open `http://localhost:8501` in your browser.

---

## Individual Contribution

The **Modeling & Analysis** page is the individual extension. The group project used a GAM model for inference across the whole market. This app extends that with **Quantile Regression** — a method that reveals whether price drivers behave differently across cheap vs. premium vehicles, which a single-model approach cannot show.

Key questions answered:
- Does vehicle age depreciate budget cars differently than premium ones?
- Does engine capacity matter more at the high end of the market?
- Are comfort features and fuel type priced differently across market segments?

---

## Tech Stack

See [`requirements.txt`](./requirements.txt) for full details. Key libraries:

- **App framework:** `streamlit`
- **Data:** `pandas`, `numpy`, `scipy`
- **Modeling:** `statsmodels` (quantile regression via `smf.quantreg`)
- **Visualization:** `plotly`, `matplotlib`

---

## Related

- [Group Project Repository](https://github.com/Dilmith-Anusara/ST_3011_Data_Analysis_Group_Project.git) — full statistical analysis notebook and report
- [🚀 Live Demo](https://st3011individualextension-gtqzrm5zltph6vmf3pfdzo.streamlit.app/)
