import os
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================
# HELPERS
# ============================================================

def clean_column_name(col_name: str) -> str:
    """
    Convert column names to snake_case.
    """
    col_name = str(col_name).lower()
    col_name = col_name.replace("(cc)", "_cc")
    col_name = col_name.replace("(km)", "_km")
    col_name = col_name.replace(" ", "_").replace("-", "_")
    while "__" in col_name:
        col_name = col_name.replace("__", "_")
    return col_name.strip("_")


def classify_year(yom: float) -> str:
    if pd.isna(yom):
        return "Unknown"
    yom = int(float(yom))
    if yom < 2010:
        return "Old"
    elif yom < 2018:
        return "Intermediate"
    else:
        return "Modern"


def classify_engine(cc) -> str:
    if pd.isna(cc):
        return "Unknown"
    cc = float(cc)
    if cc < 800:
        return "Micro (<800cc)"
    elif cc <= 1200:
        return "Compact (800-1200cc)"
    elif cc <= 1600:
        return "Mid-Range (1200-1600cc)"
    else:
        return "Large (>1600cc)"


def parse_leasing(x):
    """
    Convert leasing text to boolean.
    """
    if pd.isna(x):
        return np.nan

    s = str(x).lower().strip()

    if "no" in s and "leasing" in s:
        return False
    if "lease" in s:
        return True

    return np.nan


def availability_to_int(series: pd.Series) -> pd.Series:
    """
    Convert 'Available' / 'Not_Available' to 1 / 0 safely.
    Works even if values are already numeric.
    """
    mapping = {
        "Available": 1,
        "Not_Available": 0,
        "available": 1,
        "not_available": 0,
        "Not Available": 0,
        "not available": 0
    }

    if series.dtype.kind in "biufc":
        return series.astype(int)

    return series.map(mapping)


def classify_brand_segment(brand: str) -> str:
    """
    Group brands into economically meaningful segments.
    """
    economy = {
        "SUZUKI", "PERODUA", "MICRO", "TATA", "DAIHATSU",
        "PROTON", "CHERY", "ZOTYE", "BAJAJ", "MAHINDRA", "DATSUN"
    }

    premium = {
        "MERCEDES-BENZ", "BMW", "AUDI", "LEXUS", "MINI",
        "JAGUAR", "VOLVO", "LAND-ROVER", "TESLA"
    }

    if pd.isna(brand):
        return "Mid-range"

    brand = str(brand).strip().upper()

    if brand in economy:
        return "Economy"
    if brand in premium:
        return "Premium"
    return "Mid-range"


# ============================================================
# TOWN → PROVINCE MAPPING
# ============================================================

TOWN_TO_PROVINCE = {
    # Western
    "Colombo": "Western",
    "Gampaha": "Western",
    "Negombo": "Western",
    "Kalutara": "Western",
    "Panadura": "Western",
    "Moratuwa": "Western",
    "Dehiwala-Mount-Lavinia": "Western",
    "Maharagama": "Western",
    "Kotte": "Western",
    "Wattala": "Western",
    "Ja-Ela": "Western",
    "Kelaniya": "Western",
    "Kadawatha": "Western",
    "Nugegoda": "Western",
    "Piliyandala": "Western",
    "Boralesgamuwa": "Western",

    # Central
    "Kandy": "Central",
    "Matale": "Central",
    "Nuwara-Eliya": "Central",
    "Gampola": "Central",
    "Nawalapitiya": "Central",
    "Hatton": "Central",

    # Southern
    "Galle": "Southern",
    "Matara": "Southern",
    "Hambantota": "Southern",
    "Weligama": "Southern",
    "Tangalle": "Southern",
    "Hikkaduwa": "Southern",
    "Ambalangoda": "Southern",

    # Northern
    "Jaffna": "Northern",
    "Vavuniya": "Northern",
    "Kilinochchi": "Northern",
    "Mullaitivu": "Northern",

    # Eastern
    "Batticaloa": "Eastern",
    "Trincomalee": "Eastern",
    "Ampara": "Eastern",
    "Kalmunai": "Eastern",

    # North Western
    "Kurunegala": "North Western",
    "Puttalam": "North Western",
    "Kuliyapitiya": "North Western",
    "Chilaw": "North Western",

    # North Central
    "Anuradapura": "North Central",
    "Polonnaruwa": "North Central",

    # Uva
    "Badulla": "Uva",
    "Bandarawela": "Uva",
    "Haputale": "Uva",
    "Welimada": "Uva",

    # Sabaragamuwa
    "Ratnapura": "Sabaragamuwa",
    "Kegalle": "Sabaragamuwa",
    "Balangoda": "Sabaragamuwa",
}

URBAN_TOWNS = {
    "Ambalangoda", "Ampara", "Anuradapura", "Avissawella", "Badulla", "Balangoda",
    "Bandarawela", "Batticaloa", "Beruwala", "Boralesgamuwa", "Chavakacheri", "Chilaw",
    "Colombo", "Dambulla", "Dehiwala-Mount-Lavinia", "Galle", "Gampaha", "Gampola",
    "Hambantota", "Haputale", "Hatton", "Hikkaduwa", "Horana", "Ja-Ela", "Jaffna",
    "Kadugannawa", "Kaduwela", "Kalmunai", "Kalutara", "Kandy", "Kattankudy",
    "Katunayake", "Kegalle", "Kesbewa", "Kolonnawa", "Kotte", "Kuliyapitiya",
    "Kurunegala", "Maharagama", "Matale", "Matara", "Minuwangoda", "Moratuwa",
    "Nawalapitiya", "Negombo", "Nuwara-Eliya", "Panadura", "Peliyagoda", "Puttalam",
    "Ratnapura", "Tangalle", "Trincomalee", "Vavuniya", "Wattala", "Wattegama",
    "Weligama"
}


# ============================================================
# RAW CLEANING + FEATURE ENGINEERING
# ============================================================

@st.cache_data
def load_and_preprocess(path: str = "data/car_price_dataset.csv", age_reference_year: int = 2026) -> pd.DataFrame:
    """
    Load raw dataset and perform cleaning + intermediate feature engineering.

    Steps:
    - load CSV
    - remove duplicates
    - drop date column if present
    - remove NEW vehicles from condition
    - standardize column names
    - create age, year_category, engine_segment
    - parse leasing
    - create province, location_type
    - create simple_model, brand_grouped, brand_model

    Returns
    -------
    pd.DataFrame
        Clean intermediate dataset
    """

    # ----------------------------
    # Load raw data
    # ----------------------------
    df = pd.read_csv(path)

    # ----------------------------
    # Basic cleaning
    # ----------------------------
    df = df.drop_duplicates()

    for c in ["date", "Date", "DATE"]:
        if c in df.columns:
            df = df.drop(columns=[c])
            break

    if "Condition" in df.columns:
        df = df[df["Condition"] != "NEW"]
    elif "condition" in df.columns:
        df = df[df["condition"] != "NEW"]

    # ----------------------------
    # Standardize column names
    # ----------------------------
    df.columns = [clean_column_name(c) for c in df.columns]

    # ----------------------------
    # Feature engineering
    # ----------------------------

    # Age + year category
    if "yom" in df.columns:
        df["age"] = age_reference_year - df["yom"]
        df["year_category"] = df["yom"].apply(classify_year)

    # Engine segment
    if "engine_cc" in df.columns:
        df["engine_segment"] = df["engine_cc"].apply(classify_engine)

    # Brand grouped by frequency
    if "brand" in df.columns:
        brand_counts = df["brand"].value_counts(dropna=False)

        def group_brand(x):
            if pd.isna(x):
                return "Unknown"
            count = brand_counts.get(x, 0)
            if count > 100:
                return x
            elif count >= 10:
                return "OTHER"
            else:
                return "RARE"

        df["brand_grouped"] = df["brand"].apply(group_brand)

    # Leasing
    if "leasing" in df.columns:
        df["leasing"] = df["leasing"].apply(parse_leasing)

    # Province + location type
    if "town" in df.columns:
        town_norm = df["town"].astype(str).str.strip()
        town_title = town_norm.str.title()

        df["province"] = town_title.map(TOWN_TO_PROVINCE).fillna("Other")
        df["location_type"] = town_norm.apply(
            lambda x: "Urban" if str(x).strip() in URBAN_TOWNS else "Non-Urban"
        )

    # Model helpers
    if "model" in df.columns:
        df["simple_model"] = df["model"].astype(str).str.split().str[0]

    if "brand_grouped" in df.columns and "simple_model" in df.columns:
        df["brand_model"] = df["brand_grouped"].astype(str) + "_" + df["simple_model"].astype(str)

    df = df.reset_index(drop=True)
    return df


# ============================================================
# FINAL MODEL DATASET
# ============================================================

@st.cache_data
def create_model_dataset(path: str = "data/car_price_dataset.csv", age_reference_year: int = 2026) -> pd.DataFrame:
    """
    Create the final modeling dataset with exactly these columns:

    - gear
    - leasing
    - power_steering
    - power_mirror
    - power_window
    - age
    - province
    - location_type
    - brand_segment
    - fuel_segment
    - log_price
    - log_engine_cc
    """

    df = load_and_preprocess(path=path, age_reference_year=age_reference_year).copy()

    # ----------------------------
    # Brand segment
    # ----------------------------
    if "brand" in df.columns:
        df["brand_segment"] = df["brand"].apply(classify_brand_segment)

    # ----------------------------
    # Fuel segment
    # ----------------------------
    if "fuel_type" in df.columns:
        df["fuel_segment"] = df["fuel_type"].replace({
            "Hybrid": "Electrified",
            "Electric": "Electrified"
        })

    # ----------------------------
    # Log transforms
    # ----------------------------
    if "price" in df.columns:
        df = df[df["price"] > 0].copy()
        df["log_price"] = np.log(df["price"])

    if "engine_cc" in df.columns:
        df = df[df["engine_cc"] > 0].copy()
        df["log_engine_cc"] = np.log(df["engine_cc"])

    # ----------------------------
    # Binary conversions
    # ----------------------------
    if "leasing" in df.columns:
        df["leasing"] = df["leasing"].astype(int)

    if "power_steering" in df.columns:
        df["power_steering"] = availability_to_int(df["power_steering"])

    if "power_mirror" in df.columns:
        df["power_mirror"] = availability_to_int(df["power_mirror"])

    if "power_window" in df.columns:
        df["power_window"] = availability_to_int(df["power_window"])

    # ----------------------------
    # Final feature selection
    # ----------------------------
    final_cols = [
        "gear",
        "leasing",
        "power_steering",
        "power_mirror",
        "power_window",
        "age",
        "province",
        "location_type",
        "brand_segment",
        "fuel_segment",
        "log_price",
        "log_engine_cc"
    ]

    existing_cols = [c for c in final_cols if c in df.columns]
    df_final = df[existing_cols].copy()

    df_final = df_final.dropna().reset_index(drop=True)

    return df_final