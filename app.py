from __future__ import annotations

import hashlib
import html
import io
import math
import re
import warnings
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_NAME = "InsightForge"
MAX_CHART_CATEGORIES = 18
MAX_FILTER_OPTIONS = 250

st.set_page_config(
    page_title="InsightForge | Data Analytics Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Visual design -----------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root { --if-ink:#172033; --if-muted:#64748b; --if-line:#e6eaf1; --if-bg:#f5f7fb; --if-accent:#4f46e5; }
html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
.stApp { background:var(--if-bg); color:var(--if-ink); }
.block-container { padding-top:1.25rem; padding-bottom:2.5rem; max-width:1600px; }
[data-testid="stSidebar"] { background:#101827; border-right:1px solid #202b3c; }
[data-testid="stSidebar"] * { color:#e7edf7; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stCaption { color:#aebbd0 !important; }
[data-testid="stSidebar"] [data-baseweb="select"] div { color:#172033 !important; }
[data-testid="stSidebar"] input { color:#172033 !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] { background:#172235; border:1px dashed #4b5b72; border-radius:12px; padding:.4rem; }
[data-testid="stSidebar"] button { border-radius:9px; }
.if-brand { display:flex; align-items:center; gap:12px; margin:0 0 1.1rem 0; }
.if-logo { width:42px; height:42px; border-radius:13px; display:flex; justify-content:center; align-items:center; color:white; font-size:20px; background:linear-gradient(135deg,#665cf6,#14b8a6); box-shadow:0 8px 20px rgba(79,70,229,.22); }
.if-brand-title { font:800 1.28rem Manrope,sans-serif; letter-spacing:-.04em; margin:0; color:#172033; }
.if-brand-sub { margin:1px 0 0 0; font-size:.78rem; color:#76839a; }
.if-hero { padding:1.35rem 1.5rem; margin:.15rem 0 1.2rem; border:1px solid #e4e9f2; border-radius:18px; background:linear-gradient(112deg,#fff 15%,#f0f2ff 100%); box-shadow:0 5px 22px rgba(31,45,76,.035); }
.if-eyebrow { text-transform:uppercase; letter-spacing:.13em; font-weight:700; color:#5c5bd6; font-size:.7rem; margin-bottom:.35rem; }
.if-title { font:800 clamp(1.5rem,2.5vw,2.15rem) Manrope,sans-serif; letter-spacing:-.045em; line-height:1.15; color:#172033; margin:0; }
.if-subtitle { color:#64748b; font-size:.93rem; margin:.55rem 0 0 0; max-width:860px; }
.if-section { font:700 1.03rem Manrope,sans-serif; color:#202a40; letter-spacing:-.02em; margin:.2rem 0 .65rem; }
.if-card { background:white; padding:1rem 1.05rem; border-radius:14px; border:1px solid #e7ebf3; box-shadow:0 4px 16px rgba(23,32,51,.025); height:100%; }
.if-card-label { color:#728097; font-size:.78rem; font-weight:600; letter-spacing:.01em; margin-bottom:.55rem; }
.if-card-value { color:#18233a; font:800 1.45rem Manrope,sans-serif; letter-spacing:-.04em; overflow-wrap:anywhere; }
.if-card-help { color:#7b8799; font-size:.73rem; margin-top:.36rem; }
.if-pill { display:inline-block; padding:4px 9px; border-radius:99px; background:#eeefff; color:#5145cd; font-size:.72rem; font-weight:700; }
.if-muted { color:#758198; font-size:.82rem; }
.if-empty { padding:2.4rem 1.6rem; text-align:center; background:#fff; border:1px dashed #cbd5e1; border-radius:18px; }
.if-empty h3 { font-family:Manrope,sans-serif; margin:.5rem 0; color:#202a40; }
.if-empty p { color:#718096; max-width:600px; margin:.4rem auto 1rem; }
div[data-testid="stMetric"] { background:#fff; border:1px solid #e7ebf3; border-radius:14px; padding:14px 16px; box-shadow:0 4px 16px rgba(23,32,51,.025); }
div[data-testid="stMetricLabel"] p { color:#728097; font-weight:600; }
div[data-testid="stMetricValue"] { font-family:Manrope,sans-serif; letter-spacing:-.04em; }
.stTabs [data-baseweb="tab-list"] { gap:8px; background:transparent; }
.stTabs [data-baseweb="tab"] { border-radius:10px; background:#eceff5; padding:10px 15px; color:#57657a; }
.stTabs [aria-selected="true"] { background:#fff !important; color:#4f46e5 !important; box-shadow:0 2px 9px rgba(23,32,51,.06); }
div[data-testid="stDataFrame"] { border:1px solid #e7ebf3; border-radius:12px; overflow:hidden; }
hr { border-color:#e5e9f2; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------- State helpers ----------------------------
def init_state() -> None:
    defaults = {
        "datasets_raw": {},
        "datasets_inferred": {},
        "datasets_meta": {},
        "loaded_signature": None,
        "active_dataset": None,
        "bookmarks": {},
        "chart_meta": {},
        "cross_filter": None,
        "last_active_dataset": None,
        "upload_errors": [],
        "data_version": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def stable_id(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="replace")).hexdigest()[:10]


def display_name(key: str) -> str:
    return st.session_state.datasets_meta.get(key, {}).get("display_name", key)


def safe_label(value: Any) -> str:
    if pd.isna(value):
        return "(Missing)"
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    return str(value)


def normalize_col_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def unique_column_names(columns) -> list[str]:
    result, used = [], set()
    for i, col in enumerate(columns, start=1):
        name = str(col).strip() if col is not None else ""
        if not name or name.lower().startswith("unnamed:"):
            name = f"Column {i}"
        candidate, n = name, 2
        while candidate.casefold() in used:
            candidate = f"{name}_{n}"
            n += 1
        used.add(candidate.casefold())
        result.append(candidate)
    return result


def infer_types(df: pd.DataFrame) -> pd.DataFrame:
    """Create an analytical copy with conservative date/numeric inference."""
    out = df.copy(deep=True)
    out.columns = unique_column_names(out.columns)
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        non_null = series.dropna()
        if non_null.empty:
            continue
        sample = non_null.astype(str).head(200)
        name = str(col).lower()
        is_id_like = any(word in name for word in ("identifier", "_id", " id", "code", "phone", "mobile", "postal", "zip", "pincode")) or name.endswith("id")
        date_name = any(word in name for word in ("date", "time", "timestamp", "created", "updated", "admission", "payment", "order date", "dob", "birth"))
        date_pattern = sample.str.contains(r"[/\-:.]", regex=True).mean() >= 0.75
        if not is_id_like and (date_name or date_pattern):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False, format="mixed")
                except (TypeError, ValueError):
                    parsed = pd.to_datetime(series, errors="coerce", dayfirst=False)
            valid_ratio = parsed.notna().sum() / max(1, series.notna().sum())
            if valid_ratio >= 0.80 and parsed.nunique(dropna=True) > 1:
                out[col] = parsed
                continue
        if not is_id_like:
            txt = series.astype("string").str.strip()
            percent_present = txt.str.contains("%", regex=False, na=False).mean() > 0.5
            candidate = txt.str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("₹", "", regex=False).str.replace("€", "", regex=False).str.replace("£", "", regex=False).str.replace("%", "", regex=False)
            parsed_num = pd.to_numeric(candidate, errors="coerce")
            ratio = parsed_num.notna().sum() / max(1, series.notna().sum())
            if ratio >= 0.90 and parsed_num.nunique(dropna=True) > 1:
                out[col] = parsed_num
                # Percent values are left in their displayed numeric scale (e.g. 12 means 12%).
                if percent_present:
                    out.attrs.setdefault("percent_columns", []).append(col)
    return out


def header_looks_suspicious(df: pd.DataFrame) -> bool:
    if df.empty or len(df.columns) == 0:
        return False
    cols = [str(c).strip() for c in df.columns]
    unnamed = sum(c.lower().startswith("unnamed:") or c.lower().startswith("column ") for c in cols)
    numericish = sum(bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", c)) for c in cols)
    dateish = sum(bool(re.search(r"\d{1,4}[/\-]\d{1,2}", c)) for c in cols)
    return (unnamed / len(cols) > 0.5) or ((numericish + dateish) / len(cols) > 0.5)


def parse_bytes(data: bytes, file_name: str, sheet_name: str | None = None, header: int | None = 0) -> dict[str, pd.DataFrame]:
    ext = file_name.lower().rsplit(".", 1)[-1] if "." in file_name else ""
    if ext == "csv":
        last_error = None
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
            try:
                frame = pd.read_csv(io.BytesIO(data), encoding=encoding, header=header, sep=None, engine="python", on_bad_lines="warn")
                return {file_name: frame}
            except Exception as exc:  # retry common encodings
                last_error = exc
        raise ValueError(f"Could not read CSV: {last_error}")
    if ext in ("xlsx", "xls"):
        engine = "openpyxl" if ext == "xlsx" else "xlrd"
        try:
            book = pd.ExcelFile(io.BytesIO(data), engine=engine)
        except ImportError as exc:
            if ext == "xls":
                raise ValueError("Legacy .xls files need the xlrd dependency. Reinstall the app dependencies and try again.") from exc
            raise
        sheets: dict[str, pd.DataFrame] = {}
        for name in book.sheet_names:
            if sheet_name is not None and name != sheet_name:
                continue
            try:
                sheet = pd.read_excel(book, sheet_name=name, header=header)
                if not (sheet.empty and len(sheet.columns) == 0):
                    sheets[f"{file_name} · {name}"] = sheet
            except Exception as exc:
                raise ValueError(f"Could not read sheet '{name}' in {file_name}: {exc}") from exc
        return sheets
    raise ValueError("Unsupported file type. Use CSV, XLSX, or XLS.")


def reset_workspace() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith(("filter__", "filtercols__", "clean__", "bookmarkname__", "bookmarkpick__", "restore__", "savebookmark__", "trend__", "chartbuilder__", "merge__")):
            st.session_state.pop(key, None)
    st.session_state.datasets_raw = {}
    st.session_state.datasets_inferred = {}
    st.session_state.datasets_meta = {}
    st.session_state.loaded_signature = None
    st.session_state.active_dataset = None
    st.session_state.bookmarks = {}
    st.session_state.chart_meta = {}
    st.session_state.cross_filter = None
    st.session_state.upload_errors = []
    st.session_state.data_version += 1


def load_uploaded(uploaded_files) -> None:
    signatures = []
    file_content: list[tuple[str, bytes, str]] = []
    for item in uploaded_files:
        try:
            content = item.getvalue()
        except Exception:
            content = item.read()
        digest = hashlib.sha1(content).hexdigest()
        signatures.append((item.name, len(content), digest))
        file_content.append((item.name, content, digest))
    signature = tuple(signatures)
    if not file_content or signature == st.session_state.loaded_signature:
        return
    raw, inferred, meta = {}, {}, {}
    errors = []
    for file_name, content, digest in file_content:
        try:
            frames = parse_bytes(content, file_name)
            for parsed_name, frame in frames.items():
                if frame.empty and len(frame.columns) == 0:
                    errors.append(f"{parsed_name}: empty sheet/file skipped.")
                    continue
                key_base = parsed_name
                key = key_base
                if key in raw:
                    key = f"{key_base} [{digest[:6]}]"
                raw[key] = frame
                inferred[key] = infer_types(frame)
                sheet = parsed_name.split(" · ", 1)[1] if " · " in parsed_name else None
                meta[key] = {
                    "display_name": key,
                    "file_name": file_name,
                    "sheet_name": sheet,
                    "source_bytes": content,
                    "source_digest": digest,
                    "derived": False,
                    "header_mode": "header",
                    "header_warning": header_looks_suspicious(frame),
                }
        except Exception as exc:
            errors.append(f"{file_name}: {exc}")
    st.session_state.datasets_raw = raw
    st.session_state.datasets_inferred = inferred
    st.session_state.datasets_meta = meta
    st.session_state.loaded_signature = signature
    st.session_state.upload_errors = errors
    st.session_state.bookmarks = {}
    st.session_state.cross_filter = None
    st.session_state.chart_meta = {}
    st.session_state.data_version += 1
    st.session_state.active_dataset = next(iter(raw.keys()), None)
    st.session_state.pending_active_dataset = st.session_state.active_dataset


def reload_header_mode(dataset_key: str, header: int | None) -> None:
    meta = st.session_state.datasets_meta[dataset_key]
    frames = parse_bytes(meta["source_bytes"], meta["file_name"], meta.get("sheet_name"), header=header)
    frame = next(iter(frames.values()))
    frame.columns = unique_column_names(frame.columns) if header is None else frame.columns
    st.session_state.datasets_raw[dataset_key] = frame
    st.session_state.datasets_inferred[dataset_key] = infer_types(frame)
    meta["header_mode"] = "no_header" if header is None else "header"
    meta["header_warning"] = header_looks_suspicious(frame)
    st.session_state.cross_filter = None
    st.session_state.chart_meta = {}
    st.session_state.data_version += 1

def make_demo_dataset() -> pd.DataFrame:
    """Generate a clearly labeled, reproducible sample dataset for first-run exploration."""
    rng = np.random.default_rng(42)
    n = 650
    dates = pd.to_datetime(rng.integers(pd.Timestamp("2024-01-01").value // 10**9, pd.Timestamp("2026-10-01").value // 10**9, n), unit="s").normalize()
    regions = rng.choice(["North", "South", "East", "West", "Central"], n, p=[.23, .22, .18, .21, .16])
    segments = rng.choice(["Consumer", "Small Business", "Enterprise"], n, p=[.54, .31, .15])
    products = rng.choice(["Laptop", "Monitor", "Keyboard", "Desk", "Headphones", "Webcam"], n)
    unit_prices = {"Laptop": 850, "Monitor": 260, "Keyboard": 65, "Desk": 240, "Headphones": 115, "Webcam": 78}
    quantities = rng.integers(1, 9, n)
    discounts = rng.choice([0, .03, .05, .10, .15, .20], n, p=[.25, .10, .18, .22, .17, .08])
    sales = np.array([unit_prices[p] for p in products], dtype=float) * quantities * (1 - discounts)
    costs = np.array([unit_prices[p] * .62 for p in products], dtype=float) * quantities
    noise = rng.normal(0, 18, n)
    profit = sales - costs + noise
    demo = pd.DataFrame({
        "Order ID": [f"ORD-{i:05d}" for i in range(1, n + 1)],
        "Order Date": dates,
        "Region": regions,
        "Segment": segments,
        "Product": products,
        "Quantity": quantities,
        "Sales": np.round(sales, 2),
        "Profit": np.round(profit, 2),
        "Discount %": np.round(discounts * 100, 0),
    })
    # A few explicit quality issues let the user see quality checks; all are synthetic.
    missing_idx = rng.choice(n, 9, replace=False)
    demo.loc[missing_idx[:5], "Region"] = None
    demo.loc[missing_idx[5:], "Profit"] = np.nan
    demo = pd.concat([demo, demo.iloc[[13, 27]]], ignore_index=True)
    return demo


def load_demo_dataset() -> None:
    reset_workspace()
    key = "Demo sales dataset (synthetic)"
    frame = make_demo_dataset()
    st.session_state.datasets_raw[key] = frame
    st.session_state.datasets_inferred[key] = infer_types(frame)
    st.session_state.datasets_meta[key] = {"display_name": key, "file_name": None, "sheet_name": None, "source_bytes": None, "source_digest": "demo", "derived": False, "sample": True, "header_mode": "header", "header_warning": False}
    st.session_state.active_dataset = key
    st.session_state.pending_active_dataset = key
    st.session_state.data_version += 1

# ----------------------------- Data analysis -----------------------------
def is_id_column(col: str) -> bool:
    token = normalize_col_token(col)
    return token in {"id", "studentid", "customerid", "orderid", "transactionid", "paymentid", "recordid", "index"} or token.endswith(("id", "code", "phone", "mobile", "postal", "pincode", "zipcode"))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def measure_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in numeric_columns(df):
        token = normalize_col_token(col)
        if is_id_column(col) or token in {"year", "month", "day", "quarter", "index", "latitude", "longitude"}:
            continue
        cols.append(col)
    return cols


def date_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def categorical_columns(df: pd.DataFrame) -> list[str]:
    result = []
    for col in df.columns:
        if col in date_columns(df) or col in numeric_columns(df):
            continue
        result.append(col)
    return result


def likely_measure(df: pd.DataFrame) -> str | None:
    candidates = measure_columns(df)
    if not candidates:
        return None
    keywords = ("sales", "revenue", "amount", "profit", "price", "fee", "payment", "cost", "total", "quantity", "qty", "score", "marks", "value", "balance", "income", "expense", "units", "discount", "salary", "count")
    def score(col: str) -> int:
        token = normalize_col_token(col)
        points = sum(4 for word in keywords if word in token)
        if any(term in token for term in ("amount", "sales", "revenue", "profit", "fee", "payment", "total")):
            points += 3
        if df[col].nunique(dropna=True) <= 1:
            points -= 10
        return points
    return sorted(candidates, key=lambda c: (score(c), df[c].nunique(dropna=True)), reverse=True)[0]


def likely_entity_column(df: pd.DataFrame) -> str | None:
    non_numeric = [c for c in df.columns if c not in date_columns(df) and df[c].nunique(dropna=True) > 1]
    preferred = [c for c in non_numeric if any(w in normalize_col_token(c) for w in ("studentid", "customerid", "orderid", "employeeid", "productid", "transactionid", "accountid"))]
    if preferred:
        return preferred[0]
    for c in non_numeric:
        nunique = df[c].nunique(dropna=True)
        ratio = nunique / max(1, len(df))
        if ratio >= 0.25 and nunique > 1 and nunique < max(2, len(df) * 1.01):
            return c
    return None


def fmt_number(value: Any) -> str:
    try:
        if pd.isna(value):
            return "—"
        n = float(value)
        absn = abs(n)
        if absn >= 1_000_000_000:
            return f"{n / 1_000_000_000:,.2f}B"
        if absn >= 1_000_000:
            return f"{n / 1_000_000:,.2f}M"
        if absn >= 10_000:
            return f"{n:,.0f}"
        if absn >= 1_000:
            return f"{n:,.1f}"
        if math.isclose(n, round(n), abs_tol=1e-9):
            return f"{n:,.0f}"
        return f"{n:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def make_kpis(df: pd.DataFrame) -> list[dict[str, str]]:
    cards = [{"label": "Records", "value": fmt_number(len(df)), "help": "Rows in the current view"}]
    entity = likely_entity_column(df)
    if entity:
        cards.append({"label": f"Distinct {entity}", "value": fmt_number(df[entity].nunique(dropna=True)), "help": "Unique non-missing values"})
    measure = likely_measure(df)
    if measure:
        vals = pd.to_numeric(df[measure], errors="coerce")
        agg_name = "Sum" if any(w in normalize_col_token(measure) for w in ("sales", "revenue", "amount", "profit", "fee", "payment", "total", "cost", "quantity", "qty", "units", "income", "expense")) else "Average"
        agg_val = vals.sum(min_count=1) if agg_name == "Sum" else vals.mean()
        cards.append({"label": f"{agg_name} · {measure}", "value": fmt_number(agg_val), "help": f"Calculated from {vals.notna().sum():,} numeric values"})
        cards.append({"label": f"Median · {measure}", "value": fmt_number(vals.median()), "help": "Middle value; less affected by extremes"})
    else:
        cards.append({"label": "Columns", "value": fmt_number(len(df.columns)), "help": "Fields in this dataset"})
        if date_columns(df):
            col = date_columns(df)[0]
            valid = df[col].dropna()
            cards.append({"label": "Date span", "value": f"{valid.min():%Y-%m-%d} → {valid.max():%Y-%m-%d}" if len(valid) else "—", "help": col})
    while len(cards) < 4:
        cards.append({"label": "Columns", "value": fmt_number(len(df.columns)), "help": "Fields in this dataset"})
    return cards[:4]


def profile_data(df: pd.DataFrame) -> dict[str, Any]:
    rows, cols = df.shape
    missing_by_col = df.isna().sum().sort_values(ascending=False)
    numeric = numeric_columns(df)
    duplicate_rows = int(df.duplicated().sum()) if len(df) else 0
    missing_cells = int(df.isna().sum().sum())
    total_cells = max(1, rows * cols)
    outlier_counts: dict[str, int] = {}
    for col in measure_columns(df):
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) >= 4:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            outlier_counts[col] = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
    return {
        "rows": rows,
        "columns": cols,
        "missing_cells": missing_cells,
        "missing_pct": missing_cells / total_cells * 100,
        "missing_by_col": missing_by_col,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric,
        "categorical_columns": categorical_columns(df),
        "date_columns": date_columns(df),
        "outlier_counts": outlier_counts,
    }


def apply_cleaning(df: pd.DataFrame, trim_text: bool, duplicates: str, missing: str, outliers: str) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy(deep=True)
    notes = []
    if trim_text:
        text_cols = list(out.select_dtypes(include=["object", "string"]).columns)
        for col in text_cols:
            out[col] = out[col].map(lambda x: x.strip() if isinstance(x, str) else x)
        if text_cols:
            notes.append(f"Trimmed surrounding spaces in {len(text_cols)} text column(s).")
    if duplicates == "Remove exact duplicates":
        before = len(out)
        out = out.drop_duplicates().copy()
        notes.append(f"Removed {before - len(out):,} exact duplicate row(s).")
    if missing == "Drop rows with missing values":
        before = len(out)
        out = out.dropna().copy()
        notes.append(f"Dropped {before - len(out):,} row(s) containing missing values.")
    elif missing == "Fill numeric median + text mode":
        filled = 0
        for col in out.columns:
            count = int(out[col].isna().sum())
            if count == 0:
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                med = out[col].median()
                if pd.notna(med):
                    out[col] = out[col].fillna(med)
                    filled += count
            elif pd.api.types.is_datetime64_any_dtype(out[col]):
                continue
            else:
                mode = out[col].mode(dropna=True)
                if len(mode):
                    out[col] = out[col].fillna(mode.iloc[0])
                    filled += count
        notes.append(f"Filled {filled:,} missing cell(s) using numeric medians/text modes where possible.")
    if outliers in ("Clip numeric outliers (IQR)", "Remove rows with IQR outliers"):
        relevant = measure_columns(out)
        if outliers == "Clip numeric outliers (IQR)":
            changed = 0
            for col in relevant:
                s = pd.to_numeric(out[col], errors="coerce")
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    changed += int(((s < lo) | (s > hi)).sum())
                    out[col] = s.clip(lo, hi)
            notes.append(f"Clipped {changed:,} numeric value(s) to IQR boundaries in the analysis copy.")
        else:
            mask = pd.Series(False, index=out.index)
            for col in relevant:
                s = pd.to_numeric(out[col], errors="coerce")
                q1, q3 = s.quantile([0.25, 0.75])
                iqr = q3 - q1
                if pd.notna(iqr) and iqr > 0:
                    mask |= (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)
            before = len(out)
            out = out.loc[~mask].copy()
            notes.append(f"Removed {before - len(out):,} row(s) flagged as IQR outliers in at least one measure.")
    if not notes:
        notes.append("No optional cleaning applied. The imported data remains unchanged in the workspace.")
    return out, notes

# ----------------------------- Filters & bookmarks -----------------------
def default_filter_columns(df: pd.DataFrame) -> list[str]:
    dates = date_columns(df)
    cats = [c for c in categorical_columns(df) if 2 <= df[c].nunique(dropna=True) <= 30]
    nums = [c for c in measure_columns(df) if df[c].nunique(dropna=True) > 1]
    chosen = []
    for c in dates[:1] + cats[:3] + nums[:1]:
        if c not in chosen:
            chosen.append(c)
    return chosen[:5]


def filter_widget_key(dataset_key: str, col: str) -> str:
    return f"filter__{stable_id(dataset_key)}__{stable_id(col)}"


def filter_columns_key(dataset_key: str) -> str:
    return f"filtercols__{stable_id(dataset_key)}"


def bookmark_bucket(dataset_key: str) -> str:
    return stable_id(dataset_key)


def clamp_restored_filter_state(df: pd.DataFrame, dataset_key: str, selected_cols: list[str]) -> None:
    for col in selected_cols:
        key = filter_widget_key(dataset_key, col)
        if key not in st.session_state:
            continue
        value = st.session_state[key]
        if col in categorical_columns(df):
            options = {safe_label(v) for v in df[col].dropna().unique()}
            if df[col].isna().any():
                options.add("(Missing)")
            if isinstance(value, (list, tuple)):
                st.session_state[key] = [v for v in value if v in options]


def render_filter_controls(df: pd.DataFrame, dataset_key: str) -> dict[str, dict[str, Any]]:
    filters: dict[str, dict[str, Any]] = {}
    key_cols = filter_columns_key(dataset_key)
    choices = list(df.columns)
    default_cols = default_filter_columns(df)
    if key_cols not in st.session_state:
        st.session_state[key_cols] = default_cols
    selected_cols = st.multiselect("Slicers", choices, key=key_cols, help="Choose which fields should appear as dashboard filters.")
    clamp_restored_filter_state(df, dataset_key, selected_cols)
    if not selected_cols:
        st.caption("Choose one or more fields above to add slicers.")
        return filters
    for col in selected_cols:
        widget_key = filter_widget_key(dataset_key, col)
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            valid = series.dropna()
            if valid.empty:
                continue
            lo = valid.min().date()
            hi = valid.max().date()
            default = (lo, hi)
            with st.container():
                chosen = st.date_input(col, value=default, key=widget_key, help="Filter records by date range.")
            if isinstance(chosen, (tuple, list)) and len(chosen) == 2:
                filters[col] = {"type": "date", "value": (chosen[0], chosen[1])}
        elif pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 1 and not pd.api.types.is_bool_dtype(series):
            valid = pd.to_numeric(series, errors="coerce").dropna()
            if valid.empty:
                continue
            lo, hi = float(valid.min()), float(valid.max())
            if math.isclose(lo, hi):
                continue
            use_int = pd.api.types.is_integer_dtype(series) or (float(lo).is_integer() and float(hi).is_integer())
            if use_int:
                slider_min, slider_max = int(lo), int(hi)
                default = (slider_min, slider_max)
            else:
                slider_min, slider_max = lo, hi
                default = (lo, hi)
            if widget_key in st.session_state:
                try:
                    val = st.session_state[widget_key]
                    val_lo, val_hi = val
                    st.session_state[widget_key] = (max(slider_min, min(slider_max, val_lo)), max(slider_min, min(slider_max, val_hi)))
                except Exception:
                    pass
            with st.container():
                chosen = st.slider(col, min_value=slider_min, max_value=slider_max, value=default, key=widget_key, help="Drag to limit the numeric range.")
            filters[col] = {"type": "numeric", "value": chosen}
        else:
            vals = series.map(safe_label)
            counts = vals.value_counts(dropna=False)
            options = list(counts.index[:MAX_FILTER_OPTIONS])
            if widget_key in st.session_state and isinstance(st.session_state[widget_key], list):
                st.session_state[widget_key] = [v for v in st.session_state[widget_key] if v in options]
            with st.container():
                chosen = st.multiselect(col, options, key=widget_key, help="Leave empty to include all values." + (f" Showing the top {MAX_FILTER_OPTIONS} values." if len(counts) > MAX_FILTER_OPTIONS else ""))
            filters[col] = {"type": "categorical", "value": chosen}
    return filters


def apply_filters(df: pd.DataFrame, filters: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for col, spec in filters.items():
        if col not in out.columns:
            continue
        kind, value = spec.get("type"), spec.get("value")
        if kind == "categorical" and value:
            labels = out[col].map(safe_label)
            out = out.loc[labels.isin(value)]
        elif kind == "numeric" and value is not None:
            try:
                s = pd.to_numeric(out[col], errors="coerce")
                out = out.loc[s.between(float(value[0]), float(value[1]), inclusive="both") | s.isna()]
            except Exception:
                continue
        elif kind == "date" and value and len(value) == 2:
            parsed = pd.to_datetime(out[col], errors="coerce")
            lo = pd.Timestamp(value[0])
            hi = pd.Timestamp(value[1]) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            out = out.loc[parsed.between(lo, hi, inclusive="both") | parsed.isna()]
    return out.copy()


def snapshot_filters(dataset_key: str, filters: dict[str, dict[str, Any]]) -> dict[str, Any]:
    packed = {}
    for col, spec in filters.items():
        value = spec.get("value")
        if spec.get("type") == "date":
            value = [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in value]
        elif isinstance(value, tuple):
            value = list(value)
        packed[col] = {"type": spec.get("type"), "value": value}
    return {"filters": packed, "filter_columns": list(filters.keys()), "cross_filter": st.session_state.get("cross_filter")}


def apply_bookmark(dataset_key: str, bookmark: dict[str, Any], df: pd.DataFrame) -> None:
    key_cols = filter_columns_key(dataset_key)
    selected = [c for c in bookmark.get("filter_columns", []) if c in df.columns]
    st.session_state[key_cols] = selected
    for col in selected:
        widget_key = filter_widget_key(dataset_key, col)
        spec = bookmark.get("filters", {}).get(col, {})
        value = spec.get("value")
        if spec.get("type") == "date" and value:
            try:
                value = tuple(date.fromisoformat(v) if isinstance(v, str) else v for v in value)
            except Exception:
                pass
        elif spec.get("type") == "numeric" and isinstance(value, list):
            value = tuple(value)
        st.session_state[widget_key] = value
    st.session_state.cross_filter = bookmark.get("cross_filter")

# ----------------------------- Relationship helpers ----------------------
def detect_relationships(tables: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    keys = [k for k in tables if not st.session_state.datasets_meta.get(k, {}).get("derived", False)]
    rels = []
    for i, left_key in enumerate(keys):
        left = tables[left_key]
        for right_key in keys[i + 1:]:
            right = tables[right_key]
            lcols = {normalize_col_token(c): c for c in left.columns}
            rcols = {normalize_col_token(c): c for c in right.columns}
            for token in sorted(set(lcols) & set(rcols)):
                lc, rc = lcols[token], rcols[token]
                if token in {"", "name", "date", "status", "type", "amount", "value"}:
                    continue
                lv = left[lc].dropna().astype(str)
                rv = right[rc].dropna().astype(str)
                if lv.empty or rv.empty:
                    continue
                # Bound memory on high-cardinality files while retaining a useful overlap estimate.
                lu = set(lv.unique()[:50000])
                ru = set(rv.unique()[:50000])
                overlap = len(lu & ru) / max(1, min(len(lu), len(ru)))
                if overlap < 0.25:
                    continue
                left_unique = left[lc].nunique(dropna=True) / max(1, left[lc].notna().sum())
                right_unique = right[rc].nunique(dropna=True) / max(1, right[rc].notna().sum())
                if left_unique >= 0.95 and right_unique < 0.95:
                    cardinality = "one-to-many (likely)"
                elif right_unique >= 0.95 and left_unique < 0.95:
                    cardinality = "many-to-one (likely)"
                elif left_unique >= 0.95 and right_unique >= 0.95:
                    cardinality = "one-to-one (likely)"
                else:
                    cardinality = "many-to-many / uncertain"
                rels.append({"left": left_key, "right": right_key, "left_col": lc, "right_col": rc, "overlap": overlap, "cardinality": cardinality})
    return sorted(rels, key=lambda x: x["overlap"], reverse=True)


def create_join(left: pd.DataFrame, right: pd.DataFrame, lcol: str, rcol: str, how: str) -> pd.DataFrame:
    if left[lcol].isna().all() or right[rcol].isna().all():
        raise ValueError("The selected join key is empty in one of the tables.")
    if len(left) * max(1, len(right)) > 50_000_000:
        raise ValueError("This join may be too large. Choose a more selective key or a smaller table.")
    return pd.merge(left, right, left_on=lcol, right_on=rcol, how=how, suffixes=("_left", "_right"), sort=False)

# ----------------------------- Chart helpers -----------------------------
def apply_cross_filter(df: pd.DataFrame, cross_filter: dict[str, Any] | None) -> pd.DataFrame:
    if not cross_filter:
        return df
    col = cross_filter.get("column")
    value = cross_filter.get("value")
    kind = cross_filter.get("kind", "category")
    if col not in df.columns:
        return df
    try:
        if kind == "date_period":
            freq = cross_filter.get("frequency", "M")
            parsed = pd.to_datetime(df[col], errors="coerce")
            target = pd.Timestamp(value)
            return df.loc[parsed.notna() & (parsed.dt.to_period(freq) == target.to_period(freq))].copy()
        series = df[col]
        mask = series.map(safe_label).eq(str(value))
        if not mask.any() and pd.api.types.is_numeric_dtype(series):
            try:
                mask = pd.to_numeric(series, errors="coerce").eq(float(value))
            except Exception:
                pass
        return df.loc[mask].copy()
    except Exception:
        return df


def chart_callback(chart_key: str):
    def _callback() -> None:
        meta = st.session_state.get("chart_meta", {}).get(chart_key)
        if not meta:
            return
        state = st.session_state.get(chart_key)
        selection = getattr(state, "selection", None)
        if selection is None and isinstance(state, dict):
            selection = state.get("selection", state)
        points = getattr(selection, "points", None) if selection is not None else None
        if points is None and isinstance(selection, dict):
            points = selection.get("points", [])
        if not points:
            current_cross = st.session_state.get("cross_filter") or {}
            if current_cross.get("chart_key") == chart_key:
                st.session_state.cross_filter = None
            return
        point = points[0]
        value = point.get("x") if isinstance(point, dict) else None
        if value is None:
            return
        st.session_state.cross_filter = {
            "chart_key": chart_key,
            "column": meta["column"],
            "value": str(value),
            "kind": meta.get("kind", "category"),
            "frequency": meta.get("frequency", "M"),
            "label": meta.get("label", meta["column"]),
        }
    return _callback


def register_chart_meta(key: str, column: str, label: str | None = None, kind: str = "category", frequency: str = "M") -> None:
    if "chart_meta" not in st.session_state:
        st.session_state.chart_meta = {}
    st.session_state.chart_meta[key] = {"column": column, "label": label or column, "kind": kind, "frequency": frequency}


def show_selectable_chart(fig, key: str, column: str, label: str | None = None, kind: str = "category", frequency: str = "M") -> None:
    register_chart_meta(key, column, label, kind, frequency)
    st.plotly_chart(
        fig,
        use_container_width=True,
        key=key,
        on_select=chart_callback(key),
        selection_mode=("points", "box", "lasso"),
        config={"displaylogo": False, "responsive": True, "scrollZoom": True},
    )


def style_figure(fig, title: str | None = None):
    if title:
        fig.update_layout(title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 15, "color": "#23304a"}})
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=12, r=12, t=52 if title else 20, b=15),
        font=dict(family="DM Sans, sans-serif", color="#526078", size=11),
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )
    fig.update_xaxes(showgrid=False, linecolor="#e8ecf4", tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="#edf0f6", zerolinecolor="#e2e8f0", tickfont=dict(size=10))
    return fig


def aggregate_series(df: pd.DataFrame, x: str, y: str | None, agg: str) -> pd.DataFrame:
    if agg == "Count" or not y:
        result = df.groupby(x, dropna=False, observed=False).size().reset_index(name="Value")
        return result
    if agg == "Distinct count":
        # For a grouped distinct count, use row-level distinctness as the count of unique values in y.
        result = df.groupby(x, dropna=False, observed=False)[y].nunique(dropna=True).reset_index(name="Value")
        return result
    op = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max"}.get(agg, "sum")
    result = df.groupby(x, dropna=False, observed=False)[y].agg(op).reset_index(name="Value")
    return result


def make_overview_charts(df: pd.DataFrame) -> tuple[list[tuple[str, Any, dict[str, Any] | None]], list[str]]:
    charts: list[tuple[str, Any, dict[str, Any] | None]] = []
    notes: list[str] = []
    if df.empty:
        return charts, ["No records match the current filters."]
    date_col = date_columns(df)[0] if date_columns(df) else None
    measure = likely_measure(df)
    cats = [c for c in categorical_columns(df) if 2 <= df[c].nunique(dropna=True) <= max(100, len(df))]
    cat_col = cats[0] if cats else None
    if date_col and measure:
        tmp = df[[date_col, measure]].dropna().copy()
        if not tmp.empty:
            tmp["_period"] = tmp[date_col].dt.to_period("M").dt.start_time
            timeline = tmp.groupby("_period", as_index=False)[measure].sum(min_count=1).sort_values("_period")
            fig = px.line(timeline, x="_period", y=measure, markers=True)
            fig.update_traces(line_width=3, marker_size=6)
            charts.append(("Activity over time", style_figure(fig, f"{measure} over time"), {"column": date_col, "kind": "date_period", "frequency": "M", "label": date_col}))
            if len(timeline) >= 2:
                first, last = timeline[measure].iloc[0], timeline[measure].iloc[-1]
                if pd.notna(first) and pd.notna(last):
                    notes.append(f"{measure} changed from {fmt_number(first)} in the first observed month to {fmt_number(last)} in the latest observed month. This compares endpoints, not causation.")
    if cat_col:
        if measure:
            grouped = df.groupby(cat_col, dropna=False)[measure].sum(min_count=1).sort_values(ascending=False).head(MAX_CHART_CATEGORIES).reset_index()
            valcol = measure
        else:
            grouped = df[cat_col].map(safe_label).value_counts().head(MAX_CHART_CATEGORIES).rename_axis(cat_col).reset_index(name="Records")
            valcol = "Records"
        if not grouped.empty:
            grouped[cat_col] = grouped[cat_col].map(safe_label)
            fig = px.bar(grouped.sort_values(valcol, ascending=True), x=valcol, y=cat_col, orientation="h", text_auto=".3s")
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig = style_figure(fig, f"{valcol} by {cat_col}")
            charts.append((f"By {cat_col}", fig, {"column": cat_col, "kind": "category", "label": cat_col}))
            top = grouped.sort_values(valcol, ascending=False).iloc[0]
            notes.append(f"The highest {cat_col} by {valcol} is {safe_label(top[cat_col])} ({fmt_number(top[valcol])}).")
    if measure:
        vals = pd.to_numeric(df[measure], errors="coerce").dropna()
        if len(vals) > 1:
            fig = px.histogram(df, x=measure, nbins=min(35, max(8, int(np.sqrt(len(vals))))), marginal="box")
            fig.update_traces(marker_line_width=0.5, marker_line_color="white")
            charts.append((f"Distribution · {measure}", style_figure(fig, f"Distribution of {measure}"), None))
            notes.append(f"The median {measure} is {fmt_number(vals.median())}; the range is {fmt_number(vals.min())} to {fmt_number(vals.max())} across non-missing values.")
    if not charts:
        vc = df.nunique(dropna=True).sort_values(ascending=False).head(12).rename("Unique values").reset_index(names="Column")
        fig = px.bar(vc, x="Column", y="Unique values")
        charts.append(("Column diversity", style_figure(fig, "Distinct values by column"), None))
    return charts, notes


def build_custom_chart(df: pd.DataFrame, chart_type: str, x: str, y: str | None, color: str | None, agg: str):
    if df.empty:
        return style_figure(go.Figure(), "No data available")
    color = color if color and color != "(None)" else None
    if chart_type in ("Bar", "Stacked bar", "Line", "Area", "Pie"):
        if x not in df.columns:
            raise ValueError("Choose a valid X-axis field.")
        if chart_type == "Pie":
            if agg == "Count" or not y:
                grouped = df.groupby(x, dropna=False).size().reset_index(name="Value")
            else:
                grouped = df.groupby(x, dropna=False)[y].sum(min_count=1).reset_index(name="Value")
            grouped[x] = grouped[x].map(safe_label)
            grouped = grouped.sort_values("Value", ascending=False).head(MAX_CHART_CATEGORIES)
            fig = px.pie(grouped, names=x, values="Value", hole=0.42)
        elif color and color != x:
            if agg == "Count" or not y:
                grouped = df.groupby([x, color], dropna=False, observed=False).size().reset_index(name="Value")
            elif agg == "Distinct count":
                grouped = df.groupby([x, color], dropna=False, observed=False)[y].nunique().reset_index(name="Value")
            else:
                op = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max"}.get(agg, "sum")
                grouped = df.groupby([x, color], dropna=False, observed=False)[y].agg(op).reset_index(name="Value")
            topcats = grouped.groupby(x)["Value"].sum().nlargest(MAX_CHART_CATEGORIES).index
            grouped = grouped[grouped[x].isin(topcats)].copy()
            grouped[x] = grouped[x].map(safe_label)
            if chart_type == "Line":
                fig = px.line(grouped, x=x, y="Value", color=color, markers=True)
            elif chart_type == "Area":
                fig = px.area(grouped, x=x, y="Value", color=color)
            else:
                fig = px.bar(grouped, x=x, y="Value", color=color, barmode="stack" if chart_type == "Stacked bar" else "group")
        else:
            grouped = aggregate_series(df, x, y, agg)
            grouped[x] = grouped[x].map(safe_label)
            grouped = grouped.sort_values("Value", ascending=False).head(MAX_CHART_CATEGORIES) if chart_type in ("Bar", "Stacked bar") else grouped.sort_values(x)
            if chart_type == "Line":
                fig = px.line(grouped, x=x, y="Value", markers=True)
            elif chart_type == "Area":
                fig = px.area(grouped, x=x, y="Value")
            elif chart_type == "Bar":
                fig = px.bar(grouped.sort_values("Value", ascending=True), x="Value", y=x, orientation="h", text_auto=".3s")
            else:
                fig = px.bar(grouped, x=x, y="Value")
    elif chart_type == "Scatter":
        if not y:
            raise ValueError("Scatter charts need both X and Y numeric fields.")
        sample = df[[x, y] + ([color] if color else [])].dropna().head(10000)
        fig = px.scatter(sample, x=x, y=y, color=color, opacity=0.7)
    elif chart_type == "Histogram":
        fig = px.histogram(df, x=x, color=color, nbins=30, marginal="box")
    elif chart_type == "Box plot":
        fig = px.box(df, x=x if y else None, y=y or x, color=color, points="outliers")
    elif chart_type == "Correlation heatmap":
        nums = measure_columns(df)[:20]
        if len(nums) < 2:
            raise ValueError("A correlation heatmap needs at least two numeric measure columns.")
        corr = df[nums].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    else:
        raise ValueError("Unsupported chart type.")
    return style_figure(fig)

# ----------------------------- Exports -----------------------------------
def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Filtered Data"[:31], index=False)
        summary = pd.DataFrame({"Metric": ["Rows", "Columns", "Missing cells", "Exact duplicate rows"], "Value": [len(df), len(df.columns), int(df.isna().sum().sum()), int(df.duplicated().sum())]})
        summary.to_excel(writer, sheet_name="Summary", index=False)
    return buf.getvalue()


def make_pdf_report(df: pd.DataFrame, title: str, source_rows: int) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, PageBreak
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=16 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="IFTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=23, leading=27, textColor=colors.HexColor("#18233a"), alignment=TA_LEFT, spaceAfter=5))
    styles.add(ParagraphStyle(name="IFSub", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#64748b"), spaceAfter=14))
    styles.add(ParagraphStyle(name="IFHead", parent=styles["Heading2"], fontSize=13, leading=16, textColor=colors.HexColor("#3543a8"), spaceBefore=10, spaceAfter=6))
    story = [Paragraph("InsightForge Analytics Report", styles["IFTitle"]), Paragraph(f"Dataset: {title}<br/>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}<br/>Filtered records: {len(df):,} of {source_rows:,} source rows", styles["IFSub"])]
    profile = profile_data(df)
    kpis = make_kpis(df)
    table_data = [["Metric", "Value", "Description"]]
    for card in kpis:
        table_data.append([card["label"], card["value"], card["help"]])
    table = Table(table_data, colWidths=[48 * mm, 35 * mm, 88 * mm], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#29355b")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#dfe4ee")),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#f8faff")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.extend([Paragraph("Executive overview", styles["IFHead"]), table, Spacer(1, 8)])
    story.append(Paragraph("Data quality", styles["IFHead"]))
    dq = [["Check", "Result"], ["Rows / columns", f"{len(df):,} / {len(df.columns):,}"], ["Missing cells", f"{profile['missing_cells']:,} ({profile['missing_pct']:.2f}% of cells)"], ["Exact duplicate rows", f"{profile['duplicate_rows']:,}"], ["Numeric / categorical / date fields", f"{len(profile['numeric_columns'])} / {len(profile['categorical_columns'])} / {len(profile['date_columns'])}"]]
    dq_table = Table(dq, colWidths=[63 * mm, 108 * mm], repeatRows=1)
    dq_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef0ff")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("GRID", (0, 0), (-1, -1), .4, colors.HexColor("#dfe4ee")), ("FONTSIZE", (0, 0), (-1, -1), 8), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6)]))
    story.extend([dq_table, Spacer(1, 9)])

    # Create report charts with matplotlib, independent of browser/Plotly image plugins.
    if not df.empty:
        measure = likely_measure(df)
        cat = next((c for c in categorical_columns(df) if 2 <= df[c].nunique(dropna=True) <= 100), None)
        dcol = date_columns(df)[0] if date_columns(df) else None
        if dcol and measure:
            temp = df[[dcol, measure]].dropna().copy()
            if not temp.empty:
                temp["period"] = temp[dcol].dt.to_period("M").dt.start_time
                group = temp.groupby("period")[measure].sum(min_count=1).sort_index()
                fig, ax = plt.subplots(figsize=(8.2, 3.0))
                ax.plot(group.index, group.values, marker="o", linewidth=2)
                ax.set_title(f"{measure} over time", loc="left", fontsize=11, pad=12)
                ax.grid(axis="y", alpha=.2)
                ax.spines[["top", "right"]].set_visible(False)
                fig.autofmt_xdate()
                img = io.BytesIO(); fig.tight_layout(); fig.savefig(img, format="png", dpi=140, bbox_inches="tight"); plt.close(fig); img.seek(0)
                story.extend([Paragraph("Trend overview", styles["IFHead"]), RLImage(img, width=170 * mm, height=63 * mm)])
        if cat:
            if measure:
                group = df.groupby(cat, dropna=False)[measure].sum(min_count=1).sort_values(ascending=True).tail(10)
            else:
                group = df[cat].map(safe_label).value_counts().head(10).sort_values(ascending=True)
            if len(group):
                fig, ax = plt.subplots(figsize=(8.2, 3.2))
                ax.barh([safe_label(v) for v in group.index], group.values)
                ax.set_title(f"Top {cat} categories", loc="left", fontsize=11, pad=12)
                ax.grid(axis="x", alpha=.2); ax.spines[["top", "right", "left"]].set_visible(False)
                fig.tight_layout()
                img = io.BytesIO(); fig.savefig(img, format="png", dpi=140, bbox_inches="tight"); plt.close(fig); img.seek(0)
                story.extend([Paragraph("Category breakdown", styles["IFHead"]), RLImage(img, width=170 * mm, height=67 * mm)])
        if measure:
            vals = pd.to_numeric(df[measure], errors="coerce").dropna()
            if len(vals) > 1:
                fig, ax = plt.subplots(figsize=(8.2, 2.8))
                ax.hist(vals, bins=min(30, max(8, int(np.sqrt(len(vals))))), edgecolor="white")
                ax.set_title(f"Distribution of {measure}", loc="left", fontsize=11, pad=12)
                ax.grid(axis="y", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
                fig.tight_layout()
                img = io.BytesIO(); fig.savefig(img, format="png", dpi=140, bbox_inches="tight"); plt.close(fig); img.seek(0)
                story.extend([Paragraph("Distribution", styles["IFHead"]), RLImage(img, width=170 * mm, height=58 * mm)])
    story.append(Spacer(1, 8))
    story.append(Paragraph("Method note: Metrics and charts are calculated from the filtered dataset view. Automated pattern descriptions are descriptive and do not establish causation. Review source definitions and data quality before making decisions.", styles["IFSub"]))
    doc.build(story)
    return buf.getvalue()

# ----------------------------- UI ----------------------------------------
def render_sidebar() -> tuple[str | None, pd.DataFrame | None, pd.DataFrame | None, list[str], dict[str, dict[str, Any]]]:
    with st.sidebar:
        st.markdown("<div class='if-brand'><div class='if-logo'>✦</div><div><p style='font:800 1.15rem Manrope;margin:0;color:#f3f6fc'>InsightForge</p><p style='font-size:.72rem;color:#97a7c1;margin:2px 0 0'>DATA ANALYTICS STUDIO</p></div></div>", unsafe_allow_html=True)
        if st.button("Clear workspace", use_container_width=True, help="Remove uploaded tables and saved views from this browser session."):
            reset_workspace()
            st.session_state["data_upload"] = None
            st.rerun()
        uploaded = st.file_uploader("Upload your data", type=["csv", "xlsx", "xls"], accept_multiple_files=True, key="data_upload", help="Upload one or more CSV or Excel files. Excel sheets are loaded separately.")
        if uploaded:
            load_uploaded(uploaded)
        elif not st.session_state.datasets_raw:
            if st.button("Try a synthetic demo", use_container_width=True, help="Load a sample sales dataset. It will not be mixed with your files."):
                load_demo_dataset()
                st.rerun()
        for error in st.session_state.upload_errors:
            st.error(error)
        if st.session_state.datasets_raw:
            st.markdown("<p style='font-size:.72rem;color:#97a7c1;letter-spacing:.1em;font-weight:700;margin:1rem 0 .35rem'>WORKSPACE</p>", unsafe_allow_html=True)
            keys = list(st.session_state.datasets_raw.keys())
            pending = st.session_state.pop("pending_active_dataset", None)
            if pending in keys:
                # Apply a newly-created joined table before constructing the selectbox widget.
                st.session_state["active_dataset_select"] = pending
                st.session_state.active_dataset = pending
            current = st.session_state.active_dataset if st.session_state.active_dataset in keys else keys[0]
            chosen = st.selectbox("Active dataset", keys, index=keys.index(current), format_func=display_name, key="active_dataset_select")
            if chosen != st.session_state.active_dataset:
                st.session_state.last_active_dataset = st.session_state.active_dataset
                st.session_state.active_dataset = chosen
                st.session_state.cross_filter = None
            active = chosen
            original = st.session_state.datasets_raw[active]
            inferred = st.session_state.datasets_inferred[active]
            if st.session_state.datasets_meta.get(active, {}).get("sample"):
                st.caption(f"{len(original):,} rows · {len(original.columns):,} columns · synthetic demo data (separate from user uploads)")
            else:
                st.caption(f"{len(original):,} rows · {len(original.columns):,} columns · imported copy preserved")
            with st.expander("Optional cleaning (analysis copy only)", expanded=False):
                ck = stable_id(active)
                trim = st.checkbox("Trim text whitespace", value=st.session_state.get(f"clean__trim__{ck}", False), key=f"clean__trim__{ck}")
                duplicates = st.selectbox("Exact duplicates", ["Preserve", "Remove exact duplicates"], key=f"clean__duplicates__{ck}")
                missing = st.selectbox("Missing values", ["Preserve as-is", "Drop rows with missing values", "Fill numeric median + text mode"], key=f"clean__missing__{ck}")
                outliers = st.selectbox("Numeric outliers", ["Preserve as-is", "Clip numeric outliers (IQR)", "Remove rows with IQR outliers"], key=f"clean__outliers__{ck}")
            active_df, cleaning_notes = apply_cleaning(
                inferred,
                trim_text=trim,
                duplicates="Remove exact duplicates" if duplicates == "Remove exact duplicates" else "Preserve",
                missing=missing if missing != "Preserve as-is" else "Preserve",
                outliers=outliers if outliers != "Preserve as-is" else "Preserve",
            )

            # Restore/save bookmark controls run before slicer widgets are instantiated.
            with st.expander("Bookmarks", expanded=False):
                bucket = bookmark_bucket(active)
                st.session_state.bookmarks.setdefault(bucket, {})
                saved = st.session_state.bookmarks[bucket]
                bookmark_name_key = f"bookmarkname__{stable_id(active)}"
                bookmark_pick_key = f"bookmarkpick__{stable_id(active)}"
                name = st.text_input("View name", key=bookmark_name_key, placeholder="e.g. Q4 high-value customers")
                if saved:
                    picked = st.selectbox("Saved views", list(saved.keys()), key=bookmark_pick_key)
                    col_a, col_b = st.columns(2)
                    if col_a.button("Restore", key=f"restore__{stable_id(active)}", use_container_width=True):
                        apply_bookmark(active, saved[picked], active_df)
                        st.rerun()
                    if col_b.button("Delete", key=f"deletebookmark__{stable_id(active)}", use_container_width=True):
                        saved.pop(picked, None)
                        st.rerun()
                else:
                    st.caption("Save a filter view to reuse it in this session.")
                if st.button("Save current view", key=f"savebookmark__{stable_id(active)}", use_container_width=True):
                    view_name = (name or "").strip()
                    if not view_name:
                        st.warning("Enter a name for this bookmark first.")
                    else:
                        existing_filter_cols = st.session_state.get(filter_columns_key(active), default_filter_columns(active_df))
                        filter_state = {}
                        for col in existing_filter_cols:
                            if col not in active_df.columns:
                                continue
                            widget_key = filter_widget_key(active, col)
                            if widget_key not in st.session_state:
                                continue
                            value = st.session_state[widget_key]
                            if pd.api.types.is_datetime64_any_dtype(active_df[col]):
                                if isinstance(value, (tuple, list)):
                                    value = [v.isoformat() if hasattr(v, "isoformat") else str(v) for v in value]
                                typ = "date"
                            elif pd.api.types.is_numeric_dtype(active_df[col]) and active_df[col].nunique(dropna=True) > 1:
                                if isinstance(value, tuple): value = list(value)
                                typ = "numeric"
                            else:
                                typ = "categorical"
                            filter_state[col] = {"type": typ, "value": value}
                        saved[view_name] = {"filters": filter_state, "filter_columns": list(filter_state.keys()), "cross_filter": st.session_state.get("cross_filter")}
                        st.success(f"Saved '{view_name}'.")

            st.markdown("<p style='font-size:.72rem;color:#97a7c1;letter-spacing:.1em;font-weight:700;margin:1rem 0 .35rem'>DASHBOARD FILTERS</p>", unsafe_allow_html=True)
            filters = render_filter_controls(active_df, active)
            st.markdown("<div style='height:1px;background:#29364a;margin:1rem 0'></div><p style='font-size:.72rem;color:#8191aa'>Files are processed in this session. The app does not send uploaded data to an external AI service.</p>", unsafe_allow_html=True)
            return active, original, active_df, cleaning_notes, filters
        return None, None, None, [], {}


def render_empty_state() -> None:
    st.markdown("""
    <div class='if-hero'><div class='if-eyebrow'>Your self-service BI workspace</div><h1 class='if-title'>From raw files to clear decisions.</h1><p class='if-subtitle'>Upload CSV or Excel data to profile it, discover useful metrics, explore interactive charts, filter records, save bookmark views, and export a report.</p></div>
    <div class='if-empty'><div style='font-size:2.3rem'>📂</div><h3>Start with your own dataset</h3><p>Use the Upload your data control in the sidebar. Each Excel sheet becomes its own table, and multiple files stay separate so you can inspect possible relationships before joining them.</p><span class='if-pill'>CSV · XLSX · XLS</span></div>
    """, unsafe_allow_html=True)
    with st.expander("What this app can do"):
        st.markdown("""
        **Automated analysis:** KPIs, trends, category rankings, distributions, missing values, duplicates, outlier checks and correlation analysis.

        **Interactive dashboard:** date and numeric slicers, categorical filters, chart selection cross-filtering, drill-through via detailed records, saved views and downloads.

        **Data privacy:** analysis is rule-based and happens inside the app process; no external AI model is called. Uploaded data stays in the active session and is not persisted to a database by this app.
        """)


def render_kpi_cards(df: pd.DataFrame) -> None:
    cards = make_kpis(df)
    cols = st.columns(4, gap="small")
    for col, card in zip(cols, cards):
        with col:
            st.markdown(f"<div class='if-card'><div class='if-card-label'>{html.escape(card['label'])}</div><div class='if-card-value'>{html.escape(card['value'])}</div><div class='if-card-help'>{html.escape(card['help'])}</div></div>", unsafe_allow_html=True)


def render_overview(df: pd.DataFrame, source_df: pd.DataFrame, dataset_name: str) -> None:
    st.markdown("<div class='if-section'>At a glance</div>", unsafe_allow_html=True)
    render_kpi_cards(df)
    st.markdown("")
    cross = st.session_state.get("cross_filter")
    if cross:
        st.info(f"Chart cross-filter active: **{cross.get('label', cross.get('column'))} = {cross.get('value')}**. Other visuals use this selection where applicable.")
        if st.button("Clear chart selection", key="clear_cross_filter_overview"):
            st.session_state.cross_filter = None
            st.session_state.chart_meta = {}
            for key in list(st.session_state.keys()):
                if key.startswith(("overview_chart__", "detail_chart__", "trend_chart__")):
                    st.session_state.pop(key, None)
            st.rerun()
    chart_rows, insight_notes = make_overview_charts(df)
    if chart_rows:
        st.markdown("<div class='if-section'>Explore your data</div>", unsafe_allow_html=True)
        # Two-column chart layout, keeping chart-specific keys stable by dataset and chart type.
        for i in range(0, len(chart_rows), 2):
            cols = st.columns(2, gap="medium")
            for j, (title, fig, cross) in enumerate(chart_rows[i:i + 2]):
                with cols[j]:
                    st.markdown(f"<div class='if-card-label'>{html.escape(title)}</div>", unsafe_allow_html=True)
                    key = f"overview_chart__{stable_id(dataset_name)}__{i+j}"
                    if cross:
                        show_selectable_chart(fig, key, cross["column"], cross.get("label"), cross.get("kind", "category"), cross.get("frequency", "M"))
                    else:
                        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "responsive": True})
    st.markdown("<div class='if-section'>Automated insights</div>", unsafe_allow_html=True)
    if insight_notes:
        for note in insight_notes[:5]:
            st.markdown(f"- {note}")
    else:
        st.caption("Not enough data is available to infer useful patterns from this view.")
    profile = profile_data(df)
    quality_items = []
    if profile["missing_cells"]:
        quality_items.append(f"{profile['missing_cells']:,} missing cells ({profile['missing_pct']:.2f}% of all cells)")
    if profile["duplicate_rows"]:
        quality_items.append(f"{profile['duplicate_rows']:,} exact duplicate rows")
    high_missing = profile["missing_by_col"][profile["missing_by_col"] > 0]
    if not high_missing.empty:
        quality_items.append(f"Most missing values: **{high_missing.index[0]}** ({int(high_missing.iloc[0]):,})")
    if quality_items:
        st.warning("Data-quality notes: " + " · ".join(quality_items) + ". Check the Data Quality tab before interpreting results.")
    else:
        st.success("No missing cells or exact duplicate rows were detected in the current filtered view.")
    st.markdown("<div class='if-section'>Export this view</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Download filtered CSV", data=df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{stable_id(dataset_name)}_filtered.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Download Excel workbook", data=dataframe_to_excel(df), file_name=f"{stable_id(dataset_name)}_analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c3:
        try:
            pdf = make_pdf_report(df, dataset_name, len(source_df))
            st.download_button("Download PDF report", data=pdf, file_name=f"{stable_id(dataset_name)}_report.pdf", mime="application/pdf", use_container_width=True)
        except Exception as exc:
            st.warning(f"PDF export is unavailable in this environment: {exc}")


def render_trends(df: pd.DataFrame, dataset_name: str) -> None:
    st.markdown("<div class='if-section'>Time-series exploration</div>", unsafe_allow_html=True)
    dates, measures = date_columns(df), measure_columns(df)
    if not dates:
        st.info("No date/time field was detected. Review the inferred column types in Data Quality, or upload a dataset containing a date field to use time-series analysis.")
        return
    if not measures:
        st.info("No numeric measure was detected for trend analysis. Select or infer a numeric metric first.")
        return
    sid = stable_id(dataset_name)
    dkey, mkey, akey, fkey = (f"trend__date__{sid}", f"trend__measure__{sid}", f"trend__agg__{sid}", f"trend__freq__{sid}")
    cols = st.columns([1.2, 1.4, 1, 1])
    date_col = cols[0].selectbox("Date field", dates, key=dkey, index=0)
    measure = cols[1].selectbox("Metric", measures, key=mkey, index=measures.index(likely_measure(df)) if likely_measure(df) in measures else 0)
    aggregation = cols[2].selectbox("Aggregation", ["Sum", "Average", "Median", "Minimum", "Maximum"], key=akey, index=0)
    frequency_label = cols[3].selectbox("Granularity", ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly"], key=fkey, index=2)
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "M", "Quarterly": "Q", "Yearly": "Y"}[frequency_label]
    tmp = df[[date_col, measure]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp[measure] = pd.to_numeric(tmp[measure], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, measure])
    if tmp.empty:
        st.warning("No valid date and numeric values remain after the current slicers.")
        return
    tmp["_period"] = tmp[date_col].dt.to_period(freq).dt.start_time
    op = {"Sum": "sum", "Average": "mean", "Median": "median", "Minimum": "min", "Maximum": "max"}[aggregation]
    grouped = tmp.groupby("_period", as_index=False)[measure].agg(op).sort_values("_period")
    fig = px.line(grouped, x="_period", y=measure, markers=True)
    fig.update_traces(line_width=3, marker_size=6)
    fig = style_figure(fig, f"{aggregation} {measure} · {frequency_label.lower()}")
    chart_key = f"trend_chart__{sid}"
    show_selectable_chart(fig, chart_key, date_col, date_col, "date_period", freq)
    if len(grouped) >= 2:
        grouped["Previous period"] = grouped[measure].shift(1)
        grouped["Absolute change"] = grouped[measure] - grouped["Previous period"]
        grouped["Percentage change"] = np.where(grouped["Previous period"].notna() & grouped["Previous period"].ne(0), grouped["Absolute change"] / grouped["Previous period"] * 100, np.nan)
        st.markdown("<div class='if-section'>Period-over-period details</div>", unsafe_allow_html=True)
        st.dataframe(grouped.rename(columns={"_period": "Period"}), use_container_width=True, hide_index=True)
        st.download_button("Download trend table (CSV)", data=grouped.to_csv(index=False).encode("utf-8-sig"), file_name="trend_analysis.csv", mime="text/csv")


def render_detailed(df: pd.DataFrame, dataset_key: str, dataset_name: str) -> None:
    st.markdown("<div class='if-section'>Build a chart</div>", unsafe_allow_html=True)
    if df.empty or len(df.columns) == 0:
        st.info("No rows are available for chart creation. Adjust the slicers.")
        return
    sid = stable_id(dataset_key)
    col1, col2, col3, col4 = st.columns([1, 1.35, 1.35, 1.2])
    chart_types = ["Bar", "Stacked bar", "Line", "Area", "Scatter", "Histogram", "Box plot", "Pie", "Correlation heatmap"]
    chart_type = col1.selectbox("Chart type", chart_types, key=f"chartbuilder__type__{sid}")
    x_default = date_columns(df)[0] if date_columns(df) else (categorical_columns(df)[0] if categorical_columns(df) else df.columns[0])
    x_col = col2.selectbox("X / category", list(df.columns), index=list(df.columns).index(x_default), key=f"chartbuilder__x__{sid}")
    measure_cols = measure_columns(df)
    y_options = ["(Count rows)"] + measure_cols + [c for c in df.columns if c not in measure_cols and c != x_col]
    y_default = likely_measure(df) if likely_measure(df) in y_options else "(Count rows)"
    y_col = col3.selectbox("Y / metric", y_options, index=y_options.index(y_default), key=f"chartbuilder__y__{sid}")
    y_value = None if y_col == "(Count rows)" else y_col
    colors = ["(None)"] + [c for c in df.columns if c != x_col and c != y_value]
    color_col = col4.selectbox("Color / series", colors, key=f"chartbuilder__color__{sid}")
    aggregation = st.selectbox("Metric aggregation", ["Sum", "Average", "Median", "Minimum", "Maximum", "Count", "Distinct count"], index=0 if y_value else 5, key=f"chartbuilder__agg__{sid}")
    if chart_type in ("Scatter", "Histogram", "Box plot") and color_col != "(None)" and chart_type == "Scatter":
        st.caption("Scatter charts use row-level values (up to 10,000 rows) rather than aggregating by category.")
    try:
        if chart_type == "Correlation heatmap":
            fig = build_custom_chart(df, chart_type, x_col, y_value, None, aggregation)
        elif chart_type == "Histogram":
            chosen_x = x_col
            if not pd.api.types.is_numeric_dtype(df[chosen_x]):
                st.warning("For a histogram, choose a numeric column in X / category.")
            fig = build_custom_chart(df, chart_type, chosen_x, y_value, color_col, aggregation)
        elif chart_type == "Box plot":
            fig = build_custom_chart(df, chart_type, x_col, y_value, color_col, aggregation)
        elif chart_type == "Scatter":
            fig = build_custom_chart(df, chart_type, x_col, y_value, color_col, aggregation)
        else:
            fig = build_custom_chart(df, chart_type, x_col, y_value, color_col, aggregation)
        detail_key = f"detail_chart__{sid}"
        st.plotly_chart(fig, use_container_width=True, key=detail_key, config={"displaylogo": False, "responsive": True})
        st.download_button("Download interactive chart (HTML)", data=fig.to_html(full_html=True, include_plotlyjs="cdn"), file_name=f"{stable_id(dataset_name)}_chart.html", mime="text/html")
    except Exception as exc:
        st.error(f"Could not create this chart: {exc}")
    st.markdown("<div class='if-section'>Inspect records</div>", unsafe_allow_html=True)
    search_col, page_col = st.columns([1.2, 1])
    search = search_col.text_input("Search visible records", key=f"chartbuilder__search__{sid}", placeholder="Search across text values")
    page_size = page_col.selectbox("Rows per page", [25, 50, 100, 250], index=1, key=f"chartbuilder__pagesize__{sid}")
    table_df = df
    if search.strip():
        mask = table_df.astype("string").apply(lambda s: s.str.contains(re.escape(search), case=False, na=False)).any(axis=1)
        table_df = table_df.loc[mask]
    total = len(table_df)
    pages = max(1, math.ceil(total / page_size))
    pkey = f"chartbuilder__page__{sid}"
    page = st.number_input("Page", min_value=1, max_value=pages, value=min(int(st.session_state.get(pkey, 1)), pages), step=1, key=pkey)
    start = (int(page) - 1) * page_size
    st.caption(f"Showing {min(total, start + 1):,}–{min(total, start + page_size):,} of {total:,} rows")
    st.dataframe(table_df.iloc[start:start + page_size], use_container_width=True, hide_index=True)

    st.markdown("<div class='if-section'>Table relationships & optional joins</div>", unsafe_allow_html=True)
    source_keys = [k for k in st.session_state.datasets_raw if not st.session_state.datasets_meta.get(k, {}).get("derived", False)]
    if len(source_keys) < 2:
        st.caption("Upload more than one file or include multiple Excel sheets to inspect relationships and create a joined analysis view.")
        return
    relationships = detect_relationships(st.session_state.datasets_inferred)
    if relationships:
        rel_df = pd.DataFrame([{
            "Table A": display_name(r["left"]), "Key A": r["left_col"], "Table B": display_name(r["right"]), "Key B": r["right_col"],
            "Shared-value overlap": f"{r['overlap']:.0%}", "Relationship guess": r["cardinality"],
        } for r in relationships[:15]])
        st.dataframe(rel_df, use_container_width=True, hide_index=True)
        st.caption("Relationship suggestions are heuristic. Confirm that selected keys mean the same thing before joining.")
    with st.expander("Create a joined analysis view", expanded=False):
        j1, j2 = st.columns(2)
        left_key = j1.selectbox("Left table", source_keys, format_func=display_name, key=f"merge__left__{sid}")
        right_options = [k for k in source_keys if k != left_key]
        right_key = j2.selectbox("Right table", right_options, format_func=display_name, key=f"merge__right__{sid}")
        left_cols = list(st.session_state.datasets_inferred[left_key].columns)
        right_cols = list(st.session_state.datasets_inferred[right_key].columns)
        common = [(lc, rc) for lc in left_cols for rc in right_cols if normalize_col_token(lc) == normalize_col_token(rc)]
        defaults = common[0] if common else (left_cols[0], right_cols[0])
        jc1, jc2, jc3 = st.columns([1.2, 1.2, 1])
        lcol = jc1.selectbox("Left join key", left_cols, index=left_cols.index(defaults[0]), key=f"merge__lcol__{sid}")
        rcol = jc2.selectbox("Right join key", right_cols, index=right_cols.index(defaults[1]), key=f"merge__rcol__{sid}")
        how_label = jc3.selectbox("Join type", ["Left", "Inner", "Outer", "Right"], index=0, key=f"merge__how__{sid}")
        st.caption("A join can multiply rows when keys are duplicated. Inspect row counts and key uniqueness after merging.")
        if st.button("Create joined dataset", key=f"merge__run__{sid}", type="primary"):
            try:
                joined = create_join(st.session_state.datasets_inferred[left_key], st.session_state.datasets_inferred[right_key], lcol, rcol, how_label.lower())
                if len(joined) > 1_000_000:
                    st.warning("The joined table has more than one million rows; displaying it may be slow.")
                new_key = f"Joined · {display_name(left_key)} + {display_name(right_key)}"
                suffix = 2
                base = new_key
                while new_key in st.session_state.datasets_raw:
                    new_key = f"{base} ({suffix})"; suffix += 1
                st.session_state.datasets_raw[new_key] = joined.copy()
                st.session_state.datasets_inferred[new_key] = infer_types(joined)
                st.session_state.datasets_meta[new_key] = {"display_name": new_key, "derived": True, "file_name": None, "sheet_name": None, "source_bytes": None, "header_warning": False, "header_mode": "derived", "sample": False}
                st.session_state.pending_active_dataset = new_key
                st.session_state.active_dataset = new_key
                st.session_state.cross_filter = None
                st.success(f"Created '{new_key}' with {len(joined):,} rows and {len(joined.columns):,} columns. It is a separate derived table; source tables were not changed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Join failed: {exc}")


def render_data_quality(df: pd.DataFrame, raw_df: pd.DataFrame, dataset_key: str, dataset_name: str) -> None:
    source_meta = st.session_state.datasets_meta.get(dataset_key, {})
    if source_meta.get("header_warning") and not source_meta.get("derived", False):
        st.warning("The column names may have been read from a data row. If this file has no header row, use the controls below to re-read it.")
    if not source_meta.get("derived", False) and source_meta.get("source_bytes"):
        a, b = st.columns([1, 2])
        if source_meta.get("header_mode") == "header":
            if a.button("Treat first row as data (no header)", key=f"headerless__{stable_id(dataset_key)}"):
                try:
                    reload_header_mode(dataset_key, None)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reload this file: {exc}")
        else:
            if a.button("Restore first row as header", key=f"headerrestore__{stable_id(dataset_key)}"):
                try:
                    reload_header_mode(dataset_key, 0)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not reload this file: {exc}")
        b.caption("Header changes are applied to this table's analysis copy in the current session. Re-upload the file to return to its original interpretation.")
    st.markdown("<div class='if-section'>Quality summary · imported table</div>", unsafe_allow_html=True)
    profile = profile_data(raw_df)
    cols = st.columns(4)
    metrics = [
        ("Rows", f"{profile['rows']:,}"),
        ("Missing cells", f"{profile['missing_cells']:,} · {profile['missing_pct']:.2f}%"),
        ("Exact duplicate rows", f"{profile['duplicate_rows']:,}"),
        ("Potential numeric outliers", f"{sum(profile['outlier_counts'].values()):,}"),
    ]
    for c, (label, value) in zip(cols, metrics):
        c.metric(label, value)
    left, right = st.columns([1.2, 1])
    with left:
        missing = profile["missing_by_col"]
        missing = missing[missing > 0].head(25)
        if not missing.empty:
            mdf = missing.rename("Missing values").reset_index(names="Column").sort_values("Missing values")
            fig = px.bar(mdf, x="Missing values", y="Column", orientation="h", text="Missing values")
            st.plotly_chart(style_figure(fig, "Missing values by column · top 25"), use_container_width=True, config={"displaylogo": False})
        else:
            st.success("No missing cells in the imported table.")
    with right:
        if profile["outlier_counts"]:
            odf = pd.DataFrame({"Column": list(profile["outlier_counts"].keys()), "IQR outliers": list(profile["outlier_counts"].values())})
            odf = odf[odf["IQR outliers"] > 0].sort_values("IQR outliers", ascending=True)
            if not odf.empty:
                fig = px.bar(odf, x="IQR outliers", y="Column", orientation="h")
                st.plotly_chart(style_figure(fig, "Potential IQR outliers"), use_container_width=True, config={"displaylogo": False})
            else:
                st.caption("No numeric IQR outliers were flagged. This does not guarantee the data is error-free.")
        else:
            st.caption("Outlier checks need numeric measure columns with enough non-missing values.")
    st.markdown("<div class='if-section'>Column profiling</div>", unsafe_allow_html=True)
    rows = []
    for col in raw_df.columns:
        s = raw_df[col]
        inferred_col = st.session_state.datasets_inferred[dataset_key][col] if col in st.session_state.datasets_inferred[dataset_key].columns else s
        rows.append({
            "Column": str(col),
            "Imported dtype": str(s.dtype),
            "Detected dtype": "datetime" if pd.api.types.is_datetime64_any_dtype(inferred_col) else ("numeric" if pd.api.types.is_numeric_dtype(inferred_col) else ("boolean" if pd.api.types.is_bool_dtype(inferred_col) else "categorical/text")),
            "Non-null": int(s.notna().sum()),
            "Missing": int(s.isna().sum()),
            "Missing %": round(float(s.isna().mean() * 100), 2) if len(s) else 0,
            "Unique values": int(s.nunique(dropna=True)),
            "Example values": ", ".join(s.dropna().astype(str).head(3).tolist()),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    numeric = [c for c in numeric_columns(st.session_state.datasets_inferred[dataset_key]) if st.session_state.datasets_inferred[dataset_key][c].nunique(dropna=True) > 1]
    if len(numeric) >= 2:
        st.markdown("<div class='if-section'>Numeric correlation</div>", unsafe_allow_html=True)
        chosen_nums = st.multiselect("Choose numeric fields (up to 20)", numeric, default=numeric[:min(10, len(numeric))], key=f"dq_corr__{stable_id(dataset_key)}")
        if len(chosen_nums) >= 2:
            corr = st.session_state.datasets_inferred[dataset_key][chosen_nums[:20]].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(style_figure(fig, "Pearson correlation matrix"), use_container_width=True, config={"displaylogo": False})
            st.caption("Correlation measures linear association and does not establish causation.")
    st.markdown("<div class='if-section'>Current analysis view</div>", unsafe_allow_html=True)
    st.caption("This preview reflects your slicers and optional cleaning settings. Quality summary above is based on the imported table before optional cleaning.")
    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
    if len(df) > 500:
        st.caption("Preview limited to the first 500 rows for responsiveness.")


def main() -> None:
    init_state()
    dataset_key, raw_df, working_df, cleaning_notes, filters = render_sidebar()
    if dataset_key is None or raw_df is None or working_df is None:
        render_empty_state()
        return
    dataset_name = display_name(dataset_key)
    st.markdown(f"<div class='if-hero'><div class='if-eyebrow'>AUTOMATED DATA ANALYTICS</div><h1 class='if-title'>{html.escape(dataset_name)}</h1><p class='if-subtitle'>Explore key metrics, trends, categories, and data quality. All KPIs and charts are computed from the selected dataset and current filters.</p></div>", unsafe_allow_html=True)
    filtered = apply_filters(working_df, filters)
    cross = st.session_state.get("cross_filter")
    # Cross-filter selections are transient and only applied when their target column exists in this dataset.
    if cross and cross.get("column") in filtered.columns:
        filtered = apply_cross_filter(filtered, cross)
    if cleaning_notes:
        with st.expander("Applied analysis-copy operations", expanded=False):
            for note in cleaning_notes:
                st.write(f"• {note}")
    info_cols = st.columns([1, 1, 1, 1.5])
    info_cols[0].metric("Source rows", f"{len(raw_df):,}")
    info_cols[1].metric("Analysis rows", f"{len(working_df):,}")
    info_cols[2].metric("Filtered rows", f"{len(filtered):,}")
    info_cols[3].markdown(f"<div style='padding:.7rem .8rem;border:1px solid #e4e9f2;border-radius:12px;background:#fff'><span class='if-muted'>ACTIVE VIEW</span><br><b>{len(filters)} slicer(s)</b> · <b>{len(working_df.columns)} fields</b> · <b>{len(date_columns(working_df))} date field(s)</b></div>", unsafe_allow_html=True)
    tabs = st.tabs(["Overview", "Trends", "Detailed Analysis", "Data Quality"])
    with tabs[0]:
        render_overview(filtered, raw_df, dataset_name)
    with tabs[1]:
        render_trends(filtered, dataset_name)
    with tabs[2]:
        render_detailed(filtered, dataset_key, dataset_name)
    with tabs[3]:
        render_data_quality(working_df, raw_df, dataset_key, dataset_name)


if __name__ == "__main__":
    main()
