# ASCII-only script for Google Colab. No sys.exit(), no raised exceptions.
# Default command (can be overwritten by argv fallback).
COMMAND = "/report PRS.MC"

ASOF_DATE = "2026-02-16"

import os
import re
import math
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    yf = None

try:
    import requests
except Exception:
    requests = None

# -----------------------------
# Defaults (fallback only)
# -----------------------------
forecast_horizon_years = 5
wacc_default = 0.09
terminal_growth_default = 0.025
default_revenue_growth = 0.05
default_ebitda_margin = 0.20
default_tax_rate = 0.25
default_da_pct_revenue = 0.03
default_capex_pct_revenue = 0.04
default_nwc_pct_revenue = 0.00
balancing_method = "cash_plug"
output_dir = "/content/model_outputs"
USE_CACHE = True

# -----------------------------
# Helpers / formatting
# -----------------------------
def usage_error():
    print("UsageError: supported commands are /report /quick /sens /scenario /run /vs")

def safe_name(s):
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:120] if s else "output"

def ensure_output_dir():
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception:
        pass
    print("Saved outputs to: %s" % output_dir)

def fmt_money(x):
    try:
        if x is None:
            return "NA"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
    except Exception:
        return "NA"
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e12:
        return "%s$%sT" % (sign, f"{v/1e12:,.2f}")
    if v >= 1e9:
        return "%s$%sB" % (sign, f"{v/1e9:,.2f}")
    if v >= 1e6:
        return "%s$%sM" % (sign, f"{v/1e6:,.2f}")
    if v >= 1e3:
        return "%s$%sK" % (sign, f"{v/1e3:,.2f}")
    return "%s$%s" % (sign, f"{v:,.0f}")

def fmt_num(x, nd=2):
    try:
        if x is None:
            return "NA"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:,.{nd}f}"
    except Exception:
        return "NA"

def fmt_pct(x, nd=2):
    try:
        if x is None:
            return "NA"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v*100:.{nd}f}%"
    except Exception:
        return "NA"

def df_to_pretty(df, money_cols=None, pct_cols=None, num_cols=None):
    if df is None or getattr(df, "empty", True):
        return df
    out = df.copy()
    money_cols = money_cols or []
    pct_cols = pct_cols or []
    num_cols = num_cols or []
    for c in out.columns:
        if c in money_cols:
            out[c] = out[c].apply(lambda z: fmt_money(z))
        elif c in pct_cols:
            out[c] = out[c].apply(lambda z: fmt_pct(z))
        elif c in num_cols:
            out[c] = out[c].apply(lambda z: fmt_num(z))
        else:
            # keep as-is
            pass
    return out

def cache_path(ticker):
    return os.path.join(output_dir, "cache_%s_%s.pkl" % (safe_name(ticker), safe_name(ASOF_DATE)))

def load_cache(ticker):
    if not USE_CACHE:
        return None
    p = cache_path(ticker)
    try:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception:
        return None
    return None

def save_cache(ticker, obj):
    if not USE_CACHE:
        return
    p = cache_path(ticker)
    try:
        with open(p, "wb") as f:
            pickle.dump(obj, f)
    except Exception:
        pass

# -----------------------------
# ASOF close fetch (spec)
# -----------------------------
def get_asof_close(symbol):
    if yf is None:
        return (None, None, False)
    try:
        df = yf.download(
            symbol,
            start="2026-01-01",
            end="2026-02-17",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
    except Exception:
        df = None

    if df is None or getattr(df, "empty", True):
        return (None, None, False)

    close_data_candidate = None
    try:
        if "Close" in df.columns:
            close_data_candidate = df["Close"]
        elif isinstance(df.columns, pd.MultiIndex) and "Close" in df.columns.get_level_values(0):
            # For MultiIndex, df['Close'] might return a DataFrame itself if there are sub-levels.
            # Attempt to select the first sub-column under 'Close' if it's a DataFrame.
            temp_selection = df['Close']
            if isinstance(temp_selection, pd.DataFrame) and temp_selection.shape[1] > 0:
                close_data_candidate = temp_selection.iloc[:, 0]
            else:
                close_data_candidate = temp_selection
        else:
            return (None, None, True)
    except Exception:
        return (None, None, True)

    # After initial selection, `close_data_candidate` should ideally be a Series or a single-column DataFrame.
    # Use .squeeze() to convert a single-column DataFrame to a Series, or leave a Series as is.
    close_raw = close_data_candidate.squeeze()

    if not isinstance(close_raw, pd.Series):
        # This means .squeeze() could not reduce `close_data_candidate` to a Series.
        # This typically happens if `close_data_candidate` was a multi-column DataFrame.
        warnings.warn(f"UserWarning: Could not extract single 'Close' series for {symbol}. Final data structure is {type(close_raw)} with shape {getattr(close_raw, 'shape', 'N/A')}.")
        return (None, None, True)

    close = pd.to_numeric(close_raw, errors="coerce").dropna()
    try:
        close = close[close.index <= ASOF_DATE]
    except Exception:
        try:
            close.index = pd.to_datetime(close.index)
            close = close[close.index <= pd.to_datetime(ASOF_DATE)]
        except Exception:
            return (None, None, True)

    if close is None or close.empty:
        return (None, None, True)

    used_date = str(pd.to_datetime(close.index[-1]).date())
    used_px = float(close.iloc[-1])
    return (used_date, used_px, True)

# -----------------------------
# Yahoo fetch helpers
# -----------------------------
def get_ticker_info(t):
    if yf is None:
        return {}
    try:
        tk = yf.Ticker(t)
        return tk.info or {}
    except Exception:
        return {}

def get_statements(t):
    if yf is None:
        return None, None, None
    try:
        tk = yf.Ticker(t)
        is_df = tk.financials
        bs_df = tk.balance_sheet
        cf_df = tk.cashflow
        return is_df, bs_df, cf_df
    except Exception:
        return None, None, None

def pick_3_years(df):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        cols = list(df.columns)
        dt_cols = []
        for c in cols:
            try:
                dt_cols.append(pd.to_datetime(c))
            except Exception:
                dt_cols.append(None)
        good = [(c, d) for c, d in zip(cols, dt_cols) if d is not None]
        if not good:
            return df
        good_sorted = sorted(good, key=lambda x: x[1], reverse=True)
        keep = [x[0] for x in good_sorted[:3]]
        return df.loc[:, keep]
    except Exception:
        return df

def sget(df, key, col):
    if df is None or getattr(df, "empty", True):
        return None
    try:
        if key not in df.index:
            return None
        v = df.loc[key, col]
        if isinstance(v, (pd.Series, pd.DataFrame)):
            try:
                v = float(np.array(v).ravel()[0])
            except Exception:
                return None
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        try:
            return float(v)
        except Exception:
            return None
    except Exception:
        return None

# -----------------------------
# Damodaran ERP (best-effort)
# -----------------------------
def get_damodaran_erp_us():
    if requests is None:
        return None
    urls = [
        "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/histimpl.html",
        "https://pages.stern.nyu.edu/~adamodar/New_Home_Page/datafile/ERP.html",
    ]
    for u in urls:
        try:
            r = requests.get(u, timeout=10)
            if r.status_code != 200:
                continue
            txt = r.text
            m = re.search(r"Implied\s+ERP[^0-9]{0,60}([0-9]+(?:\.[0-9]+)?)\s*%", txt, re.IGNORECASE)
            if m:
                v = float(m.group(1)) / 100.0
                if 0.01 <= v <= 0.20:
                    return v
            # fallback: first reasonable % in 3-12%
            cand = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*%", txt)
            for c in cand[:80]:
                try:
                    v = float(c) / 100.0
                    if 0.03 <= v <= 0.12:
                        return v
                except Exception:
                    continue
        except Exception:
            continue
    return None

# -----------------------------
# Inputs from statements (3-year history)
# -----------------------------
def compute_tax_rate(is_3):
    if is_3 is None or getattr(is_3, "empty", True):
        return None
    try:
        cols = list(is_3.columns)
        if not cols:
            return None
        col = cols[0]
        pretax = sget(is_3, "Pretax Income", col)
        tax = sget(is_3, "Tax Provision", col)
        if pretax is not None and tax is not None and pretax != 0:
            tr = tax / pretax
            if tr < 0:
                tr = abs(tr)
            if 0.0 <= tr <= 0.5:
                return float(tr)
    except Exception:
        pass
    return None

def statement_history_print(is_3, bs_3, cf_3):
    try:
        print("\nIncome statement (annual) - head:")
        if is_3 is None or getattr(is_3, "empty", True):
            print("NA")
        else:
            print(is_3.head(14).to_string())
    except Exception:
        print("NA")
    try:
        print("\nBalance sheet (annual) - head:")
        if bs_3 is None or getattr(bs_3, "empty", True):
            print("NA")
        else:
            print(bs_3.head(14).to_string())
    except Exception:
        print("NA")
    try:
        print("\nCash flow (annual) - head:")
        if cf_3 is None or getattr(cf_3, "empty", True):
            print("NA")
        else:
            print(cf_3.head(14).to_string())
    except Exception:
        print("NA")

def compute_base_inputs(ticker):
    info = get_ticker_info(ticker)
    is_df, bs_df, cf_df = get_statements(ticker)
    is_3 = pick_3_years(is_df)
    bs_3 = pick_3_years(bs_df)
    cf_3 = pick_3_years(cf_df) # Corrected from cf_3 = pick_3_years(cf_3)

    statement_history_print(is_3, bs_3, cf_3)

    latest_col = None
    if is_3 is not None and not getattr(is_3, "empty", True):
        latest_col = list(is_3.columns)[0]
    elif bs_3 is not None and not getattr(bs_3, "empty", True):
        latest_col = list(bs_3.columns)[0]
    elif cf_3 is not None and not getattr(cf_3, "empty", True):
        latest_col = list(cf_3.columns)[0]

    revenue = None
    ebit = None
    net_income = None
    interest_expense = None
    if is_3 is not None and latest_col is not None:
        revenue = sget(is_3, "Total Revenue", latest_col)
        if revenue is None:
            revenue = sget(is_3, "TotalRevenue", latest_col)

        ebit = sget(is_3, "Operating Income", latest_col)
        if ebit is None:
            ebit = sget(is_3, "OperatingIncome", latest_col)
        if ebit is None:
            ebit = sget(is_3, "EBIT", latest_col)

        net_income = sget(is_3, "Net Income", latest_col)
        if net_income is None:
            net_income = sget(is_3, "NetIncome", latest_col)

        interest_expense = sget(is_3, "Interest Expense", latest_col)
        if interest_expense is None:
            interest_expense = sget(is_3, "InterestExpense", latest_col)

    da = None
    capex = None
    dnwc = None
    if cf_3 is not None and latest_col is not None:
        da = sget(cf_3, "Depreciation And Amortization", latest_col)
        if da is None:
            da = sget(cf_3, "Depreciation", latest_col)
        if da is None:
            da = sget(cf_3, "DepreciationAndAmortization", latest_col)

        capex = sget(cf_3, "Capital Expenditure", latest_col)
        if capex is None:
            capex = sget(cf_3, "CapitalExpenditures", latest_col)

        dnwc = sget(cf_3, "Change In Working Capital", latest_col)
        if dnwc is None:
            dnwc = sget(cf_3, "ChangeInWorkingCapital", latest_col)

    # Balance sheet items
    cash = None
    total_debt = None
    net_ppe = None
    current_assets = None
    current_liab = None
    if bs_3 is not None and latest_col is not None:
        cash = sget(bs_3, "Cash And Cash Equivalents", latest_col)
        if cash is None:
            cash = sget(bs_3, "CashAndCashEquivalents", latest_col)
        if cash is None:
            cash = sget(bs_3, "Cash", latest_col)

        st_debt = sget(bs_3, "Short Long Term Debt", latest_col)
        if st_debt is None:
            st_debt = sget(bs_3, "ShortTermDebt", latest_col)
        lt_debt = sget(bs_3, "Long Term Debt", latest_col)
        if lt_debt is None:
            lt_debt = sget(bs_3, "LongTermDebt", latest_col)
        if st_debt is not None or lt_debt is not None:
            total_debt = (st_debt or 0.0) + (lt_debt or 0.0)

        net_ppe = sget(bs_3, "Net PPE", latest_col)
        if net_ppe is None:
            net_ppe = sget(bs_3, "Property Plant Equipment", latest_col)
        if net_ppe is None:
            net_ppe = sget(bs_3, "PropertyPlantEquipment", latest_col)

        current_assets = sget(bs_3, "Total Current Assets", latest_col)
        if current_assets is None:
            current_assets = sget(bs_3, "TotalCurrentAssets", latest_col)
        current_liab = sget(bs_3, "Total Current Liabilities", latest_col)
        if current_liab is None:
            current_liab = sget(bs_3, "TotalCurrentLiabilities", latest_col)

    # Proxy start PPE if missing (requirement)
    if net_ppe is None and revenue is not None:
        net_ppe = 0.20 * float(revenue)
        print("UserWarning: Net PPE missing; proxy start PPE=20% of last revenue.")
    elif net_ppe is None:
        net_ppe = None
        print("UserWarning: Net PPE missing; proxy unavailable (missing revenue).")

    # Compute ratios (best-effort)
    tax_rate = compute_tax_rate(is_3)
    if tax_rate is None:
        tax_rate = default_tax_rate
        print("UserWarning: Using default tax_rate=%s (source unavailable)." % str(tax_rate))

    rev_growth = None
    try:
        if is_3 is not None and not getattr(is_3, "empty", True):
            cols = list(is_3.columns)
            if len(cols) >= 2:
                r0 = sget(is_3, "Total Revenue", cols[0]) or sget(is_3, "TotalRevenue", cols[0])
                r1 = sget(is_3, "Total Revenue", cols[1]) or sget(is_3, "TotalRevenue", cols[1])
                if r0 is not None and r1 is not None and r1 != 0:
                    g1 = (r0 / r1) - 1.0
                    if -0.5 <= g1 <= 0.5:
                        rev_growth = float(g1)
    except Exception:
        rev_growth = None
    if rev_growth is None:
        rev_growth = default_revenue_growth
        print("UserWarning: Using default revenue_growth=%s (source unavailable)." % str(rev_growth))

    ebit_margin = None
    if revenue is not None and revenue != 0 and ebit is not None:
        ebit_margin = float(ebit) / float(revenue)
    if ebit_margin is None:
        ebit_margin = default_ebitda_margin
        print("UserWarning: Using default ebit_margin=%s (source unavailable)." % str(ebit_margin))

    da_pct = None
    if revenue is not None and revenue != 0 and da is not None:
        da_pct = float(da) / float(revenue)
    if da_pct is None:
        da_pct = default_da_pct_revenue
        print("UserWarning: Using default da_pct_revenue=%s (source unavailable)." % str(da_pct))

    capex_pct = None
    if revenue is not None and revenue != 0 and capex is not None:
        capex_pct = abs(float(capex)) / float(revenue)
    if capex_pct is None:
        capex_pct = default_capex_pct_revenue
        print("UserWarning: Using default capex_pct_revenue=%s (source unavailable)." % str(capex_pct))

    nwc_pct = None
    if revenue is not None and revenue != 0 and dnwc is not None:
        nwc_pct = float(dnwc) / float(revenue)
    if nwc_pct is None:
        # try derive from BS (NWC level proxy -> delta approx 0)
        if current_assets is not None and current_liab is not None and revenue is not None and revenue != 0:
            # level NWC as % revenue (not delta) -> use 0 delta assumption
            nwc_pct = default_nwc_pct_revenue
        else:
            nwc_pct = default_nwc_pct_revenue
        print("UserWarning: Using default nwc_pct_revenue=%s (source unavailable)." % str(nwc_pct))

    # Market inputs
    beta = None
    try:
        beta = info.get("beta", None)
        if beta is not None:
            beta = float(beta)
    except Exception:
        beta = None
    if beta is None:
        beta = 1.0
        print("UserWarning: Using default beta=%s (source unavailable)." % str(beta))

    market_cap = None
    try:
        market_cap = info.get("marketCap", None)
        if market_cap is not None:
            market_cap = float(market_cap)
    except Exception:
        market_cap = None

    if cash is None:
        cash = 0.0
        print("UserWarning: Using default cash=%s (source unavailable)." % str(cash))
    if total_debt is None:
        total_debt = 0.0
        print("UserWarning: Using default total_debt=%s (source unavailable)." % str(total_debt))
    if market_cap is None:
        # attempt price*shares
        try:
            shares = info.get("sharesOutstanding", None)
            if shares is not None:
                shares = float(shares)
            else:
                shares = None
            if shares is not None:
                used_d, px, ok = get_asof_close(ticker)
                if ok and px is not None:
                    market_cap = float(px) * float(shares)
        except Exception:
            market_cap = None
    if market_cap is None:
        market_cap = 0.0
        print("UserWarning: Using default marketCap=%s (source unavailable)." % str(market_cap))

    # Cost of debt
    cost_debt = None
    try:
        if interest_expense is not None and float(total_debt) > 0:
            cost_debt = abs(float(interest_expense)) / float(total_debt)
            if not (0.0 <= cost_debt <= 0.25):
                cost_debt = None
    except Exception:
        cost_debt = None
    if cost_debt is None:
        cost_debt = 0.045
        print("UserWarning: Using default cost_of_debt=%s (source unavailable)." % str(cost_debt))

    # Save symbol in info for other helpers
    try:
        info["symbol"] = ticker
    except Exception:
        pass

    return {
        "info": info,
        "is_3": is_3,
        "bs_3": bs_3,
        "cf_3": cf_3,
        "revenue": revenue,
        "ebit": ebit,
        "net_income": net_income,
        "da": da,
        "capex": capex,
        "dnwc": dnwc,
        "cash": float(cash) if cash is not None else 0.0,
        "total_debt": float(total_debt) if total_debt is not None else 0.0,
        "net_ppe": net_ppe,
        "rev_growth": float(rev_growth),
        "ebit_margin": float(ebit_margin),
        "tax_rate": float(tax_rate),
        "da_pct": float(da_pct),
        "capex_pct": float(capex_pct),
        "nwc_pct": float(nwc_pct),
        "beta": float(beta),
        "market_cap": float(market_cap),
        "cost_debt": float(cost_debt),
        "current_assets": current_assets,
        "current_liab": current_liab,
    }

# -----------------------------
# WACC
# -----------------------------
def compute_wacc(base_inputs, rf, erp):
    beta = base_inputs["beta"]
    cost_equity = rf + beta * erp

    E = max(float(base_inputs["market_cap"]), 0.0)
    D = max(float(base_inputs["total_debt"]), 0.0)
    denom = E + D
    if denom <= 0:
        wE = 0.9
        wD = 0.1
        print("UserWarning: Using default capital weights (source unavailable).")
    else:
        wE = E / denom
        wD = D / denom

    tax_rate = float(base_inputs["tax_rate"])
    cost_debt = float(base_inputs["cost_debt"])
    wacc_val = wE * cost_equity + wD * cost_debt * (1.0 - tax_rate)
    return {
        "rf": rf,
        "erp": erp,
        "cost_equity": cost_equity,
        "wE": wE,
        "wD": wD,
        "wacc": wacc_val,
    }

# -----------------------------
# Forecast model (IS/BS/CF) + FCF
# -----------------------------
def build_forecast(base_inputs, horizon_years=5):
    rev0 = base_inputs["revenue"]
    if rev0 is None or not np.isfinite(rev0) or float(rev0) <= 0:
        rev0 = 1.0
        print("UserWarning: Using default revenue=%s (source unavailable)." % str(rev0))

    rev_growth = float(base_inputs["rev_growth"])
    ebit_margin = float(base_inputs["ebit_margin"])
    tax_rate = float(base_inputs["tax_rate"])
    da_pct = float(base_inputs["da_pct"])
    capex_pct = float(base_inputs["capex_pct"])
    nwc_pct = float(base_inputs["nwc_pct"])

    cash0 = float(base_inputs["cash"])
    debt0 = float(base_inputs["total_debt"])
    ppe0 = base_inputs["net_ppe"]
    if ppe0 is None:
        ppe0 = 0.0

    nwc0 = 0.0
    if base_inputs.get("current_assets", None) is not None and base_inputs.get("current_liab", None) is not None:
        try:
            ca = float(base_inputs["current_assets"])
            cl = float(base_inputs["current_liab"])
            nwc0 = max(ca - cash0 - cl, 0.0)
        except Exception:
            nwc0 = 0.0

    years = list(range(1, horizon_years + 1))
    rows_is = []
    rows_bs = []
    rows_cf = []
    rows_fcf = []

    rev_prev = float(rev0)
    cash = cash0
    debt = debt0
    ppe = float(ppe0)
    nwc_level_prev = float(nwc0)

    for y in years:
        rev = rev_prev * (1.0 + rev_growth)
        ebit = rev * ebit_margin
        tax = max(ebit, 0.0) * tax_rate
        nopat = ebit - tax
        da = rev * da_pct
        capex_spend = rev * capex_pct  # positive spend
        nwc_level = max(rev * max(nwc_pct, -0.5), -1e18)  # allow negative if nwc_pct negative
        delta_nwc = nwc_level - nwc_level_prev

        # Unlevered FCF (required definition)
        fcf_u = (ebit * (1.0 - tax_rate)) + da - capex_spend - delta_nwc

        # Cash plug + other financing plug (simple)
        cash_before_fin = cash + fcf_u
        other_fin_plug = 0.0
        if cash_before_fin < 0:
            # borrow to keep cash at zero
            other_fin_plug = -cash_before_fin
            debt += other_fin_plug
            cash = 0.0
        else:
            cash = cash_before_fin

        # PPE roll-forward (simple)
        ppe = ppe + capex_spend - da

        rows_is.append({
            "Year": y,
            "Revenue": rev,
            "EBIT": ebit,
            "EBIT margin": ebit / rev if rev != 0 else None,
            "Tax": tax,
            "NOPAT": nopat,
            "D&A": da,
        })
        rows_bs.append({
            "Year": y,
            "Cash": cash,
            "Net PPE": ppe,
            "NWC (level)": nwc_level,
            "Debt": debt,
            "Other Financing Plug": other_fin_plug,
        })
        rows_cf.append({
            "Year": y,
            "NOPAT": (ebit * (1.0 - tax_rate)),
            "D&A": da,
            "CapEx (spend)": capex_spend,
            "Delta NWC": delta_nwc,
            "Unlevered FCF": fcf_u,
            "Other Financing Plug": other_fin_plug,
            "Ending Cash": cash,
        })
        rows_fcf.append({
            "Year": y,
            "Unlevered FCF": fcf_u,
        })

        rev_prev = rev
        nwc_level_prev = nwc_level

    df_is = pd.DataFrame(rows_is)
    df_bs = pd.DataFrame(rows_bs)
    df_cf = pd.DataFrame(rows_cf)
    df_fcf = pd.DataFrame(rows_fcf)
    return df_is, df_bs, df_cf, df_fcf

# -----------------------------
# DCF valuation + bridge
# -----------------------------
def dcf_from_fcf(base_inputs, df_fcf, wacc_val, g_val):
    fcf = df_fcf["Unlevered FCF"].values.astype(float)
    n = len(fcf)
    if n <= 0:
        return None

    wacc_val = float(wacc_val)
    g_val = float(g_val)
    if wacc_val <= g_val + 1e-6:
        wacc_val = g_val + 0.01

    pv_fcf = 0.0
    for i in range(n):
        pv_fcf += fcf[i] / ((1.0 + wacc_val) ** (i + 1))

    tv = fcf[-1] * (1.0 + g_val) / (wacc_val - g_val)
    pv_tv = tv / ((1.0 + wacc_val) ** n)

    ev = pv_fcf + pv_tv
    eq = ev - float(base_inputs["total_debt"]) + float(base_inputs["cash"])

    shares = None
    try:
        shares = base_inputs["info"].get("sharesOutstanding", None)
        if shares is not None:
            shares = float(shares)
    except Exception:
        shares = None
    if shares is None or shares <= 0:
        shares = 1.0
        print("UserWarning: Using default sharesOutstanding=%s (source unavailable)." % str(shares))

    vps = eq / shares

    bridge = pd.DataFrame([
        {"Item": "PV of explicit FCF", "Value": pv_fcf},
        {"Item": "PV of terminal value", "Value": pv_tv},
        {"Item": "Enterprise value", "Value": ev},
        {"Item": "Less: Total debt", "Value": -float(base_inputs["total_debt"])}
    ])
    # Only add Equity Value and Value per share if shares are available and valid
    if shares is not None and shares > 0:
        bridge = pd.concat([bridge, pd.DataFrame([{"Item": "Add: Cash", "Value": float(base_inputs["cash"])}])], ignore_index=True)
        bridge = pd.concat([bridge, pd.DataFrame([{"Item": "Equity value", "Value": eq}])], ignore_index=True)
        bridge = pd.concat([bridge, pd.DataFrame([{"Item": "Shares outstanding", "Value": shares}])], ignore_index=True)
        bridge = pd.concat([bridge, pd.DataFrame([{"Item": "Value per share", "Value": vps}])], ignore_index=True)

    return ev, eq, vps, tv, bridge

# -----------------------------
# Peers / Comps (deterministic, no global search)
# -----------------------------
SECTOR_FALLBACK = {
    "Technology": ["MSFT", "AAPL", "NVDA", "ORCL", "ADBE", "CRM"],
    "Communication Services": ["GOOGL", "META", "NFLX", "TMUS", "VZ", "DIS"],
    "Consumer Cyclical": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX"],
    "Consumer Defensive": ["WMT", "PG", "KO", "PEP", "COST", "MDLZ"],
    "Healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABT"],
    "Financial Services": ["JPM", "BAC", "WFC", "GS", "MS", "C"],
    "Industrials": ["GE", "CAT", "HON", "BA", "UPS", "DE"],
    "Basic Materials": ["LIN", "BHP", "RIO", "SHW", "ECL", "FCX"],
    "Energy": ["XOM", "CVX", "SHEL", "BP", "COP", "EOG"],
    "Real Estate": ["PLD", "AMT", "EQIX", "SPG", "O", "PSA"],
    "Utilities": ["NEE", "DUK", "SO", "AEP", "EXC", "SRE"],
}

LAST_RESORT_FILL = ["MSFT", "GOOGL", "META", "NVDA", "AMZN", "AAPL"]

def yahoo_recommended_symbols(ticker):
    # Best-effort: use Yahoo quoteSummary 'recommendations' module.
    # If it doesn't exist, return [].
    if requests is None:
        return []
    url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/%s?modules=recommendations,quoteType,summaryProfile" % ticker
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        j = r.json()
        rs = j.get("quoteSummary", {}).get("result", None)
        if not rs or not isinstance(rs, list):
            return []
        rs0 = rs[0] if rs else {}
        rec = rs0.get("recommendations", {})
        # Common pattern: recommendations.recommendedSymbols = [{symbol:..},...]R
        out = []
        lst = rec.get("recommendedSymbols", None)
        if isinstance(lst, list):
            for it in lst:
                sym = None
                if isinstance(it, dict):
                    sym = it.get("symbol", None)
                if sym:
                    out.append(str(sym))
        # Some variants may store directly
        if not out and isinstance(rec, dict):
            for k in ["symbols", "recommended", "recommendedSymbol"]:
                v = rec.get(k, None)
                if isinstance(v, list):
                    for it in v:
                        if isinstance(it, dict) and it.get("symbol", None):
                            out.append(str(it.get("symbol")))
                        elif isinstance(it, str):
                            out.append(it)
                elif isinstance(v, str):
                    out.append(v)
        # Dedup keep order
        seen = set()
        dedup = []
        for s in out:
            if s and s not in seen:
                seen.add(s)
                dedup.append(s)
        return dedup
    except Exception:
        return []

def pick_peers(subject_ticker, subject_info):
    # Step 1: Yahoo-first recommended symbols (in returned order)
    candidates = yahoo_recommended_symbols(subject_ticker)
    peers = []
    for c in candidates:
        if len(peers) >= 4:
            break
        if c == subject_ticker:
            continue
        info = get_ticker_info(c)
        if not info:
            continue
        mc = info.get("marketCap", None)
        ev_to_ebitda = info.get("enterpriseToEbitda", None)
        ev_to_sales = info.get("enterpriseToRevenue", None)
        pe = info.get("trailingPE", None)
        ok = (mc is not None) and (ev_to_ebitda is not None or ev_to_sales is not None or pe is not None)
        if ok:
            peers.append(c)

    # Step 2: get sector/industry
    sector = None
    industry = None
    try:
        sector = subject_info.get("sector", None)
        industry = subject_info.get("industry", None)
    except Exception:
        sector = None
        industry = None

    # Step 3: sector fallback list fill to exactly 4
    if len(peers) < 4 and sector:
        fb = SECTOR_FALLBACK.get(str(sector), [])
        for c in fb:
            if len(peers) >= 4:
                break
            if c == subject_ticker or c in peers:
                continue
            info = get_ticker_info(c)
            if not info:
                continue
            mc = info.get("marketCap", None)
            ev_to_ebitda = info.get("enterpriseToEbitda", None)
            ev_to_sales = info.get("enterpriseToRevenue", None)
            pe = info.get("trailingPE", None)
            ok = (mc is not None) and (ev_to_ebitda is not None or ev_to_sales is not None or pe is not None)
            if ok:
                peers.append(c)

    # Step 4: last resort fill
    warned = False
    if len(peers) < 4:
        print("UserWarning: Yahoo did not return peers; using manual fallback peers. Edit this list.")
        warned = True
        for c in LAST_RESORT_FILL:
            if len(peers) >= 4:
                break
            if c == subject_ticker or c in peers:
                continue
            peers.append(c)

    # Deterministic output
    peers = peers[:4]
    print("\nPeers selected (in order): %s" % (", ".join(peers) if peers else "NA"))
    return peers

def build_comps(subject_ticker, base_inputs):
    subject_info = base_inputs["info"]
    peers = pick_peers(subject_ticker, subject_info)

    rows = []
    for t in peers:
        info = get_ticker_info(t)
        mc = info.get("marketCap", None)
        ev = info.get("enterpriseValue", None)
        ev_ebitda = info.get("enterpriseToEbitda", None)
        ev_sales = info.get("enterpriseToRevenue", None)
        pe = info.get("trailingPE", None)
        rows.append({
            "Ticker": t,
            "MarketCap": float(mc) if mc is not None else None,
            "EnterpriseValue": float(ev) if ev is not None else None,
            "EV/EBITDA": float(ev_ebitda) if ev_ebitda is not None else None,
            "EV/Sales": float(ev_sales) if ev_sales is not None else None,
            "P/E": float(pe) if pe is not None else None,
        })

    df = pd.DataFrame(rows)
    # Medians
    med = {
        "EV/EBITDA": pd.to_numeric(df["EV/EBITDA"], errors="coerce").dropna().median() if "EV/EBITDA" in df.columns else None,
        "EV/Sales": pd.to_numeric(df["EV/Sales"], errors="coerce").dropna().median() if "EV/Sales" in df.columns else None,
        "P/E": pd.to_numeric(df["P/E"], errors="coerce").dropna().median() if "P/E" in df.columns else None,
    }

    # Subject fundamentals for valuation
    revenue = base_inputs["revenue"]
    ebit = base_inputs["ebit"]
    da = base_inputs["da"]
    net_income = base_inputs["net_income"]

    # EBITDA proxy
    ebitda = None
    if ebit is not None and da is not None:
        ebitda = float(ebit) + float(da)
    elif revenue is not None:
        ebitda = float(revenue) * default_ebitda_margin
        print("UserWarning: Using default ebitda_margin=%s (source unavailable)." % str(default_ebitda_margin))
    else:
        ebitda = None

    # Valuations
    vals = []
    if med["EV/EBITDA"] is not None and ebitda is not None:
        ev_est = float(med["EV/EBITDA"]) * float(ebitda)
        eq_est = ev_est - float(base_inputs["total_debt"]) + float(base_inputs["cash"])
        vals.append(("EV/EBITDA", ev_est, eq_est))
    if med["EV/Sales"] is not None and revenue is not None:
        ev_est = float(med["EV/Sales"]) * float(revenue)
        eq_est = ev_est - float(base_inputs["total_debt"]) + float(base_inputs["cash"])
        vals.append(("EV/Sales", ev_est, eq_est))
    if med["P/E"] is not None and net_income is not None:
        eq_est = float(med["P/E"]) * float(net_income)
        ev_est = eq_est + float(base_inputs["total_debt"]) - float(base_inputs["cash"])
        vals.append(("P/E", ev_est, eq_est))

    # Shares
    shares = None
    try:
        shares = base_inputs["info"].get("sharesOutstanding", None)
        if shares is not None:
            shares = float(shares)
    except Exception:
        shares = None
    if shares is None or shares <= 0:
        shares = 1.0
        print("UserWarning: Using default sharesOutstanding=%s (source unavailable)." % str(shares))

    comp_rows = []
    for mname, ev_est, eq_est in vals:
        vps = eq_est / shares if shares else None
        comp_rows.append({
            "Method": mname,
            "EV implied": ev_est,
            "Equity implied": eq_est,
            "Value per share": vps,
        })
    comp_df = pd.DataFrame(comp_rows)

    # Summary low/median/high across methods (if present)
    vps_series = pd.to_numeric(comp_df["Value per share"], errors="coerce").dropna() if (comp_df is not None and not comp_df.empty) else pd.Series([], dtype=float)
    summary = None
    if vps_series is not None and not vps_series.empty:
        summary = {
            "Low": float(vps_series.min()),
            "Median": float(vps_series.median()),
            "High": float(vps_series.max()),
        }
    return df, med, comp_df, summary, peers

# -----------------------------
# Sensitivity + plots
# -----------------------------
def make_sensitivity_from_model(base_inputs, rf, erp, base_wacc, base_g):
    df_is, df_bs, df_cf, df_fcf = build_forecast(base_inputs, horizon_years=forecast_horizon_years)

    wacc_grid = np.linspace(base_wacc - 0.02, base_wacc + 0.02, 5)
    g_grid = np.linspace(base_g - 0.01, base_g + 0.01, 5)
    wacc_grid = np.clip(wacc_grid, 0.04, 0.20)
    g_grid = np.clip(g_grid, -0.01, 0.06)

    table = pd.DataFrame(index=[f"{w*100:.2f}%" for w in wacc_grid],
                         columns=[f"{gg*100:.2f}%" for gg in g_grid])
    vals = np.zeros((5, 5), dtype=float)

    for i, w in enumerate(wacc_grid):
        for j, gg in enumerate(g_grid):
            ev, eq, vps, tv, bridge = dcf_from_fcf(base_inputs, df_fcf, float(w), float(gg))
            vals[i, j] = float(vps) if vps is not None and np.isfinite(vps) else np.nan
            table.iloc[i, j] = vals[i, j]
    return table, vals, wacc_grid, g_grid

def plot_heatmap(vals, wacc_grid, g_grid, ticker):
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(vals, aspect="auto")
    ax.set_xticks(range(len(g_grid)))
    ax.set_yticks(range(len(wacc_grid)))
    ax.set_xticklabels([f"{gg*100:.2f}%" for gg in g_grid])
    ax.set_yticklabels([f"{w*100:.2f}%" for w in wacc_grid])
    ax.set_xlabel("Terminal growth (g)")
    ax.set_ylabel("WACC")
    ax.set_title("Sensitivity: Value per share (%s)" % ticker)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value per share")
    fig.tight_layout()
    path = os.path.join(output_dir, "sens_heatmap_%s_%s.png" % (safe_name(ticker), safe_name(ASOF_DATE)))
    try:
        fig.savefig(path, dpi=160)
    except Exception:
        pass
    plt.close(fig)
    return path

def scenarios(base_inputs, base_wacc, base_g):
    base = dict(base_inputs)
    bull = dict(base_inputs)
    bear = dict(base_inputs)

    bull["rev_growth"] = float(base_inputs["rev_growth"]) + 0.03
    bull["ebit_margin"] = float(base_inputs["ebit_margin"]) + 0.02
    bull_g = min(base_g + 0.005, 0.06)

    bear["rev_growth"] = float(base_inputs["rev_growth"]) - 0.03
    bear["ebit_margin"] = max(float(base_inputs["ebit_margin"]) - 0.02, -0.20)
    bear_g = max(base_g - 0.005, -0.01)

    for d in [bull, bear]:
        d["rev_growth"] = float(np.clip(d["rev_growth"], -0.20, 0.30))
        d["ebit_margin"] = float(np.clip(d["ebit_margin"], -0.20, 0.50))

    out_rows = []
    rev_paths = {}
    for name, inp, gg in [("Base", base, base_g), ("Bull", bull, bull_g), ("Bear", bear, bear_g)]:
        df_is, df_bs, df_cf, df_fcf = build_forecast(inp, horizon_years=forecast_horizon_years)
        ev, eq, vps, tv, bridge = dcf_from_fcf(inp, df_fcf, base_wacc, gg)
        out_rows.append({
            "Scenario": name,
            "Revenue growth": inp["rev_growth"],
            "EBIT margin": inp["ebit_margin"],
            "Terminal g": gg,
            "WACC": base_wacc,
            "Value per share": vps,
            "Equity value": eq,
            "Enterprise value": ev,
        })
        rev_paths[name] = df_is["Revenue"].values.astype(float)

    df = pd.DataFrame(out_rows)
    return df, rev_paths

def plot_revenue_scenarios(rev_paths, ticker):
    fig, ax = plt.subplots(figsize=(9, 5))
    years = list(range(1, forecast_horizon_years + 1))
    for k, v in rev_paths.items():
        try:
            ax.plot(years, v, label=k)
        except Exception:
            continue
    ax.set_xlabel("Forecast year")
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue scenarios (%s)" % ticker)
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "scenarios_revenue_%s_%s.png" % (safe_name(ticker), safe_name(ASOF_DATE)))
    try:
        fig.savefig(path, dpi=160)
    except Exception:
        pass
    plt.close(fig)
    return path

# -----------------------------
# Report
# -----------------------------
def handle_report(ticker):
    ensure_output_dir()

    # ASOF prices
    used_date_px, px, ok_px = get_asof_close(ticker)
    print("ASOF requested: %s | ASOF used for %s: %s" % (ASOF_DATE, ticker, used_date_px if used_date_px else "NA"))
    if (not ok_px) or px is None:
        info = get_ticker_info(ticker)
        fallback_px = None
        try:
            fallback_px = info.get("regularMarketPrice", None)
            if fallback_px is not None:
                fallback_px = float(fallback_px)
        except Exception:
            fallback_px = None
        if fallback_px is not None:
            px = fallback_px
            print("UserWarning: Using info['regularMarketPrice'] as price fallback (asof price unavailable).")
        else:
            print("UserWarning: Price unavailable; proceeding with valuation outputs only.")

    used_date_rf, tnx, ok_rf = get_asof_close("^TNX")
    print("ASOF requested: %s | ASOF used for %s: %s" % (ASOF_DATE, "^TNX", used_date_rf if used_date_rf else "NA"))
    if ok_rf and tnx is not None:
        rf = float(tnx) / 100.0
    else:
        rf = 0.04
        print("UserWarning: Using default rf=%s (source unavailable)." % str(rf))

    cached = load_cache(ticker)
    # Check if cached data is valid and contains 'net_ppe' before using
    if isinstance(cached, dict) and cached.get("ticker") == ticker and cached.get("asof") == ASOF_DATE:
        base_inputs = cached.get("base_inputs", None)
        erp = cached.get("erp", None)
        wacc_info = cached.get("wacc_info", None)
        if base_inputs is None or erp is None or wacc_info is None or "net_ppe" not in base_inputs:
            cached = None # Invalidate cache if 'net_ppe' is missing or other critical data

    if cached is None:
        base_inputs = compute_base_inputs(ticker)
        erp = get_damodaran_erp_us()
        if erp is None:
            erp = 0.05
            print("UserWarning: Using default erp=%s (source unavailable)." % str(erp))
        wacc_info = compute_wacc(base_inputs, rf, erp)
        save_cache(ticker, {"ticker": ticker, "asof": ASOF_DATE, "base_inputs": base_inputs, "erp": erp, "wacc_info": wacc_info})

    base_wacc = wacc_info.get("wacc", None)
    if base_wacc is None or (isinstance(base_wacc, float) and (math.isnan(base_wacc) or math.isinf(base_wacc))):
        base_wacc = wacc_default
        print("UserWarning: Using default wacc=%s (source unavailable)." % str(base_wacc))

    base_g = terminal_growth_default
    print("UserWarning: Using default terminal_growth=%s (source unavailable)." % str(base_g))

    print("\n--- Market inputs ---")
    print("Risk-free (from ^TNX): %s" % fmt_pct(rf, 2))
    print("ERP (Damodaran best-effort): %s" % fmt_pct(erp, 2))
    print("Beta (Yahoo): %s" % fmt_num(base_inputs["beta"], 2))
    print("Cost of equity: %s" % fmt_pct(wacc_info["cost_equity"], 2))
    print("WACC: %s (wE=%s, wD=%s)" % (fmt_pct(base_wacc, 2), fmt_pct(wacc_info["wE"], 2), fmt_pct(wacc_info["wD"], 2)))

    # Forecast model outputs
    df_is, df_bs, df_cf, df_fcf = build_forecast(base_inputs, horizon_years=forecast_horizon_years)

    print("\n--- Forecast: Income statement ---")
    print(df_to_pretty(df_is, money_cols=["Revenue", "EBIT", "Tax", "NOPAT", "D&A"], pct_cols=["EBIT margin"]).to_string(index=False))

    print("\n--- Forecast: Balance sheet (cash plug + other financing plug) ---")
    print(df_to_pretty(df_bs, money_cols=["Cash", "Net PPE", "NWC (level)", "Debt", "Other Financing Plug"]).to_string(index=False))

    print("\n--- Forecast: Cash flow ---")
    print(df_to_pretty(df_cf, money_cols=["NOPAT", "D&A", "CapEx (spend)", "Delta NWC", "Unlevered FCF", "Other Financing Plug", "Ending Cash"]).to_string(index=False))

    # DCF
    ev, eq, vps, tv, bridge = dcf_from_fcf(base_inputs, df_fcf, float(base_wacc), float(base_g))

    print("\n--- DCF bridge ---")
    bridge_pretty = bridge.copy()
    # Format: treat shares separately
    for i in range(len(bridge_pretty)):
        item = str(bridge_pretty.loc[i, "Item"])
        val = bridge_pretty.loc[i, "Value"]
        if "Shares" in item:
            bridge_pretty.loc[i, "Value"] = fmt_num(val, 0)
        elif "per share" in item.lower():
            bridge_pretty.loc[i, "Value"] = fmt_num(val, 2)
        else:
            bridge_pretty.loc[i, "Value"] = fmt_money(val)
    print(bridge_pretty.to_string(index=False))

    if px is not None and isinstance(px, (int, float)) and np.isfinite(px) and float(px) > 0 and vps is not None and np.isfinite(vps):
        upside = (float(vps) / float(px)) - 1.0
        print("\nAs-of price: %s | DCF value per share: %s | Implied upside: %s" % (fmt_num(px, 2), fmt_num(vps, 2), fmt_pct(upside, 2)))
    else:
        print("\nAs-of price: NA | DCF value per share: %s | Implied upside: NA" % (fmt_num(vps, 2) if vps is not None else "NA"))

    # Save key outputs
    try:
        df_is.to_csv(os.path.join(output_dir, "forecast_is_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
        df_bs.to_csv(os.path.join(output_dir, "forecast_bs_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
        df_cf.to_csv(os.path.join(output_dir, "forecast_cf_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
        df_fcf.to_csv(os.path.join(output_dir, "forecast_fcf_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
        bridge.to_csv(os.path.join(output_dir, "dcf_bridge_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
    except Exception:
        pass

    # Comps
    comps_df, med, comp_val_df, comp_summary, peers = build_comps(ticker, base_inputs)
    print("\n--- Comps (peers) ---")
    print(df_to_pretty(comps_df, money_cols=["MarketCap", "EnterpriseValue"], num_cols=["EV/EBITDA", "EV/Sales", "P/E"]).to_string(index=False))

    print("\nComps medians:")
    print("Median EV/EBITDA: %s | Median EV/Sales: %s | Median P/E: %s" % (fmt_num(med.get("EV/EBITDA", None), 2), fmt_num(med.get("EV/Sales", None), 2), fmt_num(med.get("P/E", None), 2)))

    print("\nComps valuation (implied):")
    if comp_val_df is None or getattr(comp_val_df, "empty", True):
        print("NA")
    else:
        print(df_to_pretty(comp_val_df, money_cols=["EV implied", "Equity implied"], num_cols=["Value per share"]).to_string(index=False))

    if comp_summary is not None:
        print("\nComps value per share summary (low/median/high): %s / %s / %s" % (fmt_num(comp_summary["Low"], 2), fmt_num(comp_summary["Median"], 2), fmt_num(comp_summary["High"], 2)))

    try:
        comps_df.to_csv(os.path.join(output_dir, "comps_peers_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
        if comp_val_df is not None and not comp_val_df.empty:
            comp_val_df.to_csv(os.path.join(output_dir, "comps_valuation_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE))), index=False)
    except Exception:
        pass

    # Sensitivity table + heatmap
    sens_table, sens_vals, wacc_grid, g_grid = make_sensitivity_from_model(base_inputs, rf, erp, float(base_wacc), float(base_g))
    sens_csv = os.path.join(output_dir, "sensitivity_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE)))
    try:
        out_csv = sens_table.copy()
        for c in out_csv.columns:
            out_csv[c] = pd.to_numeric(out_csv[c], errors="coerce")
        out_csv.to_csv(sens_csv, index=True)
    except Exception:
        pass
    heatmap_path = plot_heatmap(sens_vals, wacc_grid, g_grid, ticker)

    print("\n--- Sensitivity (Value per share) ---")
    sens_show = sens_table.copy()
    for c in sens_show.columns:
        sens_show[c] = sens_show[c].apply(lambda z: "NA" if z is None or (isinstance(z, float) and (math.isnan(z) or math.isinf(z))) else f"{float(z):.2f}")
    print(sens_show.to_string())
    print("\nSaved heatmap to: %s" % heatmap_path)
    print("Saved sensitivity CSV to: %s" % sens_csv)

    # Scenarios
    scen_df, rev_paths = scenarios(base_inputs, float(base_wacc), float(base_g))
    scen_csv = os.path.join(output_dir, "scenarios_%s_%s.csv" % (safe_name(ticker), safe_name(ASOF_DATE)))
    try:
        scen_df.to_csv(scen_csv, index=False)
    except Exception:
        pass
    rev_plot_path = plot_revenue_scenarios(rev_paths, ticker)

    scen_show = scen_df.copy()
    scen_show["Revenue growth"] = scen_show["Revenue growth"].apply(lambda z: fmt_pct(z, 2))
    scen_show["EBIT margin"] = scen_show["EBIT margin"].apply(lambda z: fmt_pct(z, 2))
    scen_show["Terminal g"] = scen_show["Terminal g"].apply(lambda z: fmt_pct(z, 2))
    scen_show["WACC"] = scen_show["WACC"].apply(lambda z: fmt_pct(z, 2))
    scen_show["Value per share"] = scen_show["Value per share"].apply(lambda z: fmt_num(z, 2))
    scen_show["Equity value"] = scen_show["Equity value"].apply(lambda z: fmt_money(z))
    scen_show["Enterprise value"] = scen_show["Enterprise value"].apply(lambda z: fmt_money(z))

    print("\n--- Scenarios ---")
    print(scen_show.to_string(index=False))
    print("\nSaved revenue scenarios plot to: %s" % rev_plot_path)
    print("Saved scenarios CSV to: %s" % scen_csv)

    # Interpretation (required block)
    print("\n--- Interpretation ---")
    interp = []
    interp.append("This model builds a 5-year forecast from the last 3 annual Yahoo statements (with defaults only where inputs are unavailable).")
    interp.append("DCF uses unlevered FCF = EBIT*(1-tax) + D&A - CapEx - DeltaNWC, discounted at WACC, with terminal value from (1+g)/(WACC-g).")
    if px is not None and isinstance(px, (int, float)) and np.isfinite(px) and float(px) > 0 and vps is not None and np.isfinite(vps):
        up = (float(vps) / float(px)) - 1.0
        interp.append("At the ASOF price, the base DCF implies %s upside." % fmt_pct(up, 2))
    else:
        interp.append("Price was unavailable from the ASOF pull; upside vs market price was not computed.")
    if comp_summary is not None:
        interp.append("Comps provide a cross-check range (low/median/high) based on peer medians for EV/EBITDA, EV/Sales, and P/E where available.")
    interp.append("Use the sensitivity heatmap to understand valuation exposure to WACC and terminal growth assumptions.")
    for line in interp:
        print("- %s" % line)

# -----------------------------
# Command parsing / main
# -----------------------------
def parse_command(cmd):
    cmd = (cmd or "").strip()
    parts = cmd.split()
    if not parts:
        return None, []
    return parts[0].lower(), parts[1:]

def main():
    global COMMAND
    try:
        import sys
        if (COMMAND is None or str(COMMAND).strip() == "") and len(sys.argv) > 1:
            COMMAND = " ".join(sys.argv[1:])
    except Exception:
        pass

    op, args = parse_command(COMMAND)
    if op not in ["/report", "/quick", "/sens", "/scenario", "/run", "/vs"]:
        usage_error()
        return

    if op == "/report":
        if len(args) != 1 or not args[0].strip():
            usage_error()
            return
        handle_report(args[0].strip())
        return

    # Other commands not implemented in this script variant: obey strict gate.
    usage_error()
    return
