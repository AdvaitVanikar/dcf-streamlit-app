import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import model_core as model

st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Model (Interactive)")

ticker = st.text_input("Ticker", value="HEIA.AS",
                       help="Yahoo Finance ticker, e.g. AAPL, HEIA.AS, VOD.L")

st.subheader("Overrides (optional)")
st.caption("Leave any field blank to use the value fetched from Yahoo Finance or the built-in default.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Market / Cost of Capital**")
    rf          = st.text_input("Risk-free rate (rf)",       value="", placeholder="e.g. 0.04")
    erp         = st.text_input("Equity risk premium (erp)", value="", placeholder="e.g. 0.05")
    beta        = st.text_input("Beta",                       value="", placeholder="e.g. 1.2")
with col2:
    st.markdown("**Debt & Capital Structure**")
    cost_debt   = st.text_input("Cost of debt",              value="", placeholder="e.g. 0.035")
    wD          = st.text_input("Debt weight (wD)",          value="", placeholder="e.g. 0.30")
    cost_equity = st.text_input("Cost of equity (optional)", value="", placeholder="e.g. 0.08")
with col3:
    st.markdown("**Operating Assumptions**")
    terminal_g  = st.text_input("Terminal growth (g)",       value="", placeholder="e.g. 0.025")
    rev_growth  = st.text_input("Revenue growth",            value="", placeholder="e.g. 0.06")
    ebit_margin = st.text_input("EBIT margin",               value="", placeholder="e.g. 0.15")
    tax_rate    = st.text_input("Tax rate",                  value="", placeholder="e.g. 0.25")

user_inputs = {
    "rf": rf, "erp": erp, "beta": beta,
    "cost_debt": cost_debt, "wD": wD, "cost_equity": cost_equity,
    "terminal_g": terminal_g, "rev_growth": rev_growth,
    "ebit_margin": ebit_margin, "tax_rate": tax_rate,
}

st.subheader("Comparable Companies (optional)")
st.caption("Enter tickers for peers. Leave blank to auto-select from Yahoo Finance.")

if "peer_count" not in st.session_state:
    st.session_state.peer_count = 4
col_add, col_remove, _ = st.columns([1, 1, 6])
with col_add:
    if st.button("+ Add peer"):
        st.session_state.peer_count += 1
with col_remove:
    if st.button("- Remove last") and st.session_state.peer_count > 1:
        st.session_state.peer_count -= 1

peer_tickers = []
peer_cols = st.columns(min(st.session_state.peer_count, 4))
for i in range(st.session_state.peer_count):
    with peer_cols[i % 4]:
        val = st.text_input(f"Peer {i+1}", value="", placeholder="e.g. UB.AS", key=f"peer_{i}")
        peer_tickers.append(val.strip())

user_peers = [p.upper() for p in peer_tickers if p.strip()]
if user_peers:
    st.info(f"Using {len(user_peers)} user-supplied peer(s): {', '.join(user_peers)}")
else:
    st.info("No peers entered — model will auto-select from Yahoo Finance.")

run = st.button("Run model", type="primary")

# ─── PLOT FUNCTIONS (embedded in app.py to bypass Streamlit module cache) ────

def _dcf_vps_local(base_inputs, wacc, g, horizon=5):
    try:
        _, _, _, df_fcf = model.build_forecast(base_inputs, horizon_years=horizon)
        _, _, vps, _, _ = model.dcf_from_fcf(base_inputs, df_fcf, wacc, g)
        if vps is not None and np.isfinite(float(vps)):
            return float(vps)
    except Exception:
        pass
    return None


def plot_comps_multiples(comps_df, ticker_label):
    multiples = ["EV/EBITDA", "EV/Sales", "P/E"]
    available = [m for m in multiples if m in comps_df.columns]
    if not available:
        return None

    fig, axes = plt.subplots(1, len(available),
                             figsize=(4.5 * len(available), 7), facecolor="white")
    if len(available) == 1:
        axes = [axes]

    BLUE = "#2563EB"
    RED  = "#DC2626"
    DOT  = "#1F2937"

    for ax, col in zip(axes, available):
        vals_raw = pd.to_numeric(comps_df[col], errors="coerce")
        mask = vals_raw.notna()
        vals = vals_raw[mask].values
        tkrs = comps_df.loc[mask, "Ticker"].tolist() if "Ticker" in comps_df.columns else []

        ax.set_facecolor("#F8FAFC")
        for sp in ["top", "right", "bottom", "left"]:
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])

        if len(vals) == 0:
            ax.set_title(col, fontsize=11, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        lo, hi = vals.min(), vals.max()
        med = float(np.median(vals))
        margin = (hi - lo) * 0.28 if hi != lo else max(abs(hi) * 0.15, 1.0)

        ax.set_xlim(-0.7, 0.8)
        ax.set_ylim(lo - margin, hi + margin)
        ax.tick_params(left=True, labelleft=True, labelsize=8, colors="#6B7280")
        ax.set_ylabel("Multiple (x)", fontsize=8.5, color="#6B7280")

        # thick vertical range line
        ax.plot([0, 0], [lo, hi], color=BLUE, linewidth=8,
                solid_capstyle="round", zorder=2, alpha=0.9)

        # min / max labels (left of line)
        ax.text(-0.28, hi, f"{hi:.1f}x", va="center", ha="right",
                fontsize=8, color=BLUE, fontweight="bold")
        ax.text(-0.28, lo, f"{lo:.1f}x", va="center", ha="right",
                fontsize=8, color=BLUE, fontweight="bold")

        # median dash on LEFT side of line
        ax.plot([-0.22, 0.0], [med, med],
                color=RED, linewidth=3.5, zorder=5, solid_capstyle="round")
        ax.text(-0.30, med, "Median\n" + f"{med:.1f}x",
                va="center", ha="right", fontsize=8,
                color=RED, fontweight="bold", linespacing=1.4)

        # peer dots + labels on RIGHT side
        for v, t in zip(vals, tkrs):
            ax.scatter(0.08, v, color=DOT, s=55, zorder=6,
                       edgecolors="white", linewidths=0.8)
            ax.text(0.16, v, f"{t}  {v:.1f}x",
                    va="center", ha="left", fontsize=7.5, color=DOT)

        ax.set_title(col, fontsize=11, fontweight="bold", pad=12, color="#111827")

    fig.suptitle(f"Peer Multiples — {ticker_label}", fontsize=13,
                 fontweight="bold", color="#111827", y=1.01)
    fig.patch.set_facecolor("white")
    fig.tight_layout(w_pad=3)
    return fig


def plot_valuation_bridge(base_inputs, base_wacc, base_g, base_vps, ticker_label):
    if base_vps is None or not np.isfinite(float(base_vps)):
        return None

    shocks = [
        ("Rev Growth\n+3pp",   "rev_growth",  +0.03,  False),
        ("EBIT Margin\n+2pp",  "ebit_margin", +0.02,  False),
        ("WACC\n-1pp",         "_wacc",       -0.01,  True),
        ("Terminal g\n+0.5pp", "_g",          +0.005, True),
        ("Tax Rate\n-3pp",     "tax_rate",    -0.03,  False),
        ("CapEx\n-1pp rev",    "capex_pct",   -0.01,  False),
    ]

    impacts = []
    for label, key, delta, is_wacc_g in shocks:
        inp = dict(base_inputs)
        if is_wacc_g:
            w = base_wacc + delta if key == "_wacc" else base_wacc
            g = base_g + delta    if key == "_g"    else base_g
            sv = _dcf_vps_local(inp, w, g)
        else:
            inp[key] = float(inp.get(key, 0)) + delta
            sv = _dcf_vps_local(inp, base_wacc, base_g)
        impacts.append((label, (sv - base_vps) if sv is not None else 0.0))

    GREEN  = "#16A34A"
    RED    = "#DC2626"
    BASE_C = "#2563EB"
    GREY   = "#D1D5DB"

    n     = len(impacts)
    bar_w = 0.55
    gap   = 1.0
    xs    = [i * gap for i in range(n + 2)]

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.set_facecolor("white")
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Base bar
    ax.bar(xs[0], base_vps, width=bar_w, color=BASE_C, alpha=0.88, zorder=3)
    ax.text(xs[0], base_vps * 1.02, f"{base_vps:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=BASE_C)
    ax.text(xs[0], base_vps * -0.08, "Base\nVPS",
            ha="center", va="top", fontsize=8.5, color="#374151", fontweight="bold")

    running = float(base_vps)
    prev_x  = xs[0]

    for i, (label, impact) in enumerate(impacts):
        x     = xs[i + 1]
        color = GREEN if impact >= 0 else RED

        bottom = min(running, running + impact)
        height = max(abs(impact), 0.001)
        ax.bar(x, height, bottom=bottom, width=bar_w, color=color, alpha=0.85, zorder=3)

        # dashed connector
        ax.plot([prev_x + bar_w / 2, x - bar_w / 2], [running, running],
                color=GREY, linewidth=1.5, linestyle="--", zorder=1)

        # value label
        offset = abs(base_vps) * 0.02
        lbl_y  = running + impact + (offset if impact >= 0 else -offset * 2.5)
        ax.text(x, lbl_y, f"{impact:+.1f}",
                ha="center", va="bottom" if impact >= 0 else "top",
                fontsize=8.5, fontweight="bold", color=color)

        # x-axis label (with real newline via multiline text)
        ax.text(x, base_vps * -0.08, label,
                ha="center", va="top", fontsize=8, color="#374151",
                linespacing=1.3, multialignment="center")

        running += impact
        prev_x   = x

    # Final "all shocks" bar
    x_fin = xs[-1]
    ax.bar(x_fin, running, width=bar_w, color=BASE_C, alpha=0.55, zorder=3,
           edgecolor=BASE_C, linewidth=1.5)
    ax.text(x_fin, running * 1.02, f"{running:.1f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=BASE_C)
    ax.text(x_fin, base_vps * -0.08, "All\nShocks",
            ha="center", va="top", fontsize=8.5, color="#374151", fontweight="bold")

    ax.axhline(base_vps, color=BASE_C, linewidth=0.8, linestyle=":", alpha=0.4, zorder=0)

    all_vals = [base_vps, running] + [base_vps + d for _, d in impacts]
    y_lo = min(all_vals)
    y_hi = max(all_vals)
    pad  = (y_hi - y_lo) * 0.25 if y_hi != y_lo else abs(y_hi) * 0.25
    ax.set_ylim(y_lo - pad * 1.8, y_hi + pad)
    ax.set_xlim(-gap * 0.6, xs[-1] + gap * 0.6)
    ax.set_xticks([])
    ax.set_ylabel("Value per Share", fontsize=9, color="#374151")
    ax.tick_params(left=True, labelleft=True, labelsize=8, colors="#6B7280")

    fig.suptitle(f"Scenario Impact on Value per Share — {ticker_label}",
                 fontsize=13, fontweight="bold", color="#111827")
    ax.set_title("Each bar = VPS change from a single driver shock, all others held constant",
                 fontsize=8, color="#6B7280", style="italic", pad=6)
    fig.tight_layout()
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────
def pretty_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return df2

def show_forecast(df, decimals=2):
    d = df.copy()
    if "Year" in d.columns:
        d = d.set_index("Year")
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return d

def style_assumptions(df):
    def row_style(row):
        src = row["Source"]
        if "User Input" in src:
            color = "#d4edda"
        elif any(x in src for x in ["Yahoo", "Damodaran", "CAPM", "Computed"]):
            color = "#cce5ff"
        else:
            color = "#f8f9fa"
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(row_style, axis=1)


# ── Output ────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running model..."):
        res = model.handle_report(
            ticker,
            user_inputs=user_inputs,
            user_peers=user_peers if user_peers else None
        )

    st.success("Done")

    st.subheader("Key Output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("DCF Value per Share",
                  f"{res['vps']:,.2f}" if res["vps"] is not None else "NA")
    with c2:
        st.metric("As-of Price",
                  f"{res['px']:,.2f}" if res["px"] is not None else "NA")

    st.subheader("Model Assumptions")
    st.caption("Every key driver and where it came from.")
    if res.get("assumptions_df") is not None:
        st.dataframe(style_assumptions(res["assumptions_df"]),
                     use_container_width=True, hide_index=True)
        st.caption("🟢 User Input | 🔵 Yahoo / Computed | ⬜ Default")

    st.subheader("Forecast Income Statement")
    st.table(show_forecast(res["df_is"]))

    st.subheader("Forecast Balance Sheet")
    st.table(show_forecast(res["df_bs"]))

    st.subheader("Forecast Cash Flow")
    st.table(show_forecast(res["df_cf"]))

    st.subheader("DCF Bridge")
    st.table(pretty_df(res["bridge"], decimals=2))

    st.subheader("Comps (Peers)")
    if res.get("comps_df") is not None and not res["comps_df"].empty:
        st.table(pretty_df(res["comps_df"], decimals=2))
    else:
        st.info("No comps data available.")

    st.subheader("Comps Valuation")
    if res.get("comp_val_df") is not None and not res["comp_val_df"].empty:
        st.table(pretty_df(res["comp_val_df"], decimals=2))

    st.subheader("Sensitivity Table (Value per share)")
    st.dataframe(res["sens_table"], use_container_width=True)

    st.subheader("Sensitivity Heatmap")
    st.pyplot(res["heatmap_fig"])

    # Peer Multiples — built fresh in app.py
    st.subheader("Peer Multiples")
    if res.get("comps_df") is not None and not res["comps_df"].empty:
        mfig = plot_comps_multiples(res["comps_df"], ticker)
        if mfig is not None:
            st.pyplot(mfig)
            plt.close(mfig)
        else:
            st.info("Not enough peer data to plot multiples.")
    else:
        st.info("No peer data available.")

    # Scenario Impact Bridge — built fresh in app.py
    st.subheader("Scenario Impact on Value per Share")
    st.caption(
        "Each bar = VPS change from shocking that driver alone. "
        "Green = value-additive. Red = value-destructive."
    )
    base_inputs_for_plot = res.get("base_inputs")
    base_wacc_for_plot   = res.get("base_wacc")
    base_g_for_plot      = res.get("base_g")
    vps_for_plot         = res.get("vps")

    if all(x is not None for x in [base_inputs_for_plot, base_wacc_for_plot,
                                    base_g_for_plot, vps_for_plot]):
        sfig = plot_valuation_bridge(
            base_inputs_for_plot, base_wacc_for_plot,
            base_g_for_plot, vps_for_plot, ticker
        )
        if sfig is not None:
            st.pyplot(sfig)
            plt.close(sfig)
        else:
            st.info("Scenario impact chart unavailable.")
    else:
        st.info("Scenario impact chart unavailable (base_inputs not in result).")

    st.subheader("Scenarios")
    st.table(pretty_df(res["scen_df"], decimals=2))
