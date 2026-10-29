import io
import re
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
    PageBreak
)

st.set_page_config(
    page_title="Hispanic Heritage Historical Data Explorer",
    page_icon="🌎",
    layout="wide",
)

# Required countries from the project brief, plus Costa Rica for the focused crime-rate investigation.
COUNTRIES = {
    "Mexico": "MEX",
    "Brazil": "BRA",
    "Argentina": "ARG",
    "Colombia": "COL",
    "Chile": "CHL",
    "Peru": "PER",
    "Venezuela": "VEN",
    "Ecuador": "ECU",
    "Uruguay": "URY",
    "Costa Rica": "CRI",
}

INDICATORS = {
    "Intentional homicide rate": {
        "code": "VC.IHR.PSRC.P5",
        "units": "intentional homicides per 100,000 people",
        "description": "Intentional homicides per 100,000 people, as reported by the World Bank/UNODC series.",
        "minimum": 0,
    },
    "Life expectancy at birth": {
        "code": "SP.DYN.LE00.IN",
        "units": "years",
        "description": "Life expectancy at birth, total population.",
        "minimum": 0,
    },
    "Unemployment rate (national estimate)": {
        "code": "SL.UEM.TOTL.NE.ZS",
        "units": "% of total labor force",
        "description": "Unemployment as a percentage of the total labor force, national estimate. National methods can differ.",
        "minimum": 0,
    },
    "GDP per capita": {
        "code": "NY.GDP.PCAP.CD",
        "units": "current US dollars per person",
        "description": "Gross domestic product divided by midyear population, in current US dollars; not household income or average wealth.",
        "minimum": 0,
    },
    "Crude birth rate": {
        "code": "SP.DYN.CBRT.IN",
        "units": "births per 1,000 people",
        "description": "Annual number of births per 1,000 people; this is a rate, not a birth count.",
        "minimum": 0,
    },
    "Net migration": {
        "code": "SM.POP.NETM",
        "units": "people",
        "description": "Net migration: immigrants minus emigrants for the source period; not a count of people leaving.",
        "minimum": None,
    },
    "Consumer price inflation": {
        "code": "FP.CPI.TOTL.ZG",
        "units": "% annual change",
        "description": "Annual percentage change in consumer prices.",
        "minimum": None,
    },
    "Central government debt": {
        "code": "GC.DOD.TOTL.GD.ZS",
        "units": "% of GDP",
        "description": "Central government debt as a percentage of GDP; not all public-sector debt.",
        "minimum": 0,
    },
    "Total reserves, including gold": {
        "code": "FI.RES.TOTL.CD",
        "units": "current US dollars",
        "description": "Total reserves including gold, in current US dollars.",
        "minimum": 0,
    },
    "External balance: goods and services": {
        "code": "NE.RSB.GNFS.ZS",
        "units": "% of GDP",
        "description": "External balance on goods and services as a percentage of GDP; not the current account.",
        "minimum": None,
    },
}

WORLD_BANK_ROOT = "https://api.worldbank.org/v2/country"


def fetch_world_bank(country_code, indicator_code, first_year, last_year):
    """Fetch authentic World Bank observations. Returns a tidy dataframe and retrieval timestamp."""
    url = (
        f"{WORLD_BANK_ROOT}/{country_code}/indicator/{indicator_code}"
        f"?date={first_year}:{last_year}&format=json&per_page=200"
    )
    response = requests.get(
        url,
        timeout=20,
        headers={"User-Agent": "Hispanic-Heritage-Data-Explorer/1.0"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        raise ValueError("The World Bank API returned an unexpected response format.")
    rows = []
    for item in payload[1]:
        if not isinstance(item, dict):
            continue
        year = item.get("date")
        value = item.get("value")
        try:
            year = int(year)
        except (TypeError, ValueError):
            continue
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            rows.append({"Year": year, "Value": value})
    df = pd.DataFrame(rows, columns=["Year", "Value"])
    if not df.empty:
        df = df.drop_duplicates("Year", keep="last").sort_values("Year").reset_index(drop=True)
    return df, datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"), url


def fit_polynomial(df, degree):
    if len(df) < degree + 1:
        raise ValueError(f"Degree {degree} needs at least {degree + 1} distinct valid years; only {len(df)} are available.")
    years = df["Year"].to_numpy(dtype=float)
    values = df["Value"].to_numpy(dtype=float)
    base_year = int(np.min(years))
    t = years - base_year
    coeffs = np.polynomial.polynomial.polyfit(t, values, degree)
    fitted = np.polynomial.polynomial.polyval(t, coeffs)
    residuals = values - fitted
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mae = float(np.mean(np.abs(residuals)))
    return {
        "base_year": base_year,
        "coeffs": coeffs,
        "fitted": fitted,
        "residuals": residuals,
        "r2": r_squared,
        "mae": mae,
        "years": years,
        "values": values,
        "min_year": int(np.min(years)),
        "max_year": int(np.max(years)),
        "mean": float(np.mean(values)),
    }


def predict(model, years):
    years_arr = np.asarray(years, dtype=float)
    return np.polynomial.polynomial.polyval(years_arr - model["base_year"], model["coeffs"])


def format_equation(coeffs, base_year, units):
    parts = []
    for power, coefficient in enumerate(coeffs):
        if abs(coefficient) < 1e-13:
            continue
        coeff_text = f"{coefficient:.6g}"
        if power == 0:
            term = coeff_text
        elif power == 1:
            term = f"{coeff_text}·t"
        else:
            term = f"{coeff_text}·t^{power}"
        parts.append(term)
    equation = " + ".join(parts).replace("+ -", "- ")
    if not equation:
        equation = "0"
    return f"ŷ = {equation}, where t = year − {base_year} ({units})"


def classify_year(year, model, observed_years):
    if year in observed_years:
        return "Fitted value at an observed year"
    if model["min_year"] <= year <= model["max_year"]:
        return "Within historical range (interpolation/model estimate)"
    return "Outside historical range (extrapolation)"


def make_chart(df, model, indicator_label, units, country_name, horizon, sample_step, comparisons=None, printer=False):
    fig, ax = plt.subplots(figsize=(10, 5.2) if not printer else (10, 4.8))
    ax.scatter(df["Year"], df["Value"], label=f"{country_name}: observed data", s=35, zorder=4)
    min_year, max_year = model["min_year"], model["max_year"]
    curve_end = min(2100, max_year + int(horizon))
    hist_x = np.linspace(min_year, max_year, max(150, (max_year - min_year + 1) * 3))
    if curve_end > max_year:
        extra_x = np.linspace(max_year, curve_end, max(30, (curve_end - max_year + 1) * 3))
    else:
        extra_x = np.array([])
    ax.plot(hist_x, predict(model, hist_x), label="Polynomial fit (historical range)", linewidth=2)
    if len(extra_x):
        ax.plot(extra_x, predict(model, extra_x), linestyle="--", linewidth=2, label="Extrapolated model segment")
        ax.axvline(max_year, linestyle=":", linewidth=1.5, label=f"Last observed year: {max_year}")
    else:
        ax.axvline(max_year, linestyle=":", linewidth=1.2, label=f"Last observed year: {max_year}")
    sampled_years = np.arange(min_year, curve_end + 1, int(sample_step))
    if len(sampled_years):
        ax.scatter(sampled_years, predict(model, sampled_years), marker="x", s=22, label=f"Sampled model markers (every {sample_step} year(s))", zorder=3)
    if comparisons:
        for name, cdf in comparisons.items():
            if cdf is not None and not cdf.empty:
                ax.plot(cdf["Year"], cdf["Value"], marker="o", markersize=3, linewidth=1.3, label=f"{name}: observed")
    ax.set_title(f"{indicator_label} — {country_name}")
    ax.set_xlabel("Year")
    ax.set_ylabel(units)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def xlsx_bytes(summary, data_df, result_df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        data_df.to_excel(writer, sheet_name="Working Data", index=False)
        if result_df is not None and not result_df.empty:
            result_df.to_excel(writer, sheet_name="Calculated Years", index=False)
        workbook = writer.book
        header_fmt = workbook.add_format({"bold": True, "text_wrap": True, "valign": "top", "border": 1})
        for sheet_name, worksheet in writer.sheets.items():
            worksheet.freeze_panes(1, 0)
            worksheet.set_row(0, 30, header_fmt)
            worksheet.set_column(0, 0, 24)
            worksheet.set_column(1, 20, 22)
    output.seek(0)
    return output.getvalue()


def pdf_bytes(summary, data_df, result_df, chart_image_bytes, findings):
    output = io.BytesIO()
    doc = SimpleDocTemplate(
        output, pagesize=landscape(letter), rightMargin=0.45 * inch,
        leftMargin=0.45 * inch, topMargin=0.45 * inch, bottomMargin=0.45 * inch
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Hispanic Heritage Historical Data Explorer", styles["Title"]),
        Paragraph("Historical indicators, polynomial models, and limitations", styles["Heading2"]),
        Spacer(1, 8),
    ]
    summary_rows = [["Field", "Value"]] + [[str(k), str(v)] for k, v in summary.items()]
    table = Table(summary_rows, colWidths=[2.0 * inch, 7.8 * inch], repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F5F9")]),
    ]))
    story += [table, Spacer(1, 10), Paragraph("Chart", styles["Heading2"])]
    if chart_image_bytes:
        image_buffer = io.BytesIO(chart_image_bytes)
        story.append(RLImage(image_buffer, width=9.7 * inch, height=4.6 * inch))
    story += [Paragraph("Supported findings and limitations", styles["Heading2"])]
    for finding in findings:
        story.append(Paragraph("• " + str(finding), styles["BodyText"]))
        story.append(Spacer(1, 3))
    story += [Spacer(1, 8), Paragraph("Working data preview", styles["Heading2"])]
    preview = data_df.head(20).copy()
    rows = [list(preview.columns)] + preview.astype(str).values.tolist()
    preview_table = Table(rows, repeatRows=1)
    preview_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(preview_table)
    if result_df is not None and not result_df.empty:
        story += [PageBreak(), Paragraph("Calculated-year results", styles["Heading2"])]
        res_rows = [list(result_df.columns)] + result_df.astype(str).values.tolist()
        res_table = Table(res_rows, repeatRows=1)
        res_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
        ]))
        story.append(res_table)
    doc.build(story)
    output.seek(0)
    return output.getvalue()


st.title("🌎 Hispanic Heritage Historical Data Explorer")
st.caption("Explore genuine World Bank historical data, fit polynomial models, compare countries, and export your analysis.")
st.info(
    "Featured investigation: How has Costa Rica's intentional homicide rate changed over time? "
    "Costa Rica is added as an extra country for this crime-rate focus; the nine countries required by the project menu remain available. "
    "Brazil is included as a Latin American comparison, not as a Spanish-speaking Hispanic country."
)

with st.sidebar:
    st.header("Explore the data")
    country_name = st.selectbox("Country", list(COUNTRIES.keys()), index=list(COUNTRIES.keys()).index("Costa Rica"))
    indicator_label = st.selectbox(
        "Indicator",
        list(INDICATORS.keys()),
        index=list(INDICATORS.keys()).index("Intentional homicide rate"),
    )
    current_year = datetime.now().year
    first_year = st.number_input("First year to request", min_value=1900, max_value=current_year - 1, value=max(1950, current_year - 70), step=1)
    last_year = st.number_input("Last year to request", min_value=1900, max_value=current_year - 1, value=current_year - 1, step=1)
    degree = st.slider("Polynomial degree", min_value=3, max_value=8, value=3)
    sample_step = st.slider("Model marker sample increment (years)", min_value=1, max_value=10, value=1)
    horizon = st.slider("Forward display horizon (years)", min_value=0, max_value=50, value=5)
    comparison_names = st.multiselect(
        "Compare same indicator with",
        [name for name in COUNTRIES if name != country_name],
        default=["Mexico", "Colombia"] if country_name == "Costa Rica" else [],
    )
    printer_view = st.toggle("Printer-friendly report view", value=False)

if first_year > last_year:
    st.error("The first year must be less than or equal to the last year.")
    st.stop()

indicator = INDICATORS[indicator_label]
source_url = f"https://api.worldbank.org/v2/country/{COUNTRIES[country_name]}/indicator/{indicator['code']}"
st.markdown(f"**Definition:** {indicator['description']}")
st.markdown(
    f"**Indicator code:** `{indicator['code']}` · **Units:** {indicator['units']} · "
    f"**Source:** [World Bank API]({source_url})"
)

cache_key = f"source_{COUNTRIES[country_name]}_{indicator['code']}_{int(first_year)}_{int(last_year)}"
try:
    with st.spinner("Retrieving official World Bank observations..."):
        if cache_key not in st.session_state:
            fetched, retrieved_at, api_url = fetch_world_bank(
                COUNTRIES[country_name], indicator["code"], int(first_year), int(last_year)
            )
            st.session_state[cache_key] = {
                "data": fetched.copy(), "retrieved_at": retrieved_at, "url": api_url
            }
        source_df = st.session_state[cache_key]["data"].copy()
        retrieved_at = st.session_state[cache_key]["retrieved_at"]
        api_url = st.session_state[cache_key]["url"]
except requests.exceptions.Timeout:
    st.error("The World Bank API timed out. Try again in a moment. No substitute data were created.")
    st.stop()
except (requests.exceptions.RequestException, ValueError, KeyError) as exc:
    st.error(f"Could not retrieve or read the World Bank response: {exc}. No substitute data were created.")
    st.stop()

if source_df.empty:
    st.warning("The API returned no non-missing observations for this country, indicator, and period. Try a different indicator or year window.")
    st.caption(f"Request: {api_url}")
    st.stop()

st.session_state.setdefault("working_source_key", None)
if st.session_state.working_source_key != cache_key:
    st.session_state.working_df = source_df.copy()
    st.session_state.working_source_key = cache_key
    st.session_state.calculated_results = pd.DataFrame()

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Available observations", len(source_df))
col_b.metric("First observed year", int(source_df["Year"].min()))
col_c.metric("Last observed year", int(source_df["Year"].max()))
col_d.metric("Retrieved (UTC)", retrieved_at)

st.caption(f"Actual coverage: {int(source_df['Year'].min())}–{int(source_df['Year'].max())}. Missing years are not fabricated. API request: {api_url}")

st.subheader("1. Review and edit the working data")
st.write("Edit the table below to test a scenario. The original source values remain available through **Reset to source**.")
edit_col, reset_col = st.columns([1, 5])
with edit_col:
    if st.button("Reset to source"):
        st.session_state.working_df = source_df.copy()
        st.session_state.calculated_results = pd.DataFrame()
        st.rerun()

edited_df = st.data_editor(
    st.session_state.working_df,
    num_rows="dynamic",
    use_container_width=True,
    key=f"editor_{cache_key}",
    column_config={
        "Year": st.column_config.NumberColumn("Year", step=1, format="%d"),
        "Value": st.column_config.NumberColumn("Value", format="%.6g"),
    },
)
validation_messages = []
clean_df = edited_df.copy()
clean_df["Year"] = pd.to_numeric(clean_df["Year"], errors="coerce")
clean_df["Value"] = pd.to_numeric(clean_df["Value"], errors="coerce")
invalid_mask = (
    clean_df["Year"].isna()
    | clean_df["Value"].isna()
    | ~np.isfinite(clean_df["Year"].fillna(0))
    | ~np.isfinite(clean_df["Value"].fillna(0))
    | ((clean_df["Year"] % 1) != 0)
)
if invalid_mask.any():
    validation_messages.append(f"Rejected {int(invalid_mask.sum())} row(s) with missing/non-numeric values or non-integer years.")
    clean_df = clean_df.loc[~invalid_mask].copy()
clean_df["Year"] = clean_df["Year"].astype(int)
duplicate_mask = clean_df["Year"].duplicated(keep=False)
if duplicate_mask.any():
    dup_years = sorted(clean_df.loc[duplicate_mask, "Year"].unique().tolist())
    validation_messages.append(f"Duplicate years are not allowed. Duplicate year(s): {dup_years}. Keep one row for each year.")
    clean_df = clean_df.loc[~duplicate_mask].copy()
if (clean_df["Year"] < 1900).any() or (clean_df["Year"] > current_year - 1).any():
    validation_messages.append("Rejected rows outside the valid historical year range (1900 through last year).")
    clean_df = clean_df[(clean_df["Year"] >= 1900) & (clean_df["Year"] <= current_year - 1)]
if indicator["minimum"] is not None and (clean_df["Value"] < indicator["minimum"]).any():
    validation_messages.append(f"Rejected values below {indicator['minimum']} for this indicator.")
    clean_df = clean_df[clean_df["Value"] >= indicator["minimum"]]
clean_df = clean_df.sort_values("Year").reset_index(drop=True)
for message in validation_messages:
    st.warning(message)
is_edited = not (
    len(clean_df) == len(source_df)
    and clean_df.equals(source_df.sort_values("Year").reset_index(drop=True))
)
if is_edited:
    st.warning("User-modified scenario: the working table differs from the original source observations.")
else:
    st.success("Working table matches the retrieved source observations.")

if len(clean_df) < degree + 1:
    st.error(f"Not enough valid distinct years to fit degree {degree}. Need at least {degree + 1}; currently have {len(clean_df)}.")
    st.stop()

try:
    model = fit_polynomial(clean_df, degree)
except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
    st.error(f"Could not fit the selected polynomial: {exc}")
    st.stop()

equation = format_equation(model["coeffs"], model["base_year"], indicator["units"])
summary = {
    "Country": country_name,
    "Country code": COUNTRIES[country_name],
    "Indicator": indicator_label,
    "Indicator code": indicator["code"],
    "Units": indicator["units"],
    "Definition": indicator["description"],
    "Source": api_url,
    "Retrieved (UTC)": retrieved_at,
    "Source coverage requested": f"{int(first_year)}–{int(last_year)}",
    "Actual fitted range": f"{model['min_year']}–{model['max_year']}",
    "Observations in working model": len(clean_df),
    "Data status": "User-modified scenario" if is_edited else "Source data",
    "Polynomial degree": degree,
    "Base year": model["base_year"],
    "Equation": equation,
    "Mean": f"{model['mean']:.8g} {indicator['units']}",
    "In-sample R-squared": f"{model['r2']:.6g}",
    "In-sample MAE": f"{model['mae']:.6g} {indicator['units']}",
}
st.subheader("2. Polynomial model")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Observations", len(clean_df))
metric2.metric("Mean", f"{model['mean']:.5g}")
metric3.metric("In-sample R²", f"{model['r2']:.5f}" if np.isfinite(model["r2"]) else "Undefined")
metric4.metric("In-sample MAE", f"{model['mae']:.5g}")
st.code(equation)
st.caption(
    f"Centered time variable: t = year − {model['base_year']}. "
    "R² and MAE describe how the polynomial fits these same observations; they do not prove forecast accuracy."
)

# Chronological holdout: fit on earlier data, test on the last 20% only when enough observations exist.
if len(clean_df) >= max(degree + 4, 10):
    ordered = clean_df.sort_values("Year").reset_index(drop=True)
    holdout_n = max(3, int(math.ceil(len(ordered) * 0.2)))
    train = ordered.iloc[:-holdout_n]
    test = ordered.iloc[-holdout_n:]
    if len(train) >= degree + 1:
        try:
            holdout_model = fit_polynomial(train, degree)
            holdout_predictions = predict(holdout_model, test["Year"].to_numpy())
            holdout_mae = float(np.mean(np.abs(test["Value"].to_numpy() - holdout_predictions)))
            st.write(f"**Chronological holdout MAE:** {holdout_mae:.6g} {indicator['units']} on {len(test)} later observation(s). This is separate from the full-series in-sample MAE.")
        except Exception:
            st.caption("A chronological holdout score could not be calculated for this dataset and degree.")
else:
    st.caption("Not enough observations for a meaningful chronological holdout at this degree; no tested forecast skill is claimed.")

st.subheader("3. Graph and country comparison")
comparison_data = {}
comparison_errors = []
for comparison_name in comparison_names:
    try:
        comp_df, _, _ = fetch_world_bank(
            COUNTRIES[comparison_name], indicator["code"], int(first_year), int(last_year)
        )
        comparison_data[comparison_name] = comp_df
    except Exception as exc:
        comparison_errors.append(f"{comparison_name}: {exc}")
for err in comparison_errors:
    st.warning(f"Comparison data unavailable for {err}")
fig = make_chart(
    clean_df, model, indicator_label, indicator["units"], country_name,
    horizon, sample_step, comparison_data, printer_view
)
st.pyplot(fig, use_container_width=True)
chart_buffer = io.BytesIO()
fig.savefig(chart_buffer, format="png", dpi=160, bbox_inches="tight")
chart_image = chart_buffer.getvalue()
plt.close(fig)

if comparison_data:
    st.caption("Comparison lines show actual observations for the same indicator and units. Coverage may differ; no other country's data are substituted.")

st.subheader("4. Calculate one year or a batch of years")
tab_single, tab_batch = st.tabs(["Single year", "Batch years"])
observed_years = set(clean_df["Year"].tolist())
with tab_single:
    with st.form("single_year_form"):
        single_year = st.number_input("Year to estimate (1950–2100)", min_value=1950, max_value=2100, value=min(max(model["max_year"], 1950), 2100), step=1)
        calculate_single = st.form_submit_button("Calculate value")
    if calculate_single:
        year_int = int(single_year)
        value = float(predict(model, [year_int])[0])
        result = {
            "Year": year_int,
            "Estimate": value,
            "Units": indicator["units"],
            "Classification": classify_year(year_int, model, observed_years),
            "Observed value (if available)": float(clean_df.loc[clean_df["Year"] == year_int, "Value"].iloc[0]) if year_int in observed_years else np.nan,
        }
        st.session_state.single_result = result
    if st.session_state.get("single_result"):
        result = st.session_state.single_result
        st.write(f"**{result['Year']}:** {result['Estimate']:.8g} {result['Units']}")
        st.write(f"**Type:** {result['Classification']}")
        if pd.notna(result["Observed value (if available)"]):
            st.write(f"**Observed source/working value:** {result['Observed value (if available)']:.8g} {result['Units']}")
        st.caption(f"Fitted data range: {model['min_year']}–{model['max_year']}. The model estimate is not a guarantee of the real-world value.")
with tab_batch:
    batch_mode = st.radio("Batch input method", ["Year range", "Custom years"], horizontal=True)
    valid_years = []
    invalid_entries = []
    if batch_mode == "Year range":
        b1, b2, b3 = st.columns(3)
        with b1:
            range_start = st.number_input("Start year", 1950, 2100, max(1950, model["min_year"]), 1, key="range_start")
        with b2:
            range_end = st.number_input("End year", 1950, 2100, min(2100, max(model["max_year"], max(1950, model["min_year"])) + 5), 1, key="range_end")
        with b3:
            range_step = st.number_input("Step", 1, 10, 1, 1, key="range_step")
        if range_end <= range_start:
            st.warning("End year must be greater than start year.")
        else:
            valid_years = list(range(int(range_start), int(range_end) + 1, int(range_step)))
            st.caption("The range includes the end year only when the selected step lands on it.")
    else:
        custom_years_text = st.text_input("Comma-separated years", value=f"{model['min_year']},{model['max_year']},{min(2100, model['max_year'] + 5)}")
        seen = set()
        for token in custom_years_text.split(","):
            token = token.strip()
            if not token:
                continue
            try:
                year = int(token)
                if year < 1950 or year > 2100:
                    invalid_entries.append(f"{token} (outside 1950–2100)")
                elif year in seen:
                    invalid_entries.append(f"{token} (duplicate; duplicate removed)")
                else:
                    valid_years.append(year)
                    seen.add(year)
            except ValueError:
                invalid_entries.append(f"{token} (not an integer year)")
    if invalid_entries:
        st.warning("Input feedback: " + "; ".join(invalid_entries))
    st.write("**Valid years to calculate:**", ", ".join(map(str, valid_years)) if valid_years else "None")
    if st.button("Calculate batch", disabled=not valid_years, key="calculate_batch"):
        batch_rows = []
        for year in valid_years:
            batch_rows.append({
                "Year": int(year),
                "Estimate": float(predict(model, [year])[0]),
                "Units": indicator["units"],
                "Classification": classify_year(int(year), model, observed_years),
            })
        st.session_state.calculated_results = pd.DataFrame(batch_rows)
    results_df = st.session_state.get("calculated_results", pd.DataFrame())
    if not results_df.empty:
        st.dataframe(results_df, use_container_width=True)
        st.caption("Batch estimates use the same fitted model as single-year estimates. No statistically validated prediction intervals are provided.")

st.subheader("5. Findings and limitations")
findings = [
    f"The observed {indicator_label.lower()} series for {country_name} contains {len(clean_df)} valid observations from {model['min_year']} to {model['max_year']}.",
    f"The degree-{degree} polynomial has an in-sample MAE of {model['mae']:.5g} {indicator['units']} and an in-sample R-squared of {model['r2']:.5g}." if np.isfinite(model["r2"]) else f"The degree-{degree} polynomial has an in-sample MAE of {model['mae']:.5g} {indicator['units']}; R-squared is undefined for a constant series.",
    "The fitted polynomial describes the supplied annual observations; a high in-sample R-squared does not establish reliable future predictions.",
    "Data gaps and changes in collection, reporting, definitions, or comparability may affect interpretation. Check official metadata and historical context before making causal claims.",
]
for finding in findings:
    st.markdown(f"- {finding}")
st.warning(
    "Uncertainty note: this baseline reports residual-based MAE and a holdout score when possible. "
    "It does not provide validated 95% prediction intervals. Polynomial extrapolations, especially distant ones, can become implausible."
)

st.subheader("6. Download your current analysis")
results_df = st.session_state.get("calculated_results", pd.DataFrame())
export_data = clean_df.copy()
export_data["Data status"] = "User-modified scenario" if is_edited else "Source data"
export_summary = dict(summary)
csv_data = csv_bytes(export_data)
excel_data = xlsx_bytes(export_summary, export_data, results_df)
pdf_data = pdf_bytes(export_summary, export_data, results_df, chart_image, findings)
slug = re.sub(r"[^a-z0-9]+", "-", f"{country_name}-{indicator_label}".lower()).strip("-")
d1, d2, d3 = st.columns(3)
d1.download_button("Download CSV data", data=csv_data, file_name=f"{slug}-data.csv", mime="text/csv", use_container_width=True)
d2.download_button("Download Excel workbook", data=excel_data, file_name=f"{slug}-report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
d3.download_button("Download PDF report", data=pdf_data, file_name=f"{slug}-report.pdf", mime="application/pdf", use_container_width=True)

with st.expander("Sources, method, and research notes"):
    st.markdown("- **Primary data:** [World Bank Indicators API](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries)")
    st.markdown(f"- **Selected series:** [World Bank indicator page]({source_url})")
    st.markdown("- The homicide indicator is an official international series, but reporting coverage and revisions can vary. Confirm country-specific details before making historical claims.")
    st.markdown("- Costa Rica is an extra option added for the focused investigation; the nine countries named in the assignment remain available.")
    st.markdown("- Brazil is included as a Latin American comparison, not as a Spanish-speaking Hispanic country.")
    st.markdown("- Polynomial models are descriptive mathematical approximations. Correlation and trend do not establish causes.")
    st.markdown("- For the project write-up, add at least two credible historical/cultural sources and explain a specific heritage connection supported by those sources.")

st.caption("Built for classroom investigation. Verify source values, calculations, exports, and deployed behavior before presenting.")
