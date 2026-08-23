"""EduPro Course Demand & Revenue Intelligence - Streamlit application.

Six modules:
  1. Portfolio Overview            - what the catalogue actually did in 2025
  2. Course Demand Predictor       - score a course that does not exist yet
  3. Revenue Forecast              - course, category and platform revenue outlook
  4. Feature Importance Explorer   - what moves demand, and by how much
  5. Category Demand Comparison    - where attention and money diverge
  6. Pricing Scenario Simulator    - the lever the evidence actually supports
  7. Model Reliability             - the noise floor, stated plainly

Every artefact is precomputed by ``python -m src.run_pipeline``; this app only
reads and renders, so it starts instantly.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "reports" / "tables"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="EduPro Demand & Revenue Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Palette - one accent ramp, one neutral ramp, semantic colours for up/down.
# --------------------------------------------------------------------------- #
ACCENT = "#2563eb"
ACCENT_2 = "#0d9488"
WARN = "#d97706"
NEG = "#dc2626"
MUTED = "#7b8794"
SEQ = ["#2563eb", "#0d9488", "#7c3aed", "#d97706", "#dc2626", "#0891b2",
       "#65a30d", "#db2777", "#4f46e5", "#ca8a04", "#059669", "#9333ea"]

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1400px;}
      div[data-testid="stMetricValue"] {font-size: 1.55rem;}
      .callout {border-left: 4px solid #2563eb; background: rgba(37,99,235,.07);
                padding: .85rem 1.1rem; border-radius: 6px; margin: .6rem 0 1.1rem 0;}
      .callout-warn {border-left-color:#d97706; background: rgba(217,119,6,.08);}
      .callout-neg {border-left-color:#dc2626; background: rgba(220,38,38,.08);}
      .small {font-size:.86rem; color:#7b8794;}
    </style>
    """,
    unsafe_allow_html=True,
)


def callout(text: str, kind: str = "") -> None:
    cls = "callout" + (f" callout-{kind}" if kind else "")
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_all() -> dict:
    def csv(p, **kw):
        return pd.read_csv(p, **kw) if p.exists() else pd.DataFrame()

    d = {
        "course": csv(PROCESSED / "course_features.csv"),
        "panel": csv(PROCESSED / "panel_course_month.csv", parse_dates=["Month"]),
        "panel_feat": csv(PROCESSED / "panel_features.csv", parse_dates=["Month"]),
        "audit": csv(TABLES / "01_data_quality_audit.csv"),
        "screen": csv(TABLES / "02_feature_screen.csv"),
        "gap": csv(TABLES / "03_monetisation_gap.csv"),
        "diag": csv(TABLES / "04_predictability_diagnostics.csv"),
        "cs_models": csv(TABLES / "05_cross_sectional_model_comparison.csv"),
        "panel_models": csv(TABLES / "06_panel_model_comparison.csv"),
        "imp_enr": csv(TABLES / "07_importance_enrollment.csv"),
        "imp_rev": csv(TABLES / "08_importance_revenue.csv"),
        "coef_enr": csv(TABLES / "09_ridge_coefficients_enrollment.csv"),
        "cat_rev": csv(TABLES / "10_category_revenue.csv"),
    }
    # Two course names repeat across categories (e.g. "Deep Learning" exists in both
    # Machine Learning and AI). Plotly stacks bars sharing a categorical value, which
    # would silently double them, so every chart axis uses a disambiguated label.
    if not d["course"].empty:
        c = d["course"]
        dupes = c.CourseName.duplicated(keep=False)
        c["DisplayName"] = c.CourseName.where(~dupes, c.CourseName + " (" + c.CourseCategory + ")")

    summary_path = ROOT / "reports" / "pipeline_summary.json"
    d["summary"] = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    return d


@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    out = {}
    for key, fn in [("enrollment", "enrollment_model.joblib"), ("revenue", "revenue_model.joblib")]:
        p = MODELS / fn
        if p.exists():
            out[key] = joblib.load(p)
    return out


D = load_all()
M = load_models()

if D["course"].empty:
    st.error("Processed data not found. Run `python -m src.run_pipeline` first.")
    st.stop()

course = D["course"]
panel = D["panel"]


def diag_value(block: str, metric: str, default=np.nan):
    df = D["diag"]
    if df.empty:
        return default
    hit = df[(df.Diagnostic == block) & (df.Metric == metric)]
    if hit.empty:
        return default
    v = hit.Value.iloc[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("EduPro Intelligence")
st.sidebar.caption("Predictive modelling for course demand and revenue forecasting")

PAGES = [
    "Portfolio Overview",
    "Course Demand Predictor",
    "Revenue Forecast",
    "Feature Importance Explorer",
    "Category Demand Comparison",
    "Pricing Scenario Simulator",
    "Model Reliability",
]
page = st.sidebar.radio("Module", PAGES, label_visibility="collapsed")

st.sidebar.divider()
st.sidebar.metric("Courses", f"{len(course):,}")
st.sidebar.metric("Enrollments (2025)", f"{int(course.EnrollmentCount.sum()):,}")
st.sidebar.metric("Revenue (2025)", f"{course.CourseRevenue.sum():,.0f}")
st.sidebar.divider()
st.sidebar.caption(
    "Built from the EduPro Online Platform workbook (Users, Teachers, Courses, "
    "Transactions). All model scores shown are out-of-sample."
)


# =========================================================================== #
# 1. PORTFOLIO OVERVIEW
# =========================================================================== #
if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    st.caption("What the EduPro catalogue actually did across calendar year 2025.")

    total_rev = course.CourseRevenue.sum()
    free = course[course.CourseType == "Free"]
    free_enr_share = free.EnrollmentCount.sum() / course.EnrollmentCount.sum()

    c = st.columns(5)
    c[0].metric("Courses", f"{len(course)}", f"{len(free)} free")
    c[1].metric("Enrollments", f"{int(course.EnrollmentCount.sum()):,}")
    c[2].metric("Revenue", f"{total_rev:,.0f}")
    c[3].metric("Revenue / enrollment", f"{total_rev / course.EnrollmentCount.sum():,.2f}")
    c[4].metric("Unmonetised demand", f"{free_enr_share:.0%}", "of all enrollments", delta_color="off")

    callout(
        f"<b>Headline.</b> Demand is remarkably even across the catalogue: every course drew "
        f"{course.EnrollmentCount.mean():.0f} &plusmn; {course.EnrollmentCount.std():.0f} enrollments, and a "
        f"chi-square test cannot distinguish that spread from random allocation "
        f"(p = {diag_value('uniform_allocation_test', 'p_value'):.2f}). Revenue differences between courses are "
        f"therefore driven almost entirely by <b>price</b>, not by popularity."
    )

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Enrollments by course")
        srt = course.sort_values("EnrollmentCount", ascending=False)
        mu = course.EnrollmentCount.mean()
        sd_null = np.sqrt(course.EnrollmentCount.sum() * (1 / len(course)) * (1 - 1 / len(course)))
        fig = go.Figure()
        fig.add_bar(x=srt.DisplayName, y=srt.EnrollmentCount,
                    marker_color=[ACCENT if t == "Paid" else MUTED for t in srt.CourseType],
                    hovertemplate="<b>%{x}</b><br>Enrollments: %{y}<extra></extra>", name="Enrollments")
        # Bars descend left to right, so the annotation goes top-right where there is space.
        fig.add_hrect(y0=mu - 2 * sd_null, y1=mu + 2 * sd_null, fillcolor=WARN, opacity=0.13,
                      line_width=0, annotation_text="95% band under random allocation",
                      annotation_position="top right", annotation_font_size=11)
        fig.add_hline(y=mu, line_dash="dash", line_color="#1f2933")
        fig.update_layout(height=430, margin=dict(t=48, b=10, l=12, r=0), showlegend=False,
                          xaxis_title=None, yaxis_title="Enrollments",
                          xaxis=dict(showticklabels=False))
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue = paid, grey = free. Every course sits inside the band you would expect "
                   "if the 10,000 enrollments were sprinkled at random.")

    with right:
        st.subheader("Revenue concentration")
        cat = course.groupby("CourseCategory").CourseRevenue.sum().sort_values(ascending=False).reset_index()
        cat["Share"] = cat.CourseRevenue / cat.CourseRevenue.sum()
        fig = px.bar(cat, x="CourseRevenue", y="CourseCategory", orientation="h",
                     color_discrete_sequence=[ACCENT_2], text=cat.Share.map(lambda v: f"{v:.1%}"))
        fig.update_layout(height=430, margin=dict(t=40, b=10, l=0, r=0),
                          yaxis=dict(autorange="reversed", title=None), xaxis_title="Revenue")
        fig.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, width="stretch")
        top3 = cat.head(3)
        st.caption(f"Top three categories ({', '.join(top3.CourseCategory)}) generate "
                   f"{top3.Share.sum():.0%} of all revenue.")

    st.divider()
    st.subheader("Course catalogue")
    show = course[["CourseName", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice",
                   "CourseDuration", "CourseRating", "EnrollmentCount", "CourseRevenue"]].copy()
    f1, f2, f3 = st.columns(3)
    cats = f1.multiselect("Category", sorted(show.CourseCategory.unique()))
    typs = f2.multiselect("Type", sorted(show.CourseType.unique()))
    lvls = f3.multiselect("Level", sorted(show.CourseLevel.unique()))
    if cats:
        show = show[show.CourseCategory.isin(cats)]
    if typs:
        show = show[show.CourseType.isin(typs)]
    if lvls:
        show = show[show.CourseLevel.isin(lvls)]
    st.dataframe(
        show.sort_values("CourseRevenue", ascending=False),
        width="stretch", hide_index=True,
        column_config={
            "CoursePrice": st.column_config.NumberColumn("Price", format="%.2f"),
            "CourseDuration": st.column_config.NumberColumn("Hours", format="%.1f"),
            "CourseRating": st.column_config.NumberColumn("Rating", format="%.2f"),
            "EnrollmentCount": st.column_config.NumberColumn("Enrollments", format="%d"),
            "CourseRevenue": st.column_config.NumberColumn("Revenue", format="%.0f"),
        },
    )


# =========================================================================== #
# 2. COURSE DEMAND PREDICTOR
# =========================================================================== #
elif page == "Course Demand Predictor":
    st.title("Course Demand Predictor")
    st.caption("Score a course before it launches, using course attributes and instructor profile.")

    if "enrollment" not in M:
        st.error("Model artefacts not found. Run `python -m src.run_pipeline`.")
        st.stop()

    mdl = M["enrollment"]
    st.markdown("#### Configure the course")
    a, b, c = st.columns(3)

    with a:
        st.markdown("**Course**")
        cat_in = st.selectbox("Category", sorted(course.CourseCategory.unique()))
        lvl_in = st.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
        typ_in = st.radio("Type", ["Paid", "Free"], horizontal=True)
    with b:
        st.markdown("**Commercials**")
        price_in = st.slider("Price", 0.0, 500.0, 150.0, 5.0, disabled=(typ_in == "Free"))
        if typ_in == "Free":
            price_in = 0.0
        dur_in = st.slider("Duration (hours)", 1.0, 50.0, 25.0, 0.5)
        rate_in = st.slider("Expected course rating", 1.0, 5.0, 3.5, 0.05)
    with c:
        st.markdown("**Instructor**")
        exp_in = st.slider("Years of experience", 1.0, 25.0, 6.0, 0.5)
        trate_in = st.slider("Instructor rating", 1.0, 5.0, 3.5, 0.05)
        match_in = st.slider("Expertise-category match", 0.0, 1.0, 0.4, 0.05,
                             help="Share of the delivery team whose expertise matches the course category.")

    from src.features import (duration_bucket, experience_bucket, price_band,  # noqa: E402
                              rating_tier, DURATION_BUCKETS, EXPERIENCE_BUCKETS,
                              LEVEL_ORDER, PRICE_BANDS, RATING_TIERS)

    row = {
        "CoursePrice": price_in, "CourseDuration": dur_in, "CourseRating": rate_in,
        "InstructorAvgRating": trate_in, "InstructorAvgExperience": exp_in,
        "InstructorExpertiseMatch": match_in,
        "PricePerHour": price_in / dur_in if dur_in else 0.0,
        "QualitySignal": rate_in * trate_in,
        "PriceBandOrd": PRICE_BANDS.index(price_band(price_in)),
        "DurationBucketOrd": DURATION_BUCKETS.index(duration_bucket(dur_in)),
        "RatingTierOrd": RATING_TIERS.index(rating_tier(rate_in)),
        "CourseLevelOrd": LEVEL_ORDER[lvl_in],
        "ExperienceBucketOrd": EXPERIENCE_BUCKETS.index(experience_bucket(exp_in)),
        "CourseCategory": cat_in, "CourseType": typ_in, "CourseLevel": lvl_in,
    }
    X = pd.DataFrame([row])[mdl["numeric"] + mdl["categorical"]]
    point = float(mdl["pipeline"].predict(X)[0])
    q = mdl["residual_q"]
    lo80, hi80 = point + q[0.10], point + q[0.90]
    lo95, hi95 = point + q[0.025], point + q[0.975]
    rev_point = point * price_in

    st.divider()
    st.markdown("#### Forecast")
    k = st.columns(4)
    k[0].metric("Predicted annual enrollments", f"{point:,.0f}")
    k[1].metric("80% interval", f"{lo80:,.0f} - {hi80:,.0f}")
    k[2].metric("95% interval", f"{lo95:,.0f} - {hi95:,.0f}")
    k[3].metric("Predicted annual revenue", f"{rev_point:,.0f}")

    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo95, x1=hi95, y0=0.32, y1=0.68,
                  fillcolor=ACCENT, opacity=0.14, line_width=0)
    fig.add_shape(type="rect", x0=lo80, x1=hi80, y0=0.38, y1=0.62,
                  fillcolor=ACCENT, opacity=0.30, line_width=0)
    fig.add_trace(go.Scatter(x=[point], y=[0.5], mode="markers+text",
                             marker=dict(size=17, color=ACCENT, line=dict(width=2, color="white")),
                             text=[f"{point:,.0f}"], textposition="top center", showlegend=False))
    fig.add_vline(x=course.EnrollmentCount.mean(), line_dash="dash", line_color=MUTED,
                  annotation_text=f"Catalogue average ({course.EnrollmentCount.mean():.0f})",
                  annotation_position="bottom right")
    fig.update_layout(height=190, margin=dict(t=30, b=30, l=10, r=10),
                      xaxis_title="Annual enrollments", yaxis=dict(visible=False, range=[0, 1]))
    st.plotly_chart(fig, width="stretch")

    spread = (hi80 - lo80) / point
    callout(
        f"<b>Read the interval, not the point.</b> The 80% interval spans "
        f"{hi80 - lo80:,.0f} enrollments ({spread:.0%} of the point estimate), and the model's "
        f"out-of-sample R&sup2; is {mdl['cv_r2']:.3f}. Course attributes carry almost no information "
        f"about demand in this dataset, so the honest forecast for <i>any</i> configuration is "
        f"'about the catalogue average'. Use this module to size a launch, not to rank candidates. "
        f"See <b>Model Reliability</b> for why.",
        "warn",
    )

    st.markdown("#### Sensitivity: what happens if you change one input")
    sens_var = st.selectbox("Sweep", ["Price", "Duration", "Course rating", "Instructor rating",
                                      "Instructor experience"])
    grids = {
        "Price": ("CoursePrice", np.linspace(0, 500, 40)),
        "Duration": ("CourseDuration", np.linspace(1, 50, 40)),
        "Course rating": ("CourseRating", np.linspace(1, 5, 40)),
        "Instructor rating": ("InstructorAvgRating", np.linspace(1, 5, 40)),
        "Instructor experience": ("InstructorAvgExperience", np.linspace(1, 25, 40)),
    }
    col, grid = grids[sens_var]
    rows = []
    for v in grid:
        r = dict(row)
        r[col] = v
        if col == "CoursePrice":
            r["PricePerHour"] = v / dur_in if dur_in else 0.0
            r["PriceBandOrd"] = PRICE_BANDS.index(price_band(v))
            r["CourseType"] = "Free" if v == 0 else "Paid"
        if col == "CourseDuration":
            r["PricePerHour"] = price_in / v if v else 0.0
            r["DurationBucketOrd"] = DURATION_BUCKETS.index(duration_bucket(v))
        if col == "CourseRating":
            r["RatingTierOrd"] = RATING_TIERS.index(rating_tier(v))
            r["QualitySignal"] = v * trate_in
        if col == "InstructorAvgRating":
            r["QualitySignal"] = rate_in * v
        if col == "InstructorAvgExperience":
            r["ExperienceBucketOrd"] = EXPERIENCE_BUCKETS.index(experience_bucket(v))
        rows.append(r)
    sweep = pd.DataFrame(rows)
    sweep["Predicted"] = mdl["pipeline"].predict(sweep[mdl["numeric"] + mdl["categorical"]])
    sweep["Revenue"] = sweep.Predicted * (sweep.CoursePrice if col == "CoursePrice" else price_in)

    s1, s2 = st.columns(2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=sweep.Predicted, mode="lines", line=dict(color=ACCENT, width=3)))
    fig.add_hrect(y0=lo80, y1=hi80, fillcolor=ACCENT, opacity=0.10, line_width=0,
                  annotation_text="80% prediction interval", annotation_position="top left")
    fig.update_layout(height=330, margin=dict(t=40, b=10), title="Predicted enrollments",
                      xaxis_title=sens_var, yaxis_title="Enrollments")
    s1.plotly_chart(fig, width="stretch")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=sweep.Revenue, mode="lines", line=dict(color=ACCENT_2, width=3)))
    fig.update_layout(height=330, margin=dict(t=40, b=10), title="Predicted revenue",
                      xaxis_title=sens_var, yaxis_title="Revenue")
    s2.plotly_chart(fig, width="stretch")
    st.caption("The demand curve is nearly flat in every input; the revenue curve rises with price "
               "almost linearly, because revenue is price multiplied by a demand that barely moves.")


# =========================================================================== #
# 3. REVENUE FORECAST
# =========================================================================== #
elif page == "Revenue Forecast":
    st.title("Revenue Forecast")
    st.caption("Course, category and platform revenue outlook built on the fitted demand model.")

    monthly = panel.groupby("Month").agg(Enrollments=("EnrollmentCount", "sum"),
                                         Revenue=("Revenue", "sum")).reset_index()
    horizon = st.slider("Forecast horizon (months)", 1, 12, 6)

    mu_e, mu_r = monthly.Enrollments.mean(), monthly.Revenue.mean()
    sd_e, sd_r = monthly.Enrollments.std(ddof=1), monthly.Revenue.std(ddof=1)
    future = pd.date_range(monthly.Month.max() + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    tab1, tab2, tab3 = st.tabs(["Platform outlook", "By course", "By category"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly.Month, y=monthly.Revenue, mode="lines+markers",
                                 name="Actual 2025", line=dict(color=ACCENT, width=3)))
        fig.add_trace(go.Scatter(x=future, y=[mu_r] * horizon, mode="lines+markers",
                                 name="Forecast", line=dict(color=ACCENT_2, width=3, dash="dash")))
        fig.add_trace(go.Scatter(x=list(future) + list(future[::-1]),
                                 y=[mu_r + 1.96 * sd_r] * horizon + [mu_r - 1.96 * sd_r] * horizon,
                                 fill="toself", fillcolor="rgba(13,148,136,.15)", line_width=0,
                                 name="95% interval", hoverinfo="skip"))
        fig.update_layout(height=420, margin=dict(t=30), yaxis_title="Monthly revenue", xaxis_title=None,
                          legend=dict(orientation="h", y=1.12, x=0))
        st.plotly_chart(fig, width="stretch")

        k = st.columns(3)
        k[0].metric(f"Forecast revenue, next {horizon} mo", f"{mu_r * horizon:,.0f}")
        k[1].metric("95% interval", f"{(mu_r - 1.96 * sd_r) * horizon:,.0f} - {(mu_r + 1.96 * sd_r) * horizon:,.0f}")
        k[2].metric(f"Forecast enrollments, next {horizon} mo", f"{mu_e * horizon:,.0f}")

        trend_p = diag_value("monthly_trend", "trend_p_value")
        callout(
            f"<b>Why the forecast is a flat line.</b> Monthly demand shows no trend "
            f"(slope p = {trend_p:.2f}) and no course-specific seasonality "
            f"(p = {diag_value('panel_independence', 'p_value'):.2f}). Out of sample, no machine-learning "
            f"model beat the historical mean. When the data-generating process is stationary noise, "
            f"the mean <i>is</i> the best forecast - and saying so is more useful than drawing a "
            f"confident curve that will not hold."
        )

    with tab2:
        n_show = st.slider("Courses to show", 5, 60, 20)
        top = course.nlargest(n_show, "CourseRevenue")
        # Forward revenue = price x the catalogue's average monthly demand x horizon,
        # which is the only defensible per-course projection given flat demand.
        monthly_demand = course.EnrollmentCount.mean() / 12
        proj = top.assign(ProjectedRevenue=top.CoursePrice * monthly_demand * horizon)
        fig = go.Figure()
        fig.add_bar(y=proj.DisplayName, x=proj.CourseRevenue, orientation="h",
                    name="2025 actual", marker_color=MUTED)
        fig.add_bar(y=proj.DisplayName, x=proj.ProjectedRevenue, orientation="h",
                    name=f"Next {horizon} mo forecast", marker_color=ACCENT)
        fig.update_layout(barmode="group", height=max(400, 22 * n_show),
                          yaxis=dict(autorange="reversed", title=None), xaxis_title="Revenue",
                          margin=dict(t=30), legend=dict(orientation="h", y=1.06, x=0))
        st.plotly_chart(fig, width="stretch")

    with tab3:
        if not D["cat_rev"].empty:
            cr = D["cat_rev"].copy()
            fig = go.Figure()
            fig.add_bar(x=cr.CourseCategory, y=cr.ActualRevenue, name="Actual", marker_color=MUTED)
            fig.add_bar(x=cr.CourseCategory, y=cr.PredictedRevenue, name="Model (out-of-fold)",
                        marker_color=ACCENT)
            fig.update_layout(barmode="group", height=430, margin=dict(t=30),
                              yaxis_title="Annual revenue", xaxis_title=None,
                              legend=dict(orientation="h", y=1.1, x=0))
            fig.update_xaxes(tickangle=-35)
            st.plotly_chart(fig, width="stretch")
            st.dataframe(cr, width="stretch", hide_index=True,
                         column_config={
                             "ActualRevenue": st.column_config.NumberColumn("Actual", format="%.0f"),
                             "PredictedRevenue": st.column_config.NumberColumn("Predicted", format="%.0f"),
                             "AbsError": st.column_config.NumberColumn("Abs error", format="%.0f"),
                             "APE_pct": st.column_config.NumberColumn("APE %", format="%.1f"),
                         })
            st.caption("Category revenue is predicted well because price is known and demand is "
                       "flat - the model is effectively summing price x average demand.")


# =========================================================================== #
# 4. FEATURE IMPORTANCE EXPLORER
# =========================================================================== #
elif page == "Feature Importance Explorer":
    st.title("Feature Importance Explorer")
    st.caption("Which levers move demand and revenue - and which only look like they do.")

    target = st.radio("Target", ["Enrollment demand", "Course revenue"], horizontal=True)
    imp = D["imp_enr"] if target == "Enrollment demand" else D["imp_rev"]

    if imp.empty:
        st.warning("Importance tables not found. Run the pipeline.")
        st.stop()

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Permutation importance")
        top = imp.head(14).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=top.Feature, x=top.Importance, orientation="h",
                    error_x=dict(type="data", array=top.Importance_sd, color="#1f2933", thickness=1.2),
                    marker_color=[ACCENT if v > 0 else NEG for v in top.Importance])
        fig.add_vline(x=0, line_color="#1f2933")
        fig.update_layout(height=470, margin=dict(t=30, l=0, r=0),
                          xaxis_title="Reduction in held-out MSE when the feature is intact",
                          yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption("Measured on held-out folds. Values at or below zero mean shuffling the feature "
                   "does not hurt - i.e. the model was not using it to generalise.")

    with right:
        st.subheader("Statistical screen")
        scr = D["screen"]
        if not scr.empty:
            show = scr[["Feature", "Test", "Statistic", "p_value", "p_bonferroni", "Significant_adj"]]
            st.dataframe(show, width="stretch", hide_index=True, height=470,
                         column_config={
                             "Statistic": st.column_config.NumberColumn(format="%.3f"),
                             "p_value": st.column_config.NumberColumn("p (raw)", format="%.4f"),
                             "p_bonferroni": st.column_config.NumberColumn("p (corrected)", format="%.3f"),
                             "Significant_adj": st.column_config.CheckboxColumn("Significant"),
                         })
            n_sig = int(scr.Significant_adj.sum())
            callout(
                f"{n_sig} of {len(scr)} candidate drivers survive Bonferroni correction for multiple "
                f"testing. Raw p-values below 0.05 appear for several features - that is exactly what "
                f"you expect from screening 15 variables against noise.", "warn" if n_sig <= 1 else "")

    st.divider()
    st.subheader("Directional effects (standardised Ridge coefficients)")
    coef = D["coef_enr"]
    if not coef.empty:
        top = coef.head(14).iloc[::-1]
        fig = go.Figure()
        fig.add_bar(y=top.Feature, x=top.Coefficient, orientation="h",
                    marker_color=[ACCENT_2 if v > 0 else WARN for v in top.Coefficient])
        fig.add_vline(x=0, line_color="#1f2933")
        fig.update_layout(height=430, margin=dict(t=30),
                          xaxis_title="Change in annual enrollments per 1 standard deviation of the feature")
        st.plotly_chart(fig, width="stretch")
        biggest = coef.iloc[0]
        st.caption(f"The largest effect in the catalogue is {biggest.Feature} at "
                   f"{biggest.Coefficient:+.1f} enrollments per standard deviation - "
                   f"{abs(biggest.Coefficient) / course.EnrollmentCount.mean():.1%} of average course demand.")


# =========================================================================== #
# 5. CATEGORY DEMAND COMPARISON
# =========================================================================== #
elif page == "Category Demand Comparison":
    st.title("Category Demand Comparison")
    st.caption("Where learner attention goes, versus where revenue comes from.")

    gap = D["gap"]
    if gap.empty:
        st.warning("Run the pipeline first.")
        st.stop()

    g = gap.sort_values("MonetisationGap", ascending=False)
    fig = go.Figure()
    fig.add_bar(y=g.CourseCategory, x=g.EnrollmentShare * 100, orientation="h",
                name="Share of enrollments", marker_color=ACCENT)
    fig.add_bar(y=g.CourseCategory, x=g.RevenueShare * 100, orientation="h",
                name="Share of revenue", marker_color=ACCENT_2)
    fig.update_layout(barmode="group", height=520, margin=dict(t=30),
                      xaxis_title="Share of platform total (%)", yaxis=dict(title=None, autorange="reversed"),
                      legend=dict(orientation="h", y=1.07, x=0))
    st.plotly_chart(fig, width="stretch")

    worst = g.iloc[0]
    callout(
        f"<b>{worst.CourseCategory}</b> takes {worst.EnrollmentShare:.1%} of all enrollments and returns "
        f"{worst.RevenueShare:.2%} of revenue - {int(worst.FreeCourses)} of its {int(worst.Courses)} courses "
        f"are free. Demand is uniform across categories, so a category earns little for one reason only: "
        f"it is not priced.")

    st.subheader("Category scorecard")
    st.dataframe(
        g[["CourseCategory", "Courses", "FreeCourses", "Enrollments", "Revenue",
           "RevenuePerEnrollment", "AvgPaidPrice", "MonetisationGap"]],
        width="stretch", hide_index=True,
        column_config={
            "FreeCourses": st.column_config.NumberColumn("Free courses", format="%d"),
            "Enrollments": st.column_config.NumberColumn(format="%d"),
            "Revenue": st.column_config.NumberColumn(format="%.0f"),
            "RevenuePerEnrollment": st.column_config.NumberColumn("Rev / enrollment", format="%.2f"),
            "AvgPaidPrice": st.column_config.NumberColumn("Avg paid price", format="%.2f"),
            "MonetisationGap": st.column_config.NumberColumn("Gap (enrol. share - rev. share)", format="%.3f"),
        })

    st.divider()
    st.subheader("Demand distribution within each category")
    fig = px.box(course, x="CourseCategory", y="EnrollmentCount", points="all",
                 color="CourseCategory", color_discrete_sequence=SEQ)
    fig.add_hline(y=course.EnrollmentCount.mean(), line_dash="dash", line_color="#1f2933")
    fig.update_layout(height=440, showlegend=False, margin=dict(t=30),
                      xaxis_title=None, yaxis_title="Enrollments per course")
    fig.update_xaxes(tickangle=-35)
    st.plotly_chart(fig, width="stretch")
    st.caption(f"An ANOVA across categories gives p = "
               f"{D['screen'].set_index('Feature').loc['CourseCategory', 'p_value']:.2f}: "
               f"category does not predict demand.")


# =========================================================================== #
# 6. PRICING SCENARIO SIMULATOR
# =========================================================================== #
elif page == "Pricing Scenario Simulator":
    st.title("Pricing Scenario Simulator")
    st.caption("The one lever the evidence in this dataset actually supports.")

    elast = diag_value("price_elasticity", "elasticity")
    lo, hi = diag_value("price_elasticity", "ci_low"), diag_value("price_elasticity", "ci_high")
    callout(
        f"<b>Measured price elasticity of demand: {elast:+.3f}</b> (95% CI {lo:+.3f} to {hi:+.3f}) across the "
        f"observed price range. Demand did not fall measurably as price rose. That is an observational "
        f"estimate, not a randomised test - so treat it as a reason to <i>run a pricing experiment</i>, "
        f"not as licence to reprice the whole catalogue overnight.")

    free = course[course.CourseType == "Free"]
    paid = course[course.CourseType == "Paid"]
    base_rev = course.CourseRevenue.sum()
    mean_demand = course.EnrollmentCount.mean()

    st.markdown("#### Scenario levers")
    a, b, c = st.columns(3)
    convert = a.slider("Free courses converted to paid", 0, len(free), 10)
    new_price = b.slider("Introductory price for converted courses", 5.0, 300.0, 49.0, 1.0)
    uplift = c.slider("Price change on existing paid courses (%)", -50, 100, 0, 5)
    assumed_elast = st.slider("Assumed elasticity for the scenario", -2.0, 0.0, float(round(elast, 2)), 0.05,
                              help="0 = demand unaffected by price. -1 = a 1% price rise loses 1% of demand. "
                                   "The measured value is near zero; drag it negative to stress-test.")
    churn = st.slider("Assumed demand lost when a free course starts charging (%)", 0, 100, 40, 5)

    conv_rev = convert * mean_demand * (1 - churn / 100) * new_price
    price_factor = 1 + uplift / 100
    demand_factor = price_factor ** assumed_elast
    paid_rev_new = (paid.CoursePrice * price_factor * paid.EnrollmentCount * demand_factor).sum()
    paid_rev_old = paid.CourseRevenue.sum()
    total_new = paid_rev_new + conv_rev

    k = st.columns(4)
    k[0].metric("Revenue today", f"{base_rev:,.0f}")
    k[1].metric("Scenario revenue", f"{total_new:,.0f}", f"{(total_new / base_rev - 1):+.1%}")
    k[2].metric("From conversions", f"{conv_rev:,.0f}")
    k[3].metric("From repricing", f"{paid_rev_new - paid_rev_old:+,.0f}")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Revenue today", "Convert free courses", "Reprice paid courses", "Scenario revenue"],
        y=[base_rev, conv_rev, paid_rev_new - paid_rev_old, None],
        text=[f"{base_rev:,.0f}", f"+{conv_rev:,.0f}", f"{paid_rev_new - paid_rev_old:+,.0f}", f"{total_new:,.0f}"],
        connector=dict(line=dict(color=MUTED)),
        increasing=dict(marker_color=ACCENT_2), decreasing=dict(marker_color=NEG),
        totals=dict(marker_color=ACCENT)))
    fig.update_layout(height=420, margin=dict(t=40), yaxis_title="Annual revenue", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Conversion sweep")
    shares = np.arange(0, len(free) + 1)
    curves = {}
    for pr in (25.0, new_price, 99.0):
        curves[f"at {pr:,.0f}"] = base_rev + shares * mean_demand * (1 - churn / 100) * pr
    fig = go.Figure()
    for (lab, ys), col in zip(curves.items(), [WARN, ACCENT, ACCENT_2]):
        fig.add_trace(go.Scatter(x=shares, y=ys, mode="lines", name=lab, line=dict(color=col, width=3)))
    fig.add_hline(y=base_rev, line_dash="dash", line_color=MUTED, annotation_text="Today")
    fig.update_layout(height=380, margin=dict(t=30), xaxis_title="Number of free courses converted to paid",
                      yaxis_title="Annual revenue", legend=dict(orientation="h", y=1.1, x=0))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Assumes {churn}% of a converted course's demand walks away. "
               f"{len(free)} of {len(course)} courses are currently free and carry "
               f"{free.EnrollmentCount.sum() / course.EnrollmentCount.sum():.0%} of all enrollments.")


# =========================================================================== #
# 7. MODEL RELIABILITY
# =========================================================================== #
elif page == "Model Reliability":
    st.title("Model Reliability")
    st.caption("How much of this data is signal, how much is noise, and what that means for the forecasts.")

    k = st.columns(4)
    k[0].metric("Uniform-allocation chi-square p", f"{diag_value('uniform_allocation_test', 'p_value'):.3f}")
    k[1].metric("Index of dispersion", f"{diag_value('uniform_allocation_test', 'index_of_dispersion'):.3f}",
                "1.00 = pure noise", delta_color="off")
    k[2].metric("Noise-floor max R2", f"{diag_value('noise_floor', 'max_attainable_r2'):.3f}")
    k[3].metric("Course x month independence p", f"{diag_value('panel_independence', 'p_value'):.3f}")

    callout(
        "<b>What the noise floor means.</b> The 10,000 enrollments observed in 2025 are spread across "
        "60 courses. Even if every course were <i>equally</i> attractive, random arrival alone would "
        "produce a standard deviation of about "
        f"{diag_value('uniform_allocation_test', 'multinomial_sd'):.1f} enrollments per course. The "
        f"observed standard deviation is {diag_value('uniform_allocation_test', 'observed_sd'):.1f} - "
        "no larger. There is no systematic variance left for a model to explain, so an out-of-sample "
        "R-squared near zero is the <i>correct</i> result, not a modelling failure.", "neg")

    st.subheader("Cross-sectional model comparison")
    cs = D["cs_models"]
    if not cs.empty:
        tgt = st.selectbox("Target", cs.Target.unique())
        sub = cs[cs.Target == tgt]
        fig = go.Figure()
        fig.add_bar(y=sub.Model, x=sub.CV_R2, orientation="h",
                    error_x=dict(type="data", array=sub.CV_R2_sd, color="#1f2933", thickness=1.2),
                    marker_color=[MUTED if m.startswith("Baseline") else (ACCENT if v > 0 else NEG)
                                  for m, v in zip(sub.Model, sub.CV_R2)])
        fig.add_vline(x=0, line_color="#1f2933")
        fig.update_layout(height=360, margin=dict(t=30), xaxis_title="Cross-validated R2",
                          yaxis=dict(title=None, autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(sub.drop(columns=["Target"]), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in sub.columns if c not in ("Target", "Model")})

    st.subheader("Out-of-time panel forecast")
    pm = D["panel_models"]
    if not pm.empty:
        tgt2 = st.selectbox("Panel target", pm.Target.unique(), key="pm")
        sub = pm[pm.Target == tgt2]
        st.dataframe(sub.drop(columns=["Target"]), width="stretch", hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in sub.columns if c not in ("Target", "Model")})
        st.caption("Trained on the earlier months, tested on the final three. Negative skill means the "
                   "model does worse than simply predicting the historical mean.")

    st.subheader("Data quality audit")
    st.dataframe(D["audit"], width="stretch", hide_index=True)

    with st.expander("What would make these models work"):
        st.markdown(
            """
The dataset records **outcomes** (who enrolled, what they paid) but none of the
**drivers** of demand. To move R-squared off the floor, EduPro needs to capture:

| Signal | Why it matters | Where it comes from |
|---|---|---|
| Catalogue impressions & click-through | Separates "nobody saw it" from "nobody wanted it" | Web analytics |
| Marketing spend and channel, per course | The largest controllable demand driver | Ad platforms |
| Search and wishlist volume by topic | Leading indicator of latent demand | Site search logs |
| Price *changes* over time | Elasticity needs price variation within a course | Pricing history table |
| Enrollment timestamps at day resolution | Recovers launch curves and seasonality | Already available, aggregate less |
| Completion, refund and repeat-purchase | Distinguishes durable demand from churn | LMS + billing |
| Cohort / campaign identifiers | Attributes demand to the action that caused it | CRM |

With those fields, the same pipeline in this repository re-runs unchanged - only
the feature table grows.
            """
        )
