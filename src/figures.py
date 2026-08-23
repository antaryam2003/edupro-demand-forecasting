"""Stage 9 - Publication figures for the research paper.

Renders every figure referenced in ``reports/research_paper.md`` to
``reports/figures`` at 200 dpi. Run after ``src.run_pipeline``.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config

# A single restrained palette keeps the figure set reading as one system.
INK = "#1f2933"
MUTED = "#7b8794"
GRID = "#e4e7eb"
ACCENT = "#2563eb"
ACCENT_2 = "#0d9488"
WARN = "#d97706"
NEG = "#dc2626"

plt.rcParams.update(
    {
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name: str):
    path = config.FIGURES / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------- #
def fig_demand_distribution(course: pd.DataFrame):
    """The central EDA result: realised demand vs the uniform-random null."""
    obs = course.EnrollmentCount.to_numpy(float)
    n, k = obs.sum(), len(obs)
    mu = n / k
    sd = np.sqrt(n * (1 / k) * (1 - 1 / k))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    ax.hist(obs, bins=14, color=ACCENT, alpha=0.75, edgecolor="white")
    xs = np.linspace(obs.min() - 15, obs.max() + 15, 300)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd) * len(obs) * (obs.max() - obs.min()) / 14,
            color=NEG, lw=2, label=f"Uniform-random null\nN({mu:.0f}, {sd:.1f})")
    ax.axvline(mu, color=INK, ls="--", lw=1.2, label=f"Mean = {mu:.0f}")
    ax.set_xlabel("Enrollments per course (2025)")
    ax.set_ylabel("Number of courses")
    ax.set_title("A. Demand is indistinguishable from random allocation")
    ax.legend(fontsize=8)
    _despine(ax)

    ax = axes[1]
    order = np.argsort(-obs)
    ax.bar(range(k), obs[order], color=ACCENT, alpha=0.85, width=0.85)
    ax.axhline(mu, color=INK, ls="--", lw=1.2)
    ax.fill_between([-1, k], mu - 2 * sd, mu + 2 * sd, color=WARN, alpha=0.15,
                    label="95% band under the random null")
    ax.set_xlim(-1, k)
    ax.set_ylim(0, obs.max() * 1.12)
    ax.set_xlabel("Courses, ranked by enrollments")
    ax.set_ylabel("Enrollments")
    ax.set_title("B. Every course falls inside the noise band")
    ax.legend(fontsize=8, loc="upper right")
    _despine(ax)

    fig.suptitle("Figure 1  Course-level enrollment demand vs the random-allocation null",
                 y=1.03, fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig01_demand_vs_null.png")


def fig_revenue_identity(course: pd.DataFrame):
    """Revenue is price times a near-constant demand - an accounting identity."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    paid = course[course.CoursePrice > 0]
    free = course[course.CoursePrice == 0]
    ax.scatter(paid.CoursePrice, paid.CourseRevenue, s=48, color=ACCENT, alpha=0.85,
               edgecolor="white", label=f"Paid ({len(paid)})")
    ax.scatter(free.CoursePrice, free.CourseRevenue, s=48, color=MUTED, alpha=0.7,
               edgecolor="white", label=f"Free ({len(free)}) - zero revenue")
    xs = np.linspace(0, course.CoursePrice.max() * 1.05, 50)
    ax.plot(xs, xs * course.EnrollmentCount.mean(), color=NEG, lw=2, ls="--",
            label=f"Price x mean demand ({course.EnrollmentCount.mean():.0f})")
    ax.set_xlabel("Course price (currency units)")
    ax.set_ylabel("Realised annual revenue")
    ax.set_title("A. Revenue = price x (near-constant) demand")
    ax.legend(fontsize=8)
    _despine(ax)

    ax = axes[1]
    x = np.log(paid.CoursePrice)
    y = np.log(paid.EnrollmentCount)
    sl, ic, r, p, se = stats.linregress(x, y)
    ax.scatter(x, y, s=48, color=ACCENT_2, alpha=0.85, edgecolor="white")
    xs = np.linspace(x.min(), x.max(), 50)
    tcrit = stats.t.ppf(0.975, len(paid) - 2)
    ax.plot(xs, ic + sl * xs, color=INK, lw=2)
    ax.fill_between(xs, ic + (sl - tcrit * se) * xs, ic + (sl + tcrit * se) * xs,
                    color=INK, alpha=0.12, label="95% CI on the slope")
    ax.set_xlabel("log(price)")
    ax.set_ylabel("log(enrollments)")
    ax.set_title(f"B. Price elasticity = {sl:+.3f}  (95% CI {sl - tcrit * se:+.3f} to {sl + tcrit * se:+.3f})")
    ax.legend(fontsize=8)
    _despine(ax)

    fig.suptitle("Figure 2  Revenue structure and price sensitivity",
                 y=1.03, fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig02_revenue_and_elasticity.png")


def fig_monetisation_gap(gap: pd.DataFrame):
    """Where attention goes vs where money comes from."""
    g = gap.sort_values("MonetisationGap")
    y = np.arange(len(g))
    fig, ax = plt.subplots(figsize=(9.5, 5.6))

    ax.barh(y - 0.2, g.EnrollmentShare * 100, height=0.38, color=ACCENT, label="Share of enrollments")
    ax.barh(y + 0.2, g.RevenueShare * 100, height=0.38, color=ACCENT_2, label="Share of revenue")
    ax.set_yticks(y)
    ax.set_yticklabels(g.CourseCategory)
    ax.set_xlabel("Share of platform total (%)")
    ax.set_title("Figure 3  The monetisation gap: demand share vs revenue share by category",
                 pad=26)
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), ncol=2)

    for i, row in enumerate(g.itertuples()):
        ax.text(max(row.EnrollmentShare, row.RevenueShare) * 100 + 0.6, i,
                f"{row.FreeCourses}/{row.Courses} free", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, max(g.EnrollmentShare.max(), g.RevenueShare.max()) * 100 + 6)
    _despine(ax)
    save(fig, "fig03_monetisation_gap.png")


def fig_model_comparison(cs: pd.DataFrame):
    """Model scores plotted against the noise floor they cannot cross."""
    # The three panels share one model ordering, so only the left panel carries
    # tick labels - otherwise they render across the neighbouring subplot.
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    panels = [
        ("EnrollmentCount", "A. Enrollment demand", axes[0]),
        ("CourseRevenue", "B. Course revenue (price included)", axes[1]),
        ("CourseRevenue (no price)", "C. Course revenue (price removed)", axes[2]),
    ]
    for idx, (target, title, ax) in enumerate(panels):
        sub = cs[cs.Target == target].copy().iloc[::-1]
        colors = [MUTED if m == "Baseline (mean)" else (ACCENT if v > 0 else NEG)
                  for m, v in zip(sub.Model, sub.CV_R2)]
        ax.barh(sub.Model, sub.CV_R2, color=colors, alpha=0.9)
        ax.axvline(0, color=INK, lw=1.2)
        ax.set_xlabel("Cross-validated $R^2$")
        ax.set_title(title, fontsize=11)
        for i, (v, sd) in enumerate(zip(sub.CV_R2, sub.CV_R2_sd)):
            ax.errorbar(v, i, xerr=sd, color=INK, capsize=2.5, lw=1)
        _despine(ax)
        if idx > 0:
            ax.tick_params(axis="y", length=0)
        if target != "CourseRevenue":
            lo = min(-0.6, sub.CV_R2.min() * 1.25)
            ax.set_xlim(lo, max(0.18, sub.CV_R2.max() * 1.35))

    axes[0].text(0.03, 0.94, "Noise floor: max attainable $R^2$ = 0.00",
                 transform=axes[0].transAxes, fontsize=8.5, color=NEG, style="italic",
                 va="top")
    fig.suptitle("Figure 4  Out-of-sample model performance against the mean baseline",
                 y=1.04, fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig04_model_comparison.png")


def fig_feature_importance(imp_enr: pd.DataFrame, coef: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax = axes[0]
    top = imp_enr.head(12).iloc[::-1]
    colors = [ACCENT if v > 0 else NEG for v in top.Importance]
    ax.barh(top.Feature, top.Importance, color=colors, alpha=0.9)
    ax.errorbar(top.Importance, range(len(top)), xerr=top.Importance_sd, fmt="none", color=INK, capsize=2.5, lw=1)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("Permutation importance (drop in MSE, held-out folds)")
    ax.set_title("A. Nothing improves held-out error", fontsize=11)
    _despine(ax)

    ax = axes[1]
    top = coef.head(12).iloc[::-1]
    colors = [ACCENT_2 if v > 0 else WARN for v in top.Coefficient]
    ax.barh(top.Feature, top.Coefficient, color=colors, alpha=0.9)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_xlabel("Standardised Ridge coefficient (enrollments per 1 SD)")
    ax.set_title("B. In-sample directions, all economically small", fontsize=11)
    _despine(ax)

    fig.suptitle("Figure 5  Feature importance for enrollment demand",
                 y=1.02, fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig05_feature_importance.png")


def fig_temporal(panel: pd.DataFrame):
    monthly = panel.groupby("Month").agg(Enrollments=("EnrollmentCount", "sum"), Revenue=("Revenue", "sum"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    ax = axes[0]
    mu = monthly.Enrollments.mean()
    sd = np.sqrt(mu)
    ax.plot(monthly.index, monthly.Enrollments, marker="o", color=ACCENT, lw=2)
    ax.axhline(mu, color=INK, ls="--", lw=1.2, label=f"Mean = {mu:.0f}")
    ax.fill_between(monthly.index, mu - 2 * sd, mu + 2 * sd, color=WARN, alpha=0.15,
                    label="95% Poisson band")
    ax.set_ylabel("Enrollments")
    ax.set_title("A. Platform demand is flat - no trend, no seasonality", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    _despine(ax)

    ax = axes[1]
    ax.plot(monthly.index, monthly.Revenue / 1000, marker="o", color=ACCENT_2, lw=2)
    ax.axhline(monthly.Revenue.mean() / 1000, color=INK, ls="--", lw=1.2,
               label=f"Mean = {monthly.Revenue.mean() / 1000:.1f}k")
    ax.set_ylabel("Revenue (thousands)")
    ax.set_title("B. Monthly revenue tracks the same flat process", fontsize=11)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    _despine(ax)

    fig.suptitle("Figure 6  Temporal structure of demand and revenue, 2025",
                 y=1.03, fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig06_temporal.png")


def fig_scenario(course: pd.DataFrame):
    """What pricing coverage is worth, given demonstrated price insensitivity."""
    paid_price = course.loc[course.CoursePrice > 0, "CoursePrice"].median()
    mean_demand = course.EnrollmentCount.mean()
    free = course[course.CoursePrice == 0]
    base_rev = course.CourseRevenue.sum()

    shares = np.linspace(0, 1, 21)
    prices = [0.25, 0.5, 1.0]
    fig, ax = plt.subplots(figsize=(9, 5))

    for mult, col in zip(prices, [WARN, ACCENT, ACCENT_2]):
        p = paid_price * mult
        uplift = [base_rev + s * len(free) * mean_demand * p for s in shares]
        ax.plot(shares * 100, np.array(uplift) / 1000, lw=2.2, color=col,
                label=f"Price at {mult:.0%} of median paid price ({p:,.0f})")

    ax.axhline(base_rev / 1000, color=INK, ls="--", lw=1.2, label=f"Today = {base_rev / 1000:,.0f}k")
    ax.set_xlabel("Share of the 38 free courses converted to paid (%)")
    ax.set_ylabel("Annual revenue (thousands)")
    ax.set_title("Figure 7  Revenue scenarios from closing the monetisation gap\n"
                 "(assumes zero demand loss, consistent with the measured elasticity)", fontsize=11)
    ax.legend(fontsize=8.5)
    _despine(ax)
    save(fig, "fig07_pricing_scenarios.png")


def run():
    course = pd.read_csv(config.PROCESSED / "course_features.csv")
    panel = pd.read_csv(config.PROCESSED / "panel_course_month.csv", parse_dates=["Month"])
    cs = pd.read_csv(config.TABLES / "05_cross_sectional_model_comparison.csv")
    gap = pd.read_csv(config.TABLES / "03_monetisation_gap.csv")
    imp = pd.read_csv(config.TABLES / "07_importance_enrollment.csv")
    coef = pd.read_csv(config.TABLES / "09_ridge_coefficients_enrollment.csv")

    print("Rendering figures...")
    fig_demand_distribution(course)
    fig_revenue_identity(course)
    fig_monetisation_gap(gap)
    fig_model_comparison(cs)
    fig_feature_importance(imp, coef)
    fig_temporal(panel)
    fig_scenario(course)
    print("done.")


if __name__ == "__main__":
    run()
