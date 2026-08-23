"""Stage 3 - Predictability diagnostics.

Before fitting any model we quantify how much of the variation in enrollment
demand is *systematic* (explainable by course, instructor or time attributes)
versus *sampling noise* (the irreducible randomness of allocating a finite
number of enrollments across a catalogue).

This yields a **noise floor**: an upper bound on the R-squared any model can
achieve. Reporting model scores against that bound is what separates
"the model is broken" from "the signal is not in the data".
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src import config


def uniform_allocation_test(enrollments: pd.Series) -> dict:
    """Chi-square goodness-of-fit against equal demand for every course.

    Under the null, the N enrollments are spread multinomially and uniformly
    across the K courses. The index of dispersion (variance / mean) is 1 under
    a Poisson-like null; values near 1 mean the observed spread is pure noise.
    """
    obs = enrollments.to_numpy(dtype=float)
    n, k = obs.sum(), len(obs)
    expected = n / k
    chi2 = ((obs - expected) ** 2 / expected).sum()
    dof = k - 1
    return {
        "n_enrollments": int(n),
        "n_courses": int(k),
        "expected_per_course": expected,
        "observed_sd": float(obs.std(ddof=1)),
        "multinomial_sd": float(np.sqrt(n * (1 / k) * (1 - 1 / k))),
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(stats.chi2.sf(chi2, dof)),
        "index_of_dispersion": float(obs.var(ddof=1) / obs.mean()),
    }


def noise_floor_r2(enrollments: pd.Series) -> dict:
    """Maximum attainable out-of-sample R-squared given multinomial noise.

    Total variance decomposes as systematic + sampling. Sampling variance for a
    uniform multinomial is ``N*(1/K)*(1-1/K)``. Whatever is left is the most a
    perfect model could ever explain.
    """
    obs = enrollments.to_numpy(dtype=float)
    n, k = obs.sum(), len(obs)
    total_var = obs.var(ddof=1)
    sampling_var = n * (1 / k) * (1 - 1 / k)
    systematic_var = max(total_var - sampling_var, 0.0)
    return {
        "total_variance": float(total_var),
        "sampling_variance": float(sampling_var),
        "systematic_variance": float(systematic_var),
        "max_attainable_r2": float(systematic_var / total_var) if total_var > 0 else 0.0,
    }


def panel_independence_test(panel: pd.DataFrame) -> dict:
    """Chi-square test of independence between course and calendar month.

    A significant result would mean some courses have their own seasonality.
    A non-significant result means month-to-month movement is common noise.
    """
    tab = panel.pivot_table(index="CourseID", columns="Month", values="EnrollmentCount", aggfunc="sum").fillna(0)
    chi2, p, dof, _ = stats.chi2_contingency(tab)
    return {
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p),
        "interpretation": "course-specific seasonality" if p < 0.05 else "no course-specific seasonality",
    }


def monthly_trend_test(panel: pd.DataFrame) -> dict:
    """Is platform-wide monthly demand trending, seasonal, or flat?"""
    monthly = panel.groupby("Month").EnrollmentCount.sum().sort_index()
    t = np.arange(len(monthly))
    slope, intercept, r, p_trend, se = stats.linregress(t, monthly.to_numpy())
    chi2 = ((monthly - monthly.mean()) ** 2 / monthly.mean()).sum()
    return {
        "monthly_mean": float(monthly.mean()),
        "monthly_sd": float(monthly.std(ddof=1)),
        "poisson_sd": float(np.sqrt(monthly.mean())),
        "trend_slope_per_month": float(slope),
        "trend_p_value": float(p_trend),
        "dispersion_chi2": float(chi2),
        "dispersion_dof": int(len(monthly) - 1),
        "dispersion_p_value": float(stats.chi2.sf(chi2, len(monthly) - 1)),
    }


def feature_screen(df: pd.DataFrame, target: str = "EnrollmentCount") -> pd.DataFrame:
    """Univariate association screen with Bonferroni correction.

    With 60 courses and a dozen candidate drivers, uncorrected p-values will
    throw up false positives; the corrected column is the one to read.
    """
    numeric = ["CoursePrice", "CourseDuration", "CourseRating", "InstructorAvgRating",
               "InstructorAvgExperience", "InstructorExpertiseMatch", "InstructorCount",
               "PricePerHour", "QualitySignal"]
    categorical = ["CourseCategory", "CourseType", "CourseLevel", "PriceBand",
                   "DurationBucket", "RatingTier"]

    rows = []
    for f in numeric:
        if f not in df:
            continue
        rho, p = stats.spearmanr(df[f], df[target])
        rows.append({"Feature": f, "Test": "Spearman rho", "Statistic": rho, "p_value": p})
    for f in categorical:
        if f not in df:
            continue
        groups = [g[target].to_numpy() for _, g in df.groupby(f) if len(g) > 1]
        if len(groups) < 2:
            continue
        F, p = stats.f_oneway(*groups)
        rows.append({"Feature": f, "Test": "ANOVA F", "Statistic": F, "p_value": p})

    out = pd.DataFrame(rows)
    m = len(out)
    out["p_bonferroni"] = np.minimum(out.p_value * m, 1.0)
    out["Significant_adj"] = out.p_bonferroni < 0.05
    return out.sort_values("p_value").reset_index(drop=True)


def price_elasticity(df: pd.DataFrame) -> dict:
    """Constant-elasticity demand curve on the paid catalogue.

    log(enrollments) = a + b * log(price); b is the price elasticity of demand.
    The confidence interval matters more than the point estimate: a tight
    interval around zero is evidence of price *insensitivity*, which is a
    different and far more useful statement than "not significant".
    """
    paid = df[df.CoursePrice > 0]
    if len(paid) < 5:
        return {}
    x = np.log(paid.CoursePrice.to_numpy())
    y = np.log(paid.EnrollmentCount.to_numpy())
    slope, intercept, r, p, se = stats.linregress(x, y)
    tcrit = stats.t.ppf(0.975, len(paid) - 2)
    return {
        "n_paid_courses": int(len(paid)),
        "elasticity": float(slope),
        "ci_low": float(slope - tcrit * se),
        "ci_high": float(slope + tcrit * se),
        "p_value": float(p),
        "r_squared": float(r ** 2),
        "price_min": float(paid.CoursePrice.min()),
        "price_max": float(paid.CoursePrice.max()),
    }


def revenue_identity(course_df: pd.DataFrame) -> dict:
    """Confirm Revenue == Price x Enrollments exactly (an accounting identity).

    If it holds, any revenue model that is given price and enrollments is
    re-deriving arithmetic, not learning demand behaviour - and its R-squared
    must be reported with that caveat attached.
    """
    implied = course_df.CoursePrice * course_df.EnrollmentCount
    err = (course_df.CourseRevenue - implied).abs()
    return {
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "identity_holds": bool(err.max() < 1e-6),
    }


def monetisation_gap(course_df: pd.DataFrame) -> pd.DataFrame:
    """Enrollment share vs revenue share, by category - the pricing lever."""
    g = course_df.groupby("CourseCategory").agg(
        Courses=("CourseID", "count"),
        FreeCourses=("IsPaid", lambda s: int((s == 0).sum())),
        Enrollments=("EnrollmentCount", "sum"),
        Revenue=("CourseRevenue", "sum"),
        AvgPaidPrice=("CoursePrice", lambda s: s[s > 0].mean()),
    )
    g["EnrollmentShare"] = g.Enrollments / g.Enrollments.sum()
    g["RevenueShare"] = g.Revenue / g.Revenue.sum()
    g["MonetisationGap"] = g.EnrollmentShare - g.RevenueShare
    g["RevenuePerEnrollment"] = g.Revenue / g.Enrollments
    return g.sort_values("MonetisationGap", ascending=False).reset_index()


def run(course_df: pd.DataFrame, panel: pd.DataFrame) -> dict:
    results = {
        "uniform_allocation_test": uniform_allocation_test(course_df.EnrollmentCount),
        "noise_floor": noise_floor_r2(course_df.EnrollmentCount),
        "panel_independence": panel_independence_test(panel),
        "monthly_trend": monthly_trend_test(panel),
        "price_elasticity": price_elasticity(course_df),
        "revenue_identity": revenue_identity(course_df),
    }
    screen = feature_screen(course_df)
    gap = monetisation_gap(course_df)

    screen.to_csv(config.TABLES / "02_feature_screen.csv", index=False)
    gap.to_csv(config.TABLES / "03_monetisation_gap.csv", index=False)
    pd.DataFrame(
        [{"Diagnostic": k, "Metric": mk, "Value": mv} for k, d in results.items() for mk, mv in d.items()]
    ).to_csv(config.TABLES / "04_predictability_diagnostics.csv", index=False)

    results["_feature_screen"] = screen
    results["_monetisation_gap"] = gap
    return results


if __name__ == "__main__":
    cdf = pd.read_csv(config.PROCESSED / "course_features.csv")
    pnl = pd.read_csv(config.PROCESSED / "panel_course_month.csv", parse_dates=["Month"])
    res = run(cdf, pnl)
    for key, block in res.items():
        if key.startswith("_"):
            continue
        print(f"\n--- {key} ---")
        for k, v in block.items():
            print(f"  {k}: {v}")
    print("\n--- feature screen ---")
    print(res["_feature_screen"].to_string(index=False))
    print("\n--- monetisation gap ---")
    print(res["_monetisation_gap"].to_string(index=False))
