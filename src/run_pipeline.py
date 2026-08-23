"""End-to-end pipeline: prepare -> engineer -> diagnose -> model -> report.

Run with::

    python -m src.run_pipeline

Writes every result table to ``reports/tables`` and every deployable model to
``models/``. The Streamlit app reads only those artefacts, so the app never
recomputes anything and starts instantly.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src import config, data_prep, diagnostics, features, models

CS_NUMERIC = [
    "CoursePrice", "CourseDuration", "CourseRating",
    "InstructorAvgRating", "InstructorAvgExperience", "InstructorExpertiseMatch",
    "PricePerHour", "QualitySignal",
    "PriceBandOrd", "DurationBucketOrd", "RatingTierOrd", "CourseLevelOrd", "ExperienceBucketOrd",
]
CS_CATEGORICAL = ["CourseCategory", "CourseType", "CourseLevel"]

PANEL_NUMERIC = config.PANEL_LAG_FEATURES + [
    "CoursePrice", "CourseDuration", "CourseRating",
    "InstructorAvgRating", "InstructorAvgExperience", "InstructorExpertiseMatch",
]
PANEL_CATEGORICAL = ["CourseCategory", "CourseType", "CourseLevel"]


def banner(msg: str) -> None:
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")


def main() -> dict:
    t0 = time.time()
    summary: dict = {}

    # ---------------------------------------------------------------- Stage 1
    banner("STAGE 1  Data preparation")
    frames = data_prep.run()
    audit = frames["_audit"]
    print(audit.to_string(index=False))

    # ---------------------------------------------------------------- Stage 2
    banner("STAGE 2  Feature engineering")
    course_feat, panel_feat = features.run()
    print(f"course-level table : {course_feat.shape[0]} rows x {course_feat.shape[1]} cols")
    print(f"panel table        : {panel_feat.shape[0]} rows x {panel_feat.shape[1]} cols "
          f"({panel_feat.Month.nunique()} usable months after lag warm-up)")

    panel_raw = frames["_panel"]

    # ---------------------------------------------------------------- Stage 3
    banner("STAGE 3  Predictability diagnostics")
    diag = diagnostics.run(course_feat, panel_raw)
    ua, nf = diag["uniform_allocation_test"], diag["noise_floor"]
    print(f"Uniform-allocation chi2 = {ua['chi2']:.2f} (dof {ua['dof']}), p = {ua['p_value']:.4f}")
    print(f"Index of dispersion     = {ua['index_of_dispersion']:.3f}  (1.00 = pure Poisson noise)")
    print(f"Noise-floor max R2      = {nf['max_attainable_r2']:.4f}")
    print(f"Course x month independence p = {diag['panel_independence']['p_value']:.4f}")
    pe = diag["price_elasticity"]
    print(f"Price elasticity        = {pe['elasticity']:+.4f}  95% CI [{pe['ci_low']:+.4f}, {pe['ci_high']:+.4f}]")
    print(f"Revenue identity holds  = {diag['revenue_identity']['identity_holds']}")
    summary["diagnostics"] = {k: v for k, v in diag.items() if not k.startswith("_")}

    # ---------------------------------------------------------------- Stage 4
    banner("STAGE 4a  Cold-start model - Enrollment Count")
    enr = models.evaluate_cross_sectional(course_feat, "EnrollmentCount", CS_NUMERIC, CS_CATEGORICAL,
                                          label="EnrollmentCount")
    print(enr.drop(columns=["Target"]).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    banner("STAGE 4b  Cold-start model - Course Revenue (with price)")
    rev = models.evaluate_cross_sectional(course_feat, "CourseRevenue", CS_NUMERIC, CS_CATEGORICAL,
                                          label="CourseRevenue")
    print(rev.drop(columns=["Target"]).to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    banner("STAGE 4c  ABLATION - Course Revenue without any price feature")
    no_price = [f for f in CS_NUMERIC if f not in ("CoursePrice", "PricePerHour", "PriceBandOrd")]
    no_price_cat = [c for c in CS_CATEGORICAL if c != "CourseType"]
    rev_abl = models.evaluate_cross_sectional(course_feat, "CourseRevenue", no_price, no_price_cat,
                                              label="CourseRevenue (no price)")
    print(rev_abl.drop(columns=["Target"]).to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    cs_results = pd.concat([enr, rev, rev_abl], ignore_index=True)
    cs_results.to_csv(config.TABLES / "05_cross_sectional_model_comparison.csv", index=False)

    # ---------------------------------------------------------------- Stage 5
    banner("STAGE 5  Operational panel forecast (out-of-time split)")
    pnl_enr, pred_enr = models.evaluate_panel(panel_feat, "EnrollmentCount", PANEL_NUMERIC, PANEL_CATEGORICAL)
    print("--- monthly enrollments ---")
    print(pnl_enr.drop(columns=["Target"]).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    pnl_rev, pred_rev = models.evaluate_panel(panel_feat, "Revenue", PANEL_NUMERIC, PANEL_CATEGORICAL)
    print("\n--- monthly revenue ---")
    print(pnl_rev.drop(columns=["Target"]).to_string(index=False, float_format=lambda v: f"{v:10.3f}"))

    pd.concat([pnl_enr, pnl_rev], ignore_index=True).to_csv(
        config.TABLES / "06_panel_model_comparison.csv", index=False)
    pred_enr.to_csv(config.PROCESSED / "panel_predictions_enrollment.csv", index=False)
    pred_rev.to_csv(config.PROCESSED / "panel_predictions_revenue.csv", index=False)

    # ---------------------------------------------------------------- Stage 6
    banner("STAGE 6  Feature importance")
    imp_enr = models.feature_importance(course_feat, "EnrollmentCount", CS_NUMERIC, CS_CATEGORICAL)
    imp_rev = models.feature_importance(course_feat, "CourseRevenue", CS_NUMERIC, CS_CATEGORICAL)
    coef_enr = models.linear_coefficients(course_feat, "EnrollmentCount", CS_NUMERIC, CS_CATEGORICAL)

    print("--- permutation importance, EnrollmentCount (top 10) ---")
    print(imp_enr.head(10).to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    print("\n--- permutation importance, CourseRevenue (top 10) ---")
    print(imp_rev.head(10).to_string(index=False, float_format=lambda v: f"{v:12.1f}"))
    print("\n--- standardised Ridge coefficients, EnrollmentCount (top 10) ---")
    print(coef_enr.head(10).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    imp_enr.to_csv(config.TABLES / "07_importance_enrollment.csv", index=False)
    imp_rev.to_csv(config.TABLES / "08_importance_revenue.csv", index=False)
    coef_enr.to_csv(config.TABLES / "09_ridge_coefficients_enrollment.csv", index=False)

    # ---------------------------------------------------------------- Stage 7
    banner("STAGE 7  Category-level revenue aggregation")
    best_cs = models.make_pipeline(
        models.RidgeCV(alphas=np.logspace(-3, 3, 25)), CS_NUMERIC, CS_CATEGORICAL)
    oof = models.cross_val_predict(
        best_cs, course_feat[CS_NUMERIC + CS_CATEGORICAL], course_feat.CourseRevenue,
        cv=models.KFold(5, shuffle=True, random_state=config.RANDOM_STATE))
    cat = (
        course_feat.assign(PredRevenue=oof)
        .groupby("CourseCategory")
        .agg(ActualRevenue=("CourseRevenue", "sum"), PredictedRevenue=("PredRevenue", "sum"),
             Enrollments=("EnrollmentCount", "sum"), Courses=("CourseID", "count"))
        .reset_index()
    )
    cat["AbsError"] = (cat.ActualRevenue - cat.PredictedRevenue).abs()
    cat["APE_pct"] = np.where(cat.ActualRevenue > 0, cat.AbsError / cat.ActualRevenue * 100, np.nan)
    cat = cat.sort_values("ActualRevenue", ascending=False)
    print(cat.to_string(index=False, float_format=lambda v: f"{v:12.1f}"))

    # A category earning ~100 currency units makes percentage error meaningless,
    # so report the material categories separately from the raw mean.
    material = cat[cat.ActualRevenue >= 1000]
    print(f"\nMAPE, all revenue-earning categories : {cat.APE_pct.mean():.1f}%  "
          f"(inflated by near-zero-revenue categories)")
    print(f"MAPE, categories above 1,000 revenue : {material.APE_pct.mean():.1f}%  "
          f"| median APE: {material.APE_pct.median():.1f}%  (n={len(material)})")
    tot_a, tot_p = cat.ActualRevenue.sum(), cat.PredictedRevenue.sum()
    print(f"Portfolio total  actual = {tot_a:,.0f} | predicted = {tot_p:,.0f} "
          f"| error = {(tot_p / tot_a - 1):+.2%}")
    if (cat.PredictedRevenue < 0).any():
        neg = cat.loc[cat.PredictedRevenue < 0, "CourseCategory"].tolist()
        print(f"NOTE: unconstrained linear model predicts negative revenue for {neg} "
              f"- a floor at zero is required before operational use.")
    cat.to_csv(config.TABLES / "10_category_revenue.csv", index=False)

    # ---------------------------------------------------------------- Stage 8
    banner("STAGE 8  Persisting deployable models")
    # Each target is deployed with whichever family won its own cross-validation,
    # not with one blanket choice.
    best_enr = enr.loc[enr.Model != "Baseline (mean)"].nlargest(1, "CV_R2").Model.iloc[0]
    best_rev = rev.loc[rev.Model != "Baseline (mean)"].nlargest(1, "CV_R2").Model.iloc[0]
    print(f"selected for EnrollmentCount : {best_enr}")
    print(f"selected for CourseRevenue   : {best_rev}")

    zoo = models.model_zoo(len(course_feat))
    models.fit_and_save(course_feat, "EnrollmentCount", CS_NUMERIC, CS_CATEGORICAL,
                        "enrollment_model.joblib", estimator=zoo[best_enr])
    models.fit_and_save(course_feat, "CourseRevenue", CS_NUMERIC, CS_CATEGORICAL,
                        "revenue_model.joblib", estimator=zoo[best_rev])
    models.fit_and_save(panel_feat, "EnrollmentCount", PANEL_NUMERIC, PANEL_CATEGORICAL,
                        "panel_enrollment_model.joblib",
                        estimator=models.model_zoo(len(panel_feat))["Ridge Regression"])
    print("saved:", *[p.name for p in sorted(config.MODELS.glob('*.joblib'))])
    summary["deployed_models"] = {"EnrollmentCount": best_enr, "CourseRevenue": best_rev}

    # ---------------------------------------------------------------- Summary
    summary["headline"] = {
        "courses": int(len(course_feat)),
        "enrollments": int(course_feat.EnrollmentCount.sum()),
        "total_revenue": float(course_feat.CourseRevenue.sum()),
        "free_courses": int((course_feat.CourseType == "Free").sum()),
        "free_enrollment_share": float(course_feat.loc[course_feat.CourseType == "Free", "EnrollmentCount"].sum()
                                       / course_feat.EnrollmentCount.sum()),
        "best_enrollment_cv_r2": float(enr.loc[enr.Model != "Baseline (mean)", "CV_R2"].max()),
        "best_revenue_cv_r2": float(rev.loc[rev.Model != "Baseline (mean)", "CV_R2"].max()),
        "best_revenue_noprice_cv_r2": float(rev_abl.loc[rev_abl.Model != "Baseline (mean)", "CV_R2"].max()),
        "best_panel_enrollment_r2": float(pnl_enr.loc[~pnl_enr.Model.str.startswith(("Baseline", "Naive")), "R2"].max()),
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (config.REPORTS / "pipeline_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    banner("HEADLINE RESULTS")
    for k, v in summary["headline"].items():
        print(f"  {k:32s} {v}")
    return summary


if __name__ == "__main__":
    main()
