"""Stage 2 - Feature engineering.

Implements the three feature families specified in the brief:

1. **Course features**      - price bands, duration buckets, rating tiers, level encoding
2. **Instructor features**  - experience buckets, rating score, expertise-category match
3. **Historical features**  - lagged enrollments, past average revenue, revenue per enrollment

Course/instructor features are computed from *attributes only* so the trained
model can score a course that does not exist yet (the launch-decision use case).
Historical features are lag-only and are therefore causally safe: every value
is known strictly before the month being predicted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# --- Bucket definitions (exposed so the Streamlit app applies identical logic) --
PRICE_BANDS = ["Free", "Low", "Medium", "High"]
DURATION_BUCKETS = ["Short", "Medium", "Long", "Extended"]
RATING_TIERS = ["Poor", "Fair", "Good", "Excellent"]
LEVEL_ORDER = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
EXPERIENCE_BUCKETS = ["Junior", "Mid", "Senior", "Veteran"]


def price_band(price: float) -> str:
    if price <= 0:
        return "Free"
    if price < 100:
        return "Low"
    if price < 300:
        return "Medium"
    return "High"


def duration_bucket(hours: float) -> str:
    if hours < 10:
        return "Short"
    if hours < 25:
        return "Medium"
    if hours < 40:
        return "Long"
    return "Extended"


def rating_tier(rating: float) -> str:
    if rating < 2:
        return "Poor"
    if rating < 3:
        return "Fair"
    if rating < 4:
        return "Good"
    return "Excellent"


def experience_bucket(years: float) -> str:
    if years < 3:
        return "Junior"
    if years < 7:
        return "Mid"
    if years < 12:
        return "Senior"
    return "Veteran"


def add_course_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach engineered course + instructor features to a course-level frame."""
    out = df.copy()

    out["PriceBand"] = out.CoursePrice.apply(price_band)
    out["DurationBucket"] = out.CourseDuration.apply(duration_bucket)
    out["RatingTier"] = out.CourseRating.apply(rating_tier)
    out["ExperienceBucket"] = out.InstructorAvgExperience.apply(experience_bucket)

    out["PriceBandOrd"] = out.PriceBand.map({b: i for i, b in enumerate(PRICE_BANDS)})
    out["DurationBucketOrd"] = out.DurationBucket.map({b: i for i, b in enumerate(DURATION_BUCKETS)})
    out["RatingTierOrd"] = out.RatingTier.map({b: i for i, b in enumerate(RATING_TIERS)})
    out["CourseLevelOrd"] = out.CourseLevel.map(LEVEL_ORDER)
    out["ExperienceBucketOrd"] = out.ExperienceBucket.map({b: i for i, b in enumerate(EXPERIENCE_BUCKETS)})

    # Interaction / density features
    out["PricePerHour"] = out.CoursePrice / out.CourseDuration.replace(0, np.nan)
    out["PricePerHour"] = out.PricePerHour.fillna(0.0)
    out["QualitySignal"] = out.CourseRating * out.InstructorAvgRating
    out["IsPaid"] = (out.CourseType == "Paid").astype(int)
    return out


def add_panel_features(panel: pd.DataFrame, course_level: pd.DataFrame) -> pd.DataFrame:
    """Attach lagged historical-performance features to the course-month panel.

    All lags use ``shift`` inside each course, so row *t* only ever sees data
    from months strictly earlier than *t*. Rows whose lag window is not yet
    available are dropped by :func:`panel_modelling_frame`.
    """
    p = panel.sort_values(["CourseID", "Month"]).copy()
    g = p.groupby("CourseID")

    for lag in (1, 2, 3):
        p[f"Enroll_lag{lag}"] = g.EnrollmentCount.shift(lag)
    p["Revenue_lag1"] = g.Revenue.shift(1)

    # Rolling means over the three months preceding the target month.
    p["Enroll_roll3"] = g.EnrollmentCount.shift(1).rolling(3).mean().reset_index(level=0, drop=True)
    p["Revenue_roll3"] = g.Revenue.shift(1).rolling(3).mean().reset_index(level=0, drop=True)

    # Expanding (all history to date) course averages.
    p["Enroll_expanding_mean"] = g.EnrollmentCount.shift(1).expanding().mean().reset_index(level=0, drop=True)
    rev_exp = g.Revenue.shift(1).expanding().sum().reset_index(level=0, drop=True)
    enr_exp = g.EnrollmentCount.shift(1).expanding().sum().reset_index(level=0, drop=True)
    p["RevenuePerEnroll_expanding"] = (rev_exp / enr_exp.replace(0, np.nan)).fillna(0.0)

    static_cols = [
        "CourseID", "CourseCategory", "CourseType", "CourseLevel", "CoursePrice",
        "CourseDuration", "CourseRating", "InstructorAvgRating",
        "InstructorAvgExperience", "InstructorExpertiseMatch", "PriceBandOrd",
        "DurationBucketOrd", "RatingTierOrd", "CourseLevelOrd", "ExperienceBucketOrd",
    ]
    return p.merge(course_level[static_cols], on="CourseID", how="left")


def panel_modelling_frame(panel_feat: pd.DataFrame) -> pd.DataFrame:
    """Drop warm-up months where the 3-month lag window is incomplete."""
    return panel_feat.dropna(subset=["Enroll_lag3", "Enroll_roll3"]).reset_index(drop=True)


def run() -> tuple[pd.DataFrame, pd.DataFrame]:
    course_level = pd.read_csv(config.PROCESSED / "course_level.csv")
    panel = pd.read_csv(config.PROCESSED / "panel_course_month.csv", parse_dates=["Month"])

    course_feat = add_course_features(course_level)
    panel_feat = panel_modelling_frame(add_panel_features(panel, course_feat))

    course_feat.to_csv(config.PROCESSED / "course_features.csv", index=False)
    panel_feat.to_csv(config.PROCESSED / "panel_features.csv", index=False)
    return course_feat, panel_feat


if __name__ == "__main__":
    c, p = run()
    print("course_features:", c.shape)
    print("panel_features :", p.shape, "| months kept:", p.Month.nunique())
    print(c[["CourseID", "PriceBand", "DurationBucket", "RatingTier", "ExperienceBucket"]].head().to_string(index=False))
