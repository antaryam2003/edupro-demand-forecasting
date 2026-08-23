"""Stage 1 - Data preparation.

Loads the four raw sheets, validates referential integrity, and builds the two
analysis tables used downstream:

* ``course_level``   - one row per course (60 rows), the cross-sectional
  "cold start" table used to answer *will a course we launch attract demand?*
* ``panel``          - one row per course-month (60 x 12 = 720 rows), the
  time-indexed table used for operational demand / revenue forecasting.
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_raw() -> dict[str, pd.DataFrame]:
    """Read every sheet of the source workbook into a dict of DataFrames."""
    xls = pd.ExcelFile(config.RAW_XLSX)
    frames = {s: pd.read_excel(xls, s) for s in config.SHEETS}
    frames["Transactions"]["TransactionDate"] = pd.to_datetime(
        frames["Transactions"]["TransactionDate"]
    )
    return frames


def validate(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Referential-integrity and quality checks; returns an audit table."""
    users, teachers = frames["Users"], frames["Teachers"]
    courses, txns = frames["Courses"], frames["Transactions"]

    checks = [
        ("Users rows", len(users)),
        ("Teachers rows", len(teachers)),
        ("Courses rows", len(courses)),
        ("Transactions rows", len(txns)),
        ("Duplicate CourseID", int(courses.CourseID.duplicated().sum())),
        ("Duplicate TeacherID", int(teachers.TeacherID.duplicated().sum())),
        ("Duplicate TransactionID", int(txns.TransactionID.duplicated().sum())),
        ("Txn CourseID not in Courses", int((~txns.CourseID.isin(courses.CourseID)).sum())),
        ("Txn TeacherID not in Teachers", int((~txns.TeacherID.isin(teachers.TeacherID)).sum())),
        ("Txn UserID not in Users", int((~txns.UserID.isin(users.UserID)).sum())),
        ("Courses with zero transactions", int((~courses.CourseID.isin(txns.CourseID)).sum())),
        ("Total missing cells (all sheets)", int(sum(f.isna().sum().sum() for f in frames.values()))),
        ("Negative amounts", int((txns.Amount < 0).sum())),
        ("Duplicate (User, Course) pairs", int(txns.duplicated(["UserID", "CourseID"]).sum())),
    ]

    merged = txns.merge(courses[["CourseID", "CoursePrice"]], on="CourseID")
    identity_err = (merged.Amount - merged.CoursePrice).abs().max()
    checks.append(("Max |Amount - CoursePrice|", float(identity_err)))

    return pd.DataFrame(checks, columns=["Check", "Value"])


def instructor_profile(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Course-level instructor attributes.

    Transactions link a course to many teachers, so a course's "instructor
    profile" is summarised over the *distinct* teachers who delivered it.
    Unweighted means are used deliberately: weighting by transaction volume
    would let the target (enrollment count) leak into the features.
    """
    txns, teachers, courses = frames["Transactions"], frames["Teachers"], frames["Courses"]

    pairs = (
        txns[["CourseID", "TeacherID"]]
        .drop_duplicates()
        .merge(teachers[["TeacherID", "Expertise", "YearsOfExperience", "TeacherRating"]], on="TeacherID")
        .merge(courses[["CourseID", "CourseCategory"]], on="CourseID")
    )
    pairs["ExpertiseMatch"] = (pairs.Expertise == pairs.CourseCategory).astype(int)

    prof = pairs.groupby("CourseID").agg(
        InstructorCount=("TeacherID", "nunique"),
        InstructorAvgRating=("TeacherRating", "mean"),
        InstructorMaxRating=("TeacherRating", "max"),
        InstructorAvgExperience=("YearsOfExperience", "mean"),
        InstructorMaxExperience=("YearsOfExperience", "max"),
        InstructorExpertiseMatch=("ExpertiseMatch", "mean"),
    )

    # Delivery concentration (Herfindahl index over each course's teacher mix).
    shares = txns.groupby(["CourseID", "TeacherID"]).size().rename("n").reset_index()
    shares["share"] = shares.n / shares.groupby("CourseID").n.transform("sum")
    prof["InstructorHHI"] = shares.groupby("CourseID").share.apply(lambda s: (s**2).sum())

    return prof.reset_index()


def build_course_level(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per course with realised demand, revenue and instructor profile."""
    courses, txns = frames["Courses"], frames["Transactions"]

    outcomes = txns.groupby("CourseID").agg(
        EnrollmentCount=("TransactionID", "count"),
        CourseRevenue=("Amount", "sum"),
        FirstEnrollment=("TransactionDate", "min"),
        LastEnrollment=("TransactionDate", "max"),
        UniqueLearners=("UserID", "nunique"),
    )

    df = courses.merge(outcomes, on="CourseID", how="left").merge(
        instructor_profile(frames), on="CourseID", how="left"
    )
    df[["EnrollmentCount", "CourseRevenue", "UniqueLearners"]] = df[
        ["EnrollmentCount", "CourseRevenue", "UniqueLearners"]
    ].fillna(0)
    df["RevenuePerEnrollment"] = df.CourseRevenue / df.EnrollmentCount.replace(0, pd.NA)
    df["RevenuePerEnrollment"] = df.RevenuePerEnrollment.fillna(0.0)
    return df


def build_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Dense course x month panel; empty cells are filled with zero demand."""
    courses, txns = frames["Courses"], frames["Transactions"]

    t = txns.copy()
    t["Month"] = t.TransactionDate.values.astype("datetime64[M]")

    obs = t.groupby(["CourseID", "Month"]).agg(
        EnrollmentCount=("TransactionID", "count"), Revenue=("Amount", "sum")
    )

    months = pd.date_range(t.Month.min(), t.Month.max(), freq="MS")
    idx = pd.MultiIndex.from_product([courses.CourseID, months], names=["CourseID", "Month"])
    panel = obs.reindex(idx, fill_value=0).reset_index()

    panel["MonthIndex"] = panel.groupby("CourseID").cumcount()
    panel["MonthNum"] = panel.Month.dt.month
    return panel.sort_values(["CourseID", "Month"]).reset_index(drop=True)


def run() -> dict[str, pd.DataFrame]:
    frames = load_raw()
    audit = validate(frames)
    course_level = build_course_level(frames)
    panel = build_panel(frames)

    audit.to_csv(config.TABLES / "01_data_quality_audit.csv", index=False)
    course_level.to_csv(config.PROCESSED / "course_level.csv", index=False)
    panel.to_csv(config.PROCESSED / "panel_course_month.csv", index=False)

    frames["_audit"] = audit
    frames["_course_level"] = course_level
    frames["_panel"] = panel
    return frames


if __name__ == "__main__":
    out = run()
    print(out["_audit"].to_string(index=False))
    print("\ncourse_level:", out["_course_level"].shape)
    print("panel:", out["_panel"].shape)
