"""Central configuration for the EduPro predictive-modelling project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_XLSX = ROOT / "data" / "raw" / "EduPro Online Platform.xlsx"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

for _d in (PROCESSED, MODELS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Sheet names in the source workbook
SHEETS = ["Users", "Teachers", "Courses", "Transactions"]

# Feature groups used by the course-level ("cold start") demand model.
NUMERIC_FEATURES = [
    "CoursePrice",
    "CourseDuration",
    "CourseRating",
    "InstructorAvgRating",
    "InstructorAvgExperience",
    "InstructorExpertiseMatch",
    "PriceBandOrd",
    "DurationBucketOrd",
    "RatingTierOrd",
    "CourseLevelOrd",
    "ExperienceBucketOrd",
]
CATEGORICAL_FEATURES = ["CourseCategory", "CourseType", "CourseLevel"]

# Panel (course x month) model
PANEL_LAG_FEATURES = [
    "Enroll_lag1",
    "Enroll_lag2",
    "Enroll_lag3",
    "Enroll_roll3",
    "Enroll_expanding_mean",
    "Revenue_lag1",
    "Revenue_roll3",
    "RevenuePerEnroll_expanding",
    "MonthIndex",
]

PANEL_TEST_MONTHS = 3  # hold out the final 3 months for out-of-time evaluation
