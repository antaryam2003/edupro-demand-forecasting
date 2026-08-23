# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

An end-to-end data science project that turns EduPro's historical course, instructor
and transaction records into forward-looking demand and revenue intelligence — and,
just as importantly, quantifies **how far that data can honestly be trusted**.

> **Headline result.** Course-level enrollment demand in this dataset is statistically
> indistinguishable from random allocation (χ² = 55.52, df = 59, *p* = 0.60; index of
> dispersion = 0.94). No model — linear, penalised or ensemble — beats a mean baseline
> out of sample. Revenue, by contrast, is predictable to R² ≈ 0.99, but only because
> `Revenue ≡ Price × Enrollments` holds exactly; strip price out and R² collapses to
> −0.10. The actionable finding is therefore not *which course will be popular* but
> that **64% of all enrollments are monetised at zero**, and measured price elasticity
> is statistically indistinguishable from zero (−0.010, 95% CI [−0.035, +0.015]).

---

## Contents

| Deliverable | Location |
|---|---|
| Research paper | [`reports/research_paper.md`](reports/research_paper.md) |
| Executive summary (government / public-sector stakeholders) | [`reports/executive_summary.md`](reports/executive_summary.md) |
| Streamlit dashboard | [`app.py`](app.py) |
| Reproducible pipeline | [`src/`](src/) |
| Result tables (CSV) | [`reports/tables/`](reports/tables/) |
| Figures (PNG, 200 dpi) | [`reports/figures/`](reports/figures/) |
| Full run log | [`reports/pipeline_log.txt`](reports/pipeline_log.txt) |

---

## Quick start

```bash
git clone <your-repo-url>
cd EduPro

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt

python -m src.run_pipeline    # ~3.5 min: prepare, diagnose, model, persist
python -m src.figures         # render the paper figures
streamlit run app.py          # launch the dashboard
```

The app reads only precomputed artefacts, so it starts instantly. Commit
`data/processed/`, `models/` and `reports/tables/` if you deploy to Streamlit
Community Cloud.

---

## The data

`data/raw/EduPro Online Platform.xlsx`, four sheets:

| Sheet | Rows | Key fields |
|---|---|---|
| Users | 3,000 | UserID, Age, Gender |
| Teachers | 60 | TeacherID, Expertise, YearsOfExperience, TeacherRating |
| Courses | 60 | CourseID, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating |
| Transactions | 10,000 | TransactionID, UserID, CourseID, TeacherID, TransactionDate, Amount, PaymentMethod |

Coverage is calendar year 2025. The data passes every referential-integrity check:
no missing cells, no orphan foreign keys, no duplicate transactions, no negative
amounts, and every course has transactions
([`reports/tables/01_data_quality_audit.csv`](reports/tables/01_data_quality_audit.csv)).

---

## Pipeline

```
src/
├── config.py       paths, feature groups, random seed
├── data_prep.py    load → validate → course-level table (60) + course-month panel (720)
├── features.py     price bands, duration buckets, rating tiers, level/experience
│                   encodings, expertise-match score, lagged history features
├── diagnostics.py  noise-floor and predictability tests (the analytical core)
├── models.py       preprocessing, model zoo, CV harness, importance, persistence
├── figures.py      publication figures
└── run_pipeline.py orchestrates stages 1–8
```

### Two modelling problems, kept separate

They answer different questions and deserve different validation.

**1. Cold start (cross-sectional, n = 60).** *"If we launch this course, what will it do?"*
Attributes only — no history, because a new course has none. Validated with
repeated 5-fold CV (5 repeats) **and** leave-one-out CV, because a single hold-out
split of 60 rows is far too noisy to interpret.

**2. Operational forecast (panel, n = 540 course-months).** *"What will each course do next month?"*
Adds lagged history: `Enroll_lag1..3`, 3-month rolling mean, expanding course mean,
`Revenue_lag1`, rolling revenue, expanding revenue-per-enrollment. Validated on a
strict **out-of-time split** — trained on the earlier months, tested on the final three.

Every comparison table carries a **mean baseline**, plus (for the panel) a
last-month-naive and a course-history-mean baseline. A model that does not beat
those has learned nothing regardless of its architecture.

### Guarding against leakage

- Instructor features average over the *distinct* teachers who delivered a course, never
  weighted by transaction volume — volume weighting would leak the target into the features.
- Every panel lag is produced with `groupby(...).shift(...)`, so row *t* only ever sees
  months strictly before *t*. Warm-up rows with an incomplete lag window are dropped.
- Scaling and one-hot encoding live inside the `Pipeline`, so they are refit within each
  CV fold rather than on the full dataset.

---

## Results

### Enrollment demand — cold start (out-of-sample)

| Model | CV MAE | CV RMSE | CV R² | LOO R² | RMSE skill vs baseline |
|---|---|---|---|---|---|
| Baseline (mean) | 10.17 | 12.60 | −0.029 | −0.034 | 0.0% |
| Linear Regression | 11.84 | 15.01 | −0.466 | −0.219 | −19.1% |
| Ridge Regression | 10.11 | 12.27 | 0.023 | 0.085 | +2.6% |
| Lasso Regression | 10.27 | 12.45 | −0.005 | 0.040 | +1.2% |
| **Random Forest** | **9.72** | **12.10** | **0.051** | **0.092** | **+4.0%** |
| Gradient Boosting | 11.33 | 13.73 | −0.223 | −0.032 | −9.0% |

The best model explains ~5% of variance (±3.5 SD across repeats) and cuts mean
absolute error by 0.45 enrollments on a mean of 167 — **0.3%**. Statistically
fragile, economically irrelevant.

### Course revenue — and the ablation that explains it

| Feature set | Best model | CV R² |
|---|---|---|
| All features (price included) | Lasso | **0.991** |
| **Price features removed** | Ridge | **−0.104** |

Revenue looks brilliantly predictable and is not. `Revenue ≡ Price × Enrollments`
holds to floating-point exactness (max abs error 3.6 × 10⁻¹²), so the model is
recovering arithmetic. Remove price and nothing is left. Any revenue R² reported
without this ablation is misleading.

### Operational forecast — out-of-time

Monthly enrollments: **every model scores negative R²**; the best result of any kind
is the course-history-mean baseline at −0.014. Monthly revenue reaches R² = 0.923
(Lasso) — but the course-history-mean baseline alone reaches 0.898, so machine
learning adds 2.5 points on top of "this course charges what it charged last month."

### Category revenue

Out-of-fold predictions aggregated to category level land within **1–11%** for the ten
categories earning more than 1,000 units, and the portfolio total is accurate to
**+1.7%**. The unconstrained linear model predicts *negative* revenue for Marketing
(a zero-revenue category); a non-negativity floor is required before operational use.

---

## Why the models cannot do better: the noise floor

Ten thousand enrollments spread over 60 courses produce, **under a null of equal
attractiveness**, a standard deviation of √(N·(1/K)·(1−1/K)) = 12.80 enrollments per
course. The observed standard deviation is **12.52** — no larger. Total variance
(156.8) is actually *below* the sampling variance (163.9), leaving systematic
variance of zero and a **maximum attainable R² of 0.00**.

Supporting tests:

| Test | Result | Reading |
|---|---|---|
| Uniform-allocation χ² | χ²=55.52, df=59, *p*=0.60 | Cannot reject equal demand |
| Index of dispersion | 0.941 | 1.00 = pure Poisson noise |
| Course × month independence | *p* = 0.84 | No course-specific seasonality |
| Monthly trend | slope +0.97/mo, *p* = 0.79 | No trend |
| Feature screen (15 candidates) | 1 survives Bonferroni | Consistent with screening noise |
| Price elasticity | −0.010, 95% CI [−0.035, +0.015] | Demand is price-insensitive here |

One caveat stated plainly: `QualitySignal` (course rating × instructor rating) does
survive Bonferroni correction (ρ = 0.38, adjusted *p* = 0.04). The omnibus dispersion
test and the out-of-sample results both argue it is a selection artefact — the single
strongest of fifteen screened correlations, in a dataset whose overall spread is
*narrower* than the random null. Out-of-sample validation is the tiebreaker, and it
sides with noise.

---

## What the analysis does support

1. **Monetisation coverage, not course selection, is the revenue lever.** 38 of 60
   courses are free; they absorb 64% of all enrollments and return nothing.
   Marketing (5/5 free) earns 0 on 806 enrollments; Machine Learning (4/5 free) earns 133 on 819.
2. **Price can rise with little measured demand risk.** Elasticity is bounded near zero
   across a 0.78–490.90 price range. This is observational, not experimental — the
   right response is a randomised pricing test, not a catalogue-wide reprice.
3. **Forecast the platform, not the course.** Aggregate monthly volume is stable
   (833 ± 40); per-course monthly forecasts are not defensible.
4. **Delivery load is dangerously concentrated.** Two instructors handle 61% of all
   transactions across 55 of 60 courses — a key-person risk that is invisible in
   revenue reporting and unrelated to demand (r = 0.002).
5. **Instrument the drivers.** The dataset records outcomes, not causes. Impressions,
   click-through, marketing spend per course, search volume, price *change* history,
   completions and refunds are what would move R² off the floor. The pipeline
   re-runs unchanged when those columns arrive.

---

## Dashboard modules

`streamlit run app.py`

1. **Portfolio Overview** — catalogue KPIs, demand against the random-allocation band, revenue concentration
2. **Course Demand Predictor** — score an unlaunched course; returns calibrated 80% / 95% prediction intervals and one-factor sensitivity sweeps
3. **Revenue Forecast** — platform, course and category outlook with uncertainty bands
4. **Feature Importance Explorer** — permutation importance, the Bonferroni-corrected screen, and standardised Ridge coefficients
5. **Category Demand Comparison** — enrollment share vs revenue share, the monetisation gap
6. **Pricing Scenario Simulator** — conversion and repricing waterfall with an adjustable elasticity stress test
7. **Model Reliability** — the noise floor, every model comparison table, the data audit, and the instrumentation roadmap

---

## Reproducibility

`RANDOM_STATE = 42` throughout. `python -m src.run_pipeline` regenerates every table,
model and number quoted above from the raw workbook in roughly 3.5 minutes; the
complete console transcript is committed at `reports/pipeline_log.txt`.

---

## Project context

Built for the Unified Mentor data science programme. The public-sector framing in the
executive summary is grounded in the City of Toronto's
[Parks, Forestry & Recreation division](https://www.toronto.ca/city-government/accountability-operations-customer-service/city-administration/staff-directory-divisions-and-customer-service/parks-forestry-recreation/),
which runs a registered-program catalogue across more than 1,500 parks and 8,000
hectares — the same catalogue-planning problem in a public setting.
