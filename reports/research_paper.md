# Predictive Modeling for Course Demand and Revenue Forecasting on EduPro

### A Noise-Floor Analysis of Catalogue Demand, and What It Implies for Pricing Strategy

**Author:** Antarya Mondal
**Programme:** Unified Mentor — Data Science
**Date:** 23 August 2026
**Repository:** `EduPro/` — full pipeline, tables, figures and dashboard

---

## Abstract

Online learning platforms increasingly seek predictive models to decide which courses
to launch, how to price them, and which instructors to onboard. This study builds a
complete demand- and revenue-forecasting pipeline over the EduPro Online Platform
dataset (60 courses, 60 instructors, 3,000 learners, 10,000 transactions across
calendar 2025) and evaluates it against explicit statistical baselines.

The principal result is negative, and precisely quantified. Course-level enrollment
demand is statistically indistinguishable from uniform random allocation
(χ² = 55.52, df = 59, *p* = 0.60; index of dispersion = 0.941). A variance
decomposition shows total cross-course variance (156.8) falls *below* the sampling
variance implied by multinomial allocation (163.9), which places the maximum
attainable out-of-sample R² at **0.00**. Consistent with that bound, no model in a
six-model zoo spanning linear, penalised and ensemble families achieves practically
meaningful skill: the best (Random Forest) reaches cross-validated R² = 0.051 ± 0.035
and reduces mean absolute error by 0.45 enrollments on a base of 167 — an improvement
of 0.3%. On an out-of-time panel split, every model scores negative R².

Course revenue appears highly predictable (R² = 0.991), but an ablation demonstrates
this is an artefact: `Revenue ≡ CoursePrice × EnrollmentCount` holds to floating-point
exactness, and removing price features collapses performance to R² = −0.104. We argue
that reporting revenue R² without this ablation constitutes a material overstatement of
model capability.

Three findings do survive scrutiny and carry direct commercial consequence. First,
38 of 60 courses (63%) are free; they absorb 64.0% of all enrollments and return zero
revenue. Second, price elasticity of demand across the observed 0.78–490.90 range is
−0.010 with a 95% confidence interval of [−0.035, +0.015] — bounded near zero, meaning
demand did not measurably fall as price rose. Third, two of sixty instructors deliver
60.9% of all transactions across 55 of 60 courses, a concentration risk invisible in
revenue reporting. We conclude that for this platform the decision-relevant lever is
**monetisation coverage, not course selection**, and we specify the instrumentation
required to make demand prediction viable in future.

**Keywords:** demand forecasting, revenue prediction, noise floor, predictability
limits, price elasticity, educational analytics, baseline comparison

---

## 1. Introduction

### 1.1 Motivation

For an online learning platform, forward visibility into demand governs three
recurring decisions: which new courses to launch, how to price the existing catalogue,
and which instructors to onboard and retain. Each is currently made at EduPro from
historical intuition rather than quantified forecasts, and each carries capital or
opportunity cost when made wrongly.

The stated objective of this project is to move EduPro from reactive reporting to
proactive planning by predicting three targets — enrollment count per course, revenue
per course, and aggregated revenue per category — from course attributes, instructor
attributes and historical transaction behaviour.

### 1.2 The question behind the question

A demand model is only useful if demand is, in fact, driven by things the platform
can observe and control. Before asking *"which model predicts enrollments best?"*, a
prior question must be settled: *"how much of enrollment variation is explainable at
all?"*

This distinction matters more than it is usually given credit for. A model reporting
R² = 0.05 admits two readings that call for opposite responses:

1. **The model is inadequate** — try more features, more capacity, better tuning.
2. **The data contains no signal** — no amount of modelling effort will help, and the
   correct action is to instrument new data sources.

Machine-learning practice defaults to reading (1). This study establishes a
**noise floor** — a variance-decomposition bound on attainable R² — which
discriminates between the two readings before any model is fitted. We regard
establishing that bound as the primary methodological contribution here.

### 1.3 Contributions

1. A complete, reproducible pipeline from raw workbook to deployed dashboard,
   implementing the full specified methodology (feature engineering, baseline and
   advanced models, MAE/RMSE/R² evaluation, feature-importance analysis).
2. A noise-floor diagnostic that bounds attainable predictive performance *a priori*,
   with supporting dispersion, independence and multiple-testing-corrected screens.
3. An ablation protocol that separates genuine predictive skill from recovered
   accounting identities in revenue models.
4. A quantified statement of the commercial levers the evidence supports, and an
   instrumentation roadmap for the ones it does not yet.

---

## 2. Background

### 2.1 Demand prediction in course catalogues

Course demand is conventionally modelled as a function of content attributes (topic,
level, duration), commercial attributes (price, free/paid), quality signals (ratings)
and supply-side attributes (instructor reputation, experience). The implicit causal
model is that learners survey a catalogue and select on observable quality-adjusted
value.

That model presumes learners *see* the catalogue in a way the data records. In
practice, the dominant determinants of which course a learner enrols in are typically
placement, recommendation, search ranking and marketing exposure — none of which
appear in transaction-level records. This gap between recorded outcomes and causal
drivers is central to the results reported here.

### 2.2 Baselines and the credibility of forecasting claims

Forecasting literature has long insisted that a model be compared against naive
alternatives — the mean, the last observation, a seasonal naive — because complex
models routinely fail to beat them on noisy series. We adopt that convention
throughout: every comparison table in this paper reports a mean baseline, and the
panel tables add last-month-naive and course-history-mean baselines. Skill is reported
as proportional RMSE reduction against the baseline, not as raw R².

### 2.3 Public-sector relevance

The same catalogue-planning problem recurs in public recreation programming. The City
of Toronto's Parks, Forestry & Recreation division stewards more than 1,500 parks and
8,000 hectares, and its Community Recreation unit operates a registered-program
catalogue across recreation centres, rinks and pools. Its stated priorities include
modernising front-line client services and developing a Parks and Recreation
Facilities Master Plan — both of which require exactly the forecast this study
attempts: which programs will fill, and what they will return. The methodological
caution developed below applies with more force in that setting, where a
confidently-wrong forecast misallocates public funds.

---

## 3. Data

### 3.1 Source

`EduPro Online Platform.xlsx`, four sheets covering calendar year 2025
(2025-01-01 to 2025-12-30).

| Sheet | Rows | Fields used |
|---|---:|---|
| Users | 3,000 | UserID, Age, Gender |
| Teachers | 60 | TeacherID, Expertise, YearsOfExperience, TeacherRating |
| Courses | 60 | CourseID, CourseCategory, CourseType, CourseLevel, CoursePrice, CourseDuration, CourseRating |
| Transactions | 10,000 | TransactionID, UserID, CourseID, TeacherID, TransactionDate, Amount, PaymentMethod |

### 3.2 Quality audit

The dataset is unusually clean. All fourteen integrity checks pass: zero missing cells
across all sheets, zero orphan foreign keys in any of the three transaction
relationships, zero duplicate identifiers, zero negative amounts, zero duplicate
(user, course) pairs, and every one of the 60 courses carries transactions. Full audit:
`reports/tables/01_data_quality_audit.csv`.

Cleanliness of this order is itself informative — it is characteristic of
synthetically generated records rather than production logs, and it anticipates the
distributional findings of Section 5.

### 3.3 Descriptive structure

| Quantity | Value |
|---|---|
| Enrollments per course | mean 166.7, SD 12.5, range 140–196 |
| Total revenue | 911,323.47 |
| Free courses | 38 of 60 (63%), 6,403 enrollments, revenue 0 |
| Paid courses | 22 of 60 (37%), 3,597 enrollments, mean transaction 253.36 |
| Paid price range | 0.78 – 490.90 |
| Course rating | mean 3.10, SD 1.17 |
| Course duration | mean 27.6 h, SD 16.1 h |
| Instructor rating | mean 3.13, SD 0.95 |
| Instructor experience | mean 6.3 y, range 1–24 y |
| Categories | 12, exactly 5 courses each |

Two structural features shape the analysis:

**The revenue identity.** Every transaction Amount equals the CoursePrice of the
course exactly (maximum absolute deviation 3.6 × 10⁻¹² over 10,000 rows). There are no
discounts, coupons, refunds or partial payments. Consequently
`CourseRevenue ≡ CoursePrice × EnrollmentCount` is an accounting identity, not an
empirical relationship.

**The instructor-course mapping is many-to-many.** Transactions attach a TeacherID,
but each course is delivered by a median of 14 distinct teachers and each teacher
appears across a median of 15 courses. There is no stable course owner. We therefore
define course-level instructor features as unweighted averages over the *distinct*
teachers who delivered the course — deliberately unweighted, because weighting by
transaction volume would leak the target variable into the features.

---

## 4. Methodology

### 4.1 Analysis tables

Two tables are constructed, addressing different decisions:

- **Course-level (cross-sectional), n = 60.** One row per course, attributes only.
  Supports the launch decision: *if we build this, what will it do?* A new course has
  no history, so no historical features may be used.
- **Course-month panel, n = 720 (540 after lag warm-up).** One row per course per
  month, dense (no empty cells). Supports operational planning: *what will each course
  do next month?*

### 4.2 Feature engineering

Implemented in `src/features.py`, in the three families specified by the brief.

**Course features.** Price bands (Free / Low <100 / Medium <300 / High ≥300);
duration buckets (Short <10h / Medium <25h / Long <40h / Extended ≥40h); rating tiers
(Poor <2 / Fair <3 / Good <4 / Excellent ≥4); ordinal course-level encoding
(Beginner 0, Intermediate 1, Advanced 2); price-per-hour; a paid indicator.

**Instructor features.** Experience buckets (Junior <3y / Mid <7y / Senior <12y /
Veteran ≥12y); mean and maximum instructor rating; mean and maximum experience;
**expertise–category match score**, the share of the delivery team whose Expertise
equals the CourseCategory (catalogue mean 0.478, against a chance baseline of 0.083
given 12 categories); delivery concentration as a Herfindahl index over the teacher mix;
`QualitySignal`, the product of course rating and mean instructor rating.

**Historical performance features (panel only).** Enrollment lags at 1, 2 and 3
months; 3-month rolling mean of enrollments and revenue; expanding course mean;
revenue lag 1; expanding revenue-per-enrollment; month index.

### 4.3 Leakage controls

Three specific hazards were identified and closed:

1. *Volume-weighted instructor features* would encode the target. Averages are taken
   over distinct teachers instead.
2. *Contemporaneous history* would trivially predict the target. All panel features
   use `groupby(course).shift(k)`, so row *t* sees only months strictly before *t*;
   rows with an incomplete 3-month window are dropped (720 → 540).
3. *Preprocessing fitted on the full dataset* would leak distributional information
   across folds. Scaling and one-hot encoding are inside the scikit-learn `Pipeline`
   and are therefore refit within every fold.

### 4.4 Models

Six estimators, from the brief's baseline and advanced tiers:

| Tier | Model | Configuration |
|---|---|---|
| Baseline | Mean predictor | `DummyRegressor(strategy="mean")` |
| Baseline | Linear Regression | OLS |
| Baseline | Ridge Regression | `RidgeCV`, 25 alphas, 10⁻³–10³ |
| Baseline | Lasso Regression | `LassoCV`, 25 alphas, nested CV |
| Advanced | Random Forest | 300 trees, min_samples_leaf 2, max_features sqrt |
| Advanced | Gradient Boosting | 200 stages, lr 0.05, depth 3, subsample 0.9 |

Preprocessing: `StandardScaler` on 13 numeric features, `OneHotEncoder(drop="first")`
on 3 categoricals. `RANDOM_STATE = 42` throughout.

### 4.5 Validation

**Cross-sectional.** With n = 60, a single hold-out split of 12 rows is too unstable
to interpret. We therefore use repeated 5-fold CV (5 repeats, different shuffles),
pooling out-of-fold predictions within each repeat before scoring, and report the mean
and standard deviation of the metric across repeats. Leave-one-out CV is run alongside
as a second, lower-bias estimate.

**Panel.** Strict out-of-time split: train on the earlier months, test on the final
three. No shuffling — the test set lies entirely in the future of the training set.

**Metrics.** MAE (average error magnitude), RMSE (penalises large errors), R²
(variance explained). We add **skill**, the proportional RMSE reduction against the
mean baseline, because R² alone obscures how small an absolute improvement can be.

### 4.6 The noise floor

For a catalogue of *K* courses receiving *N* total enrollments, the null hypothesis
that all courses are equally attractive implies multinomial allocation with per-course
variance

&nbsp;&nbsp;&nbsp;&nbsp;σ²_sampling = N · (1/K) · (1 − 1/K)

Observed cross-course variance decomposes as σ²_total = σ²_systematic + σ²_sampling.
Since only the systematic component is in principle explainable by covariates, the
maximum attainable out-of-sample R² is

&nbsp;&nbsp;&nbsp;&nbsp;R²_max = σ²_systematic / σ²_total = 1 − σ²_sampling / σ²_total

This bound is computed before any model is fitted, and every model score is reported
against it.

---

## 5. Predictability diagnostics

### 5.1 Demand is indistinguishable from random allocation

With N = 10,000 and K = 60, the null predicts 166.7 enrollments per course with a
standard deviation of 12.80. Observed: mean 166.7, **standard deviation 12.52** — no
larger than chance.

| Test | Statistic | *p* | Reading |
|---|---|---|---|
| Uniform-allocation χ² | 55.52 (df 59) | 0.60 | Cannot reject equal demand |
| Index of dispersion | 0.941 | — | 1.00 = pure Poisson noise |
| Course × month independence | χ² = 612.9 (df 649) | 0.84 | No course-specific seasonality |
| Monthly trend (OLS slope) | +0.97 /month | 0.79 | No trend |
| Monthly total dispersion | χ² = 21.11 (df 11) | 0.032 | Mild platform-level over-dispersion |

**Noise floor.** σ²_total = 156.8, σ²_sampling = 163.9. Total variance is *below*
sampling variance, giving σ²_systematic = 0 and **R²_max = 0.00**.

The only test returning nominal significance is platform-wide monthly dispersion
(*p* = 0.032): aggregate volume varies by ±40 enrollments per month against a Poisson
expectation of ±29. With twelve observations this is marginal, and it concerns
platform totals rather than any course-level driver.

*(Figure 1 — demand distribution against the random-allocation null; Figure 6 —
temporal structure.)*

### 5.2 Feature screen with multiple-testing correction

Fifteen candidate drivers were screened univariately (Spearman for numeric, ANOVA for
categorical) with Bonferroni correction.

| Feature | Test | Statistic | *p* raw | *p* corrected | Significant |
|---|---|---:|---:|---:|:--:|
| QualitySignal | Spearman ρ | +0.381 | 0.0027 | **0.040** | **yes** |
| CourseRating | Spearman ρ | +0.281 | 0.029 | 0.441 | no |
| InstructorAvgExperience | Spearman ρ | +0.258 | 0.047 | 0.700 | no |
| RatingTier | ANOVA F | 2.61 | 0.060 | 0.901 | no |
| InstructorAvgRating | Spearman ρ | +0.228 | 0.080 | 1.000 | no |
| CourseType | ANOVA F | 2.27 | 0.137 | 1.000 | no |
| CoursePrice | Spearman ρ | −0.159 | 0.225 | 1.000 | no |
| CourseCategory | ANOVA F | 1.28 | 0.263 | 1.000 | no |
| CourseLevel | ANOVA F | 1.33 | 0.273 | 1.000 | no |
| CourseDuration | Spearman ρ | −0.074 | 0.577 | 1.000 | no |

*(Six further features omitted; all corrected p = 1.000. Full table:
`reports/tables/02_feature_screen.csv`.)*

**On the one survivor.** `QualitySignal` — the product of course rating and mean
instructor rating — survives correction at ρ = 0.381, adjusted *p* = 0.040. We report
this rather than suppress it, and we judge it a selection artefact for three reasons.
First, it is the largest of fifteen screened correlations, and Bonferroni bounds the
family-wise error rate rather than eliminating it. Second, the omnibus dispersion test
is decisive in the opposite direction: if a genuine driver existed, cross-course
spread would *exceed* the random null, and it does not — it is slightly narrower.
Third, and conclusively, out-of-sample validation is the arbiter: a real driver
improves held-out prediction. Section 6.1 shows it does not.

This is the reason the noise floor is computed first. Without it, ρ = 0.38 with
*p* = 0.003 reads as a finding.

### 5.3 Price elasticity

A constant-elasticity specification log(enrollments) = a + b·log(price) over the 22
paid courses gives:

&nbsp;&nbsp;&nbsp;&nbsp;**b = −0.0096, 95% CI [−0.0346, +0.0155]**, *p* = 0.44, R² = 0.031

The interval is the substantive result. Elasticity is not merely "not significant" —
it is *bounded* within ±0.035 across a price range spanning 0.78 to 490.90. A 10%
price increase is estimated to cost between 0.35% and −0.16% of demand. Over this
catalogue and this range, demand did not respond to price.

Two caveats are essential. Prices were not randomly assigned, so this is an
observational estimate vulnerable to confounding; and the finding is inseparable from
Section 5.1 — demand did not respond to price because it did not respond to anything.
The appropriate response is a randomised pricing experiment, not a catalogue-wide
reprice.

*(Figure 2 — revenue structure and elasticity.)*

---

## 6. Modelling results

### 6.1 Enrollment demand — cold start

| Model | CV MAE | CV RMSE | CV R² (± SD) | LOO R² | Skill |
|---|---:|---:|---:|---:|---:|
| Baseline (mean) | 10.17 | 12.60 | −0.029 ± 0.009 | −0.034 | 0.0% |
| Linear Regression | 11.84 | 15.01 | −0.466 ± 0.213 | −0.219 | −19.1% |
| Ridge Regression | 10.11 | 12.27 | 0.023 ± 0.064 | 0.085 | +2.6% |
| Lasso Regression | 10.27 | 12.45 | −0.005 ± 0.031 | 0.040 | +1.2% |
| **Random Forest** | **9.72** | **12.10** | **0.051 ± 0.035** | **0.092** | **+4.0%** |
| Gradient Boosting | 11.33 | 13.73 | −0.223 ± 0.067 | −0.032 | −9.0% |

Three observations.

**Nothing clears the noise floor meaningfully.** The best model reaches R² = 0.051
against a bound of 0.00 — within 1.5 standard deviations of zero across repeats, and
consistent with the sampling error of the R² estimate itself on 60 rows.

**The improvement is economically void.** Random Forest reduces MAE from 10.17 to
9.72: 0.45 enrollments on a mean of 166.7, or **0.3%**. No launch, pricing or staffing
decision changes on that margin.

**Unregularised Linear Regression is actively harmful** (R² = −0.466). With 13 numeric
features, 3 one-hot-expanded categoricals and 60 rows, OLS overfits severely. That
Ridge and Lasso recover to ≈0 is a textbook demonstration that regularisation is
doing exactly what it should — shrinking toward the mean, which is the correct answer.

The `QualitySignal` question from Section 5.2 resolves here: models with access to it
do not beat the mean baseline out of sample. Its in-sample correlation does not
generalise.

*(Figures 4 and 5.)*

### 6.2 Course revenue, and the ablation

| Feature set | Best model | CV MAE | CV RMSE | CV R² | Skill |
|---|---|---:|---:|---:|---:|
| All features (price included) | Lasso | 1,466 | 2,406 | **0.991** | +90.6% |
| **Price features removed** | Ridge | 20,868 | 26,446 | **−0.104** | −3.2% |

Taken alone, the first row is an excellent result — R² = 0.991 on out-of-fold
predictions, with linear models beating Random Forest (0.889) because the underlying
relationship is exactly linear in price.

That is the tell. Since `Revenue ≡ Price × Enrollments` holds exactly and enrollments
are near-constant at 166.7, revenue is essentially `166.7 × Price`. The model recovers
a multiplication.

The ablation confirms it. Removing CoursePrice, PricePerHour, PriceBandOrd and
CourseType drops every model below zero; the best is Ridge at −0.104, worse than
predicting the mean. **All apparent revenue predictability comes from price, none from
demand behaviour.**

We would argue this ablation should be mandatory wherever a target is an arithmetic
function of an available feature. Without it, this project could have reported
"R² = 0.99 revenue forecasting" as a headline. It would have been true and
substantively worthless.

### 6.3 Operational forecast — out-of-time panel

**Monthly enrollments** (train earlier months, test final three):

| Model | MAE | RMSE | R² | Skill |
|---|---:|---:|---:|---:|
| Baseline (global mean) | 2.833 | 3.704 | −0.022 | 0.0% |
| Baseline (course history mean) | 2.786 | 3.689 | −0.014 | +0.4% |
| Naive (last month) | 3.989 | 5.082 | −0.925 | −37.2% |
| Linear Regression | 3.119 | 3.985 | −0.184 | −7.6% |
| Ridge Regression | 2.863 | 3.721 | −0.032 | −0.5% |
| Lasso Regression | 2.833 | 3.704 | −0.022 | 0.0% |
| Random Forest | 2.884 | 3.756 | −0.052 | −1.4% |
| Gradient Boosting | 3.156 | 4.031 | −0.211 | −8.8% |

**Every model scores negative R².** The best result of any kind is a baseline. Lasso
ties the global mean exactly — it shrank every coefficient to zero, which on this data
is the correct model. Last-month-naive is markedly the worst (−0.925), the signature
of a pure-noise series: chasing last month's fluctuation doubles the error.

**Monthly revenue:**

| Model | MAE | RMSE | R² | Skill |
|---|---:|---:|---:|---:|
| Baseline (global mean) | 1,703.9 | 2,192.6 | −0.000 | 0.0% |
| Baseline (course history mean) | 298.5 | 701.2 | 0.898 | +68.0% |
| Naive (last month) | 395.1 | 991.3 | 0.796 | +54.8% |
| **Lasso Regression** | **264.6** | **607.4** | **0.923** | **+72.3%** |
| Random Forest | 260.1 | 656.9 | 0.910 | +70.0% |
| Ridge Regression | 307.1 | 626.4 | 0.918 | +71.4% |

R² = 0.923 is real but must be read correctly: the course-history-mean baseline
already reaches 0.898. Machine learning contributes **2.5 percentage points** over
"this course will charge what it charged last month." Again, price — not demand — is
doing the work.

### 6.4 Category-level revenue

Out-of-fold course predictions aggregated to category:

| Category | Actual | Predicted | APE |
|---|---:|---:|---:|
| Artificial Intelligence | 202,751 | 208,734 | 3.0% |
| Business | 181,528 | 179,470 | 1.1% |
| Project Management | 169,103 | 163,770 | 3.2% |
| Programming | 77,454 | 84,569 | 9.2% |
| Data Science | 77,052 | 80,269 | 4.2% |
| Cybersecurity | 59,139 | 63,911 | 8.1% |
| Digital Marketing | 56,261 | 57,838 | 2.8% |
| Design | 43,114 | 46,864 | 8.7% |
| Finance | 28,106 | 25,056 | 10.9% |
| Web Development | 16,682 | 15,564 | 6.7% |
| Machine Learning | 133 | 1,718 | 1188% |
| Marketing | 0 | **−1,109** | n/a |

For the ten categories earning above 1,000, mean APE is 5.8% and median 5.4%.
Portfolio total is accurate to **+1.7%** (911,323 actual vs 926,655 predicted).

Two honest caveats. Machine Learning's 1188% error reflects a 133-unit denominator
(four of its five courses are free, the fifth priced at 0.78), not a model failure —
but it does show percentage error is meaningless at near-zero revenue. More seriously,
the unconstrained linear model predicts **negative revenue** for Marketing. Revenue is
non-negative by construction; a floor at zero, or a log-link / Tweedie GLM, is
required before operational use. This is a genuine defect in the current artefact.

Aggregation works here for the same reason everything else does: summing price ×
near-constant demand over five courses, with idiosyncratic error partly cancelling.

### 6.5 Feature importance

Permutation importance on held-out folds, enrollment target:

| Feature | Importance (MSE reduction) | SD |
|---|---:|---:|
| QualitySignal | 7.42 | 13.38 |
| InstructorAvgRating | 2.84 | 6.85 |
| PricePerHour | 2.39 | 5.57 |
| CoursePrice | 2.02 | 3.83 |
| CourseRating | 1.39 | 12.42 |
| CourseCategory | 0.01 | 6.42 |
| ExperienceBucketOrd | −0.38 | 1.08 |

**Every standard deviation exceeds its mean**, most by several-fold. No feature has an
importance distinguishable from zero across folds. The ranking is noise ordering
itself.

Standardised Ridge coefficients give in-sample direction: QualitySignal +1.35,
PriceBandOrd −1.18, Data Science category +1.07, InstructorAvgExperience +1.04,
CourseRating +0.88 enrollments per standard deviation. The largest effect in the
catalogue is **1.35 enrollments per SD — 0.8% of mean course demand**. Even taking
these at face value and ignoring their statistical fragility, they are too small to
inform any decision.

For revenue, permutation importance is dominated by CoursePrice (1.56 × 10⁸),
PriceBandOrd (9.39 × 10⁷) and PricePerHour (8.76 × 10⁷) — three encodings of the same
quantity, two orders of magnitude above CourseRating (2.27 × 10⁶). The identity again.

---

## 7. Business findings

### 7.1 The monetisation gap

Demand is uniform across categories (enrollment shares span 8.1%–9.2%). Revenue is
not: the top three categories take 60.7% of it. Since demand is flat, that divergence
is caused by pricing coverage alone.

| Category | Free / total | Enrollments | Revenue | Revenue per enrollment |
|---|:--:|---:|---:|---:|
| Marketing | 5/5 | 806 | **0** | 0.00 |
| Machine Learning | 4/5 | 819 | 133 | 0.16 |
| Web Development | 3/5 | 844 | 16,682 | 19.77 |
| Finance | 3/5 | 864 | 28,106 | 32.53 |
| Design | 4/5 | 827 | 43,114 | 52.13 |
| Digital Marketing | 3/5 | 808 | 56,261 | 69.63 |
| Cybersecurity | 4/5 | 819 | 59,139 | 72.21 |
| Data Science | 3/5 | 916 | 77,052 | 84.12 |
| Programming | 4/5 | 806 | 77,454 | 96.10 |
| Project Management | 2/5 | 829 | 169,103 | 203.98 |
| Business | 2/5 | 833 | 181,528 | 217.92 |
| Artificial Intelligence | 1/5 | 829 | 202,751 | 244.57 |

Marketing attracts 806 enrollments — statistically identical to AI's 829 — and returns
nothing, because all five of its courses are free. AI returns 202,751 from the same
demand because four of its five courses are priced.

**Platform-wide: 6,403 of 10,000 enrollments (64.0%) are monetised at zero.**

*(Figure 3.)*

### 7.2 What conversion is worth

Under the measured elasticity, converting free courses to paid at a modest
introductory price:

| Courses converted | At 25 | At 49 | At 99 |
|---:|---:|---:|---:|
| 10 (of 38) | +25,000 | +49,000 | +99,000 |
| 19 (half) | +47,500 | +93,100 | +188,100 |
| 38 (all) | +95,000 | +186,200 | +376,200 |

*Assumes 40% of a converted course's demand is lost on introducing a charge — a
deliberately pessimistic assumption, since the measured elasticity implies far less.
Against a base of 911,323, converting half the free catalogue at 49 yields **+10.2%
annual revenue**; all of it at 99 yields **+41%**.*

These are scenario arithmetic, not forecasts. Their value is in ordering the options:
even under pessimistic churn, pricing coverage dominates any plausible gain from
better course selection — which the modelling shows is worth 0.3%.

*(Figure 7; interactive version in the dashboard's Pricing Scenario Simulator.)*

### 7.3 Instructor concentration

Two instructors of sixty — TC00040 (Cybersecurity, 24 years, rating 4.58) and TC00042
(Machine Learning, 21 years, rating 4.97) — account for **60.9% of all 10,000
transactions**, each appearing across 55 of the 60 courses. The remaining 58
instructors handle a median of 56 transactions each.

This is uncorrelated with demand (r = 0.002, *p* = 0.99), so it is not a driver to
model. It is an operational exposure: the departure of either individual affects the
majority of delivery. It does not appear in any revenue report, which is precisely why
it is worth surfacing.

Separately, expertise–category alignment runs at 47.8% against a chance baseline of
8.3% — assignment is far from random — but alignment shows no relationship with
enrollment (ρ = 0.080, *p* = 0.55).

---

## 8. Recommendations

**1. Do not deploy the enrollment model as a ranking tool.** It cannot distinguish a
good course from a bad one; the evidence is that within this catalogue there is no
such distinction to detect. It is fit for capacity sizing — every course draws
approximately 167 enrollments per year — and the dashboard states this on the
prediction screen rather than burying it.

**2. Run a randomised pricing experiment.** The elasticity estimate is bounded near
zero but observational. Randomly assign price changes across a matched subset for one
quarter. This is the single highest-value data-collection action available, and it
converts the largest untested assumption in Section 7.2 into measurement.

**3. Convert free courses in a staged pilot.** Begin with 8–10 free courses in
categories with proven willingness to pay (Data Science, Programming, Cybersecurity
all have paid courses selling at 200+). Price at 25–49. Measure demand loss directly
rather than assuming it. Scale on the observed result.

**4. Forecast at the platform level.** Aggregate monthly volume is stable at 833 ± 40
and forecastable; per-course monthly demand is not. Set planning cadence accordingly.

**5. Address instructor concentration.** Establish delivery coverage as a tracked
operational metric and reduce single-person exposure across the catalogue.

**6. Instrument the demand drivers.** The dataset records outcomes but not causes.
Priority order:

| Signal | Why it matters | Source |
|---|---|---|
| Catalogue impressions, click-through | Separates "unseen" from "unwanted" | Web analytics |
| Marketing spend and channel per course | The largest controllable driver | Ad platforms |
| Price change history | Elasticity needs within-course price variation | Pricing table |
| Search and wishlist volume by topic | Leading indicator of latent demand | Site search logs |
| Completion, refund, repeat purchase | Distinguishes durable demand from churn | LMS + billing |
| Campaign / cohort identifiers | Attributes demand to the action causing it | CRM |

The pipeline in this repository re-runs unchanged as these columns arrive; only the
feature table grows.

---

## 9. Limitations

**Sample size.** 60 courses is small for cross-sectional modelling. Repeated K-fold and
LOOCV mitigate but do not eliminate the resulting instability; the ±0.035 standard
deviation on the best R² reflects it honestly.

**Single year, no price variation within courses.** Twelve months cannot separate
seasonality from noise, and no course changed price, so elasticity is identified only
across courses — vulnerable to confounding by category, quality and positioning.

**Synthetic data characteristics.** Zero missing values, exact price–amount
correspondence, uniform 5-courses-per-category structure and multinomial-consistent
demand together indicate generated rather than observed records. The negative findings
are therefore properly read as a finding about *this dataset*, not about online course
demand in general. The methodology — noise floor first, ablation always, baselines
throughout — transfers regardless.

**Instructor attribution.** The many-to-many transaction–teacher mapping means "the
instructor of a course" is not well defined. Unweighted distinct-teacher averages
avoid target leakage but blur genuine instructor effects if any existed.

**Non-negativity.** The deployed revenue model is unconstrained and can predict
negative revenue (Section 6.4). A zero floor or a log-link GLM is required before
operational use. This is stated as an outstanding defect, not a resolved one.

**Observational throughout.** No causal claim is made. Every recommendation involving
a change to price or catalogue is framed as an experiment to run.

---

## 10. Conclusion

This study set out to build predictive models for course enrollment and revenue on the
EduPro platform. The models were built, evaluated properly, and — for enrollment
demand — do not work. The contribution is establishing *why*, with enough precision to
act on.

Course-level demand in this dataset is statistically indistinguishable from uniform
random allocation (*p* = 0.60, dispersion 0.941), and a variance decomposition places
the maximum attainable R² at zero. Six models across three families converge on that
bound, on both a cross-sectional and an out-of-time panel design. Revenue's apparent
R² of 0.991 is an accounting identity, demonstrated by an ablation that collapses it
to −0.104.

A negative result stated with a bound is not a failed project. Reporting R² = 0.05 as
"the model needs more tuning" would have sent EduPro after a signal that provably is
not there. Reporting it as "the maximum attainable R² is 0.00, and here is the
instrumentation needed to change that" redirects the same effort somewhere it can pay.

Meanwhile the data does support a specific, quantified commercial finding: 64% of
enrollments are monetised at zero, measured price elasticity is bounded within ±0.035,
and converting half the free catalogue at a modest price is worth roughly 10% of
annual revenue even under pessimistic churn — two orders of magnitude more than
perfect course selection could deliver.

The most valuable thing a predictive model produced here was a reliable statement
about what cannot yet be predicted.

---

## References

1. Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice*,
   3rd ed. OTexts. — baseline comparison and forecast evaluation.
2. Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
3. Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
   *Annals of Statistics*, 29(5), 1189–1232.
4. Tibshirani, R. (1996). Regression Shrinkage and Selection via the Lasso.
   *JRSS-B*, 58(1), 267–288.
5. Hoerl, A.E. & Kennard, R.W. (1970). Ridge Regression. *Technometrics*, 12(1), 55–67.
6. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
7. Altmann, A. et al. (2010). Permutation importance: a corrected feature importance
   measure. *Bioinformatics*, 26(10), 1340–1347.
8. Strobl, C. et al. (2007). Bias in random forest variable importance measures.
   *BMC Bioinformatics*, 8:25.
9. Dunn, O.J. (1961). Multiple Comparisons Among Means. *JASA*, 56(293), 52–64.
10. Cox, D.R. & Lewis, P.A.W. (1966). *The Statistical Analysis of Series of Events*.
    Methuen. — index of dispersion.
11. Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2018). Statistical and Machine
    Learning forecasting methods: Concerns and ways forward. *PLoS ONE*, 13(3).
12. City of Toronto. *Parks, Forestry & Recreation Division*. toronto.ca — public-sector
    catalogue-planning context.

---

## Appendix A — Reproducibility

```bash
pip install -r requirements.txt
python -m src.run_pipeline    # regenerates every number in this paper (~3.5 min)
python -m src.figures         # renders Figures 1–7
streamlit run app.py
```

`RANDOM_STATE = 42`. Console transcript: `reports/pipeline_log.txt`.

| Artefact | Path |
|---|---|
| Data quality audit | `reports/tables/01_data_quality_audit.csv` |
| Feature screen | `reports/tables/02_feature_screen.csv` |
| Monetisation gap | `reports/tables/03_monetisation_gap.csv` |
| Predictability diagnostics | `reports/tables/04_predictability_diagnostics.csv` |
| Cross-sectional models | `reports/tables/05_cross_sectional_model_comparison.csv` |
| Panel models | `reports/tables/06_panel_model_comparison.csv` |
| Importance (enrollment) | `reports/tables/07_importance_enrollment.csv` |
| Importance (revenue) | `reports/tables/08_importance_revenue.csv` |
| Ridge coefficients | `reports/tables/09_ridge_coefficients_enrollment.csv` |
| Category revenue | `reports/tables/10_category_revenue.csv` |

## Appendix B — Figures

| Figure | File |
|---|---|
| 1. Demand vs the random-allocation null | `reports/figures/fig01_demand_vs_null.png` |
| 2. Revenue structure and price elasticity | `reports/figures/fig02_revenue_and_elasticity.png` |
| 3. The monetisation gap by category | `reports/figures/fig03_monetisation_gap.png` |
| 4. Model performance vs the mean baseline | `reports/figures/fig04_model_comparison.png` |
| 5. Feature importance for enrollment demand | `reports/figures/fig05_feature_importance.png` |
| 6. Temporal structure, 2025 | `reports/figures/fig06_temporal.png` |
| 7. Pricing scenarios | `reports/figures/fig07_pricing_scenarios.png` |
