# Executive Summary — Course Demand & Revenue Intelligence

**Prepared for:** Government and public-sector programme stakeholders
**Subject:** Predictive modelling of course demand and revenue on the EduPro platform
**Author:** Antarya Mondal · Unified Mentor Data Science Programme
**Date:** 23 August 2026
**Full analysis:** `reports/research_paper.md` · **Live dashboard:** Streamlit application

---

## The bottom line

We were asked to predict which courses will attract enrollments and generate revenue,
so that launch, pricing and instructor decisions could move from intuition to evidence.

**We built the models. For enrollment demand, they do not work — and we can prove that
no model would.** Across 60 courses and 10,000 enrollments in 2025, demand is
statistically indistinguishable from spreading learners across the catalogue at
random. This is not a modelling shortfall we could tune away; it is a property of the
data, and we quantified it before fitting anything.

**That negative finding redirected the analysis to something the data does support.**
Sixty-four percent of all enrollments generate zero revenue, and measured price
sensitivity is effectively nil. Pricing coverage — not course selection — is where the
money is, and it is worth roughly a hundred times more than perfect course prediction
could deliver.

---

## What we found

### 1. Every course draws about the same demand

| | |
|---|---|
| Enrollments per course | 167 on average, ±13 |
| Range across the whole catalogue | 140 to 196 |
| Statistical test of "all courses equally attractive" | **Cannot be rejected** (*p* = 0.60) |

If 10,000 enrollments were sprinkled across 60 courses purely at random, we would
expect a spread of about ±13. The observed spread is ±12.5 — no wider than chance.
Price, duration, rating, subject, level and instructor credentials all fail to predict
which courses land above or below the line.

**Implication:** ranking candidate courses by predicted popularity is not currently
possible, and any tool claiming to do so would be selling noise.

### 2. Revenue is predictable, but only as arithmetic

Our revenue model reports 99% accuracy. That number is real and it is misleading.

Revenue equals price multiplied by enrollments, exactly — no discounts, refunds or
partial payments exist in the data. Since enrollments are near-constant, revenue is
essentially *price × 167*. When we removed price from the model, accuracy collapsed
from 99% to worse-than-guessing.

**Implication:** we can forecast revenue accurately, but only because we already know
the price. The forecast contains no insight into demand. We flag this explicitly
because a 99% accuracy figure quoted without that caveat would materially mislead a
decision-maker.

### 3. Two thirds of activity earns nothing

| | |
|---|---|
| Free courses | 38 of 60 (63%) |
| Enrollments on free courses | 6,403 of 10,000 (**64%**) |
| Revenue from free courses | **0** |

Because demand is uniform, the difference between a high-earning and a zero-earning
category is entirely whether its courses are priced:

| Category | Free courses | Enrollments | Revenue |
|---|:--:|---:|---:|
| Marketing | 5 of 5 | 806 | **0** |
| Machine Learning | 4 of 5 | 819 | 133 |
| Artificial Intelligence | 1 of 5 | 829 | **202,751** |

Marketing and AI attract statistically identical demand. AI earns 202,751 more.

### 4. Raising prices did not reduce demand

Across the observed price range (0.78 to 490.90), the measured price elasticity of
demand is **−0.010, with a 95% confidence interval of [−0.035, +0.015]**.

In plain terms: a 10% price increase is estimated to cost somewhere between 0.35% and
nothing at all in demand. The interval is tight and it straddles zero.

**Caveat we insist on:** prices were not experimentally assigned, so this is
correlation from historical records, not a proven causal result. It justifies
*running a pricing experiment* — not repricing the catalogue overnight.

### 5. Delivery depends on two people

Two instructors out of sixty handle **61% of all transactions**, appearing across 55 of
the 60 courses. The other 58 handle a median of 56 transactions each.

This has no bearing on demand (correlation 0.002) and appears nowhere in revenue
reporting — which is exactly why it warrants attention. It is a continuity risk, not a
performance one.

---

## What we recommend

| # | Action | Rationale | Estimated value |
|---|---|---|---|
| 1 | **Run a randomised pricing experiment** — vary price on a matched subset for one quarter | Converts the largest untested assumption into measurement | Unlocks recommendation 2 with confidence |
| 2 | **Pilot converting 8–10 free courses to paid** at 25–49, in categories with proven willingness to pay | 64% of enrollments currently monetise at zero | ~+10% annual revenue at half the catalogue, even assuming 40% demand loss |
| 3 | **Stop ranking courses by predicted demand** | The signal does not exist; use the model for capacity sizing instead | Avoids decisions made on noise |
| 4 | **Forecast at platform level, not course level** | Monthly platform volume is stable and forecastable (833 ± 40); per-course monthly demand is not | Reliable planning cadence |
| 5 | **Track instructor delivery concentration** as an operational metric | Two-person dependency across 92% of the catalogue | Continuity risk reduction |
| 6 | **Instrument the demand drivers** — impressions, click-through, marketing spend per course, search volume, price-change history, completions and refunds | The data records outcomes but not causes | The only route to genuine demand prediction |

Recommendation 6 is the one that changes the ceiling. Everything else operates within
the current data; only new instrumentation makes demand itself predictable. The
pipeline we built re-runs unchanged once those fields exist.

---

## Relevance to public-sector programme planning

Public recreation divisions face this problem in the same shape. The City of Toronto's
**Parks, Forestry & Recreation** division stewards more than 1,500 parks and 8,000
hectares, and its Community Recreation unit runs a registered-programme catalogue
across recreation centres, rinks and pools. Its published priorities include
modernising front-line client services and developing a Parks and Recreation
Facilities Master Plan — both of which turn on the same question EduPro asked: *which
programmes will fill, and what will they return?*

Three transfers are direct:

**Verify that the signal exists before commissioning a prediction system.** The
noise-floor test used here costs minutes and can be run on any registration dataset
before procurement. A public body that buys a demand-forecasting system for a dataset
with no demand signal has spent money for a number that cannot be right.

**Demand a baseline in every vendor claim.** Our best model beat "just predict the
average" by 0.3%. That comparison is what made the result legible; an accuracy figure
quoted without a baseline is uninterpretable. The same applies to accuracy figures in
procurement documents.

**Separate accounting identities from predictions.** Our 99% revenue accuracy was
arithmetic wearing a model's clothing. Where fee revenue is mechanically price ×
registrations, a "revenue forecasting model" may be re-deriving multiplication. Ask
what happens when the price field is removed.

The public-sector stake is higher than the commercial one. A confidently-wrong demand
forecast in recreation programming misallocates facility time, staffing and public
funds — and the affected residents have no alternative supplier.

---

## What was delivered

| Deliverable | Description |
|---|---|
| **Reproducible pipeline** | Raw workbook → validated tables → engineered features → diagnostics → six models → persisted artefacts. One command, ~3.5 minutes. |
| **Research paper** | Full methodology, results, limitations and recommendations, with every figure and table regenerable from source. |
| **Streamlit dashboard** | Seven modules: portfolio overview, course demand predictor with calibrated uncertainty intervals, revenue forecast, feature-importance explorer, category comparison, pricing scenario simulator, and a model-reliability panel that states the limits plainly. |
| **This summary** | Non-technical decision brief. |

---

## A note on what this project is

The brief asked for predictive models. We delivered them, evaluated them rigorously,
and report that the enrollment models do not work — with a mathematical bound showing
why no model could, on this data.

That is the useful answer. Reporting 5% accuracy as "promising, needs tuning" would
have sent this organisation chasing a signal that provably is not there. Reporting it
as "the ceiling is zero, here is the instrumentation that raises it, and here is the
64% monetisation gap you can act on today" turns the same analysis into two decisions
that can be made this quarter.

The most valuable output of a predictive model is sometimes a reliable statement about
what cannot yet be predicted.
