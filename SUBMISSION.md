# Submission Guide

The form asks for four links. **Three are done and verified live.** The fourth is a
video only you can record — see section 4.

| # | Field | Status | Link |
|---|---|---|---|
| 1 | GitHub Repository | **done** | https://github.com/antaryam2003/edupro-demand-forecasting |
| 2 | Deployed project | **done** | https://edupro-noise-floor.streamlit.app |
| 3 | Research paper | **done** | https://claude.ai/code/artifact/84562d71-971e-4a9c-8322-494a259ef695 |
| 4 | Feedback video | **you record** | — |

> **Before submitting link 3:** artifacts are private by default. Open it, click
> **Share**, and enable link access, or a reviewer will get a 404. Links 1 and 2 are
> already public.

---

## 1. GitHub Repository Link — done

**https://github.com/antaryam2003/edupro-demand-forecasting**

Public, branch `main`, 46 files. Verified on the remote: `app.py`, `requirements.txt`,
and all of `data/processed/`, `models/` and `reports/tables/` are present — the
deployed app reads those and will not start without them.

To push future changes: `git push`.

---

## 2. Deployed Project Link — done

**https://edupro-noise-floor.streamlit.app**

Deployed on Streamlit Community Cloud from `main` / `app.py`, Python 3.13, no secrets
required. Verified live: all seven modules load, the joblib models unpickle correctly
on the cloud runtime, and the Course Demand Predictor returns the same numbers as
local (166 enrollments, 80% interval 153–181, revenue 24,849).

The app auto-redeploys on every push to `main`.

**If a future build fails**, it is almost always a package version without a wheel for
the selected Python runtime. Every current pin was checked against PyPI and has a
cp313/cp314 wheel or is pure Python. If you bump a version, re-check, or loosen that
pin to `>=` — but keep `scikit-learn==1.9.0` exact, because the saved models in
`models/` were pickled with it.

---

## 3. Research Paper Link — done

**https://claude.ai/code/artifact/84562d71-971e-4a9c-8322-494a259ef695**

Full paper with all seven figures embedded. **It is private until you share it** —
open it, click **Share**, enable link access, then confirm in a private window that a
logged-out visitor can read it.

Alternative if you would rather not deal with sharing settings:
`https://github.com/antaryam2003/edupro-demand-forecasting/blob/main/reports/research_paper.md`
(public already, renders as Markdown on GitHub).

---

## 4. Project Feedback Video Link — you record this

This one has to be you: it is your experience and your learnings, in your voice.
Record 5–8 minutes, screen + voice. Loom, Google Drive (set to "anyone with the link"),
or an unlisted YouTube upload all work. **Check the sharing permission before
submitting** — a private link is the most common reason a submission gets returned.

### Script

**(0:00–0:45) Who and what**

> "I'm Antarya Mondal. This is my Unified Mentor project: predictive modelling for
> course demand and revenue forecasting on the EduPro platform. The dataset is 60
> courses, 60 instructors, 3,000 learners and 10,000 transactions across 2025. The
> brief was to predict enrollment count, course revenue and category revenue, and to
> ship a Streamlit dashboard on top of it.
>
> I'll tell you upfront where this ended up, because it shaped everything: the
> enrollment models don't work — and the interesting part is that I can prove no model
> would."

**(0:45–2:15) The pipeline** *(screen: `src/`, then `reports/pipeline_log.txt`)*

> "The pipeline is six modules. Data prep loads the four sheets and runs fourteen
> integrity checks — all pass, the data is unusually clean. Feature engineering builds
> the three families the brief specified: course features like price bands and rating
> tiers, instructor features including an expertise-to-category match score, and
> lagged historical features for the time-series side.
>
> I built two separate modelling tables, because they answer different questions. The
> cross-sectional table — 60 courses, attributes only — answers 'if we launch this,
> what will it do?' A new course has no history, so it can't use historical features.
> The panel table — 540 course-months — answers 'what will this course do next
> month?', and that one gets the lags.
>
> Three leakage controls matter here. Instructor features average over distinct
> teachers, not weighted by transaction volume, because volume weighting would leak the
> target. Every lag uses a groupby-shift so no row sees its own future. And scaling and
> encoding live inside the sklearn Pipeline, so they refit inside each fold."

**(2:15–4:00) The core finding** *(screen: dashboard → Model Reliability, then Figure 1)*

> "Before fitting anything, I asked how much of the variation is explainable at all.
>
> Ten thousand enrollments across 60 courses. If every course were equally attractive
> and learners just landed at random, you'd expect a spread of about 12.8 enrollments
> per course. The observed spread is 12.5 — no wider than chance. A chi-square
> goodness-of-fit test gives p equals 0.60: I cannot reject the hypothesis that all 60
> courses are equally attractive.
>
> Then the variance decomposition. Total variance is 156.8. Sampling variance under the
> random null is 163.9. Total is *below* sampling, so systematic variance is zero — the
> maximum attainable R-squared is 0.00. That's the noise floor, and it's computed
> before any model exists.
>
> And that's exactly where the models land. Six models, linear through gradient
> boosting. The best is Random Forest at R-squared 0.051, plus or minus 0.035 — within
> a couple of standard deviations of zero. It cuts mean absolute error by 0.45
> enrollments on a base of 167. That's a third of one percent. On the out-of-time panel
> split, every single model scores negative R-squared — none of them beats predicting
> the mean."

**(4:00–5:15) The ablation** *(screen: Model Reliability → Course Revenue targets)*

> "Revenue looked completely different — R-squared 0.991. I nearly reported that as the
> headline.
>
> Then I checked the identity. Revenue equals price times enrollments, exactly — zero
> error across all 10,000 rows. No discounts, no refunds. And since enrollments are
> near-constant at 167, revenue is basically price times 167. The model was recovering
> a multiplication.
>
> So I ran the ablation: same models, price features removed. R-squared drops from
> 0.991 to minus 0.104 — worse than guessing. All of the apparent predictability was
> price. None of it was demand.
>
> I think that ablation should be mandatory any time your target is an arithmetic
> function of a feature you have. Without it I'd have shipped a 99% accuracy claim
> that was true and worthless."

**(5:15–6:45) What the data does support** *(screen: Category Comparison → Pricing Simulator)*

> "The negative result redirected me to what the data actually supports.
>
> Thirty-eight of 60 courses are free. They carry 64% of all enrollments and return
> zero. And because demand is uniform across categories, the gap between a rich
> category and a poor one is purely pricing coverage. Marketing: all five courses free,
> 806 enrollments, zero revenue. AI: four of five priced, 829 enrollments — statistically
> the same demand — 202,751 in revenue.
>
> Then elasticity. Across a price range from under a dollar to 490, measured elasticity
> is minus 0.010, with a 95% confidence interval from minus 0.035 to plus 0.015.
> Demand didn't fall as price rose. I'm careful about this one — prices weren't randomly
> assigned, so it's observational. It justifies running a pricing experiment, not
> repricing the catalogue.
>
> This simulator puts numbers on it. Converting half the free catalogue at 49, assuming
> a pessimistic 40% demand loss, is worth about 10% of annual revenue. Perfect course
> selection was worth 0.3%. That's the two-orders-of-magnitude comparison that decides
> where the effort goes."

**(6:45–7:45) What I learned**

> "Three things I'd carry into the next project.
>
> First: establish the predictability ceiling before you model. R-squared of 0.05 reads
> as 'needs tuning' by default, and I'd have spent days tuning. With a noise floor
> computed first, the same number reads as 'you're at the ceiling, go get different
> data.' Same result, opposite action.
>
> Second: baselines aren't a formality. Lasso on the panel tied the global mean
> exactly, because it shrank every coefficient to zero — the model correctly concluded
> there was nothing to learn. I'd never have seen that without the baseline in the table.
>
> Third — the one I actually struggled with — a negative result is a deliverable. It
> was tempting to lead with the 99% revenue figure and stay quiet about the ablation.
> The honest version is more useful: here's the ceiling, here's the instrumentation
> that raises it, and here's a 64% monetisation gap you can act on this quarter.
>
> The dashboard reflects that. The prediction screen shows the interval and states the
> model's R-squared next to it, rather than presenting a confident point estimate the
> data doesn't support.
>
> Known gap I'd fix first: the revenue model is unconstrained and predicts negative
> revenue for the zero-revenue Marketing category. It needs a floor at zero or a
> log-link GLM before anyone operates on it. It's flagged in the paper's limitations.
>
> Thanks for watching."

### Recording tips

- Have the app running (`streamlit run app.py`) and the repo open before you hit record.
- Say the numbers out loud — reviewers grade on whether you understand your own results.
- Do not read this script verbatim; it is a route through the material, not a teleprompter.
- One clean take beats a heavily edited one.

---

## Final checklist

- [x] GitHub repo pushed, `data/processed/` + `models/` + `reports/tables/` present
- [x] Streamlit app deployed and loads without error
- [x] Every dashboard module clicks through cleanly on the deployed URL
- [ ] **Research paper set to link-accessible** (Share button) and checked logged-out
- [ ] Video recorded and sharing set to "anyone with the link"
- [ ] All four URLs pasted into the form, each starting with `https://`
