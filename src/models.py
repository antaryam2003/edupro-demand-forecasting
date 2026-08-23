"""Stage 4 - Model development and evaluation.

Two modelling problems, deliberately kept separate because they answer
different business questions:

* **Cold-start (cross-sectional)** - 60 courses, attributes only. Answers
  *"if we launch this course, how many enrollments and how much revenue?"*
  Evaluated with repeated K-fold and leave-one-out cross-validation because
  n=60 makes a single hold-out split far too noisy to trust.

* **Operational forecast (panel)** - 540 course-months with lagged history.
  Answers *"how much demand will each course see next month?"* Evaluated with
  an out-of-time split so the test set is strictly in the future.

Every table reports a **naive baseline** (predict the mean). A model that does
not beat that baseline has learned nothing, however sophisticated it is.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RS = config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #
def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """One-hot the categoricals, standardise the numerics.

    Scaling matters for the penalised linear models (Ridge/Lasso shrink
    coefficients on the raw scale) and is harmless for the tree ensembles.
    """
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def model_zoo(n_samples: int) -> dict[str, object]:
    """Baseline -> penalised linear -> ensemble, as specified in the brief."""
    cv_inner = min(5, max(3, n_samples // 12))
    return {
        "Baseline (mean)": DummyRegressor(strategy="mean"),
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "Lasso Regression": LassoCV(alphas=np.logspace(-3, 2, 25), cv=cv_inner, random_state=RS, max_iter=20000),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=RS, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=3, subsample=0.9, random_state=RS
        ),
    }


def make_pipeline(estimator, numeric: list[str], categorical: list[str]) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(numeric, categorical)), ("model", estimator)])


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Cross-sectional evaluation
# --------------------------------------------------------------------------- #
def evaluate_cross_sectional(
    df: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    n_repeats: int = 5,
    label: str = "",
) -> pd.DataFrame:
    """Repeated 5-fold + leave-one-out CV for every model in the zoo.

    Out-of-fold predictions are pooled per repeat before scoring, which is more
    stable than averaging per-fold R-squared on 12-row folds.
    """
    X, y = df[numeric + categorical], df[target].to_numpy(dtype=float)
    rows = []

    for name, est in model_zoo(len(df)).items():
        pipe = make_pipeline(est, numeric, categorical)

        per_repeat = []
        for rep in range(n_repeats):
            kf = KFold(n_splits=5, shuffle=True, random_state=RS + rep)
            oof = cross_val_predict(pipe, X, y, cv=kf, n_jobs=1)
            per_repeat.append(metrics(y, oof))
        rep_df = pd.DataFrame(per_repeat)

        loo_pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)
        print(f"    [{label or target}] {name}: done", flush=True)
        loo = metrics(y, loo_pred)

        rows.append(
            {
                "Target": label or target,
                "Model": name,
                "CV_MAE": rep_df.MAE.mean(),
                "CV_MAE_sd": rep_df.MAE.std(),
                "CV_RMSE": rep_df.RMSE.mean(),
                "CV_RMSE_sd": rep_df.RMSE.std(),
                "CV_R2": rep_df.R2.mean(),
                "CV_R2_sd": rep_df.R2.std(),
                "LOO_MAE": loo["MAE"],
                "LOO_RMSE": loo["RMSE"],
                "LOO_R2": loo["R2"],
            }
        )

    out = pd.DataFrame(rows)
    base = out.loc[out.Model == "Baseline (mean)", "CV_RMSE"].iloc[0]
    out["Skill_vs_baseline"] = 1 - out.CV_RMSE / base
    return out


# --------------------------------------------------------------------------- #
# Panel (out-of-time) evaluation
# --------------------------------------------------------------------------- #
def evaluate_panel(
    panel: pd.DataFrame,
    target: str,
    numeric: list[str],
    categorical: list[str],
    test_months: int = config.PANEL_TEST_MONTHS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on the earliest months, test on the final ``test_months``."""
    months = np.sort(panel.Month.unique())
    cutoff = months[-test_months]
    train, test = panel[panel.Month < cutoff], panel[panel.Month >= cutoff]

    Xtr, ytr = train[numeric + categorical], train[target].to_numpy(dtype=float)
    Xte, yte = test[numeric + categorical], test[target].to_numpy(dtype=float)

    rows = []
    # Domain baselines a forecasting model must beat to be worth deploying.
    rows.append({"Model": "Baseline (global mean)", **metrics(yte, np.full_like(yte, ytr.mean()))})
    rows.append({"Model": "Baseline (course history mean)", **metrics(yte, test["Enroll_expanding_mean"] if target == "EnrollmentCount" else test["Revenue_roll3"])})
    rows.append({"Model": "Naive (last month)", **metrics(yte, test["Enroll_lag1"] if target == "EnrollmentCount" else test["Revenue_lag1"])})

    preds = {"actual": yte}
    for name, est in model_zoo(len(train)).items():
        if name == "Baseline (mean)":
            continue
        pipe = make_pipeline(est, numeric, categorical)
        pipe.fit(Xtr, ytr)
        p = pipe.predict(Xte)
        preds[name] = p
        rows.append({"Model": name, **metrics(yte, p)})

    res = pd.DataFrame(rows)
    base_rmse = res.loc[res.Model == "Baseline (global mean)", "RMSE"].iloc[0]
    res["Skill_vs_baseline"] = 1 - res.RMSE / base_rmse
    res.insert(0, "Target", target)

    pred_df = test[["CourseID", "Month"]].reset_index(drop=True).assign(**{k: np.asarray(v) for k, v in preds.items()})
    return res, pred_df


# --------------------------------------------------------------------------- #
# Feature importance
# --------------------------------------------------------------------------- #
def feature_importance(
    df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str], estimator=None
) -> pd.DataFrame:
    """Permutation importance (model-agnostic, measured on held-out folds).

    Permutation importance is preferred over impurity-based importance, which
    is biased toward high-cardinality and continuous features.
    """
    est = estimator if estimator is not None else RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=RS, n_jobs=1
    )
    pipe = make_pipeline(est, numeric, categorical)
    X, y = df[numeric + categorical], df[target].to_numpy(dtype=float)

    rows = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    for tr, te in kf.split(X):
        pipe.fit(X.iloc[tr], y[tr])
        r = permutation_importance(pipe, X.iloc[te], y[te], n_repeats=10, random_state=RS, scoring="neg_mean_squared_error")
        rows.append(pd.Series(r.importances_mean, index=X.columns))

    imp = pd.concat(rows, axis=1)
    return (
        pd.DataFrame({"Feature": imp.index, "Importance": imp.mean(axis=1), "Importance_sd": imp.std(axis=1)})
        .sort_values("Importance", ascending=False)
        .reset_index(drop=True)
    )


def linear_coefficients(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """Standardised Ridge coefficients - directional, interpretable drivers."""
    pipe = make_pipeline(RidgeCV(alphas=np.logspace(-3, 3, 25)), numeric, categorical)
    pipe.fit(df[numeric + categorical], df[target])
    names = pipe.named_steps["prep"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_
    return (
        pd.DataFrame({"Feature": [n.split("__", 1)[-1] for n in names], "Coefficient": coefs})
        .assign(AbsCoefficient=lambda d: d.Coefficient.abs())
        .sort_values("AbsCoefficient", ascending=False)
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Persist deployable artefacts
# --------------------------------------------------------------------------- #
def fit_and_save(df: pd.DataFrame, target: str, numeric: list[str], categorical: list[str], filename: str, estimator=None):
    """Fit on all rows, then store the model with its honest error profile.

    The saved payload carries out-of-fold residual quantiles so the app can show
    a *calibrated* prediction interval rather than a bare point estimate. On a
    dataset this noisy the interval is the part a decision-maker should read.
    """
    # `estimator or Default()` would evaluate the estimator's truthiness, and an
    # unfitted forest raises on __len__. Test for None explicitly.
    est = estimator if estimator is not None else RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt", random_state=RS, n_jobs=1
    )
    X, y = df[numeric + categorical], df[target].to_numpy(dtype=float)

    oof = cross_val_predict(make_pipeline(est, numeric, categorical), X, y,
                            cv=KFold(5, shuffle=True, random_state=RS), n_jobs=1)
    residuals = y - oof

    pipe = make_pipeline(est, numeric, categorical)
    pipe.fit(X, y)

    payload = {
        "pipeline": pipe,
        "numeric": numeric,
        "categorical": categorical,
        "target": target,
        "estimator_name": type(est).__name__,
        "train_mean": float(y.mean()),
        "train_sd": float(y.std(ddof=1)),
        "n_train": int(len(df)),
        "cv_mae": float(mean_absolute_error(y, oof)),
        "cv_rmse": float(np.sqrt(mean_squared_error(y, oof))),
        "cv_r2": float(r2_score(y, oof)),
        "residual_q": {q: float(np.quantile(residuals, q)) for q in (0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975)},
    }
    joblib.dump(payload, config.MODELS / filename)
    return payload
