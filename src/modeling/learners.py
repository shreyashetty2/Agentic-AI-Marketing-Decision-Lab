"""What the Modeling Agent trains. Two kinds of candidate, both returning a `Fitted` model:

- owner candidates: the owner's own training function, called unchanged (see owners.py);
- agent alternatives: built here from the instruction file (family + params + column list).

A Fitted model scores rows: `predict(rows)` returns {"score": ...} and, for uplift models,
"p_treated" / "p_control" too. It holds only estimators and column names, so it can be saved for
scoring.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def signed_log1p(X):
    """log(1 + |x|), keeping the sign. Counts and dollars are very skewed (a few heavy coupon users sit far out),
    which a straight-line model handles badly; on Model 1 this alone lifted the owner's logistic regression
    from 53.9% to 59.3% of redeemers caught (reports/model1_feature_investigation.md). 0 stays 0."""
    return np.sign(X) * np.log1p(np.abs(X))


FAMILIES = {
    ("logistic", "classify"): lambda p: make_pipeline(StandardScaler(), LogisticRegression(**p)),
    ("logistic_log", "classify"): lambda p: make_pipeline(FunctionTransformer(signed_log1p), StandardScaler(),
                                                          LogisticRegression(**p)),
    ("linear", "regress"): lambda p: make_pipeline(StandardScaler(), LinearRegression(**p)),
    # a straight line fit through the middle (median) of the answers, as Model 2's current model is
    ("median_linear", "regress"): lambda p: make_pipeline(StandardScaler(), QuantileRegressor(**p)),
    ("forest", "classify"): lambda p: RandomForestClassifier(**p),
    ("forest", "regress"): lambda p: RandomForestRegressor(**p),
    ("lightgbm", "classify"): lambda p: LGBMClassifier(**p),
    ("lightgbm", "regress"): lambda p: LGBMRegressor(**p),
}


def build_estimator(family: str, task: str, params: dict):
    """task is "classify" or "regress"; an uplift model is two classifiers (see fit_t_learner)."""
    try:
        return FAMILIES[(family, task)](dict(params))
    except KeyError:
        raise ValueError(f"no {family!r} learner for task {task!r}") from None


@dataclass
class Fitted:
    """A trained candidate. kind: "classify" (probability), "regress" (dollars) or "uplift"."""
    candidate_id: str
    kind: str
    features: list[str]
    estimator: object = None
    treated_model: object = None
    control_model: object = None
    extra: dict = field(default_factory=dict)

    def predict(self, rows: pd.DataFrame) -> dict:
        X = rows[self.features]
        if self.kind == "classify":
            return {"score": self.estimator.predict_proba(X)[:, 1]}
        if self.kind == "regress":
            return {"score": np.asarray(self.estimator.predict(X), dtype=float)}
        p_t = self.treated_model.predict_proba(X)[:, 1]
        p_c = self.control_model.predict_proba(X)[:, 1]
        return {"score": p_t - p_c, "p_treated": p_t, "p_control": p_c}


def _importance(estimator, n: int) -> np.ndarray:
    """One model's reliance on each column, summing to 1. Trees: how much each column's splits reduce
    error. Straight-line models: size of each column's weight (columns are scaled first, so comparable)."""
    last = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if n == 1:                           # one column carries all of it (e.g. Model 2's calibrated usual spend)
        return np.ones(1)
    if hasattr(last, "booster_"):        # LightGBM: by gain, not by its default (how often a column is split on)
        raw = np.asarray(last.booster_.feature_importance(importance_type="gain"), dtype=float)
    elif hasattr(last, "feature_importances_"):
        raw = np.asarray(last.feature_importances_, dtype=float)
    else:
        raw = np.abs(np.ravel(last.coef_)).astype(float)
    return raw / raw.sum() if raw.sum() > 0 else np.zeros(n)


def importances(fitted: Fitted) -> dict[str, float]:
    """How much a trained model leans on each column (shares summing to 1), largest first. For an uplift
    pair it is the average of the two models: what predicts buying, NOT what predicts the mailer's effect.
    Reliance is not added value: a column the model leans on may add little if another column says the
    same thing; only a drop-one test shows added value."""
    n = len(fitted.features)
    if fitted.kind == "uplift":
        share = (_importance(fitted.treated_model, n) + _importance(fitted.control_model, n)) / 2
    else:
        share = _importance(fitted.estimator, n)
    return dict(sorted(zip(fitted.features, share.round(4).tolist()), key=lambda kv: -kv[1]))


def fit_agent_candidate(train: pd.DataFrame, candidate_id: str, task: str, family: str, params: dict,
                        features: list[str], label: str, treatment: str | None = None) -> Fitted:
    """Train one agent alternative. Uplift = a T-learner: one classifier on mailed rows, one on the rest."""
    if task == "uplift":
        mailed = train[treatment] == 1
        model_t = build_estimator(family, "classify", params).fit(train.loc[mailed, features], train.loc[mailed, label])
        model_c = build_estimator(family, "classify", params).fit(train.loc[~mailed, features], train.loc[~mailed, label])
        return Fitted(candidate_id, "uplift", list(features), treated_model=model_t, control_model=model_c)
    model = build_estimator(family, task, params).fit(train[features], train[label])
    return Fitted(candidate_id, task, list(features), estimator=model)


def fill_blanks(table: pd.DataFrame, features: list[str], reasons_by_column: dict[str, list[str]],
                fill_zero_when: list[str]) -> pd.DataFrame:
    """Fill a blank with 0 only when every reason 2A allows that column to be blank is one where 0 is
    the right reading (e.g. "no history" -> no spend). A blank with any other possible reason is a
    question for a person, so the run stops instead of guessing."""
    out = table.copy()
    for col in features:
        if not out[col].isna().any():
            continue
        reasons = reasons_by_column.get(col, [])
        bad = [r for r in reasons if r not in fill_zero_when]
        if not reasons or bad:
            raise SystemExit(f"[STOP] {col} has blanks that 0 does not describe (reasons {reasons or 'unknown'}); "
                             "decide how to fill it before using it as an input")
        out[col] = out[col].fillna(0)
    return out
