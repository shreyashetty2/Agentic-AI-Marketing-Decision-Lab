"""Owner adapters: each model's default candidate IS its owner's code, called unchanged.

The Modeling Agent orchestrates; it does not re-implement anyone's model. These adapters only call
the owners' own load_table / train functions and wrap the result as a Fitted model, so the agent can
test it the same way as every other candidate. Each model's CURRENT script (modeling_spec.yaml,
`scripts.current`) supplies the table; the others stay in as comparisons.

  Model 1 (Yuxin, John): model1_baseline.load_table + train_models  ("Random forest", "Logistic regression")
  Model 2 (Yuxin):       model2_value.load_table + CalibratedUsualSpend (current: usual spend, calibrated);
                         model2_baseline.train_models ("Random forest", "Linear regression") for comparison
  Model 3 (Shreya):      model3_uplift.load_table + train_t_learner (two random forests, uplift = difference)
"""
from pathlib import Path

import pandas as pd

from modeling import model1_baseline, model2_baseline, model2_value, model3_uplift
from modeling.learners import Fitted

OWNERS = {
    1: {"table": model1_baseline, "kind": "classify", "features": model1_baseline.FEATURES,
        "learners": {"forest": (model1_baseline, "Random forest"), "logistic": (model1_baseline, "Logistic regression")}},
    2: {"table": model2_value, "kind": "regress", "features": model2_baseline.FEATURES,
        "learners": {"forest": (model2_baseline, "Random forest"), "linear": (model2_baseline, "Linear regression")}},
    3: {"table": model3_uplift, "kind": "uplift", "features": model3_uplift.FEATURES, "learners": {}},
}


def owner_features(model_id: int) -> list[str]:
    """The owners' main column list (the one `features: owner` in the spec refers to)."""
    return list(OWNERS[model_id]["features"])


def owner_learner_features(model_id: int, owner_learner: str) -> list[str]:
    """The columns one owner model actually uses: the calibrated Model 2 uses usual spend only."""
    return [model2_value.USUAL] if owner_learner == "calibrated" else owner_features(model_id)


def load_owner_table(model_id: int, processed: Path) -> pd.DataFrame:
    """The current script's own load_table: same rows, same blank filling, same derived columns."""
    return OWNERS[model_id]["table"].load_table(processed)


def fit_owner(model_id: int, owner_learner: str, train: pd.DataFrame, candidate_id: str) -> Fitted:
    owner = OWNERS[model_id]
    features = owner_learner_features(model_id, owner_learner)
    if owner["kind"] == "uplift":
        model_t, model_c, _, _ = model3_uplift.train_t_learner(train)
        return Fitted(candidate_id, "uplift", features, treated_model=model_t, control_model=model_c)
    if owner_learner == "calibrated":
        return Fitted(candidate_id, "regress", features, estimator=model2_value.CalibratedUsualSpend().fit(train))
    module, name = owner["learners"][owner_learner]
    return Fitted(candidate_id, owner["kind"], features, estimator=module.train_models(train)[name])
