"""Owner adapters: each model's default candidate IS its owner's code, called unchanged.

The Modeling Agent orchestrates; it does not re-implement anyone's model. These adapters only call
the owners' own load_table / train functions and wrap the result as a Fitted model, so the agent can
test it the same way as every other candidate.

  Model 1 (Yuxin, John): model1_baseline.load_table + train_models  ("Random forest", "Logistic regression")
  Model 2 (Yuxin):       model2_baseline.load_table + train_models  ("Random forest", "Linear regression")
  Model 3 (Shreya):      model3_uplift.load_table + train_t_learner (two random forests, uplift = difference)
"""
from pathlib import Path

import pandas as pd

from modeling import model1_baseline, model2_baseline, model3_uplift
from modeling.learners import Fitted

OWNERS = {
    1: {"module": model1_baseline, "kind": "classify",
        "learners": {"forest": "Random forest", "logistic": "Logistic regression"}},
    2: {"module": model2_baseline, "kind": "regress",
        "learners": {"forest": "Random forest", "linear": "Linear regression"}},
    3: {"module": model3_uplift, "kind": "uplift", "learners": {"forest": "T-learner"}},
}


def owner_features(model_id: int) -> list[str]:
    return list(OWNERS[model_id]["module"].FEATURES)


def load_owner_table(model_id: int, processed: Path) -> pd.DataFrame:
    """The owner's own load_table: same rows, same blank filling, same derived columns (e.g. type flags)."""
    return OWNERS[model_id]["module"].load_table(processed)


def fit_owner(model_id: int, owner_learner: str, train: pd.DataFrame, candidate_id: str) -> Fitted:
    owner = OWNERS[model_id]
    module, features = owner["module"], owner_features(model_id)
    if owner["kind"] == "uplift":
        model_t, model_c, _, _ = module.train_t_learner(train)
        return Fitted(candidate_id, "uplift", features, treated_model=model_t, control_model=model_c)
    name = owner["learners"][owner_learner]
    return Fitted(candidate_id, owner["kind"], features, estimator=module.train_models(train)[name])
