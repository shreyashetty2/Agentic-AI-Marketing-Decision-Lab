"""The Evaluation Agent's decision rules, as pure functions: numbers in, verdict out. No files, no data.

Verdicts (every one returns who decided and why):
  approve              the owner's model reliably beats its baseline and its numbers are trustworthy
  send_back_modeling   the model, not the columns, looks like the problem
  send_back_features   no model does better than the baseline: the signal is not in the columns
  human_needed         a rule cannot settle it (or it needs sign-off)
"""


def gap_status(low: float, high: float) -> str:
    """95% interval of (model better than baseline): above 0 -> pass, below 0 -> fail, else inconclusive."""
    if low > 0:
        return "pass"
    if high < 0:
        return "fail"
    return "inconclusive"


def route(facts: dict, cfg: dict) -> tuple[str, str, list[str]]:
    """facts: gap_status, calibration_ok, comparison_ok (uplift only: enough overlap AND weighting balanced the
    groups; None otherwise), better_alternative (id or None),
    stability (share of folds that picked the most common candidate), all_within_noise, iteration.
    cfg: max_iterations, stability_min_share, approve_requires_human.
    Checked in order; the first rule that applies decides. Returns (verdict, decided_by, reasons)."""
    gap, calib = facts["gap_status"], facts["calibration_ok"]
    if facts.get("comparison_ok") is False:
        return "human_needed", "human_needed", [
            "EVA-02: mailed and unmailed households are not comparable enough (too little overlap, or still unlike "
            "each other after weighting); uplift numbers cannot be trusted without a person's judgement"]
    if facts["iteration"] >= cfg["max_iterations"]:
        return "human_needed", "human_needed", [
            f"EVA-04: iteration {facts['iteration']} >= {cfg['max_iterations']}; re-judging the same test campaigns "
            "again risks passing by luck"]
    if gap == "pass" and calib:
        if cfg["approve_requires_human"]:
            return "human_needed", "human_needed", [
                "EVA-05: beats its baseline with a well-calibrated level; approval needs a person's sign-off for this model"]
        return "approve", "rule", ["EVA-05: beats its baseline (95% interval above 0) and its numbers match reality"]
    if gap == "pass":
        return "send_back_modeling", "rule", [
            "EVA-06: ranks well but the predicted level does not match reality; widen forecast ranges, do not "
            "recalibrate on past errors (that made it worse when tested)"]
    signs = []
    if facts.get("better_alternative"):
        signs.append(f"alternative {facts['better_alternative']} beats the owner's model beyond noise")
    if facts["stability"] < cfg["stability_min_share"]:
        signs.append(f"the agent's choice flips between folds (most common pick in {facts['stability']:.0%} of folds)")
    if signs:
        return "send_back_modeling", "rule", ["EVA-07: " + "; ".join(signs)]
    if facts["all_within_noise"]:
        return "send_back_features", "rule", [
            f"EVA-08: does not reliably beat its baseline ({gap}), and every candidate model is within noise of the "
            "others -- a different model will not help; the columns need new information"]
    return "human_needed", "human_needed", [f"no rule covers this combination (gap {gap}, calibration ok {calib})"]
