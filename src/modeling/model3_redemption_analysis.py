"""Model 3 redemption-segmentation analysis (mentor feedback, 10/9 meeting).

Model 3's main outcome counts any qualifying purchase as a campaign "success", even if the
household never used the coupon. This script splits treated purchasers by whether their purchase
was actually tied to a redeemed coupon, using bought_campaign_product_without_redemption
(Feature Agent 2B, src/features/outcomes.py).

Run from the repo folder:
    python src/modeling/model3_redemption_analysis.py --processed data/processed

Input : data/processed/model3_table.parquet (from src/features/build_2b.py)
Output: prints the breakdown to the console and writes
        data/processed/model3_redemption_analysis.md
"""
import argparse
from pathlib import Path

import pandas as pd

LABEL = "bought_campaign_product_excl_display_flyer"
WITHOUT_REDEMPTION = "bought_campaign_product_without_redemption"
TREATMENT = "mailed_flag"


def analyze(table: pd.DataFrame) -> dict:
    treated = table[table[TREATMENT] == 1]
    treated_bought = treated[treated[LABEL] == 1]
    without = treated_bought[WITHOUT_REDEMPTION] == 1
    via_redemption = len(treated_bought) - int(without.sum())

    by_campaign = (
        treated_bought.groupby("campaign")[WITHOUT_REDEMPTION]
        .agg(n_bought="count", n_without_redemption="sum")
        .assign(pct_without_redemption=lambda d: (d["n_without_redemption"] / d["n_bought"] * 100).round(1))
        .query("n_bought >= 20")   # drop tiny campaigns where the percentage is noisy
        .sort_values("pct_without_redemption", ascending=False)
    )

    return {
        "n_treated": len(treated),
        "n_treated_bought": len(treated_bought),
        "n_via_redemption": via_redemption,
        "n_without_redemption": int(without.sum()),
        "pct_without_redemption": round(without.mean() * 100, 1),
        "by_campaign": by_campaign,
    }


def render_report(result: dict) -> str:
    table = result["by_campaign"].to_string()
    return f"""# Model 3: redemption-segmentation analysis

Mentor feedback (10/9 meeting): check directly whether treated households who bought the
eligible product actually used the coupon, rather than inferring it from spend level alone.

## Headline result

Of {result['n_treated']:,} treated TypeB/C households, {result['n_treated_bought']:,} bought an
eligible product during the campaign window.

- Bought **via an actually-redeemed coupon**: {result['n_via_redemption']:,}
  ({100 - result['pct_without_redemption']:.1f}%)
- Bought **without ever touching the coupon**: {result['n_without_redemption']:,}
  ({result['pct_without_redemption']:.1f}%)

**{result['pct_without_redemption']:.1f}% of what Model 3's main outcome currently counts as a
campaign "success" is purchase behavior that happened regardless of the mailer.** This is direct,
observed evidence for the "sure things vs. persuadables" pattern the T-learner baseline inferred
from spend levels -- not an inference this time, a direct count from redemption records.

## By campaign (campaigns with at least 20 treated purchasers)

```
{table}
```

## What this means for Model 3 going forward

The current main outcome (`bought_campaign_product_excl_display_flyer`) overstates how often the
mailer is actually what drove a purchase. `bought_campaign_product_without_redemption` is now
available as a second, stricter outcome (Model 3's sensitivity label in
`model3_table.parquet`) for training a model that specifically targets persuadable households,
rather than ones who would have converted regardless.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    table = pd.read_parquet(args.processed / "model3_table.parquet")
    result = analyze(table)

    print(f"Treated households: {result['n_treated']:,}")
    print(f"Treated households who bought: {result['n_treated_bought']:,}")
    print(f"  via an actually-redeemed coupon: {result['n_via_redemption']:,} "
          f"({100 - result['pct_without_redemption']:.1f}%)")
    print(f"  without ever touching the coupon: {result['n_without_redemption']:,} "
          f"({result['pct_without_redemption']:.1f}%)")
    print("\nBy campaign (>= 20 treated purchasers):")
    print(result["by_campaign"])

    report_path = args.processed / "model3_redemption_analysis.md"
    report_path.write_text(render_report(result))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
