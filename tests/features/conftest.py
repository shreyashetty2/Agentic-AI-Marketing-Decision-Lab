"""Tiny hand-built dataset for 2A tests. Every expected value in the tests is worked out by hand from it.

Campaigns (cutoff = start - 7):
    c1  TypeB  days 100-130  cutoff 93
    c2  TypeA  days 120-160  cutoff 113   (overlaps c1)
    c3  TypeC  days 200-230  cutoff 193
Households:
    1  long history (first purchase day 10), mailed c1 + c2, redeems c1 on day 105, has demographics
    2  short history (first purchase day 80), mailed c1, never redeems
    3  first purchase day 150 (no history before c1/c2), never mailed
Products: 101 PRODUCE (c1 + c3 coupons), 102 MEAT (c2 pool), 103 GROCERY (no coupon)
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from features.data import Tables  # noqa: E402

CAMPAIGNS = [(1, "B", 100, 130), (2, "A", 120, 160), (3, "C", 200, 230)]
MAILED = {(1, 1), (1, 2), (2, 1)}
REDEEMED = {(1, 1)}


def _spine():
    rows = []
    for rank, (c, t, s, e) in enumerate(CAMPAIGNS, start=1):
        for h, first in [(1, 10), (2, 80), (3, 150)]:
            red = int((h, c) in REDEEMED)
            rows.append({
                "household_key": h, "campaign": c, "campaign_type": t, "start_day": s, "end_day": e,
                "chrono_rank": rank, "duration_days": e - s + 1, "n_overlapping_campaigns": 0,
                "window_truncated_flag": 0, "mailed_flag": int((h, c) in MAILED),
                "has_demographics": int(h == 1), "first_txn_day": first,
                "history_days_before_start": s - first,
                "n_redemptions": red, "n_redemptions_in_window": red, "redeemed_flag": red,
            })
    return pd.DataFrame(rows)


def _transactions():
    cols = ["household_key", "basket_id", "day", "product_id", "sales_value", "store_id",
            "retail_disc", "coupon_disc"]
    rows = [
        # household 1
        (1, 1, 10, 103, 5.00, 1, 0.0, 0.0),
        (1, 2, 60, 101, 4.00, 1, -1.0, 0.0),     # on loyalty discount
        (1, 2, 60, 103, 6.00, 1, 0.0, -0.5),     # coupon line
        (1, 3, 92, 103, 10.00, 1, 0.0, 0.0),     # last day before c1 cutoff -> included
        (1, 4, 93, 101, 100.00, 1, 0.0, 0.0),    # ON c1 cutoff day -> excluded from c1 profile
        (1, 5, 110, 101, 3.00, 1, 0.0, 0.0),     # c1 window, store 1 week 16: on display
        (1, 6, 120, 101, 2.00, 1, 0.0, 0.0),     # c1 window, week 18: not promoted
        (1, 7, 130, 101, 1.00, 1, 0.0, 0.0),     # c1 END day -> inside window
        (1, 8, 131, 101, 50.00, 1, 0.0, 0.0),    # day after c1 ends -> outside
        # household 2
        (2, 20, 80, 103, 8.00, 1, 0.0, 0.0),
        (2, 21, 115, 101, 7.00, 2, 0.0, 0.0),    # c1 window, store 2 week 17: in flyer only
        # household 3
        (3, 30, 150, 102, 9.00, 1, 0.0, 0.0),    # c2 (TypeA) window, pool product
    ]
    tx = pd.DataFrame(rows, columns=cols)
    tx["quantity"] = 1
    tx["coupon_match_disc"] = 0.0
    tx["trans_time"] = 1200
    tx["week_no"] = (tx["day"] + 2 + 6) // 7          # = ceil((day + 2) / 7)
    return tx


@pytest.fixture
def tables():
    coupons = pd.DataFrame(
        [("U1", 101, 1), ("U1", 101, 1),             # exact duplicate row, must not double count
         ("U2", 102, 2), ("U3", 101, 3)],
        columns=["coupon_upc", "product_id", "campaign"])
    redemptions = pd.DataFrame([(1, 105, "U1", 1)], columns=["household_key", "day", "coupon_upc", "campaign"])
    products = pd.DataFrame(
        [(101, "PRODUCE"), (102, "MEAT"), (103, "GROCERY")], columns=["product_id", "department"])
    demographics = pd.DataFrame([{
        "household_key": 1, "classification_1": "Age Group4", "classification_2": "X",
        "classification_3": "Level5", "classification_4": "2", "classification_5": "Group5",
        "homeowner_desc": "Homeowner", "kid_category_desc": "None/Unknown"}])
    return Tables(spine=_spine(), transactions=_transactions(), coupons=coupons,
                  redemptions=redemptions, products=products, demographics=demographics)


@pytest.fixture
def causal():
    # week 16 = days 110-116, week 17 = days 117-123 under week = ceil((day + 2) / 7)
    return pd.DataFrame(
        [(101, 1, 16, "1", "0"),
         (101, 1, 16, "0", "0"),     # conflicting duplicate: promoted if ANY row says so
         (101, 2, 17, "0", "A")],    # flyer only
        columns=["product_id", "store_id", "week_no", "display", "mailer"])
