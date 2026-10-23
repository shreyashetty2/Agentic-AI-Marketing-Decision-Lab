"""Inputs for step 2A: the Data Agent's spine plus the clean tables it was built from."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# Gas-kiosk, miscellaneous and coupon-item lines. Same rule as src/preprocessing/outlier_rules.py:
# their "quantity" is not an item count and their dollars are not grocery spend (open question Q7b).
NON_MERCHANDISE_DEPARTMENTS = ["KIOSK-GAS", "MISC SALES TRAN"]
NON_MERCHANDISE_COMMODITIES = ["COUPON/MISC ITEMS"]


@dataclass
class Tables:
    spine: pd.DataFrame
    transactions: pd.DataFrame
    coupons: pd.DataFrame
    redemptions: pd.DataFrame
    products: pd.DataFrame
    demographics: pd.DataFrame
    # Non-merchandise lines taken out of `transactions` when exclude_non_merchandise=True (empty otherwise).
    # Kept so fuel-and-misc features can still be built from them.
    non_merchandise: pd.DataFrame = field(default_factory=pd.DataFrame)

    def copy(self) -> "Tables":
        return copy.deepcopy(self)

    def campaigns(self) -> pd.DataFrame:
        return (self.spine[["campaign", "campaign_type", "start_day", "end_day"]]
                .drop_duplicates().reset_index(drop=True))

    def mailings(self) -> pd.DataFrame:
        mailed = self.spine[self.spine["mailed_flag"] == 1]
        return mailed[["household_key", "campaign", "start_day", "end_day"]].reset_index(drop=True)

    def campaign_products(self) -> pd.DataFrame:
        """Each campaign's DISTINCT product list, so duplicate coupon rows never double count."""
        return self.coupons[["campaign", "product_id"]].drop_duplicates().reset_index(drop=True)


def _read(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.columns = [c.lower() for c in df.columns]      # ingest.py keeps the raw upper-case names
    return df[columns] if columns else df


def non_merchandise_product_ids(products: pd.DataFrame) -> set:
    """Product ids of gas-kiosk, miscellaneous and coupon-item products (needs department and commodity_desc)."""
    is_dept = products["department"].isin(NON_MERCHANDISE_DEPARTMENTS)
    is_commodity = products["commodity_desc"].isin(NON_MERCHANDISE_COMMODITIES)
    return set(products.loc[is_dept | is_commodity, "product_id"])


def load_tables(processed: Path, exclude_non_merchandise: bool = False) -> Tables:
    """exclude_non_merchandise=False (default) keeps today's behavior: every transaction line is used.
    True moves gas-kiosk, miscellaneous and coupon-item lines out of `transactions` into `non_merchandise`."""
    product_info = _read(processed / "product.parquet", ["product_id", "department", "commodity_desc"])
    transactions = _read(processed / "transaction_data.parquet")
    non_merchandise = pd.DataFrame(columns=transactions.columns)
    if exclude_non_merchandise:
        # A product a campaign issued a coupon for stays in: its spend is campaign-product spend (Model 2 target).
        campaign_product_ids = set(_read(processed / "coupon.parquet", ["product_id"])["product_id"])
        is_non_merch = transactions["product_id"].isin(non_merchandise_product_ids(product_info) - campaign_product_ids)
        non_merchandise = transactions[is_non_merch]
        transactions = transactions[~is_non_merch].reset_index(drop=True)
    return Tables(
        spine=_read(processed / "spine_household_campaign.parquet"),
        transactions=transactions,
        coupons=_read(processed / "coupon.parquet"),
        redemptions=_read(processed / "coupon_redempt.parquet"),
        products=product_info,
        demographics=_read(processed / "hh_demographic.parquet"),
        non_merchandise=non_merchandise,
    )
