"""Inputs for step 2A: the Data Agent's spine plus the clean tables it was built from."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class Tables:
    spine: pd.DataFrame
    transactions: pd.DataFrame
    coupons: pd.DataFrame
    redemptions: pd.DataFrame
    products: pd.DataFrame
    demographics: pd.DataFrame

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


def load_tables(processed: Path) -> Tables:
    return Tables(
        spine=_read(processed / "spine_household_campaign.parquet"),
        transactions=_read(processed / "transaction_data.parquet"),
        coupons=_read(processed / "coupon.parquet"),
        redemptions=_read(processed / "coupon_redempt.parquet"),
        products=_read(processed / "product.parquet", ["product_id", "department"]),
        demographics=_read(processed / "hh_demographic.parquet"),
    )
