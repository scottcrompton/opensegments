"""
features.py

FeatureBuilder:
---------------
Transforms raw ZCTA-level ACS-derived data into a clean numeric feature matrix
for segmentation. Uses the exact columns present in the Delaware schema—
ensuring cross-state consistency.

Outputs:
    - engineered DataFrame (one row per ZCTA)
    - scaled numeric matrix (StandardScaler)
    - fitted scaler (for model save/load)

This is the backbone of the segmentation pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    from .config import SegmentationConfig
    from .utils import safe_div, log
except ImportError:
    from config import SegmentationConfig
    from utils import safe_div, log


class FeatureBuilder:
    """
    Build engineered features from raw ACS-like ZCTA data.
    """

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self.scaler = None  # Will be assigned after fitting

    # -------------------------------------------------------------
    # Core Feature Engineering
    # -------------------------------------------------------------

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features using your exact schema.

        Parameters
        ----------
        df : pd.DataFrame
            Raw state/national dataset

        Returns
        -------
        feat_df : pd.DataFrame
            Engineered feature DataFrame (non-scaled)
        """

        log("Building engineered features...")

        feat = pd.DataFrame()
        feat[self.cfg.id_col] = df[self.cfg.id_col].astype(str)

        # -------------------------
        # Race Shares
        # -------------------------
        total = df[self.cfg.race_total]
        for col in self.cfg.race_columns:
            feat[f"pct_{col}"] = safe_div(df[col], total)

        # Hispanic share
        if self.cfg.hispanic_total in df.columns:
            feat["pct_hispanic"] = safe_div(df[self.cfg.hispanic_total], total)

        # -------------------------
        # Age Shares
        # -------------------------
        pop = df[self.cfg.population_col]
        for col in self.cfg.age_columns:
            feat[f"pct_{col}"] = safe_div(df[col], pop)

        feat[self.cfg.median_age_col] = df[self.cfg.median_age_col]

        # -------------------------
        # Education: Bachelors+
        # -------------------------
        edu_total = df[self.cfg.education_total]
        edu_plus = df[self.cfg.education_bachelors_plus].sum(axis=1)
        feat["pct_bachelors_plus"] = safe_div(edu_plus, edu_total)

        # -------------------------
        # Poverty
        # -------------------------
        feat["pct_below_poverty"] = safe_div(
            df[self.cfg.poverty_below], df[self.cfg.poverty_total]
        )

        # -------------------------
        # Household
        # -------------------------
        feat["pct_1_person"] = safe_div(
            df[self.cfg.household_size_1], df[self.cfg.household_total]
        )
        feat["pct_households_with_children"] = safe_div(
            df[self.cfg.household_with_children], df[self.cfg.household_total]
        )

        # -------------------------
        # Housing
        # -------------------------
        feat["median_home_value"] = df[self.cfg.median_home_value]
        feat["median_gross_rent"] = df[self.cfg.median_gross_rent]

        # -------------------------
        # Income Signals
        # -------------------------
        for col in self.cfg.income_columns:
            if col in df.columns:
                feat[col] = df[col]

        # -------------------------
        # Industry Mix
        # -------------------------
        industry_total = df[self.cfg.industry_total]
        for col in self.cfg.industry_columns:
            if col in df.columns:
                feat[f"pct_{col}"] = safe_div(df[col], industry_total)

        # -------------------------
        # Commute
        # -------------------------
        commute_total = df[self.cfg.commute_total]
        for col in self.cfg.commute_columns:
            feat[f"pct_{col}"] = safe_div(df[col], commute_total)

        # -------------------------
        # Vehicles
        # -------------------------
        feat["pct_no_vehicle"] = safe_div(
            df[self.cfg.vehicles_no], df[self.cfg.vehicles_total]
        )

        # -------------------------
        # Occupation Earnings
        # -------------------------
        for col in self.cfg.occupation_earnings_columns:
            if col in df.columns:
                feat[col] = df[col]

        # Final cleaning
        feat = feat.replace([np.inf, -np.inf], 0)
        feat = feat.fillna(0)

        log(f"Engineered features created: {feat.shape[1]} columns.")

        return feat

    # -------------------------------------------------------------
    # Scaling
    # -------------------------------------------------------------

    def fit_scaler(self, feat_df: pd.DataFrame) -> np.ndarray:
        """
        Fit StandardScaler using numeric feature columns.

        Returns:
            scaled matrix (np.ndarray)
        """
        log("Fitting StandardScaler...")

        numeric_df = feat_df.drop(columns=[self.cfg.id_col])
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(numeric_df)

        log("Scaler fitted.")
        return X_scaled

    def transform(self, feat_df: pd.DataFrame) -> np.ndarray:
        """
        Apply existing scaler to new feature data.
        """
        if self.scaler is None:
            raise ValueError("Scaler is not fitted. Call fit_scaler() first.")

        numeric_df = feat_df.drop(columns=[self.cfg.id_col])
        X_scaled = self.scaler.transform(numeric_df)
        return X_scaled

