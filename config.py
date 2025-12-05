"""
config.py

Central configuration for the segmentation engine.

This configuration is fully aligned with the actual dataset schema
provided (Delaware sample), and will generalize to all states since
the schema is consistent nationwide.

Model directory resolution:
    - Default: backend/modules/segmentation/models/
    - Override with environment variable: SEGMENTATION_MODEL_DIR
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict


# -------------------------------------------------------------
# Model Storage Paths
# -------------------------------------------------------------

def get_model_dir() -> str:
    """Resolve model directory from env or default."""
    env_dir = os.getenv("SEGMENTATION_MODEL_DIR")
    if env_dir:
        return env_dir
    return "backend/modules/segmentation/models"


# -------------------------------------------------------------
# Segmentation Configuration Dataclass
# -------------------------------------------------------------

@dataclass
class SegmentationConfig:

    # -----------------------------
    # Model directories
    # -----------------------------
    model_root: str = field(default_factory=get_model_dir)
    macro_model_dir: str = field(init=False)
    micro_model_dir: str = field(init=False)

    def __post_init__(self):
        self.macro_model_dir = os.path.join(self.model_root, "macro")
        self.micro_model_dir = os.path.join(self.model_root, "micro")

    # -----------------------------
    # Simple segmentation settings
    # -----------------------------
    n_segments: int = 25  # Number of segments to create
    top_n_segments: int = 5  # Top N segments to assign per ZCTA

    # -----------------------------
    # Column Groups [ALIGNED TO DELAWARE SCHEMA]
    # -----------------------------

    # Identifiers
    id_col: str = "zcta"

    # Race
    race_total: str = "race_total"
    race_columns: List[str] = field(default_factory=lambda: [
        "race_white",
        "race_black",
        "race_asian",
        "race_native",
        "race_pacific",
        "race_other",
        "race_two_or_more"
    ])

    hispanic_total: str = "hispanic_total"

    # Age
    population_col: str = "population"
    age_columns: List[str] = field(default_factory=lambda: [
        "age_0_17",
        "age_18_34",
        "age_35_54",
        "age_55_74",
        "age_75_plus"
    ])
    median_age_col: str = "median_age"

    # Education
    education_total: str = "total_education_population"
    education_bachelors_plus: List[str] = field(default_factory=lambda: [
        "bachelors_complete",
        "masters",
        "professional",
        "doctorate"
    ])

    # Poverty
    poverty_total: str = "poverty_total"
    poverty_below: str = "income_below_poverty"

    # Household
    household_total: str = "household_total"
    household_size_1: str = "household_size_1"
    household_with_children: str = "households_with_children_total"

    # Housing
    median_home_value: str = "median_home_value"
    median_gross_rent: str = "median_gross_rent"

    # Vehicle availability
    vehicles_total: str = "vehicles_available_total"
    vehicles_no: str = "no_vehicle"

    # Commute
    commute_total: str = "commute_total"
    commute_columns: List[str] = field(default_factory=lambda: [
        "commute_drove_alone",
        "commute_carpool",
        "commute_public_transit",
        "commute_walked",
        "worked_from_home",
    ])

    # Industry mix (we use industry_* rather than employment_*)
    industry_total: str = "industry_total"
    industry_columns: List[str] = field(default_factory=lambda: [
        "industry_manufacturing",
        "industry_retail_trade",
        "industry_finance_insurance_real_estate",
        "industry_educational_services",
        "industry_healthcare_social_assistance",
        "industry_construction",
        "industry_transportation_utilities",
        "industry_public_administration",
    ])

    # Collapse income brackets into broader custom bins (Option B)
    # Only two bins exist in your uploaded dataset: under_10k, 200k_plus.
    # We will treat these as signals rather than create full distributions.
    income_columns: List[str] = field(default_factory=lambda: [
        "income_under_10k",
        "income_200k_plus",
        "median_household_income",     # main driver
        "per_capita_income"
    ])

    # Earnings (as direct numeric signals)
    occupation_earnings_columns: List[str] = field(default_factory=lambda: [
        "occupation_median_earnings_management",
        "occupation_median_earnings_business_science_arts",
        "occupation_median_earnings_service",
        "occupation_median_earnings_sales_office",
        "occupation_median_earnings_natural_resources_construction",
        "occupation_median_earnings_production_transportation",
    ])

    # For labeling (max features to consider)
    n_label_features: int = 5
    
    # Multi-segment assignments
    top_n_segments: int = 5  # Number of top segments to assign per ZCTA

