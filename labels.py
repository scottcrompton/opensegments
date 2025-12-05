"""
labels.py

SegmentLabelGenerator
---------------------

Creates human-readable labels for:
    - Macro clusters
    - Micro clusters within macro groups
    - Final combined hierarchical segment IDs

Label structure:
    Segment_<MacroLabel>_<MicroSubtypeLabel>

Example:
    Segment_Affluent_Professional_Suburbs_Young_Families
"""

import re
from typing import Dict, List, Tuple

try:
    from .utils import top_n_features, log
except ImportError:
    from utils import top_n_features, log


class SegmentLabelGenerator:
    """
    Generates human-readable labels for macro and micro clusters
    using top z-scored features.
    """

    def __init__(self, cfg):
        self.cfg = cfg

    # -------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------

    def create_macro_labels(self, macro_zscores: Dict[int, Dict[str, float]]):
        """
        Generate human-readable labels for each macro cluster.
        """
        labels = {}

        for macro_id, zdict in macro_zscores.items():
            pos, neg = top_n_features(zdict, self.cfg.n_label_features)
            labels[macro_id] = self._macro_label_from_features(pos, neg)

        return labels

    def create_micro_labels(self, micro_zscores_per_macro):
        """
        Generate human-readable micro labels per macro.
        """
        micro_labels = {}

        for macro_id, micro_dict in micro_zscores_per_macro.items():
            micro_labels[macro_id] = {}
            for micro_id, zdict in micro_dict.items():
                pos, neg = top_n_features(zdict, self.cfg.n_label_features)
                micro_labels[macro_id][micro_id] = self._micro_label_from_features(pos, neg)

        return micro_labels

    def combine_labels(self, macro_label: str, micro_label: str) -> str:
        """
        Create the final stable segment ID, normalized:
            <MacroLabel>_<MicroLabel>
        """
        combined = f"{macro_label}_{micro_label}"
        return self._normalize(combined)
    
    def simplify_segment_name(self, macro_label: str, micro_label: str) -> str:
        """
        Convert complex feature-based labels to simple, intuitive segment names.
        Creates more granular segments by combining macro and micro characteristics.
        """
        # Combine labels for analysis (labels are already keyword-based)
        combined = f"{macro_label}_{micro_label}".lower()
        macro_lower = macro_label.lower()
        micro_lower = micro_label.lower()
        
        # College/University Students - check for high concentration of young adults
        # This should catch areas like 19717 (UDEL)
        # Prioritize this check - college areas have high young adult concentration
        if any(x in combined for x in ["young_adults", "age_18_34"]):
            # High education + young = college area (preferred)
            if any(x in combined for x in ["highly_educated", "bachelors_plus", "education"]):
                if any(x in combined for x in ["carpool", "carpoolers"]):
                    return "College Students - Carpoolers"
                elif any(x in combined for x in ["transit", "transit_users"]):
                    return "College Students - Transit Users"
                elif any(x in combined for x in ["walkable", "walked"]):
                    return "College Students - Walkable"
                elif any(x in combined for x in ["asian_presence"]):
                    return "College Students - Asian Community"
                else:
                    return "College Students"
            # Even without explicit education keywords, if it's primarily young adults,
            # it's likely a college/university area - prioritize this over ethnic labels
            # Check if young adults is a dominant feature (not just a minor one)
            elif "young_adults" in macro_lower or "age_18_34" in macro_lower:
                # This is primarily a young adult area - likely students
                if any(x in combined for x in ["walkable", "walked", "transit"]):
                    return "Students & Young Adults - Walkable"
                elif any(x in combined for x in ["single_households", "1_person"]):
                    return "Students & Young Adults"
                else:
                    return "Students & Young Adults"
        
        # Elderly/Retirees (check first as it's most specific)
        if any(x in combined for x in ["elderly_population", "older_adults"]):
            if any(x in combined for x in ["government_workers", "public_administration"]):
                if any(x in combined for x in ["car_commuters"]):
                    return "Retirees - Government Workers"
                else:
                    return "Retirees"
            elif any(x in combined for x in ["high_home_value", "highly_educated"]):
                if any(x in combined for x in ["walkable_area"]):
                    return "Affluent Retirees - Walkable"
                else:
                    return "Affluent Retirees"
            else:
                return "Retirees"
        
        # Young Families / Millennials - more granular
        if any(x in combined for x in ["young_families"]):
            if any(x in combined for x in ["single_households", "1_person"]):
                if any(x in combined for x in ["car_commuters"]):
                    return "Young Professionals - Suburban"
                else:
                    return "Young Professionals - Urban"
            elif any(x in combined for x in ["family_households", "households_with_children"]):
                if any(x in combined for x in ["car_commuters"]):
                    return "Millennials - Suburban Families"
                elif any(x in combined for x in ["midlife_adults"]):
                    return "Millennials - Established Families"
                else:
                    return "Millennials"
            else:
                return "Young Families"
        
        # Affluent/Professional segments - more granular
        if any(x in combined for x in ["high_rent", "high_home_value", "highly_educated"]):
            if any(x in combined for x in ["finance_workers"]):
                if any(x in combined for x in ["walkable_area"]):
                    return "Affluent Professionals - Urban Finance"
                else:
                    return "Affluent Professionals - Finance"
            elif any(x in combined for x in ["walkable_area", "transit_users"]):
                if any(x in combined for x in ["asian_presence"]):
                    return "Urban Professionals - Asian Community"
                elif any(x in combined for x in ["healthcare_workers"]):
                    return "Urban Professionals - Healthcare"
                else:
                    return "Urban Professionals"
            elif any(x in combined for x in ["asian_presence"]):
                if any(x in combined for x in ["manufacturing_workers", "construction_workers"]):
                    return "Affluent Professionals - Asian Blue Collar"
                else:
                    return "Affluent Professionals - Asian Community"
            elif any(x in combined for x in ["healthcare_workers"]):
                return "Affluent Professionals - Healthcare"
            else:
                return "Affluent Suburbs"
        
        # Low-income segments - more granular
        if any(x in combined for x in ["low_income"]):
            if any(x in combined for x in ["transit_users", "transit_dependent", "no_vehicle"]):
                if any(x in combined for x in ["mostly_black"]):
                    return "Urban Working Class - Black Community"
                elif any(x in combined for x in ["asian_presence"]):
                    return "Urban Working Class - Asian Community"
                elif any(x in combined for x in ["carpool", "carpoolers"]):
                    return "Urban Working Class - Carpoolers"
                else:
                    return "Urban Working Class"
            elif any(x in combined for x in ["mostly_black"]):
                if any(x in combined for x in ["transit_users"]):
                    return "Urban Working Class - Black Transit Users"
                else:
                    return "Urban Working Class - Black Community"
            elif any(x in combined for x in ["remote_workers", "worked_from_home"]):
                if any(x in combined for x in ["retail_workers"]):
                    return "Working Class - Remote Retail"
                else:
                    return "Working Class - Remote"
            else:
                return "Low Income Communities"
        
        # Industry-based segments - more granular
        if any(x in combined for x in ["manufacturing_workers", "construction_workers"]):
            if any(x in combined for x in ["asian_presence"]):
                return "Blue Collar Workers - Asian Community"
            elif any(x in combined for x in ["car_commuters"]):
                return "Blue Collar Workers - Car Commuters"
            else:
                return "Blue Collar Workers"
        if any(x in combined for x in ["retail_workers"]):
            if any(x in combined for x in ["low_income"]):
                return "Service Workers - Low Income"
            else:
                return "Service Workers"
        if any(x in combined for x in ["healthcare_workers"]):
            if any(x in combined for x in ["urban", "walkable"]):
                return "Healthcare Workers - Urban"
            else:
                return "Healthcare Workers"
        if any(x in combined for x in ["education_workers", "educational"]):
            return "Educators"
        
        # Commute-based segments - more granular
        if any(x in combined for x in ["car_commuters", "drove_alone"]):
            if any(x in combined for x in ["midlife_adults"]):
                if any(x in combined for x in ["family_households"]):
                    return "Suburban Families - Car Commuters"
                else:
                    return "Suburban - Car Commuters"
            else:
                return "Car-Dependent Communities"
        
        if any(x in combined for x in ["walkable_area", "walked"]):
            if any(x in combined for x in ["elderly"]):
                return "Walkable Urban - Retirees"
            else:
                return "Walkable Urban"
        
        if any(x in combined for x in ["remote_workers", "worked_from_home"]):
            if any(x in combined for x in ["construction_workers"]):
                return "Remote Workers - Construction"
            else:
                return "Remote Workers"
        
        # Default: use simplified macro label with micro characteristics
        # Extract key words and create a more descriptive name
        keywords = []
        if "asian_presence" in combined:
            keywords.append("Asian")
        if "hispanic_community" in combined:
            keywords.append("Hispanic")
        if "mostly_black" in combined:
            keywords.append("Black")
        if "mostly_white" in combined:
            keywords.append("White")
        
        if keywords:
            # Prioritize "Students & Young Adults" over ethnic labels for young adult areas
            # This catches cases like 19717 (UDEL) where young adults is the dominant feature
            if any(x in combined for x in ["young_adults", "age_18_34"]) or any(x in micro_lower for x in ["young_adults", "age_18_34"]):
                if any(x in macro_lower for x in ["young_adults", "age_18_34"]) or any(x in combined for x in ["single_households", "1_person"]):
                    return "Students & Young Adults"
            # Add micro characteristics if available
            if any(x in micro_lower for x in ["young", "families"]):
                return " ".join(keywords) + " - Young Communities"
            elif any(x in micro_lower for x in ["elderly", "older"]):
                return " ".join(keywords) + " - Retiree Communities"
            else:
                return " ".join(keywords) + " Communities"
        
        return "General Population"

    # -------------------------------------------------------------
    # Macro Label Builder
    # -------------------------------------------------------------

    def _macro_label_from_features(self, top_pos, top_neg):
        """
        Build macro-level lifestyle label from highest positive z-scores.
        Macro labels are broad (Tapestry-style).
        """
        keywords = []

        for feat, _ in top_pos:
            kw = self._keyword(feat)
            if kw:
                keywords.append(kw)

        if not keywords:
            keywords.append("General_Population")

        # First 2–3 keywords form the macro name
        macro_label = "_".join(keywords[:3])
        return self._normalize(macro_label)

    # -------------------------------------------------------------
    # Micro Label Builder
    # -------------------------------------------------------------

    def _micro_label_from_features(self, top_pos, top_neg):
        """
        Build micro-level subtype label (more detailed).
        """
        keywords = []

        for feat, _ in top_pos:
            kw = self._keyword(feat)
            if kw:
                keywords.append(kw)

        if not keywords:
            keywords.append("General")

        micro_label = "_".join(keywords[:3])  # allow more detail
        return self._normalize(micro_label)

    # -------------------------------------------------------------
    # Feature Keyword Mapping
    # -------------------------------------------------------------

    def _keyword(self, feat_name: str) -> str:
        """
        Convert engineered feature names into meaningful English.
        """

        # Race
        if "race_white" in feat_name:
            return "Mostly_White"
        if "race_black" in feat_name:
            return "Mostly_Black"
        if "race_asian" in feat_name:
            return "Asian_Presence"
        if "race_hispanic" in feat_name or "pct_hispanic" in feat_name:
            return "Hispanic_Community"

        # Age
        if "age_0_17" in feat_name:
            return "Young_Families"
        if "age_18_34" in feat_name:
            return "Young_Adults"
        if "age_35_54" in feat_name:
            return "Midlife_Adults"
        if "age_55_74" in feat_name:
            return "Older_Adults"
        if "age_75_plus" in feat_name:
            return "Elderly_Population"

        if feat_name == "median_age":
            return "Age_Skew"

        # Education
        if "bachelors_plus" in feat_name:
            return "Highly_Educated"

        # Poverty
        if "below_poverty" in feat_name:
            return "Low_Income"

        # Household
        if "1_person" in feat_name:
            return "Single_Households"
        if "households_with_children" in feat_name:
            return "Family_Households"

        # Housing
        if "home_value" in feat_name:
            return "High_Home_Value"
        if "gross_rent" in feat_name:
            return "High_Rent"

        # Vehicles
        if "no_vehicle" in feat_name:
            return "Transit_Dependent"

        # Commute
        if "drove_alone" in feat_name:
            return "Car_Commuters"
        if "carpool" in feat_name:
            return "Carpoolers"
        if "public_transit" in feat_name:
            return "Transit_Users"
        if "walked" in feat_name:
            return "Walkable_Area"
        if "worked_from_home" in feat_name:
            return "Remote_Workers"

        # Industry sectors
        if "manufacturing" in feat_name:
            return "Manufacturing_Workers"
        if "retail_trade" in feat_name:
            return "Retail_Workers"
        if "finance" in feat_name:
            return "Finance_Workers"
        if "educational_services" in feat_name:
            return "Education_Workers"
        if "healthcare" in feat_name:
            return "Healthcare_Workers"
        if "construction" in feat_name:
            return "Construction_Workers"
        if "transportation_utilities" in feat_name:
            return "Transport_Utility_Workers"
        if "public_administration" in feat_name:
            return "Government_Workers"

        return None  # fallback

    # -------------------------------------------------------------
    # Normalization
    # -------------------------------------------------------------

    def _normalize(self, text: str) -> str:
        """
        Convert text into a stable segment identifier:
            - Replace spaces with underscores
            - Remove invalid characters
        """
        text = text.replace(" ", "_")
        text = re.sub(r"[^A-Za-z0-9_]", "", text)
        text = re.sub(r"_+", "_", text)
        return text.strip("_")
