"""
trainer.py

SegmentationTrainer
-------------------

Provides TWO training entry points:

1. train(session)
   - Loads ZCTA data from PostgreSQL (production mode)

2. train_from_dataframe(df_raw)
   - Trains directly from any DataFrame (CSV mode / proof of concept)

Both pipelines then:
    - Build engineered features
    - Fit macro model (12 clusters)
    - Compute macro z-scores
    - Fit micro models (auto-K per macro group)
    - Compute micro z-scores
    - Generate human-readable labels
    - Save CSV outputs
    - Save macro + micro model artifacts
"""

import os
import sys
import numpy as np
import pandas as pd

from sqlalchemy.orm import Session

# Handle both relative imports (when used as package) and absolute imports (when run as script)
try:
    from .config import SegmentationConfig
    from .features import FeatureBuilder
    from .labels import SegmentLabelGenerator
    from .utils import (
        log,
        ensure_dir,
        compute_cluster_zscores,
        compute_segment_scores
    )
except ImportError:
    # Running as script - use absolute imports
    from config import SegmentationConfig
    from features import FeatureBuilder
    from labels import SegmentLabelGenerator
    from utils import (
        log,
        ensure_dir,
        compute_cluster_zscores,
        compute_segment_scores
    )


class SegmentationTrainer:
    """
    Orchestrates the complete segmentation model creation process.

    Supports:
        • train(session)           → PostgreSQL → full pipeline
        • train_from_dataframe(df) → CSV / local testing → full pipeline
    """

    def __init__(self, cfg: SegmentationConfig = None):
        self.cfg = cfg or SegmentationConfig()

        # Core components
        self.feature_builder = FeatureBuilder(self.cfg)
        self.labeler = SegmentLabelGenerator(self.cfg)

    # -------------------------------------------------------------------------
    # PUBLIC API — TRAINING ENTRY POINTS
    # -------------------------------------------------------------------------

    def train(self, session: Session, output_dir: str = None):
        """
        Production training.
        Loads all ZCTAs from PostgreSQL, then trains the complete segmentation.
        """
        log("==== SEGMENTATION TRAINING (DB MODE) START ====")

        if output_dir is None:
            output_dir = os.path.join(self.cfg.model_root, "outputs")

        ensure_dir(output_dir)

        # 1. Load from DB
        df_raw = self.feature_builder.load_from_db(session)

        return self._run_training_pipeline(df_raw, output_dir)

    def train_from_dataframe(self, df_raw: pd.DataFrame, output_dir: str = None):
        """
        Train from DataFrame (CSV mode).
        """
        log("==== SEGMENTATION TRAINING START ====")

        if output_dir is None:
            output_dir = os.path.join(self.cfg.model_root, "outputs")

        ensure_dir(output_dir)

        return self._run_training_pipeline(df_raw, output_dir)

    # -------------------------------------------------------------------------
    # INTERNAL PIPELINE (SHARED BY DB + CSV MODES)
    # -------------------------------------------------------------------------

    def _run_training_pipeline(self, df_raw: pd.DataFrame, output_dir: str):
        """
        Simplified segmentation pipeline - single-level clustering.
        """

        # ---------------------------------------------------------
        # 1. Build engineered features
        # ---------------------------------------------------------
        log(f"Building features for {len(df_raw)} ZCTAs...")
        feat_df = self.feature_builder.build_features(df_raw)

        # Fit scaler + scale features
        X_scaled = self.feature_builder.fit_scaler(feat_df)
        feature_names = list(feat_df.drop(columns=[self.cfg.id_col]).columns)

        # ---------------------------------------------------------
        # 2. Fit Single-Level Segmentation Model
        # ---------------------------------------------------------
        from sklearn.cluster import KMeans
        
        # Use a reasonable number of clusters (20-30 for good granularity)
        n_clusters = getattr(self.cfg, 'n_segments', 25)
        log(f"Fitting {n_clusters} segments...")
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=20
        )
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Save model
        import joblib
        model_dir = os.path.join(self.cfg.model_root, "segments")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(kmeans, os.path.join(model_dir, "kmeans.joblib"))
        joblib.dump(self.feature_builder.scaler, os.path.join(model_dir, "scaler.joblib"))
        
        log(f"Segmentation model saved to {model_dir}")

        # ---------------------------------------------------------
        # 3. Compute Z-Scores and Create Segment Labels
        # ---------------------------------------------------------
        log("Computing segment characteristics...")
        segment_zscores = compute_cluster_zscores(X_scaled, cluster_labels, feature_names)
        
        # Create simplified segment names
        segment_labels_map = {}
        for cluster_id, zscores in segment_zscores.items():
            # Get top features for this segment
            sorted_features = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
            top_pos = sorted_features[:5]
            
            # Create a simple label based on top features
            # We'll use a simplified approach - just use the top feature keyword
            segment_name = self._create_simple_segment_name(zscores, feature_names)
            segment_labels_map[cluster_id] = segment_name
        
        log(f"Created {len(segment_labels_map)} unique segments")

        # ---------------------------------------------------------
        # 4. Compute Segment Scores for All ZCTAs
        # ---------------------------------------------------------
        log("Computing segment scores...")
        
        # Get cluster centroids
        segment_centroids = {}
        for cluster_id in range(n_clusters):
            segment_centroids[cluster_id] = kmeans.cluster_centers_[cluster_id]
        
        # Compute scores (similarity to each segment)
        segment_scores, segment_keys = compute_segment_scores(X_scaled, segment_centroids)
        
        # Normalize scores to 0-1 range (softmax)
        exp_scores = np.exp(segment_scores - segment_scores.max(axis=1, keepdims=True))
        normalized_scores = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        
        # ---------------------------------------------------------
        # 5. Create Final Segment Assignments
        # ---------------------------------------------------------
        log("Creating segment assignments...")
        
        top_n = getattr(self.cfg, 'top_n_segments', 5)
        segment_data = []
        
        for i, row in feat_df.iterrows():
            zcta = row[self.cfg.id_col]
            scores = normalized_scores[i]
            
            # Sort segments by score (descending)
            sorted_indices = np.argsort(scores)[::-1]
            
            # Get top N segments
            for rank, idx in enumerate(sorted_indices[:top_n], 1):
                if scores[idx] > 0.01:  # Only include segments with meaningful score (>1%)
                    segment_id = segment_keys[idx]
                    segment_name = segment_labels_map[segment_id]
                    score = float(scores[idx])
                    
                    segment_data.append({
                        self.cfg.id_col: zcta,
                        'segment_id': segment_name,
                        'score': score,
                        'rank': rank
                    })
        
        segments_df = pd.DataFrame(segment_data)

        # ---------------------------------------------------------
        # 6. Save Output
        # ---------------------------------------------------------
        log("Saving segmentation outputs...")
        
        segments_df.to_csv(
            os.path.join(output_dir, "final_segments.csv"), index=False
        )
        
        log(f"Saved final_segments.csv with {len(segments_df)} segment assignments")
        log("==== SEGMENTATION TRAINING COMPLETE ====")

        return segments_df
    
    def _create_simple_segment_name(self, zscores: dict, feature_names: list) -> str:
        """
        Create a simple segment name from z-scores.
        Uses the simplified naming logic from SegmentLabelGenerator.
        """
        # Get top positive features
        sorted_features = sorted(zscores.items(), key=lambda x: x[1], reverse=True)
        top_pos = sorted_features[:5]
        
        # Extract keywords from top features
        keywords = []
        for feat_name, zscore in top_pos:
            if zscore > 0.5:  # Only use significantly positive features
                kw = self.labeler._keyword(feat_name)
                if kw:
                    keywords.append(kw)
        
        # Create macro and micro labels (simplified)
        if not keywords:
            return "General Population"
        
        macro_keywords = keywords[:2] if len(keywords) >= 2 else keywords
        micro_keywords = keywords[2:4] if len(keywords) >= 4 else keywords[1:3] if len(keywords) >= 3 else keywords
        
        macro_label = "_".join(macro_keywords[:2])
        micro_label = "_".join(micro_keywords[:2]) if micro_keywords else macro_label
        
        # Use simplified naming
        return self.labeler.simplify_segment_name(macro_label, micro_label)
