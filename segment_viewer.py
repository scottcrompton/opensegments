"""
segment_viewer.py

Visualization tool for the OpenSegments hierarchical segmentation system.

Features:
---------
✓ Load macro + micro models from disk
✓ Load data (CSV or DB)
✓ Generate PCA 2D scatter plots colored by macro or micro clusters
✓ Show centroid radar charts for selected segments
✓ Summaries of top z-score features per segment
✓ Exportable interactive HTML visualizations (Plotly)

Usage:
------
python segment_viewer.py --csv delaware.csv
python segment_viewer.py --db   (requires DB session)

"""

import os
import argparse
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.decomposition import PCA

from sqlalchemy.orm import Session

# Handle both relative imports (when used as package) and absolute imports (when run as script)
try:
    from .config import SegmentationConfig
    from .features import FeatureBuilder
    from .labels import SegmentLabelGenerator
    from .utils import compute_cluster_zscores, log
except ImportError:
    from config import SegmentationConfig
    from features import FeatureBuilder
    from labels import SegmentLabelGenerator
    from utils import compute_cluster_zscores, log


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def load_data(csv_path, session, feature_builder):
    """Load dataset from CSV or DB."""
    if csv_path:
        log(f"Loading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        return df

    if session:
        log("Loading dataset from DB...")
        return feature_builder.load_from_db(session)

    raise ValueError("Must provide either CSV path or DB session.")


def pca_transform(X_scaled, n_components=2):
    """Compute PCA for 2D visualization."""
    pca = PCA(n_components=n_components)
    pca_coords = pca.fit_transform(X_scaled)
    return pca_coords, pca.explained_variance_ratio_


def generate_macro_scatter(df_vis, output="macro_pca.html"):
    """2D PCA scatter plot for macro clusters."""
    fig = px.scatter(
        df_vis,
        x="pc1",
        y="pc2",
        color="macro",
        hover_data=["zcta", "segment_id"],
        title="Macro Segmentation — PCA Visualization",
        width=950,
        height=650
    )
    fig.write_html(output)
    print(f"[✓] Saved macro PCA visualization → {output}")


def generate_micro_scatter(df_vis, output="micro_pca.html"):
    """2D PCA scatter plot for micro clusters."""
    fig = px.scatter(
        df_vis,
        x="pc1",
        y="pc2",
        color="micro",
        hover_data=["zcta", "segment_id"],
        title="Micro Segmentation — PCA Visualization",
        width=950,
        height=650
    )
    fig.write_html(output)
    print(f"[✓] Saved micro PCA visualization → {output}")


def radar_chart(label, zscores, output_path=None):
    """Generate a radar chart of top z-score features."""
    sorted_feats = sorted(zscores.items(), key=lambda x: -abs(x[1]))[:8]
    labels = [f[0] for f in sorted_feats]
    values = [f[1] for f in sorted_feats]

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself'
        )
    )
    fig.update_layout(
        title=f"Feature Profile: {label}",
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=False
    )

    if output_path:
        fig.write_html(output_path)
        print(f"[✓] Saved radar chart → {output_path}")

    return fig


# -----------------------------------------------------------------------------
# MAIN VISUALIZATION PIPELINE
# -----------------------------------------------------------------------------

def visualize(csv_path=None, session=None):
    cfg = SegmentationConfig()
    fb = FeatureBuilder(cfg)

    # ---------------------------------------------------------------------
    # Load Models (simplified - single model)
    # ---------------------------------------------------------------------
    import joblib
    import os
    
    model_dir = os.path.join(cfg.model_root, "segments")
    kmeans_path = os.path.join(model_dir, "kmeans.joblib")
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"Model not found at {kmeans_path}. Please train the model first.")
    
    kmeans_model = joblib.load(kmeans_path)
    scaler = joblib.load(scaler_path)
    
    # Load feature names from metadata if available, otherwise use config
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        from .utils import read_json
        try:
            from utils import read_json
        except ImportError:
            pass
        metadata = read_json(metadata_path)
        feature_names = metadata.get("feature_names", [])
    else:
        # Fallback: would need to reconstruct from config
        feature_names = []
    
    labeler = SegmentLabelGenerator(cfg)

    # ---------------------------------------------------------------------
    # Load Dataset (CSV or DB)
    # ---------------------------------------------------------------------
    df_raw = load_data(csv_path, session, fb)

    # Build features + scale
    feat_df = fb.build_features(df_raw)
    X_scaled = fb.scaler.transform(
        feat_df.drop(columns=[cfg.id_col])
    )

    # ---------------------------------------------------------------------
    # Apply macro + micro cluster assignments
    # ---------------------------------------------------------------------
    macro_labels = macro_model.predict(X_scaled)
    feat_df["macro"] = macro_labels

    final_micro = np.zeros(len(df_raw), dtype=int)
    for macro_id in np.unique(macro_labels):
        mask = (macro_labels == macro_id)
        X_sub = X_scaled[mask]
        m = micro_model.models[macro_id]
        final_micro[mask] = m.predict(X_sub)
    feat_df["micro"] = final_micro

    # ---------------------------------------------------------------------
    # Compute labels
    # ---------------------------------------------------------------------
    macro_z = compute_cluster_zscores(X_scaled, macro_labels, feature_names)
    macro_text = labeler.create_macro_labels(macro_z)

    micro_z_all = {}
    for macro_id in np.unique(macro_labels):
        mask = (macro_labels == macro_id)
        X_sub = X_scaled[mask]
        micro_z_all[macro_id] = compute_cluster_zscores(
            X_sub, final_micro[mask], feature_names
        )

    micro_text = labeler.create_micro_labels(micro_z_all)

    segment_ids = []
    for _, row in feat_df.iterrows():
        seg = labeler.combine_labels(
            macro_text[int(row["macro"])],
            micro_text[int(row["macro"])][int(row["micro"])]
        )
        segment_ids.append(seg)
    feat_df["segment_id"] = segment_ids

    # ---------------------------------------------------------------------
    # PCA Visualization
    # ---------------------------------------------------------------------
    pca_coords, var_ratio = pca_transform(X_scaled)

    df_vis = feat_df.copy()
    df_vis["pc1"] = pca_coords[:, 0]
    df_vis["pc2"] = pca_coords[:, 1]
    df_vis["zcta"] = df_raw[cfg.id_col].astype(str)

    print(f"[INFO] PCA variance explained: {var_ratio}")

    generate_macro_scatter(df_vis, "macro_pca.html")
    generate_micro_scatter(df_vis, "micro_pca.html")

    # ---------------------------------------------------------------------
    # Example: Generate radar charts for each macro cluster
    # ---------------------------------------------------------------------
    for macro_id in np.unique(macro_labels):
        radar_chart(
            label=f"Macro {macro_id}",
            zscores=macro_z[macro_id],
            output_path=f"macro_{macro_id}_radar.html"
        )

    print("\n[✓] Visualization outputs generated!")
    print("Open the following files in a browser:")
    print(" - macro_pca.html")
    print(" - micro_pca.html")
    print(" - macro_<id>_radar.html\n")


# -----------------------------------------------------------------------------
# CLI SUPPORT
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Visualization Tool")

    parser.add_argument("--csv", type=str, help="Path to CSV file for testing")
    parser.add_argument("--use-db", action="store_true", help="Load data from DB instead of CSV")

    args = parser.parse_args()

    if args.csv and args.use_db:
        raise ValueError("Choose either --csv OR --use-db, not both.")

    if args.use_db:
        from db import SessionLocal
        session = SessionLocal()
        vis
