# OpenSegments

A machine learning-powered segmentation engine for ZCTA (ZIP Code Tabulation Area) demographic data using American Community Survey (ACS) statistics. This system uses unsupervised learning (K-Means clustering) to create human-readable market segments automatically from demographic patterns.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

Train the segmentation model using the sample data:

```bash
python run_training.py
```

Or specify your own CSV file and output directory:

```bash
python run_training.py --csv data/delaware.csv --output outputs
```

### 3. View Results

After training completes, check the `outputs/` directory for:
- **`final_segments.csv`**: ZCTA segment assignments with scores and rankings

## What It Does

OpenSegments transforms raw demographic data into actionable market segments by:

1. **Feature Engineering**: Converts raw ACS data into normalized, meaningful features
2. **Clustering**: Creates segments using K-Means clustering (default: 25 segments)
3. **Label Generation**: Produces human-readable segment names (e.g., "Students & Young Adults", "Affluent Retirees", "Urban Working Class")
4. **Scoring**: Assigns each ZCTA to multiple segments with similarity scores

## Machine Learning Approach

OpenSegments uses **unsupervised machine learning** to discover patterns in demographic data without requiring labeled examples.

### Core ML Techniques

1. **K-Means Clustering** (Unsupervised Learning)
   - Groups ZCTAs into segments based on demographic similarity
   - Uses scikit-learn's `KMeans` algorithm
   - Default: 25 clusters (configurable)
   - Finds natural groupings in high-dimensional demographic space

2. **Feature Scaling** (Preprocessing)
   - Uses `StandardScaler` to normalize features (mean=0, std=1)
   - Essential for clustering when features have different scales (e.g., income vs. age percentages)
   - Ensures all features contribute equally to distance calculations

3. **Distance-Based Similarity Scoring**
   - Computes Euclidean distance between ZCTAs and segment centroids
   - Converts distances to similarity scores (higher = more similar)
   - Uses softmax normalization to create probability-like scores

### Why Unsupervised Learning?

- **No labels needed**: Discovers segments automatically from data patterns
- **Data-driven**: Finds natural groupings that may not be obvious
- **Scalable**: Works with any demographic dataset following the ACS schema
- **Interpretable**: Generates human-readable labels from discovered patterns

### Model Training Process

The model learns by:
1. Analyzing demographic patterns across all ZCTAs
2. Identifying clusters of similar areas
3. Computing representative centroids for each segment
4. Learning which features distinguish each segment

Once trained, the model can assign new ZCTAs to segments based on their demographic characteristics.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup Steps

1. **Clone or download the repository**

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line (Recommended)

The easiest way to run training is using the provided script:

```bash
# Use default data file (data/delaware.csv)
python run_training.py

# Specify custom input and output
python run_training.py --csv path/to/your/data.csv --output my_outputs
```

### Python API

You can also use the training API directly in Python:

```python
import pandas as pd
from trainer import SegmentationTrainer

# Load your data
df = pd.read_csv('data/delaware.csv')

# Initialize trainer
trainer = SegmentationTrainer()

# Train from DataFrame
results = trainer.train_from_dataframe(df, output_dir='outputs')

# View results
print(results.head())
```

### Training from PostgreSQL (Advanced)

```python
from sqlalchemy.orm import Session
from trainer import SegmentationTrainer

# Assuming you have a SQLAlchemy session
trainer = SegmentationTrainer()
results = trainer.train(session, output_dir='outputs')
```

## Project Structure

```
OpenSegments/
├── run_training.py          # Main training script (use this!)
├── trainer.py               # Training orchestration
├── config.py                # Configuration and schema
├── features.py              # Feature engineering
├── labels.py                # Human-readable label generation
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── data/
│   └── delaware.csv         # Sample dataset
├── outputs/                 # Training outputs (created after training)
│   └── final_segments.csv   # Segment assignments
└── backend/modules/segmentation/models/  # Saved models
    └── segments/            # Model artifacts
```

## Output Files

After training, you'll find:

### `outputs/final_segments.csv`

Contains segment assignments for each ZCTA with:
- `zcta`: ZIP Code Tabulation Area identifier
- `segment_id`: Human-readable segment name (e.g., "Students & Young Adults")
- `score`: Similarity score (0-1, higher = more similar)
- `rank`: Ranking of this segment for the ZCTA (1 = best match)

Each ZCTA can be assigned to multiple segments (top 5 by default), allowing for nuanced understanding of demographic characteristics.

### Model Artifacts

Trained models are saved to `backend/modules/segmentation/models/segments/`:
- `kmeans.joblib`: Trained clustering model
- `scaler.joblib`: Feature scaler for new predictions

## Data Requirements

Your input CSV file must contain columns matching the ACS schema. Required column groups include:

- **Identifiers**: `zcta` (or configurable via `id_col` in config)
- **Race**: `race_total`, `race_white`, `race_black`, `race_asian`, etc.
- **Age**: `population`, `age_0_17`, `age_18_34`, `age_35_54`, `age_55_74`, `age_75_plus`, `median_age`
- **Education**: `total_education_population`, `bachelors_complete`, `masters`, etc.
- **Household**: `household_total`, `household_size_1`, `households_with_children_total`
- **Housing**: `median_home_value`, `median_gross_rent`
- **Income**: `median_household_income`, `per_capita_income`, `income_under_10k`, `income_200k_plus`
- **Industry**: `industry_total`, `industry_manufacturing`, `industry_retail_trade`, etc.
- **Commute**: `commute_total`, `commute_drove_alone`, `commute_public_transit`, `worked_from_home`
- **Vehicles**: `vehicles_available_total`, `no_vehicle`
- **Poverty**: `poverty_total`, `income_below_poverty`

See `config.py` for the complete schema definition. The sample `data/delaware.csv` demonstrates the expected format.

## Configuration

Key configuration options in `SegmentationConfig` (in `config.py`):

- **Clustering**:
  - `n_segments`: Number of segments to create (default: 25)
  - `top_n_segments`: Number of top segments to assign per ZCTA (default: 5)

- **Model Storage**:
  - `model_root`: Base directory for saved models (default: `backend/modules/segmentation/models`)
  - Override via `SEGMENTATION_MODEL_DIR` environment variable

- **Labeling**:
  - `n_label_features`: Number of top features used for label generation (default: 5)

## How It Works

### Training Pipeline

The machine learning pipeline processes data through these stages:

```
Raw Data (CSV)
    ↓
Feature Engineering (FeatureBuilder)
    - Converts raw counts to percentages/ratios
    - Creates normalized demographic features
    ↓
Feature Scaling (StandardScaler - ML Preprocessing)
    - Normalizes all features to same scale
    - Required for accurate clustering
    ↓
K-Means Clustering (Unsupervised ML)
    - Learns 25 segment centroids from data
    - Groups similar ZCTAs together
    ↓
Z-Score Computation
    - Calculates how each segment differs from average
    - Identifies distinguishing characteristics
    ↓
Label Generation (Rule-based)
    - Creates human-readable names from top features
    - Maps demographic patterns to segment labels
    ↓
Segment Scoring (Distance-based ML)
    - Computes similarity scores for each ZCTA-segment pair
    - Uses Euclidean distance in feature space
    ↓
CSV Output + Model Artifacts
    - Saves trained model for future predictions
    - Exports segment assignments with scores
```

### Segment Labels

The system automatically generates human-readable segment names based on top demographic features. Examples:

- **"Students & Young Adults"**: High concentration of 18-34 year olds, often in college areas
- **"Affluent Retirees"**: High home values, older population, highly educated
- **"Urban Working Class - Black Community"**: Low income, transit users, predominantly Black
- **"Blue Collar Workers - Asian Community"**: Manufacturing/construction workers, Asian presence
- **"General Population"**: Mixed characteristics, no dominant features

## Dependencies

- **numpy** (≥1.21.0): Numerical operations
- **pandas** (≥1.3.0): Data manipulation
- **scikit-learn** (≥1.0.0): Machine learning (KMeans, StandardScaler)
- **sqlalchemy** (≥1.4.0): Database ORM (for PostgreSQL mode)
- **joblib** (≥1.0.0): Model serialization

## Troubleshooting

### Common Issues

**"CSV file not found"**
- Make sure your CSV file path is correct
- Use absolute paths if relative paths don't work

**"Missing required columns"**
- Check that your CSV has all required columns (see Data Requirements)
- Compare with `data/delaware.csv` as a reference

**Import errors**
- Make sure you've installed all dependencies: `pip install -r requirements.txt`
- Activate your virtual environment if using one

## Examples

### Example 1: Train with Default Data

```bash
python run_training.py
```

This will:
- Load `data/delaware.csv`
- Train the model
- Save results to `outputs/final_segments.csv`

### Example 2: Custom Input and Output

```bash
python run_training.py --csv my_data.csv --output my_results
```

### Example 3: Python Script

```python
import pandas as pd
from trainer import SegmentationTrainer

# Load data
df = pd.read_csv('data/delaware.csv')

# Train
trainer = SegmentationTrainer()
results = trainer.train_from_dataframe(df, output_dir='outputs')

# Filter results for specific ZCTA
zcta_19717 = results[results['zcta'] == 19717]
print(zcta_19717)
```

## Development

### Adding New Features

To extend the feature set:

1. Add column names to `SegmentationConfig` in `config.py`
2. Add feature engineering logic in `FeatureBuilder.build_features()` in `features.py`
3. Add keyword mapping in `SegmentLabelGenerator._keyword()` in `labels.py` for label generation

### Module Structure

- **config.py**: Centralized configuration and schema definitions
- **features.py**: Data transformation and feature engineering
- **trainer.py**: High-level training orchestration
- **labels.py**: Human-readable segment generation
- **utils.py**: Shared utility functions

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Contact

[Add contact information if desired]
