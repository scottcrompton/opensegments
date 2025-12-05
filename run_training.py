#!/usr/bin/env python
"""
run_training.py

Runnable script for training the segmentation model from CSV data.

Usage:
    python run_training.py [--csv path/to/data.csv] [--output outputs/]
"""

import os
import sys
import argparse
import pandas as pd

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from trainer import SegmentationTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Train segmentation model from CSV data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/delaware.csv',
        help='Path to input CSV file (default: data/delaware.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for results (default: outputs)'
    )
    
    args = parser.parse_args()
    
    # Check if CSV file exists
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        sys.exit(1)
    
    # Load data
    print(f"Loading data from {args.csv}...")
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows")
    
    # Initialize trainer
    trainer = SegmentationTrainer()
    
    # Train
    print(f"\nStarting training...")
    print(f"Output directory: {args.output}")
    results = trainer.train_from_dataframe(df, output_dir=args.output)
    
    print(f"\nTraining complete!")
    print(f"Results saved to: {args.output}")
    print(f"\nSample segment assignments:")
    print(results.head(20))


if __name__ == '__main__':
    main()
