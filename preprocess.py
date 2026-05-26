"""
Step 2: Clean raw_data.csv and produce features.csv
that is ready for the deep learning model.

What this does:
  - Separates condition columns (mill input) from target columns (roll output)
  - Fills missing values sensibly
  - Encodes categorical IDs
  - Normalises numeric columns to [0, 1]
  - Saves scalers/encoders for use at inference time
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ------------------------------------------------------------------ #
# COLUMN GROUPS
# These match exactly what extract_features.py selected.
# ------------------------------------------------------------------ #

# Conditions = what the engineer knows BEFORE designing the roll.
# These are the INPUT to the CVAE at inference time.
CONDITION_NUMERIC = [
    'RPM',
    'MaxTension', 'MinTension',
    'MaxWrapAngle', 'MinWrapAngle',
    'MaxStripTemp',
]

CONDITION_BOOL = []       # none available from WPF
CONDITION_CATEGORICAL = [] # none available from WPF

TARGET_NUMERIC = [
    'TotalNoOfZones',
    'NoOf52mmZones',
    'NoOf26mmZones',
    'BearingCentreDistance',
]

TARGET_BOOL = []

TARGET_CATEGORICAL = [
    'RollDiameterID',
]

TEXT_CATEGORICAL = []  # remove StripCooling, GearBox




def preprocess(input_csv="raw_data.csv", output_csv="features.csv"):
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} rows from {input_csv}")

    # ── 1. Drop rows where ALL target columns are null ────────────
    # (order exists but no roll was ever configured - not useful)
    target_cols = TARGET_NUMERIC + TARGET_BOOL + TARGET_CATEGORICAL
    before = len(df)
    df = df.dropna(subset=['TotalNoOfZones', 'RPM'], how='all')
    print(f"After dropping empty rows: {len(df)} (removed {before - len(df)})")

    # ── 2. Fill missing NUMERIC values with column median ─────────
    all_numeric = CONDITION_NUMERIC + TARGET_NUMERIC
    for col in all_numeric:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # ── 3. Fill missing BOOL values with 0 (False) ───────────────
    all_bool = CONDITION_BOOL + TARGET_BOOL
    for col in all_bool:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── 4. Fill missing CATEGORICAL IDs with 0 (= "unknown") ─────
    all_categorical = CONDITION_CATEGORICAL + TARGET_CATEGORICAL
    for col in all_categorical:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # ── 5. Encode text categoricals ───────────────────────────────
    label_encoders = {}
    all_label_cols = (
            CONDITION_CATEGORICAL +
            TARGET_CATEGORICAL +
            TEXT_CATEGORICAL
        )

    for col in all_label_cols:
        if col in df.columns:
            # Fill NaN with the most common value instead of 'Unknown'
            most_common = df[col].mode()[0] if df[col].notna().any() else 'None'
            df[col] = df[col].fillna(most_common)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded '{col}': {len(le.classes_)} unique values")

    # ── 6. Normalise numeric columns to [0, 1] ────────────────────
    scalers = {}
    for col in all_numeric:
        if col in df.columns:
            scaler = MinMaxScaler()
            df[col] = scaler.fit_transform(df[[col]])
            scalers[col] = scaler

    # ── 7. Save preprocessors for inference time ──────────────────
    with open("preprocessors.pkl", "wb") as f:
        pickle.dump({
            'scalers': scalers,
            'label_encoders': label_encoders,
        }, f)
    print("✓ Saved preprocessors.pkl")

    # ── 8. Save column group metadata ─────────────────────────────
    metadata = {
        'condition_cols': CONDITION_NUMERIC + CONDITION_BOOL + CONDITION_CATEGORICAL,
        'target_cols': (TARGET_NUMERIC + TARGET_BOOL +
                        TARGET_CATEGORICAL + TEXT_CATEGORICAL),
        'condition_numeric': CONDITION_NUMERIC,
        'condition_bool': CONDITION_BOOL,
        'condition_categorical': CONDITION_CATEGORICAL,
        'target_numeric': TARGET_NUMERIC,
        'target_bool': TARGET_BOOL,
        'target_categorical': TARGET_CATEGORICAL,
    }
    with open("column_metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    print("✓ Saved column_metadata.pkl")

    # ── 9. Save processed CSV ─────────────────────────────────────
    df.to_csv(output_csv, index=False)
    print(f"✓ Saved {output_csv}")
    print(f"\nFinal shape: {df.shape}")
    print(f"Condition columns: {len(metadata['condition_cols'])}")
    print(f"Target columns:    {len(metadata['target_cols'])}")

    return df, metadata


if __name__ == "__main__":
    df, meta = preprocess()
    print("\nSample (first 3 rows, condition cols only):")
    print(df[meta['condition_cols'][:5]].head(3))