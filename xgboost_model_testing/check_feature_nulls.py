"""
check_feature_nulls.py

Utility to:
  - Fetch raw PlayerGameLogs for the configured SEASONS / SEASON_TYPE,
  - Build the full feature table via build_feature_table(),
  - Report which columns contain missing values *before* any fillna(0)
    that the training scripts apply.

Usage:
    python3 check_feature_nulls.py
"""

from train_xgboost_player import (
    SEASONS,
    SEASON_TYPE,
    fetch_league_player_logs,
    build_feature_table,
)


def main() -> None:
    print(f"Checking feature nulls for seasons: {SEASONS}, season_type={SEASON_TYPE!r}")

    # 1. Load raw logs
    df_raw = fetch_league_player_logs(SEASONS, SEASON_TYPE)

    # 2. Build feature table (no fillna here)
    df_feat = build_feature_table(df_raw)
    total_rows, total_cols = df_feat.shape
    print(f"Feature table shape after feature engineering: {total_rows} rows x {total_cols} columns")

    # 3. Inspect nulls before training scripts replace them
    null_counts = df_feat.isna().sum()
    cols_with_nulls = null_counts[null_counts > 0]

    if cols_with_nulls.empty:
        print("No missing values found in the feature table (pre-fill).")
    else:
        print("Columns with missing values (pre-fill):")
        for col, cnt in cols_with_nulls.items():
            print(f"  {col}: {cnt} null(s)")


if __name__ == "__main__":
    main()

