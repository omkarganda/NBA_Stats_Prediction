"""
train_player_models_ensemble.py

Train an ensemble of XGBoost, LightGBM, and CatBoost models for
minutes and player stat targets, with hyperparameter tuning via Optuna.

For each target (MIN, PTS, REB, AST, FG3M, PRA), this script:
  - Tunes separate XGB, LGBM, and CatBoost regressors on a time-based
    train/validation split, using RECENCY_WEIGHT as sample_weight.
  - Trains final models on train+val with the best hyperparameters.
  - Builds an EnsembleRegressor that averages the three models with
    weights proportional to 1 / (val_RMSE^2).
  - Evaluates the ensemble on a held-out test set.
  - Saves ensemble models as:
        minutes_model_ensemble.joblib
        stat_model_<lower_target>_ensemble.joblib

Usage:
    python3 train_player_models_ensemble.py

Requirements:
    pip install optuna lightgbm catboost
"""

from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from joblib import dump
from pathlib import Path

from train_xgboost_player import (
    RANDOM_STATE,
    SEASONS,
    SEASON_TYPE,
    TARGET_STATS,
    build_feature_table,
    fetch_league_player_logs,
    get_minutes_feature_cols,
    get_stat_feature_cols,
    time_based_split,
)
from ensemble_regressor import EnsembleRegressor

# Third-party model families
try:
    from xgboost import XGBRegressor
except ImportError as exc:
    raise ImportError("xgboost is required. Install with `pip install xgboost`.") from exc

try:
    from lightgbm import LGBMRegressor
except ImportError as exc:
    raise ImportError("lightgbm is required. Install with `pip install lightgbm`.") from exc

try:
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise ImportError("catboost is required. Install with `pip install catboost`.") from exc


# Reduce Optuna logging to warnings only; rely on progress bars instead of per-trial logs.
optuna.logging.set_verbosity(optuna.logging.WARNING)


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_raw = fetch_league_player_logs(SEASONS, SEASON_TYPE)
    df_feat = build_feature_table(df_raw)
    # Avoid NaNs in features
    df_feat = df_feat.fillna(0)
    df_train, df_val, df_test = time_based_split(df_feat)
    return df_feat, df_train, df_val, df_test


def tune_model_family(
    family: str,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    sample_weight_col: str = "RECENCY_WEIGHT",
    n_trials: int = 40,
) -> optuna.study.Study:
    """
    Tune one model family (xgb, lgbm, cat) for a given target/feature set.
    Returns the Optuna Study (best_params and best_value available).
    """
    # Use DataFrames for X so all model families see consistent feature names,
    # which also avoids "X does not have valid feature names" warnings.
    X_train = df_train[feature_cols]
    y_train = df_train[target_col].values
    w_train = df_train[sample_weight_col].values

    X_val = df_val[feature_cols]
    y_val = df_val[target_col].values
    w_val = df_val[sample_weight_col].values

    def objective(trial: optuna.Trial) -> float:
        if family == "xgb":
            # Tighter, more regularized search space for XGBoost
            params: Dict = {
                "objective": "reg:squarederror",
                "eval_metric": "rmse",
                "tree_method": "hist",
                "random_state": RANDOM_STATE,
                "max_depth": trial.suggest_int("max_depth", 3, 6),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 1200, 2500),
                "min_child_weight": trial.suggest_float("min_child_weight", 10.0, 30.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            }
            model = XGBRegressor(**params)
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                sample_weight_eval_set=[w_val],
                verbose=False,
            )

        elif family == "lgbm":
            # Tighter, more regularized search space for LightGBM
            params = {
                "objective": "regression",
                "metric": "rmse",
                "random_state": RANDOM_STATE,
                "num_leaves": trial.suggest_int("num_leaves", 31, 128),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 1200, 2500),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "verbosity": -1,
            }
            model = LGBMRegressor(**params)
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                eval_sample_weight=[w_val],
            )

        elif family == "cat":
            # Tighter, more regularized search space for CatBoost
            params = {
                "loss_function": "RMSE",
                "eval_metric": "RMSE",
                "random_seed": RANDOM_STATE,
                "depth": trial.suggest_int("depth", 4, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 1200, 2500),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 10.0),
                "subsample": trial.suggest_float("subsample", 0.6, 0.9),
                "logging_level": "Silent",
                "allow_writing_files": False,
            }
            model = CatBoostRegressor(**params)
            model.fit(
                X_train,
                y_train,
                sample_weight=w_train,
                eval_set=(X_val, y_val),
                verbose=False,
            )
        else:
            raise ValueError(f"Unknown model family: {family}")

        y_pred = model.predict(X_val)
        rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
        return rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


def train_final_model_family(
    family: str,
    df_trainval: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    best_params: Dict,
    sample_weight_col: str = "RECENCY_WEIGHT",
):
    """
    Train a final model of given family on train+val with best hyperparameters.
    """
    X_train = df_trainval[feature_cols]
    y_train = df_trainval[target_col].values
    w_train = df_trainval[sample_weight_col].values

    if family == "xgb":
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "random_state": RANDOM_STATE,
        }
        params.update(best_params)
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        return model

    if family == "lgbm":
        params = {
            "objective": "regression",
            "metric": "rmse",
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }
        params.update(best_params)
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train)
        return model

    if family == "cat":
        params = {
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": RANDOM_STATE,
            "logging_level": "Silent",
            "allow_writing_files": False,
        }
        params.update(best_params)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
        return model

    raise ValueError(f"Unknown model family: {family}")


def evaluate_rmse_mae(
    model,
    df_test: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[float, float]:
    X_test = df_test[feature_cols]
    y_test = df_test[target_col].values
    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(np.mean((y_pred - y_test) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_test)))
    return rmse, mae


def main() -> None:
    print("Preparing data for ensemble training...")
    df_feat, df_train, df_val, df_test = prepare_data()
    df_trainval = pd.concat([df_train, df_val], ignore_index=True)

    minutes_features = get_minutes_feature_cols(df_feat)
    stat_features = get_stat_feature_cols(df_feat)

    # -------------------------
    # Minutes model ensemble
    # -------------------------
    print("\n=== Tuning ensemble for MINUTES ===")
    family_names = ["xgb", "lgbm", "cat"]
    best_params_minutes: Dict[str, Dict] = {}
    best_vals_minutes: Dict[str, float] = {}

    for fam in family_names:
        print(f"\nTuning {fam.upper()} for MIN_NUM (minutes)...")
        study = tune_model_family(
            fam,
            df_train,
            df_val,
            feature_cols=minutes_features,
            target_col="MIN_NUM",
        )
        best_params_minutes[fam] = study.best_params
        best_vals_minutes[fam] = study.best_value
        print(f"  Best {fam.upper()} val RMSE: {study.best_value:.4f}")

    # Train final models on train+val
    minutes_models = []
    minutes_weights = []
    for fam in family_names:
        model = train_final_model_family(
            fam,
            df_trainval,
            feature_cols=minutes_features,
            target_col="MIN_NUM",
            best_params=best_params_minutes[fam],
        )
        minutes_models.append(model)
        # Use inverse squared val RMSE as weight; guard against zero
        val_rmse = max(best_vals_minutes[fam], 1e-6)
        minutes_weights.append(1.0 / (val_rmse ** 2))

    minutes_ensemble = EnsembleRegressor(minutes_models, weights=minutes_weights)
    rmse_min, mae_min = evaluate_rmse_mae(
        minutes_ensemble,
        df_test,
        feature_cols=minutes_features,
        target_col="MIN_NUM",
    )
    print(f"\nMinutes ensemble - Test RMSE: {rmse_min:.3f}, MAE: {mae_min:.3f}")
    models_dir = Path(__file__).resolve().parent / "models"
    models_dir.mkdir(exist_ok=True)
    dump(minutes_ensemble, models_dir / "minutes_model_ensemble.joblib")

    # -------------------------
    # Stat models ensemble
    # -------------------------
    tuned_stat_ensembles: Dict[str, EnsembleRegressor] = {}

    for stat in TARGET_STATS:
        print(f"\n=== Tuning ensemble for {stat} ===")
        best_params: Dict[str, Dict] = {}
        best_vals: Dict[str, float] = {}

        for fam in family_names:
            print(f"\nTuning {fam.upper()} for {stat} ...")
            study = tune_model_family(
                fam,
                df_train,
                df_val,
                feature_cols=stat_features,
                target_col=stat,
            )
            best_params[fam] = study.best_params
            best_vals[fam] = study.best_value
            print(f"  Best {fam.upper()} val RMSE for {stat}: {study.best_value:.4f}")

        stat_models = []
        stat_weights = []
        for fam in family_names:
            model = train_final_model_family(
                fam,
                df_trainval,
                feature_cols=stat_features,
                target_col=stat,
                best_params=best_params[fam],
            )
            stat_models.append(model)
            val_rmse = max(best_vals[fam], 1e-6)
            stat_weights.append(1.0 / (val_rmse ** 2))

        ensemble = EnsembleRegressor(stat_models, weights=stat_weights)
        rmse, mae = evaluate_rmse_mae(
            ensemble,
            df_test,
            feature_cols=stat_features,
            target_col=stat,
        )
        print(f"{stat} ensemble - Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")

        tuned_stat_ensembles[stat] = ensemble
        dump(ensemble, models_dir / f"stat_model_{stat.lower()}_ensemble.joblib")

    print("\nEnsemble training complete. Ensemble models saved as *_ensemble.joblib")


if __name__ == "__main__":
    main()
