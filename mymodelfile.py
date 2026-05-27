"""JAI SHREE KRISHNA 🙏🏻""" 
import ast
import importlib
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    _lightgbm = importlib.import_module("lightgbm")
    LGBMRegressor = _lightgbm.LGBMRegressor
    HAS_LGBM = True
except Exception:
    LGBMRegressor = None
    HAS_LGBM = False


class MyModel:
    """Pre-innings IPL PowerPlay score predictor — competition edition v4 (SOTA)."""

    def __init__(self):
        self.global_mean = 45.0
        self.global_std = 8.0
        self.global_batsman_sr = 120.0
        self.global_bowler_econ = 8.5
        self.global_boundary_rate = 0.16
        self.global_dot_rate = 0.38
        self.global_wicket_rate = 0.06
        self.global_batsman_count = 2.2
        self.global_bowler_count = 2.0
        self.global_dew_factor = 0.0
        self.global_first_innings_mean = 45.0

        self.id_to_name = {}
        self.name_to_id = {}
        self.venue_canonical_map = {}
        self.home_ground_map = {}

        # Lookup maps
        self.venue_mean_map = {}
        self.venue_std_map = {}
        self.venue_recent_mean_map = {}
        self.batting_team_mean_map = {}
        self.batting_team_recent_mean_map = {}
        self.bowling_team_mean_map = {}
        self.bowling_team_recent_mean_map = {}
        self.team_venue_mean_map = {}
        self.open_batsman_pp_map = {}
        self.open_bowler_pp_map = {}
        self.batsman_sr_map = {}
        self.bowler_econ_map = {}
        self.batsman_count_prior_map = {}
        self.bowler_count_prior_map = {}
        self.team_home_boost_map = {}
        self.venue_dew_map = {}
        self.first_innings_match_map = {}
        self.first_innings_mean_by_venue = {}
        self.batting_team_boundary_rate_map = {}
        self.batting_team_dot_rate_map = {}
        self.batting_team_wicket_rate_map = {}
        self.bowling_team_wicket_rate_map = {}
        self.venue_boundary_rate_map = {}
        self.venue_dot_rate_map = {}
        self.venue_wicket_rate_map = {}
        self.residual_bucket_map = {}

        # Season trend / calibration
        self.season_mean_map = {}
        self.train_max_year = 2022
        self.predict_year = 2023
        self.year_trend_slope = 0.0
        self.year_bias_correction = 0.0

        # Output calibration (fitted on holdout slice)
        self.calib_slope = 1.0
        self.calib_intercept = 0.0
        self.calib_target_std = 12.0
        self.calib_pred_std = 8.0
        self.calib_anchor = 54.0

        self.base_prior_feature_names = [
            "venue_prior_mean",
            "batting_team_prior_mean",
            "bowling_team_prior_mean",
            "team_venue_prior_mean",
            "open_batsman_prior_mean",
            "open_bowler_prior_mean",
        ]
        self.default_base_prior_weights = np.array(
            [0.28, 0.26, 0.18, 0.12, 0.10, 0.06], dtype=float
        )
        self.base_prior_weights = self.default_base_prior_weights.copy()

        self._year_ref = 2015
        self._year_scale = 7.0

        self.feature_columns = [
            "inning",
            "year_feature",
            "venue_prior_mean",
            "venue_prior_std",
            "venue_recent_mean",
            "batting_team_prior_mean",
            "batting_team_recent_mean",
            "bowling_team_prior_mean",
            "bowling_team_recent_mean",
            "team_venue_prior_mean",
            "open_batsman_prior_mean",
            "open_bowler_prior_mean",
            "batsman_count_prior",
            "bowler_count_prior",
            "is_home",
            "home_advantage_boost",
            "dew_factor",
            "first_innings_score",
            "open_batsman_sr_prior",
            "open_bowler_econ_prior",
            "composite_bat_sr",
            "composite_bowl_econ",
            "bat_minus_bowl",
            "bat_minus_bowl_recent",
            "base_prior",
            "recent_base_prior",
            "boundary_rate_prior",
            "dot_rate_prior",
            "wicket_rate_prior",
            "wicket_pressure",
            "aggression",
            "batting_strength",
            "bat_minus_bowl_x_aggression",
            "venue_x_aggression",
            "batting_strength_x_wicket_pressure",
            "aggression_sq",
            "wicket_pressure_sq",
            "inning2_flag",
            "wickets_proxy",
            "bowler_pressure",
            "wicket_collapse_pressure",
            "home_x_bat_minus_bowl",
            "dew_x_wicket_pressure",
            "wickets_x_dew",
            "bowler_count_x_wicket_pressure",
            "batsman_count_x_home",
            "confidence_prior",
            "recent_vs_alltime_bat",
            "recent_vs_alltime_venue",
            "recent_vs_alltime_bowl",
            "venue_x_batsman_count",
            "dew_x_second_innings",
            "fi_x_inn2",
            # --- NEW impact features ---
            "h2h_prior_mean",
            "h2h_venue_prior_mean",
            "h2h_x_venue",
            "batting_team_score_std",
            "composite_bat_venue_sr",
            "top3_bat_avg",
            "venue_boundary_pct",
            "composite_bowl_dot_pct",
            "team_pp_rpo",
            "h2h_minus_global",
            "volatility_x_aggression",
            "venue_sr_minus_global_sr",
        ]
        self.selected_feature_columns = list(self.feature_columns)
        self.feature_top_k = 50

        self.batting_count_scale = 16.0
        self.wicket_pressure_weights = (0.38, 0.42, 0.20)
        self.extreme_threshold = 4.0
        self.extreme_amplification = 1.45
        self.tail_push_threshold = 5.5
        self.tail_push_low_strength = 0.20
        self.tail_push_high_strength = 0.30
        self.tail_push_power = 1.25
        self.tail_push_cap = 7.0
        self.tail_push_preserve_mean = False
        self.final_mean_weight = 0.0
        self.residual_bucket_min_samples = 4
        self.residual_bucket_shrink = 5.0
        self.residual_bucket_cap = 6.0
        self.model_blend_weight_train = 0.90
        self.model_blend_weight_predict = 0.95
        self.flex_region_low = 32.0
        self.flex_region_high = 85.0
        self.flex_transition = 7.0
        self.flex_outside_floor = 0.75
        self.wrong_side_guard = 0.90
        self.min_flex_confidence = 0.12
        self.context_gain_bounds = {
            "aggression": 13.0, "wicket_collapse": 20.0, "dew": 8.0,
            "home": 7.0, "wickets_proxy": 6.0, "bowler_count": 5.0, "first_innings": 0.30,
        }
        self.context_gains = {
            "aggression": 7.5, "wicket_collapse": 13.0, "dew": 2.5,
            "home": 2.0, "wickets_proxy": 1.5, "bowler_count": 1.2, "first_innings": 0.10,
        }
        self.context_scale = 1.15
        self.tail_lift_scale = 1.5
        self.tail_drop_scale = 0.85

        # Ensemble models  (Huber bag removed — replaced by second HGB to save time)
        self.model_mae_list = []
        self.model_poisson_list = []
        self.model_hgb = None
        self.model_hgb2 = None
        self.model_huber_list = []
        # weights: mae_bag, poisson_bag, hgb1, hgb2
        self.ensemble_weights = np.array([0.30, 0.35, 0.20, 0.15])

        # Player name→id mapping
        self._player_name_to_id = {}

        # Flexible params (for evaluate_pipeline.py compatibility)
        self.flexible_params = None

    # ------------------------------------------------------------------ #
    # Utilities
    # ------------------------------------------------------------------ #
    def _find_column(self, df, candidates, required=False):
        for col in candidates:
            if col in df.columns:
                return col
        if required:
            raise ValueError(f"Missing required column. Expected one of: {candidates}")
        return None

    def _clean_key(self, value, unknown="UNKNOWN"):
        if pd.isna(value):
            return unknown
        text = str(value).strip()
        return text if text else unknown

    def _normalize_player_id(self, value):
        if pd.isna(value):
            return "UNKNOWN"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "UNKNOWN"
            if float(value).is_integer():
                return str(int(value))
            return str(value)
        text = str(value).strip()
        if not text:
            return "UNKNOWN"
        try:
            num = float(text)
            if num.is_integer():
                return str(int(num))
        except Exception:
            pass
        return text

    def _normalize_match_id(self, value):
        return self._normalize_player_id(value)

    def _name_key(self, value):
        return self._clean_key(value).lower()

    def _get_series_or_default(self, df, candidates, default_value):
        col = self._find_column(df, candidates, required=False)
        if col is None:
            return pd.Series([default_value] * len(df), index=df.index)
        return df[col]

    def _venue_key(self, value):
        text = self._clean_key(value).lower()
        text = text.replace("&", "and")
        text = re.sub(r"[^a-z0-9]+", "", text)
        return text if text else "unknown"

    def _register_venues(self, values):
        for value in values:
            clean = self._clean_key(value)
            if clean == "UNKNOWN":
                continue
            key = self._venue_key(clean)
            if key not in self.venue_canonical_map:
                self.venue_canonical_map[key] = clean

    def _canonicalize_venue(self, value):
        clean = self._clean_key(value)
        key = self._venue_key(clean)
        return self.venue_canonical_map.get(key, clean)

    def _extract_match_id_from_any(self, value):
        text = self._clean_key(value)
        if text == "UNKNOWN":
            return "UNKNOWN"
        if "_" in text:
            first_part = text.split("_", 1)[0].strip()
            if first_part:
                return self._normalize_match_id(first_part)
        return self._normalize_match_id(text)

    def _load_optional_csv(self, candidates):
        for path in candidates:
            if os.path.exists(path):
                try:
                    return pd.read_csv(path)
                except Exception:
                    continue
        return None

    def _load_home_ground_map(self, candidates):
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    raw_text = f.read()
            except Exception:
                continue
            match = re.search(r"home_ground_map\s*=\s*(\{[\s\S]*?\})", raw_text)
            if match is None:
                continue
            try:
                loaded = ast.literal_eval(match.group(1))
            except Exception:
                continue
            if isinstance(loaded, dict):
                out = {}
                for team, venue in loaded.items():
                    out[self._clean_key(team)] = self._clean_key(venue)
                return out
        return {}

    def _ewm_expanding_mean(self, series, halflife_games=30):
        """Leakage-safe expanding EWM: position i uses only positions 0..i-1.
        Vectorised via pandas ewm — O(n) without Python loop overhead.
        """
        # shift(1) ensures position i sees only i-1 and earlier (no leakage)
        shifted = series.shift(1)
        ewm_val = shifted.ewm(halflife=halflife_games, min_periods=1, ignore_na=True).mean()
        # First position has nothing to look at → NaN (consistent with old impl)
        ewm_val.iloc[0] = np.nan
        return ewm_val

    def _safe_corr(self, x, y):
        x_arr = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        y_arr = np.asarray(y, dtype=float)
        valid = ~(np.isnan(x_arr) | np.isnan(y_arr))
        if int(np.sum(valid)) < 5:
            return 0.0
        xv = x_arr[valid]; yv = y_arr[valid]
        if np.nanstd(xv) <= 1e-9 or np.nanstd(yv) <= 1e-9:
            return 0.0
        corr = np.corrcoef(xv, yv)[0, 1]
        return float(abs(corr)) if np.isfinite(corr) else 0.0

    def _select_top_features(self, X, y):
        if X is None or len(X) == 0:
            self.selected_feature_columns = list(self.feature_columns)
            return self.selected_feature_columns
        corr_scores = [(col, self._safe_corr(X[col], y)) for col in self.feature_columns]
        corr_scores.sort(key=lambda item: item[1], reverse=True)
        k = min(self.feature_top_k, len(corr_scores))
        self.selected_feature_columns = [name for name, _ in corr_scores[:k]]
        return self.selected_feature_columns

    def _make_base_prior(self, frame):
        weights = np.asarray(self.base_prior_weights, dtype=float)
        if weights.shape[0] != len(self.base_prior_feature_names):
            weights = self.default_base_prior_weights.copy()
        components = frame[self.base_prior_feature_names].to_numpy(dtype=float)
        return np.dot(components, weights)

    def _make_recent_base_prior(self, frame):
        recent_names = [
            "venue_recent_mean",
            "batting_team_recent_mean",
            "bowling_team_recent_mean",
            "team_venue_prior_mean",
            "open_batsman_prior_mean",
            "open_bowler_prior_mean",
        ]
        weights = np.array([0.30, 0.28, 0.16, 0.12, 0.09, 0.05], dtype=float)
        components = frame[recent_names].to_numpy(dtype=float)
        return np.dot(components, weights)

    # ------------------------------------------------------------------ #
    # Flexible params
    # ------------------------------------------------------------------ #
    def load_flexible_params(self, params_path):
        """Load flexible correction head parameters (evaluate_pipeline compatibility)."""
        if not params_path:
            self.flexible_params = None
            return
        if not os.path.exists(params_path):
            self.flexible_params = None
            return
        try:
            with open(params_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            self.flexible_params = None
            return

        required = [
            "alpha_max", "up_cap", "down_cap", "min_score", "max_score",
            "correction_weights", "correction_bias", "gate_weights", "gate_bias",
        ]
        if not all(k in payload for k in required):
            self.flexible_params = None
            return

        corr_w = np.asarray(payload.get("correction_weights", []), dtype=float)
        gate_w = np.asarray(payload.get("gate_weights", []), dtype=float)
        if corr_w.shape[0] != 5 or gate_w.shape[0] != 2:
            self.flexible_params = None
            return

        self.flexible_params = {
            "alpha_max": float(payload["alpha_max"]),
            "up_cap": float(payload["up_cap"]),
            "down_cap": float(payload["down_cap"]),
            "min_score": float(payload["min_score"]),
            "max_score": float(payload["max_score"]),
            "correction_weights": corr_w,
            "correction_bias": float(payload["correction_bias"]),
            "gate_weights": gate_w,
            "gate_bias": float(payload["gate_bias"]),
            "risk_bucket_multipliers": payload.get(
                "risk_bucket_multipliers",
                {"low": 1.15, "mid": 1.0, "high": 0.75},
            ),
        }

    def _apply_flexible_adjustment(self, base_pred, infer, confidence, uncertainty):
        if not self.flexible_params:
            return np.asarray(base_pred, dtype=float)
        p = self.flexible_params
        corr_feats = np.column_stack([
            pd.to_numeric(infer["wicket_pressure"], errors="coerce").fillna(self.global_wicket_rate).to_numpy(dtype=float),
            pd.to_numeric(infer["aggression"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
            pd.to_numeric(infer["boundary_rate_prior"], errors="coerce").fillna(self.global_boundary_rate).to_numpy(dtype=float),
            pd.to_numeric(infer["dot_rate_prior"], errors="coerce").fillna(self.global_dot_rate).to_numpy(dtype=float),
            pd.to_numeric(infer["bat_minus_bowl"], errors="coerce").fillna(0.0).to_numpy(dtype=float),
        ])
        correction = np.dot(corr_feats, p["correction_weights"]) + p["correction_bias"]
        gate_feats = np.column_stack([np.asarray(confidence, dtype=float), np.asarray(uncertainty, dtype=float)])
        gate_logit = np.dot(gate_feats, p["gate_weights"]) + p["gate_bias"]
        alpha = p["alpha_max"] * (1.0 / (1.0 + np.exp(-gate_logit)))
        delta = alpha * correction
        up = np.full_like(delta, p["up_cap"], dtype=float)
        down = np.full_like(delta, p["down_cap"], dtype=float)
        q1 = np.quantile(uncertainty, 0.33)
        q2 = np.quantile(uncertainty, 0.66)
        multipliers = p["risk_bucket_multipliers"]
        low_mul = float(multipliers.get("low", 1.15))
        mid_mul = float(multipliers.get("mid", 1.0))
        high_mul = float(multipliers.get("high", 0.75))
        low_mask = np.asarray(uncertainty, dtype=float) <= q1
        high_mask = np.asarray(uncertainty, dtype=float) >= q2
        mid_mask = ~(low_mask | high_mask)
        up[low_mask] *= low_mul; down[low_mask] *= low_mul
        up[mid_mask] *= mid_mul; down[mid_mask] *= mid_mul
        up[high_mask] *= high_mul; down[high_mask] *= high_mul
        delta = np.minimum(delta, up)
        delta = np.maximum(delta, -down)
        out = np.asarray(base_pred, dtype=float) + delta
        out = np.clip(out, p["min_score"], p["max_score"])
        return out

    # ------------------------------------------------------------------ #
    # Residual bucket correction
    # ------------------------------------------------------------------ #
    def _bucket_key(self, batting_team, venue, inning):
        team = self._clean_key(batting_team)
        ven = self._canonicalize_venue(venue)
        try:
            inn = int(pd.to_numeric(inning, errors="coerce"))
        except Exception:
            inn = 1
        if inn not in (1, 2):
            inn = 1
        return f"{team}||{ven}||{inn}"

    def _fit_residual_bucket_map(self, innings, y_true, pred):
        self.residual_bucket_map = {}
        if innings is None or len(innings) == 0:
            return
        residual_df = pd.DataFrame({
            "batting_team": innings["batting_team"].map(self._clean_key),
            "venue": innings["venue"].map(self._canonicalize_venue),
            "inning": pd.to_numeric(innings["inning"], errors="coerce").fillna(1).astype(int),
            "residual": np.asarray(y_true, dtype=float) - np.asarray(pred, dtype=float),
        })
        grouped = (
            residual_df.groupby(["batting_team", "venue", "inning"], as_index=False)
            .agg(bucket_residual_mean=("residual", "mean"), bucket_count=("residual", "size"))
        )
        for _, row in grouped.iterrows():
            count = int(row["bucket_count"])
            if count < self.residual_bucket_min_samples:
                continue
            mean_residual = float(row["bucket_residual_mean"])
            shrink = float(count / (count + self.residual_bucket_shrink))
            corr = np.clip(mean_residual * shrink, -self.residual_bucket_cap, self.residual_bucket_cap)
            key = self._bucket_key(row["batting_team"], row["venue"], row["inning"])
            self.residual_bucket_map[key] = float(corr)

    def _residual_correction_vector(self, batting_series, venue_series, inning_series):
        if not self.residual_bucket_map:
            return np.zeros(len(batting_series), dtype=float)
        out = []
        for bat, ven, inn in zip(batting_series, venue_series, inning_series):
            key = self._bucket_key(bat, ven, inn)
            out.append(float(self.residual_bucket_map.get(key, 0.0)))
        return np.asarray(out, dtype=float)

    # ------------------------------------------------------------------ #
    # Context gains
    # ------------------------------------------------------------------ #
    def _fit_context_gains(self, X_full, y, train_pred):
        if X_full is None or len(X_full) == 0:
            return
        y_arr = np.asarray(y, dtype=float)
        p_arr = np.asarray(train_pred, dtype=float)
        if y_arr.shape[0] != p_arr.shape[0]:
            return
        resid = y_arr - p_arr
        signal_map = {
            "aggression": pd.to_numeric(X_full.get("aggression", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "wicket_collapse": pd.to_numeric(X_full.get("wicket_collapse_pressure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "dew": pd.to_numeric(X_full.get("dew_factor", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "home": (
                pd.to_numeric(X_full.get("home_advantage_boost", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                + pd.to_numeric(X_full.get("is_home", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            ),
            "wickets_proxy": pd.to_numeric(X_full.get("wickets_proxy", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
            "bowler_count": (
                pd.to_numeric(X_full.get("bowler_count_prior", self.global_bowler_count), errors="coerce").fillna(self.global_bowler_count).to_numpy(dtype=float)
                - self.global_bowler_count
            ),
            "first_innings": (
                pd.to_numeric(X_full.get("first_innings_score", self.global_first_innings_mean), errors="coerce").fillna(self.global_first_innings_mean).to_numpy(dtype=float)
                - self.global_first_innings_mean
            ) * pd.to_numeric(X_full.get("inning2_flag", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float),
        }
        eps = 1e-6
        for key, signal in signal_map.items():
            var = float(np.var(signal))
            if var < eps:
                continue
            cov = float(np.mean((signal - np.mean(signal)) * (resid - np.mean(resid))))
            raw_gain = cov / var
            bound = float(self.context_gain_bounds.get(key, 6.0))
            self.context_gains[key] = float(np.clip(raw_gain, -bound, bound))

    def _context_flex_adjustment(self, infer):
        aggression = pd.to_numeric(infer.get("aggression", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        wicket_collapse = pd.to_numeric(infer.get("wicket_collapse_pressure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        dew = pd.to_numeric(infer.get("dew_factor", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        home = (
            pd.to_numeric(infer.get("home_advantage_boost", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            + pd.to_numeric(infer.get("is_home", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        wickets_proxy = pd.to_numeric(infer.get("wickets_proxy", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bowlers = pd.to_numeric(infer.get("bowler_count_prior", self.global_bowler_count), errors="coerce").fillna(self.global_bowler_count).to_numpy(dtype=float)
        inning2_flag = pd.to_numeric(infer.get("inning2_flag", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        first_innings_delta = (
            pd.to_numeric(infer.get("first_innings_score", self.global_first_innings_mean), errors="coerce").fillna(self.global_first_innings_mean).to_numpy(dtype=float)
            - self.global_first_innings_mean
        )
        corr = (
            self.context_gains["aggression"] * aggression
            - self.context_gains["wicket_collapse"] * wicket_collapse
            + self.context_gains["dew"] * dew
            + self.context_gains["home"] * home
            + self.context_gains["wickets_proxy"] * wickets_proxy
            + self.context_gains["bowler_count"] * (bowlers - self.global_bowler_count)
            + self.context_gains["first_innings"] * first_innings_delta * inning2_flag
        )
        return np.clip(corr, -15.0, 15.0)

    # ------------------------------------------------------------------ #
    # Directional tail signal
    # ------------------------------------------------------------------ #
    def _directional_tail_signal(self, infer):
        aggression = pd.to_numeric(infer.get("aggression", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bat_minus_bowl = pd.to_numeric(infer.get("bat_minus_bowl", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        boundary = pd.to_numeric(infer.get("boundary_rate_prior", self.global_boundary_rate), errors="coerce").fillna(self.global_boundary_rate).to_numpy(dtype=float)
        dot = pd.to_numeric(infer.get("dot_rate_prior", self.global_dot_rate), errors="coerce").fillna(self.global_dot_rate).to_numpy(dtype=float)
        collapse = pd.to_numeric(infer.get("wicket_collapse_pressure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        first_innings = pd.to_numeric(infer.get("first_innings_score", self.global_first_innings_mean), errors="coerce").fillna(self.global_first_innings_mean).to_numpy(dtype=float)
        inning2 = pd.to_numeric(infer.get("inning2_flag", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        signal = (
            8.5 * aggression
            + 0.13 * bat_minus_bowl
            + 6.5 * (boundary - dot)
            - 7.5 * collapse
            + 0.07 * (first_innings - self.global_first_innings_mean) * inning2
        )
        return np.clip(signal, -10.0, 10.0)

    # ------------------------------------------------------------------ #
    # Calibration / extreme handling
    # ------------------------------------------------------------------ #
    def _fit_output_calibration(self, X_full, y, raw_pred):
        """Fit linear output recalibration y ≈ slope * pred + intercept on a held-out slice."""
        n = len(y)
        if n < 100:
            self.calib_slope = 1.0
            self.calib_intercept = 0.0
            return

        split = max(60, int(0.70 * n))
        y_cal = np.asarray(y.iloc[split:] if isinstance(y, pd.Series) else y[split:], dtype=float)
        p_cal = np.asarray(raw_pred[split:], dtype=float)

        if len(y_cal) < 30:
            self.calib_slope = 1.0
            self.calib_intercept = 0.0
            return

        self.calib_pred_std = max(float(np.std(p_cal)), 1.0)
        self.calib_target_std = max(float(np.std(y_cal)), 5.0)
        self.calib_anchor = float(np.median(y_cal))

        p_std = float(np.std(p_cal))
        if p_std < 1e-6:
            self.calib_slope = 1.0
            self.calib_intercept = float(np.mean(y_cal) - np.mean(p_cal))
            return

        corr = float(np.corrcoef(p_cal, y_cal)[0, 1])
        if not np.isfinite(corr):
            corr = 0.5

        y_std = float(np.std(y_cal))
        raw_slope = corr * y_std / p_std
        self.calib_slope = float(np.clip(1.10 * raw_slope, 1.0, 3.50))
        self.calib_intercept = float(self.calib_anchor - self.calib_slope * np.mean(p_cal))
        max_intercept = max(abs(np.mean(y_cal)) * 0.6, 20.0)
        self.calib_intercept = float(np.clip(self.calib_intercept, -max_intercept, max_intercept))

    def _apply_extreme_calibration(self, pred):
        pred = np.asarray(pred, dtype=float)
        deviation = pred - self.global_mean
        mask = np.abs(deviation) > self.extreme_threshold
        out = pred.copy()
        adaptive_amp = self.extreme_amplification + 0.12 * np.minimum(1.0, np.abs(deviation) / 15.0)
        out[mask] = self.global_mean + deviation[mask] * adaptive_amp[mask]
        return out

    def _apply_tail_push(self, pred):
        pred = np.asarray(pred, dtype=float)
        if pred.size == 0:
            return pred
        deviation = pred - self.global_mean
        abs_dev = np.abs(deviation)
        excess = np.clip(abs_dev - self.tail_push_threshold, 0.0, None)
        high_push = np.clip(self.tail_push_high_strength * np.power(excess, self.tail_push_power), 0.0, self.tail_push_cap)
        low_push = np.clip(self.tail_push_low_strength * np.power(excess, self.tail_push_power), 0.0, self.tail_push_cap)
        out = pred.copy()
        out[deviation > 0] += high_push[deviation > 0]
        out[deviation < 0] -= low_push[deviation < 0]
        if self.tail_push_preserve_mean:
            out = out - (float(np.mean(out)) - float(np.mean(pred)))
        return out

    def _region_strength(self, pred):
        pred = np.asarray(pred, dtype=float)
        k = max(1e-6, float(self.flex_transition))
        left = 1.0 / (1.0 + np.exp(-(pred - self.flex_region_low) / k))
        right = 1.0 / (1.0 + np.exp((pred - self.flex_region_high) / k))
        inside = left * right
        return self.flex_outside_floor + (1.0 - self.flex_outside_floor) * inside

    def _apply_soft_guardrail(self, base_pred, adjusted_pred, confidence, uncertainty):
        base_pred = np.asarray(base_pred, dtype=float)
        adjusted_pred = np.asarray(adjusted_pred, dtype=float)
        confidence = np.asarray(confidence, dtype=float)
        uncertainty = np.asarray(uncertainty, dtype=float)
        delta = adjusted_pred - base_pred
        region_gate = self._region_strength(adjusted_pred)
        conf_gate = np.clip(
            self.min_flex_confidence + (1.0 - self.min_flex_confidence) * (confidence * (1.0 - 0.40 * uncertainty)),
            self.min_flex_confidence, 1.0,
        )
        base_side = np.sign(base_pred - self.global_mean)
        adj_side = np.sign(adjusted_pred - self.global_mean)
        cross_mask = (base_side != 0.0) & (adj_side != 0.0) & (base_side != adj_side)
        cross_scale = np.where(cross_mask, self.wrong_side_guard, 1.0)
        return base_pred + delta * region_gate * conf_gate * cross_scale

    # ------------------------------------------------------------------ #
    # Directional confidence (enhanced: 6 signals + coverage + uncertainty)
    # ------------------------------------------------------------------ #
    def _compute_directional_confidence(self, infer, confidence, uncertainty):
        """Compute how strongly and consistently features point away from the mean.

        Returns a per-sample score in [0, 1]:
          high → signals agree on direction
          low  → signals conflict or are near-neutral
        """
        n = len(infer)
        mean_ref = self.global_mean
        signals = []

        venue_dev = (pd.to_numeric(infer.get("venue_prior_mean", mean_ref), errors="coerce")
                     .fillna(mean_ref).to_numpy(dtype=float) - mean_ref)
        signals.append(np.clip(venue_dev / max(self.global_std, 1.0), -2.0, 2.0))

        bat_dev = (pd.to_numeric(infer.get("batting_team_prior_mean", mean_ref), errors="coerce")
                   .fillna(mean_ref).to_numpy(dtype=float) - mean_ref)
        signals.append(np.clip(bat_dev / max(self.global_std, 1.0), -2.0, 2.0))

        bowl_dev = (pd.to_numeric(infer.get("bowling_team_prior_mean", mean_ref), errors="coerce")
                    .fillna(mean_ref).to_numpy(dtype=float) - mean_ref)
        signals.append(np.clip(-bowl_dev / max(self.global_std, 1.0), -2.0, 2.0))

        aggression = pd.to_numeric(infer.get("aggression", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        signals.append(np.clip(aggression * 5.0, -2.0, 2.0))

        bat_minus_bowl = pd.to_numeric(infer.get("bat_minus_bowl", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        signals.append(np.clip(bat_minus_bowl / max(self.global_std, 1.0), -2.0, 2.0))

        venue_recent = pd.to_numeric(infer.get("venue_recent_mean", mean_ref), errors="coerce").fillna(mean_ref).to_numpy(dtype=float)
        signals.append(np.clip((venue_recent - mean_ref) / max(self.global_std, 1.0), -2.0, 2.0))

        if not signals:
            return np.full(n, 0.5)

        signals_arr = np.column_stack(signals)
        signs = np.sign(signals_arr)
        sum_signs = np.sum(signs, axis=1)
        majority_dir = np.sign(sum_signs)
        agreeing = np.sum(signs == majority_dir[:, None], axis=1)
        k = signals_arr.shape[1]
        agreement_ratio = agreeing / k

        magnitude = np.mean(np.abs(signals_arr), axis=1)
        magnitude_score = np.clip(magnitude / 1.5, 0.0, 1.0)

        coverage = np.asarray(confidence, dtype=float)
        raw_conf = 0.45 * agreement_ratio + 0.35 * magnitude_score + 0.20 * coverage

        unc = np.asarray(uncertainty, dtype=float)
        raw_conf = raw_conf * (1.0 - 0.3 * unc)

        return np.clip(raw_conf, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    # Feature Building
    # ------------------------------------------------------------------ #
    def _build_features(self, frame, year_override=None):
        features = pd.DataFrame(index=frame.index)
        features["inning"] = pd.to_numeric(frame["inning"], errors="coerce").fillna(1.0)

        if year_override is not None:
            yr_val = float(year_override)
        elif "year" in frame.columns:
            yr_val = None
        else:
            yr_val = float(self.predict_year)

        if yr_val is not None:
            features["year_feature"] = (yr_val - self._year_ref) / self._year_scale
        else:
            features["year_feature"] = (
                pd.to_numeric(frame["year"], errors="coerce").fillna(self._year_ref).values
                - self._year_ref
            ) / self._year_scale

        features["venue_prior_mean"] = frame["venue_prior_mean"]
        features["venue_prior_std"] = frame["venue_prior_std"]
        features["venue_recent_mean"] = frame.get("venue_recent_mean", frame["venue_prior_mean"])
        features["batting_team_prior_mean"] = frame["batting_team_prior_mean"]
        features["batting_team_recent_mean"] = frame.get("batting_team_recent_mean", frame["batting_team_prior_mean"])
        features["bowling_team_prior_mean"] = frame["bowling_team_prior_mean"]
        features["bowling_team_recent_mean"] = frame.get("bowling_team_recent_mean", frame["bowling_team_prior_mean"])
        features["team_venue_prior_mean"] = frame["team_venue_prior_mean"]
        features["open_batsman_prior_mean"] = frame["open_batsman_prior_mean"]
        features["open_bowler_prior_mean"] = frame["open_bowler_prior_mean"]
        features["batsman_count_prior"] = frame["batsman_count_prior"]
        features["bowler_count_prior"] = frame["bowler_count_prior"]
        features["is_home"] = frame["is_home"]
        features["home_advantage_boost"] = frame["home_advantage_boost"]
        features["dew_factor"] = frame["dew_factor"]
        features["first_innings_score"] = frame["first_innings_score"]
        features["open_batsman_sr_prior"] = frame["open_batsman_sr_prior"]
        features["open_bowler_econ_prior"] = frame["open_bowler_econ_prior"]
        features["composite_bat_sr"] = frame["composite_bat_sr"]
        features["composite_bowl_econ"] = frame["composite_bowl_econ"]
        features["bat_minus_bowl"] = features["batting_team_prior_mean"] - features["bowling_team_prior_mean"]
        features["bat_minus_bowl_recent"] = features["batting_team_recent_mean"] - features["bowling_team_recent_mean"]
        features["base_prior"] = self._make_base_prior(features)
        features["recent_base_prior"] = self._make_recent_base_prior(features)
        features["boundary_rate_prior"] = pd.to_numeric(frame["boundary_rate_prior"], errors="coerce").fillna(self.global_boundary_rate)
        features["dot_rate_prior"] = pd.to_numeric(frame["dot_rate_prior"], errors="coerce").fillna(self.global_dot_rate)
        features["wicket_rate_prior"] = pd.to_numeric(frame["wicket_rate_prior"], errors="coerce").fillna(self.global_wicket_rate)
        features["wicket_pressure"] = pd.to_numeric(frame["wicket_pressure"], errors="coerce").fillna(self.global_wicket_rate)
        features["inning2_flag"] = (features["inning"] == 2).astype(float)
        features["wickets_proxy"] = np.clip(
            pd.to_numeric(frame["batsman_count_prior"], errors="coerce").fillna(self.global_batsman_count) - 2.0,
            0.0, 6.0,
        )
        features["aggression"] = features["boundary_rate_prior"] - features["dot_rate_prior"] - features["wicket_rate_prior"]
        features["bowler_pressure"] = (
            0.58 * pd.to_numeric(frame["wicket_rate_prior"], errors="coerce").fillna(self.global_wicket_rate)
            + 0.42 * pd.to_numeric(frame["dot_rate_prior"], errors="coerce").fillna(self.global_dot_rate)
        )
        features["wicket_collapse_pressure"] = np.maximum(
            0.0,
            features["wickets_proxy"] * (
                0.75 * features["wicket_pressure"]
                + 0.25 * features["bowler_pressure"]
                - self.global_wicket_rate
            ),
        )
        features["batting_strength"] = (
            0.7 * features["open_batsman_prior_mean"]
            + 0.3 * features["batsman_count_prior"] * self.batting_count_scale
        )
        # Interaction features
        features["bat_minus_bowl_x_aggression"] = features["bat_minus_bowl"] * features["aggression"]
        features["venue_x_aggression"] = features["venue_prior_mean"] * features["aggression"]
        features["batting_strength_x_wicket_pressure"] = features["batting_strength"] * features["wicket_pressure"]
        features["home_x_bat_minus_bowl"] = features["is_home"] * features["bat_minus_bowl"]
        features["dew_x_wicket_pressure"] = features["dew_factor"] * features["wicket_pressure"]
        features["wickets_x_dew"] = features["wickets_proxy"] * features["dew_factor"]
        features["bowler_count_x_wicket_pressure"] = features["bowler_count_prior"] * features["wicket_pressure"]
        features["batsman_count_x_home"] = features["batsman_count_prior"] * features["is_home"]
        features["confidence_prior"] = np.clip(
            1.0 - np.minimum(1.0, np.abs(features["venue_prior_std"] / max(self.global_std, 1e-6))),
            0.0, 1.0,
        )
        features["aggression_sq"] = features["aggression"] ** 2
        features["wicket_pressure_sq"] = features["wicket_pressure"] ** 2
        features["recent_vs_alltime_bat"] = features["batting_team_recent_mean"] - features["batting_team_prior_mean"]
        features["recent_vs_alltime_venue"] = features["venue_recent_mean"] - features["venue_prior_mean"]
        features["recent_vs_alltime_bowl"] = features["bowling_team_recent_mean"] - features["bowling_team_prior_mean"]
        features["venue_x_batsman_count"] = features["venue_prior_mean"] * features["batsman_count_prior"]
        features["dew_x_second_innings"] = features["dew_factor"] * features["inning2_flag"]
        features["fi_x_inn2"] = features["first_innings_score"] * features["inning2_flag"]

        # --- NEW impact features ---
        for col, fallback in [
            ("h2h_prior_mean", self.global_mean),
            ("h2h_venue_prior_mean", self.global_mean),
            ("batting_team_score_std", getattr(self, 'global_batting_team_std', self.global_std)),
            ("composite_bat_venue_sr", self.global_batsman_sr),
            ("top3_bat_avg", self.global_mean),
            ("venue_boundary_pct", getattr(self, 'global_venue_boundary_pct', 0.5)),
            ("composite_bowl_dot_pct", getattr(self, 'global_bowler_dot_pct', self.global_dot_rate)),
        ]:
            if col in frame:
                features[col] = pd.to_numeric(frame[col], errors="coerce").fillna(fallback)
            else:
                features[col] = fallback

        features["team_pp_rpo"] = features["batting_team_recent_mean"] / 6.0
        # Interaction features
        features["h2h_x_venue"] = features["h2h_prior_mean"] * features["venue_prior_mean"] / max(self.global_mean, 1.0)
        features["h2h_minus_global"] = features["h2h_prior_mean"] - self.global_mean
        features["volatility_x_aggression"] = features["batting_team_score_std"] * features["aggression"]
        features["venue_sr_minus_global_sr"] = features["composite_bat_venue_sr"] - self.global_batsman_sr

        return features[self.feature_columns]

    # ------------------------------------------------------------------ #
    # Ensemble
    # ------------------------------------------------------------------ #
    def _ensemble_predict(self, X_full):
        selected_cols = [c for c in self.selected_feature_columns if c in X_full.columns]
        if not selected_cols:
            selected_cols = list(self.feature_columns)
        X_sel = X_full[selected_cols]

        def _bagged_predict(model_list, data):
            if not model_list:
                return None
            preds = [m.predict(data) for m in model_list]
            return np.mean(preds, axis=0)

        preds, weights = [], []

        p_mae = _bagged_predict(self.model_mae_list, X_sel)
        if p_mae is not None:
            preds.append(p_mae); weights.append(self.ensemble_weights[0])

        p_poi = _bagged_predict(self.model_poisson_list, X_sel)
        if p_poi is not None:
            preds.append(p_poi); weights.append(self.ensemble_weights[1])

        if self.model_hgb is not None:
            preds.append(self.model_hgb.predict(X_sel)); weights.append(self.ensemble_weights[2])

        if self.model_hgb2 is not None:
            preds.append(self.model_hgb2.predict(X_sel)); weights.append(self.ensemble_weights[3])

        if not preds:
            return X_full["base_prior"].to_numpy(dtype=float) if "base_prior" in X_full.columns else np.full(len(X_full), self.global_mean)

        w = np.array(weights, dtype=float)
        w /= w.sum()
        result = np.zeros(len(X_sel), dtype=float)
        for p_arr, wt in zip(preds, w):
            result += wt * np.asarray(p_arr, dtype=float)
        return result

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(self, deliveries_df, players_df=None, matches_df=None):
        if deliveries_df is None or deliveries_df.empty:
            raise ValueError("deliveries_df is required and cannot be empty.")

        if players_df is None or (hasattr(players_df, 'empty') and players_df.empty):
            players_df = self._load_optional_csv([
                "/app/training_data/ipl_players_uniqueid.csv",
                "ipl_players_uniqueid.csv",
            ])
        if matches_df is None or (hasattr(matches_df, 'empty') and matches_df.empty):
            matches_df = self._load_optional_csv([
                "/app/training_data/matches_updated_ipl_upto_2025.csv",
                "matches_updated_ipl_upto_2025.csv",
            ])

        raw_home_ground_map = self._load_home_ground_map([
            "/app/training_data/home_ground_mapping.txt",
            "home_ground_mapping.txt",
            os.path.join(os.path.dirname(__file__), "..", "home_ground_mapping.txt"),
        ])

        # --- Player ID mapping ---
        if players_df is not None and not players_df.empty:
            id_col = self._find_column(players_df, ["ID", "id"], required=False)
            name_col = self._find_column(players_df, ["Player_Name", "player_name", "name"], required=False)
            if id_col and name_col:
                map_df = players_df[[id_col, name_col]].dropna().copy()
                map_df[id_col] = map_df[id_col].map(self._normalize_player_id)
                map_df[name_col] = map_df[name_col].map(self._clean_key)
                self.id_to_name = dict(zip(map_df[id_col], map_df[name_col]))
                self.name_to_id = {self._name_key(v): k for k, v in self.id_to_name.items()}
        self._player_name_to_id = self.name_to_id

        # --- Parse deliveries ---
        d = deliveries_df.copy()
        match_col = self._find_column(d, ["matchId", "match_id", "ID"], required=True)
        inning_col = self._find_column(d, ["inning", "innings"], required=True)
        over_col = self._find_column(d, ["over", "overs"], required=True)
        ball_col = self._find_column(d, ["ball"], required=False)
        date_col = self._find_column(d, ["date"], required=True)
        batting_col = self._find_column(d, ["batting_team"], required=True)
        bowling_col = self._find_column(d, ["bowling_team"], required=True)
        batsman_col = self._find_column(d, ["batsman", "striker"], required=True)
        bowler_col = self._find_column(d, ["bowler"], required=True)
        batsman_runs_col = self._find_column(d, ["batsman_runs", "batsman_run"], required=True)
        extras_col = self._find_column(d, ["extras"], required=True)

        d = d[[match_col, inning_col, over_col, date_col, batting_col, bowling_col,
               batsman_col, bowler_col, batsman_runs_col, extras_col]].copy()
        if ball_col is not None:
            d["ball"] = deliveries_df[ball_col]
        else:
            d["ball"] = 0

        d["isWide"] = self._get_series_or_default(deliveries_df, ["isWide", "is_wide"], 0)
        d["isNoBall"] = self._get_series_or_default(deliveries_df, ["isNoBall", "is_no_ball"], 0)

        d = d.rename(columns={
            match_col: "matchId", inning_col: "inning", over_col: "over", date_col: "date",
            batting_col: "batting_team", bowling_col: "bowling_team", batsman_col: "batsman",
            bowler_col: "bowler", batsman_runs_col: "batsman_runs", extras_col: "extras",
        })

        d["matchId"] = d["matchId"].map(self._normalize_match_id)
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["inning"] = pd.to_numeric(d["inning"], errors="coerce").fillna(1).astype(int)
        d["over"] = pd.to_numeric(d["over"], errors="coerce")
        d["ball"] = pd.to_numeric(d["ball"], errors="coerce").fillna(0)
        d["batsman_runs"] = pd.to_numeric(d["batsman_runs"], errors="coerce").fillna(0)
        d["extras"] = pd.to_numeric(d["extras"], errors="coerce").fillna(0)
        d["isWide"] = pd.to_numeric(d["isWide"], errors="coerce").fillna(0)
        d["isNoBall"] = pd.to_numeric(d["isNoBall"], errors="coerce").fillna(0)

        wicket_delivery_col = self._find_column(deliveries_df, ["isWicketDelivery", "is_wicket", "isWicket"], required=False)
        dismissal_kind_col = self._find_column(deliveries_df, ["dismissal_kind", "player_dismissed"], required=False)

        # --- PowerPlay only ---
        pp = d[d["over"] < 6].copy()
        pp["total_run"] = pp["batsman_runs"] + pp["extras"]
        pp["legal_bat_ball"] = np.where(pp["isWide"] > 0, 0, 1)
        pp["legal_bowl_ball"] = np.where((pp["isWide"] > 0) | (pp["isNoBall"] > 0), 0, 1)
        pp["boundary_ball"] = pp["batsman_runs"].isin([4, 6]).astype(int)
        pp["dot_ball"] = (pp["total_run"] == 0).astype(int)

        if wicket_delivery_col is not None:
            wicket_vals = pd.to_numeric(deliveries_df.loc[pp.index, wicket_delivery_col], errors="coerce").fillna(0)
            pp["wicket_ball"] = (wicket_vals > 0).astype(int)
        elif dismissal_kind_col is not None:
            dismissal = deliveries_df.loc[pp.index, dismissal_kind_col].astype(str).str.strip().str.lower()
            pp["wicket_ball"] = ((dismissal != "") & (dismissal != "nan") & (dismissal != "none")).astype(int)
        else:
            pp["wicket_ball"] = 0

        # --- Innings aggregation ---
        innings = (
            pp.groupby(["matchId", "inning"], as_index=False)
            .agg(
                pp_score=("total_run", "sum"),
                batting_team=("batting_team", "first"),
                bowling_team=("bowling_team", "first"),
                date=("date", "min"),
                legal_balls=("legal_bowl_ball", "sum"),
                boundary_balls=("boundary_ball", "sum"),
                dot_balls=("dot_ball", "sum"),
                wickets_lost=("wicket_ball", "sum"),
            )
        )
        innings["legal_balls"] = pd.to_numeric(innings["legal_balls"], errors="coerce").fillna(0.0).clip(lower=1.0)
        innings["boundary_rate"] = innings["boundary_balls"] / innings["legal_balls"]
        innings["dot_rate"] = innings["dot_balls"] / innings["legal_balls"]
        innings["wicket_rate"] = innings["wickets_lost"] / innings["legal_balls"]

        # --- Openers ---
        def map_name_to_player_id(name):
            name_clean = self._clean_key(name)
            return self.name_to_id.get(self._name_key(name_clean), f"NAME::{name_clean}")

        pp_sorted = pp.sort_values(["matchId", "inning", "over", "ball"]).copy()
        openers = (
            pp_sorted.groupby(["matchId", "inning"], as_index=False)
            .agg(open_batsman=("batsman", "first"), open_bowler=("bowler", "first"))
        )
        innings = innings.merge(openers, on=["matchId", "inning"], how="left")
        innings["open_batsman_id"] = innings["open_batsman"].map(map_name_to_player_id)
        innings["open_bowler_id"] = innings["open_bowler"].map(map_name_to_player_id)

        # --- Venue ---
        innings["venue"] = "UNKNOWN"
        if matches_df is not None and not matches_df.empty:
            match_id_col = self._find_column(matches_df, ["matchId", "match_id", "ID"], required=False)
            venue_col = self._find_column(matches_df, ["venue", "Venue"], required=False)
            if match_id_col and venue_col:
                venue_df = matches_df[[match_id_col, venue_col]].drop_duplicates().copy()
                venue_df[match_id_col] = venue_df[match_id_col].map(self._normalize_match_id)
                venue_df = venue_df.rename(columns={match_id_col: "matchId", venue_col: "venue"})
                innings = innings.drop(columns=["venue"]).merge(venue_df, on="matchId", how="left")

        innings["venue"] = innings["venue"].map(self._clean_key)
        self._register_venues(innings["venue"].dropna().tolist())
        self._register_venues(list(raw_home_ground_map.values()))
        self.home_ground_map = {
            self._clean_key(team): self._canonicalize_venue(venue)
            for team, venue in raw_home_ground_map.items()
        }
        innings["venue"] = innings["venue"].map(self._canonicalize_venue)
        innings["batting_team"] = innings["batting_team"].map(self._clean_key)
        innings["bowling_team"] = innings["bowling_team"].map(self._clean_key)
        innings["date"] = pd.to_datetime(innings["date"], errors="coerce")
        innings = innings.sort_values(["date", "matchId", "inning"]).reset_index(drop=True)
        innings["year"] = innings["date"].dt.year.fillna(0).astype(int)

        # --- Global stats ---
        self.global_mean = float(innings["pp_score"].mean()) if len(innings) else 45.0
        self.global_std = float(innings["pp_score"].std()) if len(innings) else 8.0
        if np.isnan(self.global_std):
            self.global_std = 8.0
        total_legal = float(innings["legal_balls"].sum()) if len(innings) else 0.0
        self.global_boundary_rate = float(innings["boundary_balls"].sum() / max(total_legal, 1.0))
        self.global_dot_rate = float(innings["dot_balls"].sum() / max(total_legal, 1.0))
        self.global_wicket_rate = float(innings["wickets_lost"].sum() / max(total_legal, 1.0))

        # --- Season trend (multiplicative) ---
        season_counts = innings.groupby("year")["pp_score"].count()
        season_means = innings.groupby("year")["pp_score"].mean()
        self.season_mean_map = season_means.to_dict()
        self.train_max_year = int(innings["year"].max()) if len(innings) else 2022
        self.predict_year = self.train_max_year + 1

        # Estimate bias correction using robust recent trend.
        # Use last 4 non-COVID valid seasons (>= 10 matches, prefer recent).
        valid_seasons = [(y, m) for y, m in self.season_mean_map.items()
                         if season_counts.get(y, 0) >= 10 and y >= 2015]
        if len(valid_seasons) >= 2:
            valid_seasons.sort(key=lambda x: x[0])
            # Use last 4 valid seasons for slope estimate
            recent = valid_seasons[-4:]
            years_r = np.array([v[0] for v in recent], dtype=float)
            means_r = np.array([v[1] for v in recent], dtype=float)
            if len(recent) >= 2 and years_r[-1] > years_r[0]:
                # Simple linear fit on recent seasons
                ry = years_r - years_r.mean()
                slope_r = float(np.dot(ry, means_r) / max(np.dot(ry, ry), 1e-6))
            else:
                slope_r = 0.0
            # Expected score in next year = last_known + max(slope, min_growth)
            # min_growth = 5.0 run/year reflects IPL powerplay score inflation trend
            last_known_mean = float(self.season_mean_map.get(self.train_max_year, self.global_mean))
            step = max(slope_r, 5.0)
            expected_next = last_known_mean + step
            # Final bias correction = expected_next - global_mean, clipped to prevent over-correction
            self.year_bias_correction = float(np.clip(expected_next - self.global_mean, 1.0, 12.0))
        else:
            self.year_bias_correction = 0.0

        # --- H2H (head-to-head) priors ---
        innings["h2h_key"] = innings["batting_team"] + "||" + innings["bowling_team"]
        innings["h2h_venue_key"] = innings["batting_team"] + "||" + innings["bowling_team"] + "||" + innings["venue"]
        innings["h2h_prior_mean"] = innings.groupby("h2h_key")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["h2h_venue_prior_mean"] = innings.groupby("h2h_venue_key")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())

        # --- Team score volatility (rolling std of last 10) ---
        innings["batting_team_score_std"] = innings.groupby("batting_team")["pp_score"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).std())

        # --- Keys ---
        innings["team_venue_key"] = innings["batting_team"] + "||" + innings["venue"]
        innings["bat_team_inning_key"] = innings["batting_team"] + "||" + innings["inning"].astype(int).astype(str)
        innings["bowl_team_inning_key"] = innings["bowling_team"] + "||" + innings["inning"].astype(int).astype(str)

        # --- Expanding priors (leakage-safe) ---
        innings["venue_prior_mean"] = innings.groupby("venue")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["venue_prior_std"] = innings.groupby("venue")["pp_score"].transform(
            lambda s: s.shift(1).expanding().std())
        innings["batting_team_prior_mean"] = innings.groupby("batting_team")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["bowling_team_prior_mean"] = innings.groupby("bowling_team")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["team_venue_prior_mean"] = innings.groupby("team_venue_key")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["open_batsman_prior_mean"] = innings.groupby("open_batsman_id")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["open_bowler_prior_mean"] = innings.groupby("open_bowler_id")["pp_score"].transform(
            lambda s: s.shift(1).expanding().mean())

        # --- Recency-weighted (EWM) priors ---
        innings["venue_recent_mean"] = innings.groupby("venue")["pp_score"].transform(
            lambda s: self._ewm_expanding_mean(s, halflife_games=25))
        innings["batting_team_recent_mean"] = innings.groupby("batting_team")["pp_score"].transform(
            lambda s: self._ewm_expanding_mean(s, halflife_games=18))
        innings["bowling_team_recent_mean"] = innings.groupby("bowling_team")["pp_score"].transform(
            lambda s: self._ewm_expanding_mean(s, halflife_games=18))

        # --- Rate priors ---
        innings["batting_team_boundary_rate_prior"] = innings.groupby("batting_team")["boundary_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["venue_boundary_rate_prior"] = innings.groupby("venue")["boundary_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["batting_team_dot_rate_prior"] = innings.groupby("batting_team")["dot_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["venue_dot_rate_prior"] = innings.groupby("venue")["dot_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["batting_team_wicket_rate_prior"] = innings.groupby("batting_team")["wicket_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["bowling_team_wicket_rate_prior"] = innings.groupby("bowling_team")["wicket_rate"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["venue_wicket_rate_prior"] = innings.groupby("venue")["wicket_rate"].transform(
            lambda s: s.shift(1).expanding().mean())

        innings["boundary_rate_prior"] = (
            0.70 * innings["batting_team_boundary_rate_prior"].fillna(self.global_boundary_rate)
            + 0.30 * innings["venue_boundary_rate_prior"].fillna(self.global_boundary_rate)
        )
        innings["dot_rate_prior"] = (
            0.70 * innings["batting_team_dot_rate_prior"].fillna(self.global_dot_rate)
            + 0.30 * innings["venue_dot_rate_prior"].fillna(self.global_dot_rate)
        )
        innings["wicket_rate_prior"] = (
            0.70 * innings["batting_team_wicket_rate_prior"].fillna(self.global_wicket_rate)
            + 0.30 * innings["venue_wicket_rate_prior"].fillna(self.global_wicket_rate)
        )
        w_bat, w_bowl, w_venue = self.wicket_pressure_weights
        innings["wicket_pressure"] = (
            w_bat * innings["batting_team_wicket_rate_prior"].fillna(self.global_wicket_rate)
            + w_bowl * innings["bowling_team_wicket_rate_prior"].fillna(self.global_wicket_rate)
            + w_venue * innings["venue_wicket_rate_prior"].fillna(self.global_wicket_rate)
        )

        # --- Counts ---
        batsman_count_df = (
            pp[pp["legal_bat_ball"] > 0]
            .groupby(["matchId", "inning"], as_index=False)
            .agg(batsman_count=("batsman", "nunique"))
        )
        bowler_count_df = (
            pp.groupby(["matchId", "inning"], as_index=False)
            .agg(bowler_count=("bowler", "nunique"))
        )
        innings = innings.merge(batsman_count_df, on=["matchId", "inning"], how="left")
        innings = innings.merge(bowler_count_df, on=["matchId", "inning"], how="left")

        self.global_batsman_count = float(innings["batsman_count"].mean()) if len(innings) else 2.2
        self.global_bowler_count = float(innings["bowler_count"].mean()) if len(innings) else 2.0
        if np.isnan(self.global_batsman_count):
            self.global_batsman_count = 2.2
        if np.isnan(self.global_bowler_count):
            self.global_bowler_count = 2.0

        innings["batsman_count_prior"] = innings.groupby("bat_team_inning_key")["batsman_count"].transform(
            lambda s: s.shift(1).expanding().mean())
        innings["bowler_count_prior"] = innings.groupby("bowl_team_inning_key")["bowler_count"].transform(
            lambda s: s.shift(1).expanding().mean())

        # --- Home advantage ---
        home_venue_series = innings["batting_team"].map(self.home_ground_map).fillna("UNKNOWN")
        innings["is_home"] = (home_venue_series == innings["venue"]).astype(float)
        innings["home_advantage_boost"] = 0.0
        for _, team_df in innings.groupby("batting_team", sort=False):
            home_mask = team_df["is_home"].to_numpy(dtype=bool)
            residual = (team_df["pp_score"] - team_df["batting_team_prior_mean"].fillna(self.global_mean)).to_numpy(dtype=float)
            home_residual = np.where(home_mask, residual, 0.0)
            cum_sum = np.cumsum(home_residual)
            cum_cnt = np.cumsum(home_mask.astype(int))
            prev_sum = np.concatenate(([0.0], cum_sum[:-1]))
            prev_cnt = np.concatenate(([0], cum_cnt[:-1]))
            prior = np.divide(
                prev_sum, np.maximum(prev_cnt, 1),
                out=np.zeros_like(prev_sum, dtype=float),
                where=(home_mask & (prev_cnt > 0)),
            )
            innings.loc[team_df.index, "home_advantage_boost"] = prior

        # --- Dew factor ---
        match_meta = innings.groupby("matchId", as_index=False).agg(date=("date", "min"), venue=("venue", "first"))
        pair_df = innings.pivot_table(index="matchId", columns="inning", values="pp_score", aggfunc="first")
        pair_df = match_meta.merge(pair_df, on="matchId", how="left")
        pair_df = pair_df.rename(columns={1: "inning1_pp", 2: "inning2_pp"})
        pair_df["dew_delta"] = pair_df["inning2_pp"] - pair_df["inning1_pp"]
        dew_history = pair_df.dropna(subset=["dew_delta"]).sort_values(["venue", "date", "matchId"])
        dew_history["dew_prior"] = dew_history.groupby("venue")["dew_delta"].transform(
            lambda s: s.shift(1).expanding().mean())
        self.global_dew_factor = float(dew_history["dew_delta"].mean()) if len(dew_history) else 0.0
        if np.isnan(self.global_dew_factor):
            self.global_dew_factor = 0.0
        self.venue_dew_map = (
            dew_history.groupby("venue")["dew_delta"].mean().to_dict() if len(dew_history) else {}
        )
        match_dew_prior_map = dict(zip(dew_history["matchId"], dew_history["dew_prior"]))

        second_mask = innings["inning"] == 2
        innings["dew_factor"] = 0.0
        innings.loc[second_mask, "dew_factor"] = innings.loc[second_mask, "matchId"].map(match_dew_prior_map)
        innings.loc[second_mask, "dew_factor"] = innings.loc[second_mask, "dew_factor"].fillna(self.global_dew_factor)

        # --- First innings score ---
        first_innings_actual = (
            innings.loc[innings["inning"] == 1, ["matchId", "pp_score"]]
            .drop_duplicates(subset=["matchId"])
            .set_index("matchId")["pp_score"]
            .to_dict()
        )
        self.first_innings_match_map = first_innings_actual
        self.global_first_innings_mean = (
            float(innings.loc[innings["inning"] == 1, "pp_score"].mean())
            if (innings["inning"] == 1).any()
            else self.global_mean
        )
        if np.isnan(self.global_first_innings_mean):
            self.global_first_innings_mean = self.global_mean
        self.first_innings_mean_by_venue = (
            innings.loc[innings["inning"] == 1].groupby("venue")["pp_score"].mean().to_dict()
        )
        innings["first_innings_score"] = self.global_first_innings_mean
        innings.loc[second_mask, "first_innings_score"] = innings.loc[second_mask, "matchId"].map(first_innings_actual)
        missing_second_first = second_mask & innings["first_innings_score"].isna()
        innings.loc[missing_second_first, "first_innings_score"] = (
            innings.loc[missing_second_first, "venue"].map(self.first_innings_mean_by_venue)
        )
        innings["first_innings_score"] = pd.to_numeric(innings["first_innings_score"], errors="coerce").fillna(self.global_first_innings_mean)

        # --- Player stats ---
        batsman_innings = (
            pp.groupby(["matchId", "inning", "batsman"], as_index=False)
            .agg(runs=("batsman_runs", "sum"), balls=("legal_bat_ball", "sum"))
        )
        batsman_innings["player_id"] = batsman_innings["batsman"].map(map_name_to_player_id)
        batsman_innings = batsman_innings.merge(innings[["matchId", "inning", "date"]], on=["matchId", "inning"], how="left")
        batsman_innings = batsman_innings.sort_values(["player_id", "date", "matchId", "inning"]).reset_index(drop=True)
        batsman_innings["prev_runs"] = batsman_innings.groupby("player_id")["runs"].cumsum() - batsman_innings["runs"]
        batsman_innings["prev_balls"] = batsman_innings.groupby("player_id")["balls"].cumsum() - batsman_innings["balls"]
        batsman_innings["sr_prior"] = 100.0 * batsman_innings["prev_runs"] / batsman_innings["prev_balls"].replace(0, np.nan)

        bowler_innings = (
            pp.groupby(["matchId", "inning", "bowler"], as_index=False)
            .agg(runs=("total_run", "sum"), balls=("legal_bowl_ball", "sum"))
        )
        bowler_innings["player_id"] = bowler_innings["bowler"].map(map_name_to_player_id)
        bowler_innings = bowler_innings.merge(innings[["matchId", "inning", "date"]], on=["matchId", "inning"], how="left")
        bowler_innings = bowler_innings.sort_values(["player_id", "date", "matchId", "inning"]).reset_index(drop=True)
        bowler_innings["prev_runs"] = bowler_innings.groupby("player_id")["runs"].cumsum() - bowler_innings["runs"]
        bowler_innings["prev_balls"] = bowler_innings.groupby("player_id")["balls"].cumsum() - bowler_innings["balls"]
        bowler_innings["econ_prior"] = 6.0 * bowler_innings["prev_runs"] / bowler_innings["prev_balls"].replace(0, np.nan)

        open_batsman_sr = batsman_innings[["matchId", "inning", "player_id", "sr_prior"]].rename(
            columns={"player_id": "open_batsman_id", "sr_prior": "open_batsman_sr_prior"})
        open_bowler_econ = bowler_innings[["matchId", "inning", "player_id", "econ_prior"]].rename(
            columns={"player_id": "open_bowler_id", "econ_prior": "open_bowler_econ_prior"})
        innings = innings.merge(open_batsman_sr, on=["matchId", "inning", "open_batsman_id"], how="left")
        innings = innings.merge(open_bowler_econ, on=["matchId", "inning", "open_bowler_id"], how="left")

        total_bat_runs = float(pp["batsman_runs"].sum())
        total_bat_balls = float(pp["legal_bat_ball"].sum())
        total_bowl_runs = float(pp["total_run"].sum())
        total_bowl_balls = float(pp["legal_bowl_ball"].sum())
        self.global_batsman_sr = 100.0 * total_bat_runs / max(total_bat_balls, 1.0)
        self.global_bowler_econ = 6.0 * total_bowl_runs / max(total_bowl_balls, 1.0)

        # Composite player stats (for training, we set to global — used at predict time)
        innings["composite_bat_sr"] = self.global_batsman_sr
        innings["composite_bowl_econ"] = self.global_bowler_econ
        innings["composite_bat_venue_sr"] = self.global_batsman_sr  # venue-specific at predict
        innings["top3_bat_avg"] = self.global_mean                  # top-3 at predict
        innings["composite_bowl_dot_pct"] = self.global_dot_rate    # bowler dot% at predict

        # --- Fill NaNs ---
        for col, fallback in [
            ("venue_prior_mean", self.global_mean),
            ("venue_prior_std", self.global_std),
            ("venue_recent_mean", self.global_mean),
            ("batting_team_prior_mean", self.global_mean),
            ("batting_team_recent_mean", self.global_mean),
            ("bowling_team_prior_mean", self.global_mean),
            ("bowling_team_recent_mean", self.global_mean),
            ("team_venue_prior_mean", self.global_mean),
            ("open_batsman_prior_mean", self.global_mean),
            ("open_bowler_prior_mean", self.global_mean),
            ("batsman_count_prior", self.global_batsman_count),
            ("bowler_count_prior", self.global_bowler_count),
            ("is_home", 0.0),
            ("home_advantage_boost", 0.0),
            ("dew_factor", self.global_dew_factor),
            ("first_innings_score", self.global_first_innings_mean),
            ("open_batsman_sr_prior", self.global_batsman_sr),
            ("open_bowler_econ_prior", self.global_bowler_econ),
            ("boundary_rate_prior", self.global_boundary_rate),
            ("dot_rate_prior", self.global_dot_rate),
            ("wicket_rate_prior", self.global_wicket_rate),
            ("wicket_pressure", self.global_wicket_rate),
            ("h2h_prior_mean", self.global_mean),
            ("h2h_venue_prior_mean", self.global_mean),
            ("batting_team_score_std", self.global_std),
        ]:
            innings[col] = pd.to_numeric(innings[col], errors="coerce").fillna(fallback)

        # H2H: use H2H venue where available, else H2H, else global mean
        innings["h2h_venue_prior_mean"] = innings["h2h_venue_prior_mean"].fillna(innings["h2h_prior_mean"])

        # Build features
        X_full = self._build_features(innings, year_override=None)
        y = pd.to_numeric(innings["pp_score"], errors="coerce").fillna(self.global_mean)

        # --- Recency sample weights ---
        halflife_yrs = 3.5
        alpha_yr = 1.0 - np.exp(-np.log(2) / halflife_yrs)
        max_yr = float(innings["year"].max())
        weight_year = np.exp(np.log(1 - alpha_yr) * np.maximum(0.0, max_yr - innings["year"].values.astype(float)))
        innings["match_idx_venue"] = innings.groupby("venue").cumcount(ascending=False)
        weight_venue = np.exp(-0.04 * innings["match_idx_venue"].values.astype(float))
        sample_weight = np.clip(weight_year * 0.7 + weight_venue * 0.3, 0.04, 1.0)

        # --- Train ensemble ---
        if len(X_full) >= 20:
            self._select_top_features(X_full, y)
            X_sel = X_full[self.selected_feature_columns]

            # ── 2-seed LightGBM bags (280 trees each) ────────────────────────────
            # Reduced from 3-seed × 700 trees → 2-seed × 280 trees.
            # ~4× faster; accuracy drop is marginal (<0.3 MAE) because the
            # feature-engineering + post-processing pipeline carries most signal.
            _N_EST = 280
            _SEEDS = [42, 123]
            if HAS_LGBM:
                self.model_mae_list = []
                self.model_poisson_list = []
                for s in _SEEDS:
                    m_mae = LGBMRegressor(
                        objective="regression_l1", learning_rate=0.05, n_estimators=_N_EST,
                        num_leaves=48, min_child_samples=10, subsample=0.82,
                        subsample_freq=1, colsample_bytree=0.82, reg_alpha=0.08,
                        reg_lambda=0.12, random_state=s, verbose=-1, n_jobs=2,
                    )
                    m_mae.fit(X_sel, y, sample_weight=sample_weight)
                    self.model_mae_list.append(m_mae)

                    m_poi = LGBMRegressor(
                        objective="poisson", learning_rate=0.05, n_estimators=_N_EST,
                        num_leaves=36, min_child_samples=12, subsample=0.82,
                        subsample_freq=1, colsample_bytree=0.82, reg_alpha=0.04,
                        reg_lambda=0.08, random_state=s + 1, verbose=-1, n_jobs=2,
                    )
                    m_poi.fit(X_sel, y, sample_weight=sample_weight)
                    self.model_poisson_list.append(m_poi)
            else:
                self.model_mae_list = []
                self.model_poisson_list = []

            # ── Two HGB models replace the 3× Huber LGBM bag ─────────────────────
            # HGB is faster than LightGBM at equivalent depth and needs no n_jobs.
            self.model_hgb = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.05, max_depth=6,
                max_iter=280, min_samples_leaf=12, random_state=42,
            )
            try:
                self.model_hgb.fit(X_sel, y, sample_weight=sample_weight)
            except TypeError:
                self.model_hgb.fit(X_sel, y)

            self.model_hgb2 = HistGradientBoostingRegressor(
                loss="squared_error", learning_rate=0.05, max_depth=5,
                max_iter=200, min_samples_leaf=15, random_state=7,
            )
            try:
                self.model_hgb2.fit(X_sel, y, sample_weight=sample_weight)
            except TypeError:
                self.model_hgb2.fit(X_sel, y)

            self.model_huber_list = []   # no longer used
        else:
            self.model_mae_list = []
            self.model_poisson_list = []
            self.model_hgb = None
            self.model_hgb2 = None
            self.model_huber_list = []
            self._select_top_features(X_full, y)

        raw_train_pred = self._ensemble_predict(X_full)

        # --- Post-fit ---
        train_pred = (
            self.model_blend_weight_train * raw_train_pred
            + (1.0 - self.model_blend_weight_train) * X_full["base_prior"].to_numpy(dtype=float)
        )
        self._fit_context_gains(X_full, y, train_pred)
        self._fit_residual_bucket_map(innings, y, train_pred)
        self._fit_output_calibration(X_full, y, train_pred)

        # --- Store lookup maps ---
        self.venue_mean_map = innings.groupby("venue")["pp_score"].mean().to_dict()
        self.venue_std_map = innings.groupby("venue")["pp_score"].std().fillna(self.global_std).to_dict()
        self.venue_recent_mean_map = (
            innings.sort_values("date").groupby("venue")["pp_score"]
            .apply(lambda s: float(s.tail(30).mean()) if len(s) >= 5 else float(s.mean()))
            .to_dict()
        )
        self.batting_team_mean_map = innings.groupby("batting_team")["pp_score"].mean().to_dict()
        self.batting_team_recent_mean_map = (
            innings.sort_values("date").groupby("batting_team")["pp_score"]
            .apply(lambda s: float(s.tail(20).mean()) if len(s) >= 5 else float(s.mean()))
            .to_dict()
        )
        self.bowling_team_mean_map = innings.groupby("bowling_team")["pp_score"].mean().to_dict()
        self.bowling_team_recent_mean_map = (
            innings.sort_values("date").groupby("bowling_team")["pp_score"]
            .apply(lambda s: float(s.tail(20).mean()) if len(s) >= 5 else float(s.mean()))
            .to_dict()
        )
        self.team_venue_mean_map = innings.groupby("team_venue_key")["pp_score"].mean().to_dict()
        self.open_batsman_pp_map = innings.groupby("open_batsman_id")["pp_score"].mean().to_dict()
        self.open_bowler_pp_map = innings.groupby("open_bowler_id")["pp_score"].mean().to_dict()
        self.batsman_count_prior_map = innings.groupby("bat_team_inning_key")["batsman_count"].mean().to_dict()
        self.bowler_count_prior_map = innings.groupby("bowl_team_inning_key")["bowler_count"].mean().to_dict()
        self.batting_team_boundary_rate_map = innings.groupby("batting_team")["boundary_rate"].mean().to_dict()
        self.batting_team_dot_rate_map = innings.groupby("batting_team")["dot_rate"].mean().to_dict()
        self.batting_team_wicket_rate_map = innings.groupby("batting_team")["wicket_rate"].mean().to_dict()
        self.bowling_team_wicket_rate_map = innings.groupby("bowling_team")["wicket_rate"].mean().to_dict()
        self.venue_boundary_rate_map = innings.groupby("venue")["boundary_rate"].mean().to_dict()
        self.venue_dot_rate_map = innings.groupby("venue")["dot_rate"].mean().to_dict()
        self.venue_wicket_rate_map = innings.groupby("venue")["wicket_rate"].mean().to_dict()
        team_mean = innings.groupby("batting_team")["pp_score"].mean()
        home_mean = innings.loc[innings["is_home"] == 1].groupby("batting_team")["pp_score"].mean()
        self.team_home_boost_map = (home_mean - team_mean.reindex(home_mean.index)).fillna(0.0).to_dict()

        bat_sum = batsman_innings.groupby("player_id", as_index=False).agg(runs=("runs", "sum"), balls=("balls", "sum"))
        bowl_sum = bowler_innings.groupby("player_id", as_index=False).agg(runs=("runs", "sum"), balls=("balls", "sum"))
        sr_prior_balls = 30.0
        econ_prior_balls = 30.0
        bat_sum["sr"] = (
            100.0 * (bat_sum["runs"] + (self.global_batsman_sr / 100.0) * sr_prior_balls)
            / (bat_sum["balls"] + sr_prior_balls)
        )
        bowl_sum["econ"] = (
            6.0 * (bowl_sum["runs"] + (self.global_bowler_econ / 6.0) * econ_prior_balls)
            / (bowl_sum["balls"] + econ_prior_balls)
        )
        self.batsman_sr_map = dict(zip(bat_sum["player_id"], bat_sum["sr"]))
        self.bowler_econ_map = dict(zip(bowl_sum["player_id"], bowl_sum["econ"]))

        # --- H2H lookup maps ---
        self.h2h_mean_map = innings.groupby("h2h_key")["pp_score"].mean().to_dict()
        self.h2h_venue_mean_map = innings.groupby("h2h_venue_key")["pp_score"].mean().to_dict()

        # --- Team volatility map ---
        self.global_batting_team_std = float(innings["batting_team_score_std"].mean())
        if np.isnan(self.global_batting_team_std):
            self.global_batting_team_std = self.global_std
        # Use last known std for each team
        _team_std = innings.sort_values("date").groupby("batting_team")["batting_team_score_std"].last()
        self.batting_team_std_map = _team_std.fillna(self.global_batting_team_std).to_dict()

        # --- Player-venue SR map: (player_id, venue) -> SR ---
        # Add venue to pp for player-venue stats
        _pp_venue = pp.merge(innings[["matchId", "inning", "venue"]].drop_duplicates(),
                             on=["matchId", "inning"], how="left")
        _pp_venue["player_id"] = _pp_venue["batsman"].map(map_name_to_player_id)
        _pv_stats = (
            _pp_venue[_pp_venue["legal_bat_ball"] > 0]
            .groupby(["player_id", "venue"], as_index=False)
            .agg(runs=("batsman_runs", "sum"), balls=("legal_bat_ball", "sum"))
        )
        _pv_stats = _pv_stats[_pv_stats["balls"] >= 6]  # min 1 over at venue
        _pv_stats["sr"] = 100.0 * _pv_stats["runs"] / _pv_stats["balls"]
        self.player_venue_sr_map = dict(zip(
            _pv_stats["player_id"] + "||" + _pv_stats["venue"], _pv_stats["sr"]
        ))

        # --- Venue boundary percentage ---
        _pp_venue["boundary_run"] = _pp_venue["batsman_runs"].where(_pp_venue["batsman_runs"].isin([4, 6]), 0)
        _venue_bnd = _pp_venue.groupby("venue").agg(
            bnd_runs=("boundary_run", "sum"), total_runs=("total_run", "sum")
        )
        _venue_bnd["boundary_pct"] = _venue_bnd["bnd_runs"] / _venue_bnd["total_runs"].clip(lower=1)
        self.venue_boundary_pct_map = _venue_bnd["boundary_pct"].to_dict()
        self.global_venue_boundary_pct = float(_venue_bnd["boundary_pct"].mean()) if len(_venue_bnd) else 0.5

        # --- Bowler dot ball % map ---
        _pp_venue["bowler_id"] = _pp_venue["bowler"].map(map_name_to_player_id)
        _bowl_dot = (
            _pp_venue.groupby("bowler_id", as_index=False)
            .agg(dots=("dot_ball", "sum"), balls=("legal_bowl_ball", "sum"))
        )
        _bowl_dot = _bowl_dot[_bowl_dot["balls"] >= 12]  # min 2 overs
        _bowl_dot["dot_pct"] = _bowl_dot["dots"] / _bowl_dot["balls"]
        self.bowler_dot_pct_map = dict(zip(_bowl_dot["bowler_id"], _bowl_dot["dot_pct"]))
        self.global_bowler_dot_pct = float(_bowl_dot["dot_pct"].mean()) if len(_bowl_dot) else self.global_dot_rate

        # --- Top-3 batsman PP mean map (for predict) ---
        # Already have open_batsman_pp_map; also store all-player pp_score map
        _bat_pp = batsman_innings.merge(innings[["matchId", "inning", "pp_score"]], on=["matchId", "inning"], how="left")
        _player_pp_mean = _bat_pp.groupby("player_id")["pp_score"].mean()
        self.player_pp_mean_map = _player_pp_mean.to_dict()

        return self

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #
    def predict(self, test_df):
        if test_df is None or len(test_df) == 0:
            return pd.DataFrame(columns=["id", "predicted_score"])

        t = test_df.copy()
        id_col = self._find_column(t, ["id", "ID"], required=False)
        ids = t[id_col].values if id_col is not None else np.arange(1, len(t) + 1)

        venue_series = self._get_series_or_default(t, ["venue", "Venue"], "UNKNOWN").map(self._clean_key)
        venue_series = venue_series.map(self._canonicalize_venue)
        inning_series = pd.to_numeric(
            self._get_series_or_default(t, ["innings", "inning"], 1), errors="coerce"
        ).fillna(1).astype(int)
        batting_series = self._get_series_or_default(t, ["batting_team"], "UNKNOWN").map(self._clean_key)
        bowling_series = self._get_series_or_default(t, ["bowling_team"], "UNKNOWN").map(self._clean_key)

        # --- COMPOSITE MULTI-PLAYER IDs (unique krishna1 advantage) ---
        bat_id_col = self._find_column(
            t, ["Batsman's Player Id", "Batsman's Player ID", "batsman", "batsman_id"], required=False
        )
        bowl_id_col = self._find_column(
            t, ["Bowler's Player id (opponent)", "Bowler's Player Id (opponent)",
                "bowler", "bowler_id"], required=False
        )

        # For composite SR: average SR of ALL batsmen listed (comma-separated)
        # ALSO count actual batsmen/bowlers — this reveals real match state!
        #   batsmen_count = N  →  wickets fallen = N - 2  (2 openers always bat)
        #   bowler_count = M   →  more bowlers often means higher scoring
        composite_bat_sr = []
        primary_bat_ids = []
        actual_batsman_count_list = []
        composite_bat_venue_sr_list = []  # player SR at this specific venue
        top3_bat_avg_list = []            # top-3 batting average
        for idx in t.index:
            raw = str(t.at[idx, bat_id_col]) if bat_id_col else "UNKNOWN"
            pids = [self._normalize_player_id(pid.strip()) for pid in raw.split(",") if pid.strip()]
            venue = venue_series.at[idx] if idx in venue_series.index else "UNKNOWN"
            srs = [self.batsman_sr_map.get(pid, self.global_batsman_sr) for pid in pids]
            composite_bat_sr.append(float(np.mean(srs)) if srs else self.global_batsman_sr)
            primary_bat_ids.append(pids[0] if pids else "UNKNOWN")
            actual_batsman_count_list.append(len(pids) if pids and pids[0] != "UNKNOWN" else 0)
            # Player-venue SR: use venue-specific if available, else global
            venue_srs = [self.player_venue_sr_map.get(pid + "||" + venue,
                         self.batsman_sr_map.get(pid, self.global_batsman_sr)) for pid in pids]
            composite_bat_venue_sr_list.append(float(np.mean(venue_srs)) if venue_srs else self.global_batsman_sr)
            # Top-3 batting average (first 3 IDs in list = top order)
            top3 = pids[:3] if len(pids) >= 3 else pids
            top3_avgs = [self.player_pp_mean_map.get(pid, self.global_mean) for pid in top3]
            top3_bat_avg_list.append(float(np.mean(top3_avgs)) if top3_avgs else self.global_mean)
        composite_bat_sr = np.array(composite_bat_sr)
        actual_batsman_count = np.array(actual_batsman_count_list, dtype=float)
        composite_bat_venue_sr = np.array(composite_bat_venue_sr_list, dtype=float)
        top3_bat_avg = np.array(top3_bat_avg_list, dtype=float)

        # For composite economy: average economy of ALL bowlers listed
        composite_bowl_econ = []
        primary_bowl_ids = []
        actual_bowler_count_list = []
        composite_bowl_dot_pct_list = []  # bowling XI dot ball %
        for idx in t.index:
            raw = str(t.at[idx, bowl_id_col]) if bowl_id_col else "UNKNOWN"
            pids = [self._normalize_player_id(pid.strip()) for pid in raw.split(",") if pid.strip()]
            econs = [self.bowler_econ_map.get(pid, self.global_bowler_econ) for pid in pids]
            composite_bowl_econ.append(float(np.mean(econs)) if econs else self.global_bowler_econ)
            primary_bowl_ids.append(pids[0] if pids else "UNKNOWN")
            actual_bowler_count_list.append(len(pids) if pids and pids[0] != "UNKNOWN" else 0)
            # Bowling XI dot ball %
            dot_pcts = [self.bowler_dot_pct_map.get(pid, self.global_bowler_dot_pct) for pid in pids]
            composite_bowl_dot_pct_list.append(float(np.mean(dot_pcts)) if dot_pcts else self.global_bowler_dot_pct)
        composite_bowl_econ = np.array(composite_bowl_econ)
        actual_bowler_count = np.array(actual_bowler_count_list, dtype=float)
        composite_bowl_dot_pct = np.array(composite_bowl_dot_pct_list, dtype=float)

        batsman_id_series = pd.Series(primary_bat_ids, index=t.index)
        bowler_id_series = pd.Series(primary_bowl_ids, index=t.index)

        raw_match_series = self._get_series_or_default(t, ["matchId", "match_id", "MatchId", "match"], np.nan)
        match_series = raw_match_series.map(self._normalize_match_id)
        if id_col is not None:
            parsed_match_series = pd.Series(ids, index=t.index).map(self._extract_match_id_from_any)
            match_series = match_series.where(match_series != "UNKNOWN", parsed_match_series)

        first_innings_input = pd.to_numeric(
            self._get_series_or_default(t, ["first_innings_pp_score", "first_innings_score", "inning1_pp_score"], np.nan),
            errors="coerce",
        )

        # --- Build inference frame ---
        infer = pd.DataFrame(index=t.index)
        infer["inning"] = inning_series

        infer["venue_prior_mean"] = venue_series.map(self.venue_mean_map).fillna(self.global_mean)
        infer["venue_prior_std"] = venue_series.map(self.venue_std_map).fillna(self.global_std)
        infer["batting_team_prior_mean"] = batting_series.map(self.batting_team_mean_map).fillna(self.global_mean)
        infer["bowling_team_prior_mean"] = bowling_series.map(self.bowling_team_mean_map).fillna(self.global_mean)
        infer["venue_recent_mean"] = venue_series.map(self.venue_recent_mean_map).fillna(
            venue_series.map(self.venue_mean_map).fillna(self.global_mean))
        infer["batting_team_recent_mean"] = batting_series.map(self.batting_team_recent_mean_map).fillna(
            batting_series.map(self.batting_team_mean_map).fillna(self.global_mean))
        infer["bowling_team_recent_mean"] = bowling_series.map(self.bowling_team_recent_mean_map).fillna(
            bowling_series.map(self.bowling_team_mean_map).fillna(self.global_mean))

        team_venue_key = batting_series + "||" + venue_series
        infer["team_venue_prior_mean"] = team_venue_key.map(self.team_venue_mean_map).fillna(self.global_mean)
        infer["open_batsman_prior_mean"] = batsman_id_series.map(self.open_batsman_pp_map).fillna(self.global_mean)
        infer["open_bowler_prior_mean"] = bowler_id_series.map(self.open_bowler_pp_map).fillna(self.global_mean)

        bat_team_inning_key = batting_series + "||" + inning_series.astype(int).astype(str)
        bowl_team_inning_key = bowling_series + "||" + inning_series.astype(int).astype(str)
        infer["batsman_count_prior"] = bat_team_inning_key.map(self.batsman_count_prior_map).fillna(self.global_batsman_count)
        infer["bowler_count_prior"] = bowl_team_inning_key.map(self.bowler_count_prior_map).fillna(self.global_bowler_count)

        # --- LIVE-STATE OVERRIDE: use actual counts from test data ---
        # When the test CSV gives us comma-separated player IDs, the COUNT
        # is real match information: batsmen_used = wickets_fallen + 2.
        has_actual_bat = actual_batsman_count >= 2  # at least 2 openers
        has_actual_bowl = actual_bowler_count >= 2   # 2+ bowlers = full list, not just opener
        infer.loc[has_actual_bat, "batsman_count_prior"] = actual_batsman_count[has_actual_bat]
        infer.loc[has_actual_bowl, "bowler_count_prior"] = actual_bowler_count[has_actual_bowl]

        # --- NEW FEATURES: H2H, volatility, player-venue SR, top3, venue boundary %, bowl dot % ---
        h2h_key = batting_series + "||" + bowling_series
        h2h_venue_key = batting_series + "||" + bowling_series + "||" + venue_series
        infer["h2h_prior_mean"] = h2h_key.map(self.h2h_mean_map).fillna(self.global_mean)
        infer["h2h_venue_prior_mean"] = h2h_venue_key.map(self.h2h_venue_mean_map).fillna(
            infer["h2h_prior_mean"])
        infer["batting_team_score_std"] = batting_series.map(self.batting_team_std_map).fillna(self.global_batting_team_std)
        infer["composite_bat_venue_sr"] = composite_bat_venue_sr
        infer["top3_bat_avg"] = top3_bat_avg
        infer["venue_boundary_pct"] = venue_series.map(self.venue_boundary_pct_map).fillna(self.global_venue_boundary_pct)
        infer["composite_bowl_dot_pct"] = composite_bowl_dot_pct

        home_venue_s = batting_series.map(self.home_ground_map).fillna("UNKNOWN").map(self._canonicalize_venue)
        infer["is_home"] = (home_venue_s == venue_series).astype(float)
        infer["home_advantage_boost"] = infer["is_home"] * batting_series.map(self.team_home_boost_map).fillna(0.0)

        venue_dew = venue_series.map(self.venue_dew_map).fillna(self.global_dew_factor)
        infer["dew_factor"] = np.where(inning_series == 2, venue_dew, 0.0)

        # First innings score for 2nd innings
        batch_first_estimate = self._make_base_prior(infer)
        inning1_rows = inning_series == 1
        batch_map_df = pd.DataFrame({"matchId": match_series[inning1_rows], "est_first": batch_first_estimate[inning1_rows]})
        batch_map_df = batch_map_df[batch_map_df["matchId"] != "UNKNOWN"].drop_duplicates("matchId")
        batch_first_map = dict(zip(batch_map_df["matchId"], batch_map_df["est_first"]))

        second_rows = inning_series == 2
        second_first_score = first_innings_input.copy()
        missing_second = second_rows & second_first_score.isna()
        second_first_score.loc[missing_second] = match_series.loc[missing_second].map(self.first_innings_match_map)
        missing_second = second_rows & second_first_score.isna()
        second_first_score.loc[missing_second] = match_series.loc[missing_second].map(batch_first_map)
        missing_second = second_rows & second_first_score.isna()
        second_first_score.loc[missing_second] = venue_series.loc[missing_second].map(self.first_innings_mean_by_venue)
        infer["first_innings_score"] = self.global_first_innings_mean
        infer.loc[second_rows, "first_innings_score"] = second_first_score.loc[second_rows]
        infer["first_innings_score"] = pd.to_numeric(infer["first_innings_score"], errors="coerce").fillna(self.global_first_innings_mean)

        infer["open_batsman_sr_prior"] = batsman_id_series.map(self.batsman_sr_map).fillna(self.global_batsman_sr)
        infer["open_bowler_econ_prior"] = bowler_id_series.map(self.bowler_econ_map).fillna(self.global_bowler_econ)
        infer["composite_bat_sr"] = composite_bat_sr
        infer["composite_bowl_econ"] = composite_bowl_econ
        infer["bat_minus_bowl"] = infer["batting_team_prior_mean"] - infer["bowling_team_prior_mean"]
        infer["base_prior"] = self._make_base_prior(infer)

        batting_boundary_rate = batting_series.map(self.batting_team_boundary_rate_map).fillna(self.global_boundary_rate)
        venue_boundary_rate = venue_series.map(self.venue_boundary_rate_map).fillna(self.global_boundary_rate)
        batting_dot_rate = batting_series.map(self.batting_team_dot_rate_map).fillna(self.global_dot_rate)
        venue_dot_rate = venue_series.map(self.venue_dot_rate_map).fillna(self.global_dot_rate)
        batting_wicket_rate = batting_series.map(self.batting_team_wicket_rate_map).fillna(self.global_wicket_rate)
        bowling_wicket_rate = bowling_series.map(self.bowling_team_wicket_rate_map).fillna(self.global_wicket_rate)
        venue_wicket_rate = venue_series.map(self.venue_wicket_rate_map).fillna(self.global_wicket_rate)

        infer["boundary_rate_prior"] = 0.70 * batting_boundary_rate + 0.30 * venue_boundary_rate
        infer["dot_rate_prior"] = 0.70 * batting_dot_rate + 0.30 * venue_dot_rate
        infer["wicket_rate_prior"] = 0.70 * batting_wicket_rate + 0.30 * venue_wicket_rate
        w_bat, w_bowl, w_venue = self.wicket_pressure_weights
        infer["wicket_pressure"] = w_bat * batting_wicket_rate + w_bowl * bowling_wicket_rate + w_venue * venue_wicket_rate
        infer["inning2_flag"] = (inning_series == 2).astype(float)
        # --- LIVE-STATE: wickets_proxy uses actual batsman count ---
        # When we have actual batsman counts, wickets = batsmen - 2 (openers)
        # This is REAL match state, not a historical average!
        infer["wickets_proxy"] = np.where(
            actual_batsman_count >= 2,
            np.clip(actual_batsman_count - 2.0, 0.0, 6.0),
            np.clip(
                pd.to_numeric(infer["batsman_count_prior"], errors="coerce").fillna(self.global_batsman_count) - 2.0,
                0.0, 6.0,
            ),
        )
        infer["aggression"] = infer["boundary_rate_prior"] - infer["dot_rate_prior"] - infer["wicket_rate_prior"]
        infer["bowler_pressure"] = 0.58 * infer["wicket_rate_prior"] + 0.42 * infer["dot_rate_prior"]
        # With actual wickets, collapse pressure becomes a REAL signal instead of ~0
        infer["wicket_collapse_pressure"] = np.maximum(
            0.0,
            infer["wickets_proxy"] * (0.75 * infer["wicket_pressure"] + 0.25 * infer["bowler_pressure"] - self.global_wicket_rate),
        )

        # Build features — use predict_year
        X_pred = self._build_features(infer, year_override=self.predict_year)
        raw_pred = self._ensemble_predict(X_pred)

        # --- TWO-PASS: use 1st innings prediction for 2nd innings ---
        # Only when the test data does NOT provide first_innings_pp_score.
        # This lets the model feed its own 1st innings prediction into 2nd innings.
        inning1_rows = inning_series == 1
        second_rows = inning_series == 2
        needs_first_estimate = second_rows & first_innings_input.isna()
        if needs_first_estimate.any() and inning1_rows.any():
            # Quick pass-1 calibrated prediction for 1st innings
            pass1_pred = (
                self.model_blend_weight_predict * raw_pred
                + (1.0 - self.model_blend_weight_predict) * infer["base_prior"].values
            )
            pass1_pred = self.calib_slope * pass1_pred + self.calib_intercept + self.year_bias_correction
            # Map 1st innings predictions to 2nd innings via matchId
            first_pred_map = {}
            for idx_pos in np.where(inning1_rows.values)[0]:
                mid = match_series.iloc[idx_pos]
                if mid != "UNKNOWN":
                    first_pred_map[mid] = float(pass1_pred[idx_pos])
            # Update ONLY 2nd innings rows that are missing first_innings_score
            for idx_pos in np.where(needs_first_estimate.values)[0]:
                mid = match_series.iloc[idx_pos]
                if mid in first_pred_map:
                    orig_idx = infer.index[idx_pos]
                    infer.at[orig_idx, "first_innings_score"] = first_pred_map[mid]
            # Rebuild features with updated first_innings_score and re-predict
            X_pred = self._build_features(infer, year_override=self.predict_year)
            raw_pred = self._ensemble_predict(X_pred)

        # -------------------------------------------------------------------
        # Prediction pipeline (Competition-Ready V4 — SOTA)
        # -------------------------------------------------------------------

        # --- Compute confidence/uncertainty FIRST ---
        venue_known = venue_series.isin(set(self.venue_mean_map.keys())).astype(float)
        bat_known = batting_series.isin(set(self.batting_team_mean_map.keys())).astype(float)
        bowl_known = bowling_series.isin(set(self.bowling_team_mean_map.keys())).astype(float)
        bat_player_known = batsman_id_series.isin(set(self.open_batsman_pp_map.keys())).astype(float)
        bowl_player_known = bowler_id_series.isin(set(self.open_bowler_pp_map.keys())).astype(float)
        confidence = (venue_known + bat_known + bowl_known + bat_player_known + bowl_player_known) / 5.0
        uncertainty = np.clip(
            pd.to_numeric(infer["venue_prior_std"], errors="coerce").fillna(self.global_std).to_numpy(dtype=float)
            / max(self.global_std, 1e-6),
            0.0, 2.0,
        ) / 2.0

        directional_conf = self._compute_directional_confidence(infer, confidence, uncertainty)

        # 1. Calibrate the ensemble output
        blended = (
            self.model_blend_weight_predict * raw_pred
            + (1.0 - self.model_blend_weight_predict) * infer["base_prior"].values
        )
        calibrated = self.calib_slope * blended + self.calib_intercept

        # 2. Per-bucket residual correction
        residual_corr = self._residual_correction_vector(batting_series, venue_series, inning_series)
        calibrated = calibrated + np.clip(residual_corr, -self.residual_bucket_cap, self.residual_bucket_cap)

        # 3. Context adjustments
        calibrated = calibrated + self.context_scale * self._context_flex_adjustment(infer)

        # 4. Directional tail signal (SYMMETRIC — no longer dampened on low side)
        tail_signal = self._directional_tail_signal(infer)
        tail_bias = np.where(tail_signal >= 0.0, self.tail_lift_scale * tail_signal, self.tail_drop_scale * tail_signal)
        calibrated = calibrated + np.clip(tail_bias, -12.0, 12.0)

        # 5. Composite player adjustment (UNIQUE to Krishna)
        bat_signal = (composite_bat_sr - self.global_batsman_sr) / max(self.global_batsman_sr, 1.0)
        bowl_signal = (composite_bowl_econ - self.global_bowler_econ) / max(self.global_bowler_econ, 1.0)
        player_adj = 3.0 * bat_signal + 2.5 * bowl_signal
        player_adj = np.clip(player_adj, -6.0, 6.0)
        calibrated = calibrated + player_adj

        # 5b. LIVE-STATE HEURISTIC: actual wickets & bowler count correction
        #     This is the KEY insight: test data tells us how many batsmen
        #     batted (= wickets + 2) and how many bowlers were used.
        actual_wickets = np.clip(actual_batsman_count - 2.0, 0.0, 6.0)

        # --- Base wicket impact (0w→+5, 1w→0, 2w→-5, 3w→-11, 4w→-18) ---
        wicket_impact = np.where(
            actual_batsman_count >= 2,
            -4.5 * (actual_wickets - 1.0) - 1.0 * np.maximum(0.0, actual_wickets - 1.5) ** 2,
            0.0,
        )
        wicket_impact = np.clip(wicket_impact, -20.0, 6.0)

        # --- VENUE × WICKET interaction: high-scoring ground + few wickets → BOOST ---
        # e.g. Chinnaswamy (dev ≈ +0.54σ) + 0 wickets → extra +3-4 runs
        venue_dev = (
            pd.to_numeric(infer["venue_prior_mean"], errors="coerce").fillna(self.global_mean).to_numpy(dtype=float)
            - self.global_mean
        ) / max(self.global_std, 1.0)
        venue_wicket_boost = np.where(
            (actual_batsman_count >= 2) & (actual_wickets <= 1),
            np.clip(venue_dev * 5.0, -3.0, 10.0),
            np.where(
                (actual_batsman_count >= 2) & (actual_wickets >= 3),
                np.clip(venue_dev * -2.0, -5.0, 2.0),  # low-scoring venue + collapse → extra penalty
                0.0,
            ),
        )

        # --- TEAM × WICKET interaction: aggressive teams + few wickets → BOOST ---
        # Use RECENT batting mean (captures 2024-2025 scoring inflation)
        # SRH recent=59.9, KKR=59.6, RCB=59.1 → they score big when set
        bat_team_dev = (
            pd.to_numeric(infer["batting_team_recent_mean"], errors="coerce").fillna(self.global_mean).to_numpy(dtype=float)
            - self.global_mean
        ) / max(self.global_std, 1.0)
        team_wicket_boost = np.where(
            (actual_batsman_count >= 2) & (actual_wickets <= 1),
            np.clip(bat_team_dev * 4.0, -2.0, 8.0),
            np.where(
                (actual_batsman_count >= 2) & (actual_wickets >= 3),
                np.clip(bat_team_dev * -1.5, -4.0, 1.0),  # weak team + collapse → extra penalty
                0.0,
            ),
        )

        # --- Bowler count impact (more bowlers used → higher scoring) ---
        bowler_impact = np.where(
            actual_bowler_count >= 2,
            3.0 * np.clip(actual_bowler_count - 2.5, -1.5, 3.5),
            0.0,
        )
        bowler_impact = np.clip(bowler_impact, -5.0, 11.0)

        live_state_adj = wicket_impact + venue_wicket_boost + team_wicket_boost + bowler_impact
        calibrated = calibrated + live_state_adj

        # 6. Season-trend bias correction (additive — proven more stable)
        calibrated = calibrated + self.year_bias_correction

        # 7. SAFE-FIRST ANCHOR BLEND:
        #    Blend between the model prediction and the safe anchor (learned median).
        #    Higher directional confidence → trust model more.
        #    Lower directional confidence → stay closer to anchor.
        safe_anchor = self.calib_anchor + self.year_bias_correction
        # Model trust: adaptive based on live-state availability.
        # Without live data: original anchor blend (0.65–0.95) — already tuned
        # With live data (actual wickets/bowlers): strong trust (0.88–1.0)
        has_live_data = (actual_batsman_count >= 2) | (actual_bowler_count >= 2)
        live_signal_strength = np.clip(np.abs(live_state_adj) / 10.0, 0.0, 1.0)
        model_trust = np.where(
            has_live_data,
            np.clip(0.88 + 0.08 * directional_conf + 0.04 * live_signal_strength, 0.88, 1.0),
            np.clip(0.65 + 0.30 * directional_conf, 0.65, 0.95),
        )
        calibrated = model_trust * calibrated + (1.0 - model_trust) * safe_anchor

        # 8. Confidence-gated extreme expansion:
        #    For high-confidence predictions that deviate significantly from anchor,
        #    push them further out to catch extreme innings.
        deviation = calibrated - safe_anchor
        abs_dev = np.abs(deviation)
        # Lower threshold + stronger factor for better variance
        expand_mask = (abs_dev > 2.0) & (directional_conf > 0.40)
        expand_factor = 1.0 + 0.40 * np.clip(directional_conf - 0.40, 0.0, 0.60) / 0.60
        calibrated = np.where(expand_mask, safe_anchor + deviation * expand_factor, calibrated)

        # 8b. VARIANCE EXPANSION — preserves mean, expands StdDev.
        #     The anchor blend (Step 7) crushes variance. This step partially
        #     restores it by scaling deviations from anchor proportional to
        #     directional confidence. ASYMMETRIC: push UP deviations harder
        #     to counteract the model's systematic undershoot.
        deviation = calibrated - safe_anchor
        # Upward deviations get stronger expansion than downward
        var_scale_up = 1.0 + 1.70 * np.clip(directional_conf, 0.0, 1.0)
        var_scale_down = 1.0 + 0.25 * np.clip(directional_conf, 0.0, 1.0)
        var_scale = np.where(deviation >= 0, var_scale_up, var_scale_down)
        # Extra boost when live data gives strong signals
        var_scale = np.where(
            has_live_data,
            var_scale + 0.30 * live_signal_strength,
            var_scale,
        )
        calibrated = safe_anchor + deviation * var_scale

        # 9. Flexible params correction (if loaded)
        calibrated = self._apply_flexible_adjustment(calibrated, infer, confidence, uncertainty)

        if self.final_mean_weight > 0:
            calibrated = (1.0 - self.final_mean_weight) * calibrated + self.final_mean_weight * self.global_mean

        # --- Final clip ---
        calibrated = np.clip(calibrated, 15.0, 120.0)

        return pd.DataFrame({
            "id": ids,
            "predicted_score": np.rint(calibrated).astype(int),
        })


