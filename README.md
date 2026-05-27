# 🏏 Sona Power Predict – 2026

Welcome to the official repository of **Team Pinnacle** for the **Sona Power Predict 2026** competition. This repository contains our high-performance machine learning model for predicting pre-innings IPL PowerPlay scores.

---

## 👥 Team & College Details

* **College Name:** Dayanand Sagar Academy of Technology and Management
* **Team Name:** Team Pinnacle
* **Team Members:**
  1. **Jigyasa Jaiswal**  
     * *College Year:* 2nd Year 
     * *Department:* Computer Science and Engineering (Data Science)
  2. **Aadesh Mishra**  
     * *College Year:* 2nd Year  
     * *Department:* Computer Science and Engineering

---

## ⚙️ How the Model Actually Works

At a high level, the model reads raw ball-by-ball delivery data, aggregates it into PowerPlay innings, builds a rich set of historical features, trains a gradient-boosting ensemble, and then applies a multi-step post-processing pipeline before returning a final integer score prediction.

### Training (`fit`)

1. **Data ingestion.** Ball-by-ball deliveries are filtered to the first 6 overs. Each innings is then aggregated into a single row containing the total PowerPlay score, boundary rate, dot-ball rate, and wicket rate. Venue information is joined from a separate matches file, and a home-ground mapping is loaded from a text file.

2. **Leakage-safe feature computation.** All historical priors (venue mean, team mean, player mean, etc.) are computed using `shift(1).expanding().mean()` — meaning position *i* can only see data from matches *before* it. Recency-weighted variants use an exponentially weighted moving average (EWM) with the same shift to give more weight to recent form.

3. **Feature engineering.** Roughly 60 features are built per row, covering venue priors, team priors, player strike rates/economies, home advantage, dew factor, first-innings score, head-to-head priors, team score volatility, and a range of non-linear interaction terms (e.g. `bat_minus_bowl × aggression`, `venue × wicket_dynamics`).

4. **Ensemble training.** Four models are trained on the selected top-50 features (chosen by correlation with the target):
   - Two bagged LightGBM regressors with MAE loss (seeds 42 & 123)
   - Two bagged LightGBM regressors with Poisson loss
   - One `HistGradientBoosting` with absolute-error loss
   - One `HistGradientBoosting` with squared-error loss
   
   All are trained with recency-based sample weights that down-weight older seasons.

5. **Post-fit calibration.** After training, the model fits a linear output calibration on a held-out slice (last 30% of rows), a per-bucket residual correction map keyed on `(batting_team, venue, innings)`, and context gain coefficients via ordinary least squares on the training residuals.

---

### Prediction (`predict`)

Given a test row (venue, batting team, bowling team, innings, and optionally comma-separated player ID lists), the model runs this 9-step pipeline:

**Step 1 — Raw ensemble prediction.**  
Features are built from stored lookup maps and passed through the weighted ensemble. A two-pass strategy is used: 1st-innings predictions are fed back as the "first innings score" input for any 2nd-innings rows that don't already have it.

**Step 2 — Output calibration.**  
`prediction = slope × raw_pred + intercept`, where slope and intercept were fitted on the holdout slice during training.

**Step 3 — Residual bucket correction.**  
A small shrinkage correction (capped at ±6 runs) is added per `(batting_team, venue, innings)` bucket based on the model's historical error on that specific combination.

**Step 4 — Context-flex adjustment.**  
Learned coefficients for aggression, dew, home advantage, wicket collapse pressure, and first-innings target pressure are applied additively (clipped to ±15 runs).

**Step 5 — Directional tail signal.**  
A linear combination of aggression, boundary–dot spread, wicket collapse, and first-innings delta is computed. Positive signals get a `1.5×` lift; negative signals get a `0.85×` drop (asymmetric, to counteract the model's tendency to undershoot high scores).

**Step 6 — Composite player adjustment.**  
The strike rates of *all* listed batsmen and economies of *all* listed bowlers are averaged. Deviations from the global league average translate into a ±6-run adjustment.

**Step 7 — Live-state heuristic (key innovation).**  
The number of comma-separated player IDs in the test data encodes real match state: `wickets fallen = batsmen listed − 2`. This drives a calibrated wicket-impact curve (`0w → +5`, `1w → 0`, `2w → −5`, `3w → −11`, `4w → −18`) plus venue × wicket and team × wicket interaction boosts. Bowler count feeds a separate 0–11 run adjustment.

**Step 8 — Season trend correction.**  
A bias correction (1–12 runs, fitted from recent season slope) is added to account for year-over-year IPL scoring inflation.

**Step 9 — Anchor blend + variance expansion.**  
The prediction is blended with a "safe anchor" (the learned target median + bias correction). The blend weight is directional-confidence-driven (0.65–1.0). Deviations from the anchor are then asymmetrically re-scaled (1.7× upward, 0.25× downward) to restore variance that the anchor blend collapses.

The final output is clipped to [15, 120] and rounded to the nearest integer.

---

## 🏗️ Model Architecture & Approach

Our model, **Competition Edition v4 (SOTA)**, is designed to predict pre-innings IPL PowerPlay scores (first 6 overs) with extreme accuracy and robustness. The solution combines a diverse gradient boosting ensemble with a sophisticated context-aware post-processing pipeline.

```
┌───────────────────────────────────────────────┐
│          4-MODEL WEIGHTED ENSEMBLE            │
├───────────────────────────────────────────────┤
│  LightGBM MAE (2-seed bag) │  weight: 0.30   │
│  LightGBM Poisson (2-seed) │  weight: 0.35   │
│  HGB Absolute Error        │  weight: 0.20   │
│  HGB Squared Error          │  weight: 0.15   │
├───────────────────────────────────────────────┤
│       9-STEP POST-PROCESSING PIPELINE         │
│  1. Output Calibration                        │
│  2. Residual Bucket Correction                │
│  3. Context-Flex Adjustment                   │
│  4. Directional Tail Signal                   │
│  5. Composite Player Strike Rate/Econ         │
│  6. Season Trend (Scoring Inflation)          │
│  7. Safe-First Anchor Blend                   │
│  8. Confidence-Gated Variance Expansion       │
│  9. Flexible Gated Adjustment                 │
└───────────────────────────────────────────────┘
```

### 1. Advanced Feature Engineering (60+ Features)
* **Venue Priors:** Leakage-safe expanding historical means/std (`shift(1)`), exponentially weighted moving averages for recent trends, and boundary-run percentages.
* **Team Metrics:** Batting and bowling team historical averages, recent form over past games, and team-level score volatility.
* **Player Statistics:** Average strike rates of all listed batsmen and average economy of all listed bowlers.
* **Context & Match Conditions:** Home ground advantage boost, dew factor estimates for 2nd innings, and first innings score constraints.
* **Interaction Features:** Complex non-linear features like `bat_minus_bowl_x_aggression`, `venue_x_aggression`, and `volatility_x_aggression`.

### 2. Robust Multi-Loss Ensemble
We employ a robust, multi-objective ensemble using:
* **LightGBM Regressor** with Mean Absolute Error (L1) loss (bagged over multiple seeds for stability).
* **LightGBM Regressor** with Poisson loss to handle count data.
* **HistGradientBoosting Regressor** with Absolute Error loss.
* **HistGradientBoosting Regressor** with Squared Error loss.

This combination enables the model to balance the median prediction with the overall expected value, minimizing outliers.

### 3. The 9-Step Post-Processing Pipeline
To bridge the gap between pure statistical predictions and real-world cricket dynamics:
1. **Output Calibration:** Corrects distribution drift using fitted holdout parameters.
2. **Residual Bucket Correction:** Applies localized corrections for specific `Team × Venue × Innings` combinations based on training error history.
3. **Context-Flex Adjustment:** Modulates the score based on dew, aggression index, home advantage, and first innings target pressure.
4. **Directional Tail Signal:** Adjusts predictions near the extremes when multiple features agree on direction.
5. **Composite Player Adjustment:** Integrates overall player statistics of the active XI (strike rates/economies) relative to historical league averages.
6. **Season Trend Correction:** Implements robust drift estimation to account for year-over-year IPL scoring inflation.
7. **Safe-First Anchor Blend:** Adaptive blend between prediction and learned safe anchor based on directional confidence.
8. **Asymmetric Variance Expansion:** Safely expands prediction variance (1.7× upward vs 0.25× downward) to capture high-scoring extremes without undershooting.
9. **Flexible Correction Head:** Gated neural-like adjustment driven by custom risk parameters.

---

## 🛠️ Tech Stack & Libraries Used

The model uses only optimized, competition-compliant open-source libraries:
* **Python** (Core Logic)
* **pandas** (Data Manipulation & Processing)
* **numpy** (Vectorized Calculations & Numerical Operations)
* **scikit-learn** (HistGradientBoosting, Metrics, & Preprocessing)
* **lightgbm** (Gradient Boosting Ensemble Models)

---

## 🚀 Key Innovations

1. **Composite Player Stats:** Unlike naive models that look only at the opening batsman/bowler, our model computes composite metrics across all listed squad members.
2. **Live-State Inference:** Automatically deduces actual wickets fallen from player list counts (Actual Batsmen = Wickets + 2) to dynamically adjust prediction mid-game.
3. **Venue × Wicket Dynamics:** Dynamically scales score boosts depending on the venue scoring potential combined with the wickets lost (e.g., highly aggressive team + 0 wickets at a high-scoring venue triggers a massive, calibrated boost).
4. **Asymmetric Scaling:** Counteracts systematic model regression-to-the-mean by selectively boosting high-side predictions while guarding the downside.
