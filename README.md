# 🏏 Sona Power Predict – 2026

Welcome to the official repository of **Team Pinnacle** for the **Sona Power Predict 2026** competition. This repository contains our high-performance machine learning model for predicting pre-innings IPL PowerPlay scores.

---

## 👥 Team & College Details

* **College Name:** Dayanand Sagar Academy of Technology and Management
* **Team Name:** Team Pinnacle
* **Team Members:**
  1. **Jigyasa Jaiswal**  
     * *College Year:* 2nd Year (Sophomore)  
     * *Department:* Computer Science and Engineering (Data Science)
  2. **Aadesh Mishra**  
     * *College Year:* 2nd Year (Sophomore)  
     * *Department:* Computer Science and Engineering

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
