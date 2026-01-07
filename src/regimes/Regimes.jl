# Regimes.jl - Submodule for regime classification
#
# Classify market regimes for conditional performance analysis.
#
# Knowledge Tiers:
#   [T1] Regime-switching theory (Hamilton 1989, 1994)
#   [T1] Rolling volatility as regime indicator (standard in finance)
#   [T2] Volatility of CHANGES not levels (BUG-005 fix in myga-v2)
#   [T3] 13-week window, 33/67 percentile thresholds

"""
    Regimes

Regime classification for conditional performance analysis.

This submodule provides:
- **Volatility regimes** (`classify_volatility_regime`): LOW/MED/HIGH by rolling std
- **Direction regimes** (`classify_direction_regime`): UP/DOWN/FLAT by threshold
- **Combined regimes** (`get_combined_regimes`): Vol × Direction labels
- **Stratified metrics** (`compute_stratified_metrics`): Per-regime evaluation

# Critical Implementation Note
Volatility MUST be computed on CHANGES (first differences), NOT levels.
Using levels mislabels steady drifts as "volatile":
- A series drifting steadily from 3.0 to 4.0 has high std of LEVELS
- But it has ZERO volatility of changes (constant increments)

# Knowledge Tier
[T1] Hamilton (1989, 1994). Regime-switching models.
[T2] Volatility basis='changes' correction: myga-forecasting-v2 BUG-005.
[T3] 13-week window assumes quarterly seasonality.
[T3] 33rd/67th percentiles for regime boundaries.

# Example
```julia
using TemporalValidation

# Classify volatility using changes (correct!)
vol_regimes = classify_volatility_regime(values; basis=:changes)

# Classify direction using thresholded signs
threshold = compute_move_threshold(train_actuals)
dir_regimes = classify_direction_regime(actuals, threshold)

# Combine for full regime labels
combined = get_combined_regimes(vol_regimes, dir_regimes)

# Compute stratified metrics
result = compute_stratified_metrics(predictions, actuals, vol_regimes)
println(summary(result))
```

# See Also
- `classify_volatility_regime`: Main volatility classification
- `compute_stratified_metrics`: Per-regime metrics
"""
module Regimes

using StatsBase: mean, std
using Statistics: quantile

# Import frozen constants from parent module
using ..TemporalValidation: VOLATILITY_WINDOW, VOLATILITY_LOW_PERCENTILE, VOLATILITY_HIGH_PERCENTILE

# Include implementation files
include("volatility.jl")   # classify_volatility_regime
include("direction.jl")    # classify_direction_regime, helpers
include("stratified.jl")   # StratifiedMetricsResult, compute_stratified_metrics

# =========================================================================
# Exports
# =========================================================================

# Volatility regime
export classify_volatility_regime

# Direction regime and helpers
export classify_direction_regime
export get_combined_regimes
export get_regime_counts
export mask_low_n_regimes

# Stratified metrics
export StratifiedMetricsResult
export compute_stratified_metrics

end # module Regimes
