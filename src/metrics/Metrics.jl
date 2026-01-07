# Metrics.jl - Submodule for forecast evaluation metrics
#
# Provides core metrics, move-conditional metrics (MC-SS), and direction accuracy.
#
# Knowledge Tiers:
#   [T1] Standard statistical metrics (MAE, RMSE, MASE, etc.)
#   [T2] Move-conditional metrics from myga-forecasting-v2 (MC-SS)
#   [T3] Implementation choices (scale-aware epsilon, reliability threshold)

"""
    Metrics

Forecast evaluation metrics for time-series validation.

This submodule provides:
- **Core metrics** (`compute_mae`, `compute_rmse`, etc.): Standard point forecast metrics [T1]
- **Scale-free metrics** (`compute_mase`, `compute_theils_u`): Baseline comparison [T1]
- **Move-conditional** (`compute_move_conditional_metrics`): MC-SS for high-persistence [T2]
- **Direction accuracy** (`compute_direction_accuracy`): 2-class and 3-class modes

# Knowledge Tier
[T1] Hyndman & Koehler (2006). Another look at measures of forecast accuracy.
[T1] Theil (1966). Applied Economic Forecasting.
[T2] MC-SS formula from myga-forecasting-v2 Phase 11.
[T3] 70th percentile threshold, 10-sample reliability minimum.

# Example
```julia
using TemporalValidation

# Core metrics
mae = compute_mae(predictions, actuals)
rmse = compute_rmse(predictions, actuals)

# Scale-free comparison
naive_mae = compute_naive_error(train_actuals)
mase = compute_mase(predictions, actuals, naive_mae)
println("MASE: \$mase (< 1 = beats naive)")

# Move-conditional for high-persistence
threshold = compute_move_threshold(train_actuals)
mc = compute_move_conditional_metrics(predictions, actuals; threshold=threshold)
println("MC-SS: \$(mc.skill_score)")
if mc.is_reliable
    println("Result is reliable (n_up ≥ 10, n_down ≥ 10)")
end
```

# See Also
- `compute_naive_error`: Baseline for MASE normalization
- `MoveConditionalResult`: Full move-conditional results
"""
module Metrics

using StatsBase: mean, std, cor
using Statistics: quantile

# Import frozen constants from parent module
using ..TemporalValidation: MOVE_THRESHOLD_PERCENTILE

# Include implementation files
include("core.jl")       # Core metrics (MAE, RMSE, etc.)
include("persistence.jl") # Move-conditional (MC-SS)
include("direction.jl")   # Direction accuracy

# =========================================================================
# Exports
# =========================================================================

# Core metrics
export compute_mae, compute_mse, compute_rmse
export compute_mape, compute_smape, compute_bias
export compute_mase, compute_mrae, compute_theils_u
export compute_r_squared, compute_forecast_correlation
export compute_naive_error

# Move-conditional
export MoveDirection, UP, DOWN, FLAT
export MoveConditionalResult
export compute_move_threshold, classify_moves
export compute_move_conditional_metrics
export is_reliable, n_total, n_moves, move_fraction

# Direction accuracy
export compute_direction_accuracy
export compute_move_only_mae, compute_persistence_mae

end # module Metrics
