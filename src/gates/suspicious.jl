# suspicious.jl - Suspicious improvement detection gate
#
# Detects implausibly large improvements over baseline, which often indicate
# data leakage or evaluation errors.

"""
    gate_suspicious_improvement(model_metric, baseline_metric; kwargs...) -> GateResult

Detect implausibly large improvements over baseline (leakage indicator).

A model that dramatically outperforms a persistence baseline (e.g., >20% improvement)
should trigger investigation for data leakage, lookahead bias, or evaluation errors.

# Arguments
- `model_metric::Real`: Model's error metric (lower = better, e.g., MAE, RMSE)
- `baseline_metric::Real`: Baseline error metric (e.g., persistence forecast)

# Keyword Arguments
- `threshold::Float64=$(SUSPICIOUS_IMPROVEMENT_HALT)`: HALT if improvement exceeds this (default 20%)
- `warn_threshold::Float64=$(SUSPICIOUS_IMPROVEMENT_WARN)`: WARN if improvement exceeds this (default 10%)
- `metric_name::String="MAE"`: Name of metric for reporting

# Returns
- `GateResult` with status:
  - `HALT`: improvement > threshold (>20% default) - likely leakage
  - `WARN`: warn_threshold < improvement <= threshold (10-20% default) - proceed with caution
  - `PASS`: improvement <= warn_threshold (<10% default) - acceptable
  - `SKIP`: baseline_metric <= 0 (cannot compute improvement ratio)

# Formula
```
improvement = 1 - (model_metric / baseline_metric)
```

# Example
```julia
baseline_mae = 0.15  # Persistence baseline
model_mae = 0.10     # Our model

result = gate_suspicious_improvement(model_mae, baseline_mae)
# improvement = 1 - 0.10/0.15 = 0.333 (33%)
# result.status == HALT (exceeds 20% threshold)
```

# Knowledge Tier
[T3] The 20%/10% thresholds are empirical heuristics from myga-forecasting-v2 postmortem.
For highly persistent series, even small improvements over persistence can be suspicious.

# See Also
- `gate_synthetic_ar1()`: Test against theoretical AR(1) bounds
- `gate_shuffled_target()`: Test if model exploits target position
"""
function gate_suspicious_improvement(
    model_metric::Real,
    baseline_metric::Real;
    threshold::Float64 = SUSPICIOUS_IMPROVEMENT_HALT,
    warn_threshold::Float64 = SUSPICIOUS_IMPROVEMENT_WARN,
    metric_name::String = "MAE"
)::GateResult

    # Handle degenerate case: baseline ≤ 0 means we can't compute improvement ratio
    if baseline_metric <= 0
        return GateResult(
            name = :suspicious_improvement,
            status = SKIP,
            message = "Baseline $metric_name ≤ 0 ($baseline_metric), cannot compute improvement ratio",
            details = Dict{Symbol, Any}(
                Symbol("model_$(lowercase(metric_name))") => model_metric,
                Symbol("baseline_$(lowercase(metric_name))") => baseline_metric
            )
        )
    end

    # Compute improvement ratio: positive means model is better than baseline
    # improvement = 1 - (model/baseline)
    # If model_metric = 0.10 and baseline = 0.15, improvement = 1 - 0.667 = 0.333 (33%)
    improvement = 1.0 - (model_metric / baseline_metric)

    # Build details dict
    details = Dict{Symbol, Any}(
        Symbol("model_$(lowercase(metric_name))") => model_metric,
        Symbol("baseline_$(lowercase(metric_name))") => baseline_metric,
        :improvement_ratio => improvement
    )

    # Decision logic
    if improvement > threshold
        return GateResult(
            name = :suspicious_improvement,
            status = HALT,
            message = "$(round(improvement*100, digits=1))% improvement over baseline exceeds $(round(threshold*100))% HALT threshold",
            metric_value = improvement,
            threshold = threshold,
            details = details,
            recommendation = "Check for data leakage in feature engineering. " *
                           "Verify gap >= horizon in CV. " *
                           "Test with gate_shuffled_target() and gate_synthetic_ar1()."
        )
    elseif improvement > warn_threshold
        return GateResult(
            name = :suspicious_improvement,
            status = WARN,
            message = "$(round(improvement*100, digits=1))% improvement exceeds $(round(warn_threshold*100))% WARN threshold",
            metric_value = improvement,
            threshold = threshold,
            details = details,
            recommendation = "Verify improvement with external holdout data. " *
                           "Consider additional leakage tests."
        )
    else
        return GateResult(
            name = :suspicious_improvement,
            status = PASS,
            message = "$(round(improvement*100, digits=1))% improvement is within acceptable range",
            metric_value = improvement,
            threshold = threshold,
            details = details,
            recommendation = ""
        )
    end
end
