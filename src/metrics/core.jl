# core.jl - Core forecast evaluation metrics
#
# Standard metrics for point forecast evaluation.
# All formulas are [T1] academically validated.
#
# Knowledge Tiers:
#   [T1] All metrics are standard statistical formulations.
#   [T1] MASE: Hyndman & Koehler (2006).
#   [T1] Theil's U: Theil (1966).

using StatsBase: mean, std, cor

# =============================================================================
# Input Validation
# =============================================================================

"""
    _validate_metric_inputs(predictions, actuals, name) -> (Vector, Vector)

Internal helper to validate and convert inputs.

Checks:
- No NaN values
- Equal lengths
- Converts to Float64 vectors

# Throws
- `ArgumentError` if validation fails
"""
function _validate_metric_inputs(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    name::String
)
    if length(predictions) != length(actuals)
        throw(ArgumentError(
            "$name: predictions and actuals must have same length. " *
            "Got $(length(predictions)) and $(length(actuals))."
        ))
    end

    if any(isnan, predictions)
        throw(ArgumentError(
            "$name: predictions contain NaN values. " *
            "Clean data before computing metrics."
        ))
    end

    if any(isnan, actuals)
        throw(ArgumentError(
            "$name: actuals contain NaN values. " *
            "Clean data before computing metrics."
        ))
    end

    return Float64.(predictions), Float64.(actuals)
end


# =============================================================================
# Point Forecast Metrics
# =============================================================================

"""
    compute_mae(predictions, actuals) -> Float64

Compute Mean Absolute Error.

MAE = mean(|ŷ - y|)

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
Mean absolute error (non-negative).

# Knowledge Tier
[T1] Standard error metric.

# Example
```julia
mae = compute_mae([1.0, 2.0, 3.0], [1.1, 2.2, 2.8])
```
"""
function compute_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_mae")

    if isempty(preds)
        return NaN
    end

    return mean(abs.(preds .- acts))
end


"""
    compute_mse(predictions, actuals) -> Float64

Compute Mean Squared Error.

MSE = mean((ŷ - y)²)

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
Mean squared error (non-negative).

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_mse(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_mse")

    if isempty(preds)
        return NaN
    end

    return mean((preds .- acts) .^ 2)
end


"""
    compute_rmse(predictions, actuals) -> Float64

Compute Root Mean Squared Error.

RMSE = √(mean((ŷ - y)²))

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
Root mean squared error (non-negative).

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_rmse(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    return sqrt(compute_mse(predictions, actuals))
end


"""
    compute_mape(predictions, actuals; epsilon=1e-8) -> Float64

Compute Mean Absolute Percentage Error.

MAPE = 100 × mean(|ŷ - y| / max(|y|, ε))

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values
- `epsilon::Float64=1e-8`: Small value to prevent division by zero

# Returns
MAPE as percentage (0-100+). Not bounded above.

# Notes
MAPE has known issues:
- Undefined when actuals = 0 (mitigated by epsilon)
- Asymmetric: penalizes over-prediction more
- Unbounded above 100%

Consider `compute_smape` for a bounded alternative.

# Knowledge Tier
[T1] Standard percentage error metric.
"""
function compute_mape(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    epsilon::Float64 = 1e-8
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_mape")

    if isempty(preds)
        return NaN
    end

    denom = max.(abs.(acts), epsilon)
    return 100.0 * mean(abs.(preds .- acts) ./ denom)
end


"""
    compute_smape(predictions, actuals) -> Float64

Compute Symmetric Mean Absolute Percentage Error.

SMAPE = 100 × mean(2 × |ŷ - y| / (|ŷ| + |y|))

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
SMAPE as percentage (bounded 0-200%).
When both prediction and actual are zero, that observation is excluded.

# Knowledge Tier
[T1] Armstrong (1985) symmetric alternative to MAPE.
"""
function compute_smape(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_smape")

    if isempty(preds)
        return NaN
    end

    denom = abs.(preds) .+ abs.(acts)
    # Exclude cases where both are zero
    mask = denom .> 0

    if !any(mask)
        return 0.0
    end

    numerator = 2.0 .* abs.(preds[mask] .- acts[mask])
    return 100.0 * mean(numerator ./ denom[mask])
end


"""
    compute_bias(predictions, actuals) -> Float64

Compute mean signed error (bias).

Bias = mean(ŷ - y)

Positive bias indicates over-prediction on average.
Negative bias indicates under-prediction on average.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
Mean signed error.

# Knowledge Tier
[T1] Standard error metric.
"""
function compute_bias(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_bias")

    if isempty(preds)
        return NaN
    end

    return mean(preds .- acts)
end


# =============================================================================
# Scale-Invariant Metrics
# =============================================================================

"""
    compute_naive_error(values; method=:persistence) -> Float64

Compute naive forecast MAE for scale normalization.

Used as denominator for MASE and other scale-free metrics.

# Arguments
- `values::AbstractVector{<:Real}`: Training series values
- `method::Symbol=:persistence`: Naive forecast method
  - `:persistence`: y[t] = y[t-1] (random walk)
  - `:mean`: y[t] = mean(y)

# Returns
MAE of naive forecast on training data.

# Notes
For persistence: MAE = mean(|y[t] - y[t-1]|) for t = 2, ..., n
This represents the "cost of being naive" for MASE normalization.

# Knowledge Tier
[T1] Hyndman & Koehler (2006).

# Example
```julia
naive_mae = compute_naive_error(train_values)
mase = compute_mase(predictions, actuals, naive_mae)
```
"""
function compute_naive_error(
    values::AbstractVector{<:Real};
    method::Symbol = :persistence
)::Float64
    vals = Float64.(values)

    if length(vals) < 2
        throw(ArgumentError("compute_naive_error requires at least 2 values"))
    end

    if method == :persistence
        # y[t] - y[t-1] for t >= 2
        return mean(abs.(diff(vals)))
    elseif method == :mean
        mean_val = mean(vals)
        return mean(abs.(vals .- mean_val))
    else
        throw(ArgumentError("method must be :persistence or :mean, got $method"))
    end
end


"""
    compute_mase(predictions, actuals, naive_mae) -> Float64

Compute Mean Absolute Scaled Error.

MASE = MAE / naive_MAE

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values
- `naive_mae::Float64`: MAE of naive forecast on training data
  (compute with `compute_naive_error(train_values)`)

# Returns
MASE value:
- < 1: Model beats naive forecast
- = 1: Model equals naive forecast
- > 1: Model worse than naive forecast

# Notes
MASE is scale-free and can be used to compare accuracy across
different time series with different scales.

# Knowledge Tier
[T1] Hyndman & Koehler (2006).

# Example
```julia
naive_mae = compute_naive_error(train_values)
mase = compute_mase(predictions, actuals, naive_mae)
if mase < 1.0
    println("Model beats naive baseline")
end
```
"""
function compute_mase(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    naive_mae::Float64
)::Float64
    if naive_mae <= 0
        throw(ArgumentError("naive_mae must be positive, got $naive_mae"))
    end

    mae = compute_mae(predictions, actuals)
    return mae / naive_mae
end


"""
    compute_mrae(predictions, actuals, naive_predictions) -> Float64

Compute Mean Relative Absolute Error.

MRAE = mean(|ŷ - y| / |y_naive - y|)

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `naive_predictions::AbstractVector{<:Real}`: Naive/baseline predictions

# Returns
MRAE value (< 1 means better than naive).
Points where naive error = 0 are excluded.

# Notes
MRAE compares each error to the naive error at that point.

# Knowledge Tier
[T1] Standard relative error metric.
"""
function compute_mrae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    naive_predictions::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_mrae")
    naive = Float64.(naive_predictions)

    if length(naive) != length(preds)
        throw(ArgumentError(
            "compute_mrae: naive_predictions must have same length as predictions. " *
            "Got $(length(naive)) and $(length(preds))."
        ))
    end

    model_errors = abs.(preds .- acts)
    naive_errors = abs.(naive .- acts)

    # Exclude points where naive error is zero
    mask = naive_errors .> 0

    if !any(mask)
        return NaN
    end

    return mean(model_errors[mask] ./ naive_errors[mask])
end


"""
    compute_theils_u(predictions, actuals; naive_predictions=nothing) -> Float64

Compute Theil's U statistic.

U = RMSE(model) / RMSE(naive)

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `naive_predictions::Union{AbstractVector{<:Real}, Nothing}=nothing`:
  Naive predictions. If nothing, uses persistence (y[t-1]).

# Returns
Theil's U:
- < 1: Model outperforms naive
- = 1: Model equals naive
- > 1: Model worse than naive

# Notes
This is Theil's U2 (1966), comparing to a naive forecast.
When using persistence baseline, the first observation is excluded.

# Knowledge Tier
[T1] Theil (1966).

# Example
```julia
u = compute_theils_u(predictions, actuals)
if u < 1.0
    println("Model beats persistence baseline")
end
```
"""
function compute_theils_u(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    naive_predictions::Union{AbstractVector{<:Real}, Nothing} = nothing
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_theils_u")

    if isnothing(naive_predictions)
        # Use persistence: y[t-1] as prediction for y[t]
        if length(preds) < 2
            throw(ArgumentError("Need at least 2 observations for persistence baseline"))
        end
        # y[0], y[1], ..., y[n-2] predict y[1], ..., y[n-1]
        naive = acts[1:end-1]
        preds_trimmed = preds[2:end]
        acts_trimmed = acts[2:end]

        model_rmse = sqrt(mean((preds_trimmed .- acts_trimmed) .^ 2))
        naive_rmse = sqrt(mean((naive .- acts_trimmed) .^ 2))
    else
        naive = Float64.(naive_predictions)

        if length(naive) != length(preds)
            throw(ArgumentError(
                "naive_predictions must have same length as predictions. " *
                "Got $(length(naive)) and $(length(preds))."
            ))
        end

        model_rmse = sqrt(mean((preds .- acts) .^ 2))
        naive_rmse = sqrt(mean((naive .- acts) .^ 2))
    end

    if naive_rmse == 0
        return model_rmse > 0 ? Inf : 1.0
    end

    return model_rmse / naive_rmse
end


# =============================================================================
# Correlation Metrics
# =============================================================================

"""
    compute_forecast_correlation(predictions, actuals; method=:pearson) -> Float64

Compute correlation between predictions and actuals.

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values
- `method::Symbol=:pearson`: Correlation method
  - `:pearson`: Pearson correlation
  - `:spearman`: Spearman rank correlation

# Returns
Correlation coefficient [-1, 1].

# Notes
Correlation measures association but not accuracy. A model can
have high correlation but large errors (wrong scale/offset).

# Knowledge Tier
[T1] Standard statistical measures.
"""
function compute_forecast_correlation(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    method::Symbol = :pearson
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_forecast_correlation")

    if length(preds) < 2
        return NaN
    end

    if method == :pearson
        return cor(preds, acts)
    elseif method == :spearman
        # Spearman = Pearson on ranks
        ranks_pred = sortperm(sortperm(preds))
        ranks_act = sortperm(sortperm(acts))
        return cor(Float64.(ranks_pred), Float64.(ranks_act))
    else
        throw(ArgumentError("method must be :pearson or :spearman, got $method"))
    end
end


"""
    compute_r_squared(predictions, actuals) -> Float64

Compute R² (coefficient of determination).

R² = 1 - SS_res / SS_tot

Where:
- SS_res = sum((y - ŷ)²)  [residual sum of squares]
- SS_tot = sum((y - mean(y))²)  [total sum of squares]

# Arguments
- `predictions::AbstractVector{<:Real}`: Predicted values
- `actuals::AbstractVector{<:Real}`: Actual values

# Returns
R² value:
- = 1: Perfect predictions
- = 0: Model equals mean forecast
- < 0: Model worse than mean forecast (possible for out-of-sample)

# Knowledge Tier
[T1] Standard statistical measure.

# Example
```julia
r2 = compute_r_squared(predictions, actuals)
if r2 < 0
    @warn "Model performs worse than predicting the mean"
end
```
"""
function compute_r_squared(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real}
)::Float64
    preds, acts = _validate_metric_inputs(predictions, actuals, "compute_r_squared")

    if isempty(preds)
        return NaN
    end

    ss_res = sum((acts .- preds) .^ 2)
    ss_tot = sum((acts .- mean(acts)) .^ 2)

    if ss_tot == 0
        # All actuals are the same
        return ss_res == 0 ? 1.0 : -Inf
    end

    return 1.0 - ss_res / ss_tot
end
