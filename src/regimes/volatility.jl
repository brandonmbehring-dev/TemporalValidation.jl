# volatility.jl - Volatility regime classification
#
# CRITICAL: Volatility must be computed on CHANGES (first differences), NOT levels.
#
# Knowledge Tiers:
#   [T1] Rolling volatility as regime indicator (standard in finance)
#   [T2] Volatility of CHANGES not levels (BUG-005 fix)
#   [T3] 13-week window, 33/67 percentile thresholds

using StatsBase: std
using Statistics: quantile

"""
    classify_volatility_regime(values; kwargs...) -> Vector{Symbol}

Classify volatility regime for each point using rolling window.

CRITICAL: Default `basis=:changes` computes volatility on first differences,
which is the methodologically correct approach. Using `basis=:levels` is
provided for legacy comparison only.

# Arguments
- `values::AbstractVector{<:Real}`: Time series values.
  If basis=:changes, these should be levels (differences will be computed).
  If basis=:levels, used directly.

# Keyword Arguments
- `window::Int=13`: Rolling window for volatility calculation (13 weeks ~ 1 quarter) [T3]
- `basis::Symbol=:changes`: Volatility computation basis
  - `:changes` (correct): Compute volatility on first differences
  - `:levels` (legacy): Compute volatility on raw values
- `low_percentile::Float64=33.0`: Percentile threshold for LOW volatility [T3]
- `high_percentile::Float64=67.0`: Percentile threshold for HIGH volatility [T3]

# Returns
Vector of regime symbols: `:LOW`, `:MED`, `:HIGH` for each point.
Points with insufficient history are labeled `:MED`.

# Algorithm
1. If basis=:changes: compute diff(values), pad first with NaN
2. Rolling window std with ddof=1 (sample std)
3. Compute percentiles from valid volatilities
4. Classify: vol ≤ p33 → :LOW, p33 < vol ≤ p67 → :MED, vol > p67 → :HIGH

# Why Changes, Not Levels? [T2]
The correct pattern is volatility of CHANGES, not levels:
- Steady drift (constant increases) has LOW volatility of changes
- But HIGH std of levels (increasing values)

This fixes BUG-005 from myga-forecasting-v2 where using levels led to
mislabeling steady drifts as volatile.

# Notes
Thresholds are computed from the full series, which is appropriate when
classifying a single evaluation period. For walk-forward with multiple
test periods, use training-only thresholds to prevent leakage.

# Knowledge Tier
[T1] Rolling volatility (standard in finance literature).
[T2] Volatility of changes (BUG-005 fix).
[T3] 13-week window, 33/67 percentile thresholds.

# Example
```julia
values = cumsum(randn(200) * 0.01) .+ 3.0
regimes = classify_volatility_regime(values; window=13, basis=:changes)

# Count by regime
for r in [:LOW, :MED, :HIGH]
    n = count(==(r), regimes)
    println("\$r: \$n (\$(round(100*n/length(regimes), digits=1))%)")
end
```

# See Also
- `classify_direction_regime`: Classify by direction (UP/DOWN/FLAT).
- `compute_stratified_metrics`: Run metrics per-regime.
"""
function classify_volatility_regime(
    values::AbstractVector{<:Real};
    window::Int = VOLATILITY_WINDOW,
    basis::Symbol = :changes,
    low_percentile::Float64 = VOLATILITY_LOW_PERCENTILE,
    high_percentile::Float64 = VOLATILITY_HIGH_PERCENTILE
)::Vector{Symbol}
    n = length(values)

    # Validate inputs
    if window < 2
        throw(ArgumentError("window must be >= 2, got $window"))
    end

    if !(0 < low_percentile < high_percentile <= 100)
        throw(ArgumentError(
            "Invalid percentiles: need 0 < low_percentile < high_percentile <= 100. " *
            "Got low=$low_percentile, high=$high_percentile."
        ))
    end

    if basis ∉ (:changes, :levels)
        throw(ArgumentError("basis must be :changes or :levels, got $basis"))
    end

    # Warn about near-zero data with levels basis
    if basis == :levels && std(values) < 1e-8
        @warn "Very small std ($(round(std(values), sigdigits=3))) with basis=:levels. " *
              "Regime classification may be degenerate. " *
              "Consider basis=:changes or normalizing data."
    end

    # Handle insufficient data
    if n < window + 1
        # Return all :MED for insufficient data
        return fill(:MED, n)
    end

    # Compute series for volatility calculation
    if basis == :changes
        # Compute first differences
        series_for_vol = diff(Float64.(values))
        # Pad with NaN to maintain alignment (first point has no change)
        series_for_vol = vcat(NaN, series_for_vol)
    else
        series_for_vol = Float64.(values)
    end

    # Compute rolling volatility using sample std
    rolling_vol = fill(NaN, n)

    for i in window:n
        window_data = series_for_vol[(i - window + 1):i]
        # Skip if any NaN in window
        if !any(isnan, window_data)
            rolling_vol[i] = std(window_data; corrected=true)  # ddof=1
        end
    end

    # Get valid volatility values for threshold computation
    valid_vol = rolling_vol[.!isnan.(rolling_vol)]

    if isempty(valid_vol)
        return fill(:MED, n)
    end

    # Compute thresholds
    vol_low = quantile(valid_vol, low_percentile / 100.0)
    vol_high = quantile(valid_vol, high_percentile / 100.0)

    # Classify each point
    regimes = Vector{Symbol}(undef, n)

    for i in 1:n
        vol = rolling_vol[i]
        if isnan(vol)
            regimes[i] = :MED  # Default for insufficient history
        elseif vol <= vol_low
            regimes[i] = :LOW
        elseif vol <= vol_high
            regimes[i] = :MED
        else
            regimes[i] = :HIGH
        end
    end

    return regimes
end
