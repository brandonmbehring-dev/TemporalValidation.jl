# direction.jl - Direction regime classification and helpers
#
# Classify direction using thresholded signs and provide helper functions
# for regime analysis.
#
# Knowledge Tiers:
#   [T2] 3-class direction enables fair persistence comparison (myga-v2)
#   [T3] Minimum n=10 per regime for reliability

using StatsBase: countmap

"""
    classify_direction_regime(values, threshold) -> Vector{Symbol}

Classify direction using thresholded signs.

This makes persistence (predicts 0) a meaningful baseline for
direction accuracy metrics.

# Arguments
- `values::AbstractVector{<:Real}`: Values to classify (typically actual changes)
- `threshold::Float64`: Move threshold (typically 70th percentile of |actuals| from training)

# Returns
Direction labels: `:UP`, `:DOWN`, `:FLAT`

# Classification Rules
- |value| > threshold AND value > 0 → `:UP`
- |value| > threshold AND value < 0 → `:DOWN`
- |value| <= threshold → `:FLAT`

# Notes
Using thresholded signs instead of raw signs provides a fair baseline for
persistence model comparison. Without threshold, persistence (predicts 0)
gets 0% direction accuracy, making all comparisons trivially "significant".

# Knowledge Tier
[T2] 3-class direction from myga-forecasting-v2.

# Example
```julia
actuals = [0.1, -0.1, 0.02, -0.02, 0.0]
directions = classify_direction_regime(actuals, 0.05)
# Returns [:UP, :DOWN, :FLAT, :FLAT, :FLAT]
```

# See Also
- `classify_volatility_regime`: Classify by volatility level.
- `compute_move_threshold`: Compute threshold from training data.
"""
function classify_direction_regime(
    values::AbstractVector{<:Real},
    threshold::Float64
)::Vector{Symbol}
    if threshold < 0
        throw(ArgumentError("threshold must be non-negative, got $threshold"))
    end

    n = length(values)
    directions = Vector{Symbol}(undef, n)

    for (i, v) in enumerate(values)
        if abs(v) <= threshold
            directions[i] = :FLAT
        elseif v > 0
            directions[i] = :UP
        else
            directions[i] = :DOWN
        end
    end

    return directions
end


"""
    get_combined_regimes(vol_regimes, dir_regimes) -> Vector{Symbol}

Combine volatility and direction into single label.

# Arguments
- `vol_regimes::AbstractVector{Symbol}`: Volatility regime labels (:LOW, :MED, :HIGH)
- `dir_regimes::AbstractVector{Symbol}`: Direction regime labels (:UP, :DOWN, :FLAT)

# Returns
Combined labels like `:HIGH_UP`, `:LOW_FLAT`, etc.

# Notes
Combined regimes can have very low sample counts in some cells.
Always check sample counts before interpreting conditional performance.

# Example
```julia
vol = [:HIGH, :LOW, :MED]
dir = [:UP, :DOWN, :FLAT]
combined = get_combined_regimes(vol, dir)
# Returns [:HIGH_UP, :LOW_DOWN, :MED_FLAT]
```
"""
function get_combined_regimes(
    vol_regimes::AbstractVector{Symbol},
    dir_regimes::AbstractVector{Symbol}
)::Vector{Symbol}
    if length(vol_regimes) != length(dir_regimes)
        throw(ArgumentError(
            "Arrays must have same length. " *
            "vol_regimes: $(length(vol_regimes)), dir_regimes: $(length(dir_regimes))"
        ))
    end

    return [Symbol(string(v) * "_" * string(d)) for (v, d) in zip(vol_regimes, dir_regimes)]
end


"""
    get_regime_counts(regimes) -> Dict{Symbol, Int}

Get sample counts per regime.

# Arguments
- `regimes::AbstractVector{Symbol}`: Regime labels

# Returns
Dictionary mapping regime to count, sorted by count descending.

# Example
```julia
regimes = [:HIGH, :LOW, :LOW, :MED, :LOW]
counts = get_regime_counts(regimes)
# Returns Dict(:LOW => 3, :HIGH => 1, :MED => 1)
```
"""
function get_regime_counts(regimes::AbstractVector{Symbol})::Dict{Symbol, Int}
    counts = countmap(regimes)

    # Sort by count descending
    sorted_pairs = sort(collect(counts); by=x -> x[2], rev=true)
    return Dict(sorted_pairs)
end


"""
    mask_low_n_regimes(regimes; min_n=10, mask_value=:MASKED) -> Vector{Symbol}

Mask regime labels with insufficient samples.

# Arguments
- `regimes::AbstractVector{Symbol}`: Regime labels
- `min_n::Int=10`: Minimum samples required per regime [T3]
- `mask_value::Symbol=:MASKED`: Value to use for masked regimes

# Returns
Regimes with low-n cells masked.

# Notes
Use this to identify unreliable regime-conditional metrics.
Cells with n < min_n should be masked before interpretation.

# Knowledge Tier
[T3] Minimum n=10 per regime (rule of thumb).

# Example
```julia
regimes = vcat(fill(:HIGH, 5), fill(:LOW, 15))
masked = mask_low_n_regimes(regimes; min_n=10)
# HIGH cells (n=5) become :MASKED
unique(masked)  # [:LOW, :MASKED]
```
"""
function mask_low_n_regimes(
    regimes::AbstractVector{Symbol};
    min_n::Int = 10,
    mask_value::Symbol = :MASKED
)::Vector{Symbol}
    counts = get_regime_counts(regimes)
    low_n_regimes = Set(r for (r, c) in counts if c < min_n)

    if isempty(low_n_regimes)
        return copy(regimes)
    end

    result = copy(regimes)
    for (i, r) in enumerate(result)
        if r in low_n_regimes
            result[i] = mask_value
        end
    end

    return result
end
