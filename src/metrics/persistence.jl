# persistence.jl - Move-conditional metrics for high-persistence series
#
# Specialized metrics for evaluating forecasts on high-persistence time series
# where the persistence baseline (predict no change) is trivially good.
#
# Knowledge Tiers:
#   [T1] Persistence baseline = predict no change (Tashman 2000)
#   [T1] Skill score formula: SS = 1 - (model_error / baseline_error) (Murphy 1988)
#   [T2] MC-SS = skill score computed on moves only (myga-forecasting-v2 Phase 11)
#   [T2] 70th percentile threshold defines "significant" moves (v2 empirical)
#   [T3] 10 samples per direction for reliability (rule of thumb)
#   [T3] Scale-aware epsilon for numerical stability

using StatsBase: mean
using Statistics: median, quantile

# =============================================================================
# MoveDirection Enum
# =============================================================================

"""
    MoveDirection

Direction of value change, used for 3-class classification.

Values are sign-aligned for mathematical convenience:
- `UP = 1` (positive direction)
- `DOWN = -1` (negative direction)
- `FLAT = 0` (no significant change)

# Example
```julia
dir = UP
if Int8(dir) > 0
    println("Positive move")
end
```
"""
@enum MoveDirection::Int8 begin
    UP = 1
    DOWN = -1
    FLAT = 0
end

# =============================================================================
# MoveConditionalResult
# =============================================================================

"""
    MoveConditionalResult

Move-conditional evaluation results.

Evaluates performance conditional on actual movement:
- UP: actual > threshold
- DOWN: actual < -threshold
- FLAT: |actual| <= threshold

# Fields
- `mae_up::Float64`: MAE for upward moves
- `mae_down::Float64`: MAE for downward moves
- `mae_flat::Float64`: MAE for flat periods
- `n_up::Int`: Count of upward moves
- `n_down::Int`: Count of downward moves
- `n_flat::Int`: Count of flat periods
- `skill_score::Float64`: MC-SS = 1 - (model_mae_moves / persistence_mae_moves)
- `move_threshold::Float64`: Threshold used for classification

# Properties
- `n_total`: Total sample count
- `n_moves`: Count of significant moves (UP + DOWN)
- `is_reliable`: True if n_up ≥ 10 AND n_down ≥ 10 [T3]
- `move_fraction`: Fraction of samples that are moves

# Knowledge Tier
[T2] MC-SS from myga-forecasting-v2 Phase 11.
[T3] Reliability threshold of 10 per direction.

# Example
```julia
threshold = compute_move_threshold(train_actuals)
mc = compute_move_conditional_metrics(predictions, actuals; threshold=threshold)
println("MC-SS: \$(mc.skill_score)")
println("Reliable: \$(is_reliable(mc))")
```
"""
struct MoveConditionalResult
    mae_up::Float64
    mae_down::Float64
    mae_flat::Float64
    n_up::Int
    n_down::Int
    n_flat::Int
    skill_score::Float64
    move_threshold::Float64
end

# Convenience accessors
"""
    n_total(r::MoveConditionalResult) -> Int

Total sample count.
"""
n_total(r::MoveConditionalResult) = r.n_up + r.n_down + r.n_flat

"""
    n_moves(r::MoveConditionalResult) -> Int

Count of significant moves (UP + DOWN).
"""
n_moves(r::MoveConditionalResult) = r.n_up + r.n_down

"""
    is_reliable(r::MoveConditionalResult) -> Bool

Check if results are statistically reliable.
Requires at least 10 samples per move direction [T3].
"""
is_reliable(r::MoveConditionalResult) = r.n_up >= 10 && r.n_down >= 10

"""
    move_fraction(r::MoveConditionalResult) -> Float64

Fraction of samples that are moves (not FLAT).
"""
function move_fraction(r::MoveConditionalResult)
    total = n_total(r)
    total == 0 ? 0.0 : n_moves(r) / total
end

# Pretty printing
function Base.show(io::IO, r::MoveConditionalResult)
    reliable_str = is_reliable(r) ? "" : " [unreliable]"
    print(io, "MC-SS=$(round(r.skill_score, digits=3)), n=$(n_total(r))$reliable_str")
end

function Base.show(io::IO, ::MIME"text/plain", r::MoveConditionalResult)
    println(io, "MoveConditionalResult")
    println(io, "  skill_score: $(round(r.skill_score, digits=4))")
    println(io, "  move_threshold: $(round(r.move_threshold, digits=6))")
    println(io, "  n_up: $(r.n_up) (MAE=$(round(r.mae_up, digits=4)))")
    println(io, "  n_down: $(r.n_down) (MAE=$(round(r.mae_down, digits=4)))")
    println(io, "  n_flat: $(r.n_flat) (MAE=$(round(r.mae_flat, digits=4)))")
    println(io, "  n_total: $(n_total(r))")
    println(io, "  move_fraction: $(round(move_fraction(r) * 100, digits=1))%")
    print(io, "  is_reliable: $(is_reliable(r))")
end


# =============================================================================
# Helper Functions
# =============================================================================

"""
    _get_scale_aware_epsilon(values) -> Float64

Compute scale-aware epsilon for division safety.

Uses median absolute value to determine appropriate epsilon,
avoiding issues with very small magnitude data where fixed
thresholds like 1e-10 may be inappropriate.

# Returns
Scale-appropriate epsilon (minimum 1e-10).

# Knowledge Tier
[T3] Implementation choice for numerical stability.
"""
function _get_scale_aware_epsilon(values::AbstractVector{<:Real})
    nonzero = values[values .!= 0]
    if length(nonzero) > 0
        scale = median(abs.(nonzero))
        return max(1e-10, scale * 1e-8)
    end
    return 1e-10
end


# =============================================================================
# Core Functions
# =============================================================================

"""
    compute_move_threshold(actuals; percentile=70.0) -> Float64

Compute move threshold from historical changes.

Default: 70th percentile of |actuals|.

# Arguments
- `actuals::AbstractVector{<:Real}`: Historical actual values (from training data).
  Should be *changes* (returns/differences), not raw levels.
- `percentile::Float64=70.0`: Percentile of |actuals| to use as threshold

# Returns
Move threshold value.

# Notes
CRITICAL: The threshold MUST be computed from training data only
to prevent regime threshold leakage.

Using 70th percentile means ~30% of historical changes are "moves"
and ~70% are "flat". This provides a meaningful signal-to-noise ratio.

# Knowledge Tier
[T2] 70th percentile from myga-forecasting-v2.

# Example
```julia
# Compute from training data only!
threshold = compute_move_threshold(train_actuals)
# Then use for test evaluation
mc = compute_move_conditional_metrics(preds, actuals; threshold=threshold)
```

# See Also
- `compute_move_conditional_metrics`: Main MC-SS computation using threshold.
- `classify_moves`: Classify values into UP/DOWN/FLAT.
"""
function compute_move_threshold(
    actuals::AbstractVector{<:Real};
    percentile::Float64 = 70.0
)::Float64
    if isempty(actuals)
        throw(ArgumentError("Cannot compute threshold from empty array"))
    end

    if !(0 < percentile <= 100)
        throw(ArgumentError("percentile must be in (0, 100], got $percentile"))
    end

    if any(isnan, actuals)
        throw(ArgumentError(
            "actuals contains NaN values. Clean data before processing."
        ))
    end

    return Float64(quantile(abs.(actuals), percentile / 100.0))
end


"""
    classify_moves(values, threshold) -> Vector{MoveDirection}

Classify values into UP, DOWN, FLAT categories.

# Arguments
- `values::AbstractVector{<:Real}`: Values to classify (typically actuals)
- `threshold::Float64`: Threshold for flat classification

# Returns
Vector of `MoveDirection` enums.

# Classification Rules
- UP: value > threshold
- DOWN: value < -threshold
- FLAT: |value| <= threshold

# Example
```julia
values = [0.1, -0.1, 0.02, -0.02, 0.0]
moves = classify_moves(values, 0.05)
# Returns [UP, DOWN, FLAT, FLAT, FLAT]
```
"""
function classify_moves(
    values::AbstractVector{<:Real},
    threshold::Float64
)::Vector{MoveDirection}
    if threshold < 0
        throw(ArgumentError("threshold must be non-negative, got $threshold"))
    end

    classifications = Vector{MoveDirection}(undef, length(values))

    for (i, v) in enumerate(values)
        if v > threshold
            classifications[i] = UP
        elseif v < -threshold
            classifications[i] = DOWN
        else
            classifications[i] = FLAT
        end
    end

    return classifications
end


"""
    compute_move_conditional_metrics(predictions, actuals; kwargs...) -> MoveConditionalResult

Compute move-conditional evaluation metrics.

Evaluates model performance separately for:
- UP moves: actual > threshold
- DOWN moves: actual < -threshold
- FLAT periods: |actual| <= threshold

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions (changes, not levels)
- `actuals::AbstractVector{<:Real}`: Actual values (changes, not levels)

# Keyword Arguments
- `threshold::Union{Float64, Nothing}=nothing`: Move threshold. If nothing,
  computed from actuals (NOT recommended for walk-forward; use training threshold).
- `threshold_percentile::Float64=70.0`: Percentile for threshold if computed.

# Returns
`MoveConditionalResult` with move-conditional metrics including MC-SS.

# MC-SS Formula [T2]
```
move_mask = (direction == UP) | (direction == DOWN)
model_mae_moves = mean(|pred[move_mask] - actual[move_mask]|)
persistence_mae_moves = mean(|actual[move_mask]|)  # Persistence predicts 0
skill_score = 1.0 - (model_mae_moves / persistence_mae_moves)
```

# Interpretation
- skill_score ≈ 1.0: Perfect predictions on moves
- skill_score ≈ 0.0: Model equals persistence baseline
- skill_score < 0.0: Model worse than persistence

# Notes
CRITICAL: For walk-forward evaluation, `threshold` should be computed
from training data only to prevent leakage.

# Knowledge Tier
[T2] MC-SS from myga-forecasting-v2 Phase 11.
[T3] Scale-aware epsilon for numerical stability.

# Example
```julia
# Compute threshold from training
threshold = compute_move_threshold(train_actuals)

# Evaluate on test
mc = compute_move_conditional_metrics(preds, actuals; threshold=threshold)
println("MC-SS: \$(mc.skill_score)")

if is_reliable(mc)
    println("Result is reliable")
else
    @warn "Insufficient samples per direction"
end
```

# See Also
- `compute_move_threshold`: Compute threshold from training data.
- `is_reliable`: Check result reliability.
- `MoveConditionalResult`: Full result structure.
"""
function compute_move_conditional_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    threshold::Union{Float64, Nothing} = nothing,
    threshold_percentile::Float64 = 70.0
)::MoveConditionalResult

    # Validate inputs
    if any(isnan, predictions)
        throw(ArgumentError(
            "predictions contains NaN values. Clean data before processing."
        ))
    end
    if any(isnan, actuals)
        throw(ArgumentError(
            "actuals contains NaN values. Clean data before processing."
        ))
    end

    if length(predictions) != length(actuals)
        throw(ArgumentError(
            "Arrays must have same length. " *
            "predictions: $(length(predictions)), actuals: $(length(actuals))"
        ))
    end

    n = length(predictions)

    # Handle empty case
    if n == 0
        return MoveConditionalResult(
            NaN, NaN, NaN,  # mae_up, mae_down, mae_flat
            0, 0, 0,        # n_up, n_down, n_flat
            NaN,            # skill_score
            0.0             # move_threshold
        )
    end

    # Compute or use provided threshold
    actual_threshold = isnothing(threshold) ?
        compute_move_threshold(actuals; percentile=threshold_percentile) :
        threshold

    # Classify moves based on ACTUALS
    classifications = classify_moves(actuals, actual_threshold)

    # Create masks
    up_mask = classifications .== UP
    down_mask = classifications .== DOWN
    flat_mask = classifications .== FLAT

    # Counts
    n_up = sum(up_mask)
    n_down = sum(down_mask)
    n_flat = sum(flat_mask)

    # Conditional MAEs with warnings for empty subsets
    function compute_conditional_mae(mask, name)
        if sum(mask) == 0
            @warn "No $name moves in sample (n=0). MAE will be NaN. " *
                  "Total samples: $n, threshold: $(round(actual_threshold, sigdigits=3)). " *
                  "Consider adjusting the move threshold."
            return NaN
        end
        return mean(abs.(predictions[mask] .- actuals[mask]))
    end

    mae_up = compute_conditional_mae(up_mask, "UP")
    mae_down = compute_conditional_mae(down_mask, "DOWN")
    mae_flat = compute_conditional_mae(flat_mask, "FLAT")

    # Compute MC-SS on moves only (UP + DOWN)
    move_mask = up_mask .| down_mask
    n_moves_val = n_up + n_down

    if n_moves_val > 0
        # Model MAE on moves
        model_mae_moves = mean(abs.(predictions[move_mask] .- actuals[move_mask]))

        # Persistence MAE on moves
        # Persistence predicts 0, so error = |actual|
        persistence_mae_moves = mean(abs.(actuals[move_mask]))

        # Guard against division by zero (scale-aware epsilon)
        epsilon = _get_scale_aware_epsilon(actuals[move_mask])

        if persistence_mae_moves > epsilon
            skill_score = 1.0 - (model_mae_moves / persistence_mae_moves)
        else
            @warn "Persistence MAE on moves is near zero. " *
                  "skill_score will be NaN. " *
                  "persistence_mae_moves=$(round(persistence_mae_moves, sigdigits=3)), " *
                  "epsilon=$(round(epsilon, sigdigits=3)). " *
                  "Consider raising the move threshold."
            skill_score = NaN
        end
    else
        @warn "No moves (UP or DOWN) in sample - all observations are FLAT. " *
              "skill_score will be NaN. " *
              "Total samples: $n, n_flat: $n_flat. " *
              "Consider lowering the move threshold or checking data scale."
        skill_score = NaN
    end

    return MoveConditionalResult(
        mae_up, mae_down, mae_flat,
        n_up, n_down, n_flat,
        skill_score,
        actual_threshold
    )
end
