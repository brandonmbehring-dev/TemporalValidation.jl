# direction.jl - Direction accuracy metrics
#
# Functions for computing directional accuracy and move-only metrics.
#
# Knowledge Tiers:
#   [T1] Directional accuracy is standard (Pesaran & Timmermann 1992)
#   [T2] 3-class direction makes persistence meaningful (myga-v2)

using StatsBase: mean

# =============================================================================
# Direction Accuracy
# =============================================================================

"""
    compute_direction_accuracy(predictions, actuals; move_threshold=nothing) -> Float64

Compute directional accuracy.

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `move_threshold::Union{Float64, Nothing}=nothing`:
  If provided, uses 3-class (UP/DOWN/FLAT) comparison.
  If nothing, uses 2-class (positive/negative sign) comparison.

# Returns
Direction accuracy as fraction (0-1).

# Modes

## Without threshold (2-class)
- Compares signs: both positive OR both negative = correct
- Zero actuals are excluded
- Persistence (predicts 0) gets 0% accuracy

## With threshold (3-class)
- UP: value > threshold
- DOWN: value < -threshold
- FLAT: |value| <= threshold
- Correct if both have same class (including both FLAT)
- Persistence (predicts 0 = FLAT) gets credit when actual is also FLAT

The 3-class version provides a meaningful baseline for persistence model
comparison. Without it, persistence trivially gets 0% making all
comparisons "significant".

# Knowledge Tier
[T1] Directional accuracy (Pesaran & Timmermann 1992).
[T2] 3-class mode from myga-forecasting-v2.

# Example
```julia
# 2-class (sign-based)
acc = compute_direction_accuracy(preds, actuals)

# 3-class (with threshold)
threshold = compute_move_threshold(train_actuals)
acc = compute_direction_accuracy(preds, actuals; move_threshold=threshold)
```

# See Also
- `pt_test`: Statistical test for directional accuracy significance.
- `compute_move_conditional_metrics`: Full move-conditional evaluation.
"""
function compute_direction_accuracy(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real};
    move_threshold::Union{Float64, Nothing} = nothing
)::Float64
    if length(predictions) != length(actuals)
        throw(ArgumentError(
            "Arrays must have same length. " *
            "predictions: $(length(predictions)), actuals: $(length(actuals))"
        ))
    end

    if isempty(predictions)
        return 0.0
    end

    if !isnothing(move_threshold)
        # 3-class comparison
        pred_dirs = classify_moves(predictions, move_threshold)
        actual_dirs = classify_moves(actuals, move_threshold)

        correct = pred_dirs .== actual_dirs
        return mean(correct)
    end

    # 2-class (sign) comparison
    # Exclude near-zero actuals to avoid floating-point issues
    epsilon = 1e-10
    nonzero_mask = abs.(actuals) .> epsilon

    if !any(nonzero_mask)
        return 0.0
    end

    pred_signs = sign.(predictions[nonzero_mask])
    actual_signs = sign.(actuals[nonzero_mask])

    return mean(pred_signs .== actual_signs)
end


# =============================================================================
# Move-Only Metrics
# =============================================================================

"""
    compute_move_only_mae(predictions, actuals, threshold) -> Tuple{Float64, Int}

Compute MAE only on moves (excluding FLAT).

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `threshold::Float64`: Move threshold

# Returns
Tuple of (mae, n_moves):
- `mae`: MAE on moves only
- `n_moves`: Count of moves used

# Notes
This isolates model performance on "significant" moves,
excluding periods where nothing happened (FLAT).

# Example
```julia
mae, n = compute_move_only_mae(preds, actuals, 0.05)
if n >= 20
    println("Move-only MAE: \$mae (n=\$n)")
end
```
"""
function compute_move_only_mae(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    threshold::Float64
)::Tuple{Float64, Int}
    if threshold < 0
        throw(ArgumentError("threshold must be non-negative, got $threshold"))
    end

    if length(predictions) != length(actuals)
        throw(ArgumentError(
            "Arrays must have same length. " *
            "predictions: $(length(predictions)), actuals: $(length(actuals))"
        ))
    end

    # Identify moves (UP or DOWN, not FLAT)
    move_mask = abs.(actuals) .> threshold
    n_moves_val = sum(move_mask)

    if n_moves_val == 0
        return (NaN, 0)
    end

    mae = mean(abs.(predictions[move_mask] .- actuals[move_mask]))
    return (mae, n_moves_val)
end


"""
    compute_persistence_mae(actuals; threshold=nothing) -> Float64

Compute MAE of persistence baseline.

Persistence predicts 0 (no change), so MAE = mean(|actual|).

# Arguments
- `actuals::AbstractVector{<:Real}`: Actual values
- `threshold::Union{Float64, Nothing}=nothing`: If provided, computes MAE only on moves

# Returns
Persistence baseline MAE.

# Notes
This function assumes `actuals` represent changes (differences), not levels.
For level data, the persistence baseline would predict y[t-1] for y[t].

# Example
```julia
persistence_mae = compute_persistence_mae(actuals)
persistence_mae_moves = compute_persistence_mae(actuals; threshold=0.05)
```
"""
function compute_persistence_mae(
    actuals::AbstractVector{<:Real};
    threshold::Union{Float64, Nothing} = nothing
)::Float64
    if isempty(actuals)
        return NaN
    end

    if !isnothing(threshold)
        move_mask = abs.(actuals) .> threshold
        if !any(move_mask)
            return NaN
        end
        return mean(abs.(actuals[move_mask]))
    end

    return mean(abs.(actuals))
end
