# shuffled_target.jl - Shuffled target gate for leakage detection
#
# The shuffled target test detects whether a model's performance depends on
# the temporal alignment between features and targets. If a model performs
# well on shuffled targets, it indicates the features encode target position
# information — a form of data leakage.
#
# Knowledge Tiers:
#   [T1] Block permutation from Kunsch (1989) - preserves autocorrelation
#   [T3] Gate thresholds are empirical heuristics

using Random

# =============================================================================
# Block Permutation Helper
# =============================================================================

"""
    block_permute(arr, block_size, rng) -> Vector

Permute array by shuffling non-overlapping blocks.

Block permutation preserves the autocorrelation structure within blocks
better than iid permutation, making it appropriate for time series.

# Arguments
- `arr::AbstractVector`: Array to permute
- `block_size::Int`: Size of each block (typically n^(1/3) for Kunsch 1989)
- `rng::AbstractRNG`: Random number generator for reproducibility

# Returns
A new vector with blocks shuffled (not in-place modification).

# Algorithm
1. Divide array into non-overlapping blocks of size `block_size`
2. Handle remainder as a partial block
3. Shuffle the order of blocks
4. Concatenate shuffled blocks

# Example
```julia
using StableRNGs
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9]
result = block_permute(arr, 3, StableRNG(42))
# Blocks [1,2,3], [4,5,6], [7,8,9] are shuffled
# e.g., result = [7, 8, 9, 1, 2, 3, 4, 5, 6]
```

# Knowledge Tier
[T1] Kunsch (1989). "The Jackknife and the Bootstrap for General Stationary
     Observations." Annals of Statistics 17(3):1217-1241.

# See Also
- `gate_shuffled_target()`: Main gate using this permutation method
"""
function block_permute(arr::AbstractVector, block_size::Int, rng::AbstractRNG)
    n = length(arr)

    # Edge cases
    if n == 0
        return similar(arr, 0)
    end
    if block_size <= 0
        throw(ArgumentError("block_size must be > 0, got $block_size"))
    end
    if block_size >= n
        # Single block, just return a copy
        return copy(arr)
    end

    # Calculate number of full blocks
    n_full_blocks = n ÷ block_size

    # Create block index ranges
    block_ranges = Vector{UnitRange{Int}}()

    for i in 1:n_full_blocks
        start_idx = (i - 1) * block_size + 1
        end_idx = i * block_size
        push!(block_ranges, start_idx:end_idx)
    end

    # Handle remainder (partial block at the end)
    remainder_start = n_full_blocks * block_size + 1
    if remainder_start <= n
        push!(block_ranges, remainder_start:n)
    end

    # Shuffle block order (not in-place on original array)
    shuffled_ranges = shuffle(rng, block_ranges)

    # Collect elements in new order
    result = similar(arr, n)
    write_idx = 1
    for range in shuffled_ranges
        for i in range
            result[write_idx] = arr[i]
            write_idx += 1
        end
    end

    return result
end

"""
    default_block_size(n::Int) -> Int

Compute default block size for block permutation.

Uses the Kunsch (1989) rule: block_size = floor(n^(1/3)).

# Arguments
- `n::Int`: Length of the series

# Returns
Block size (minimum 1).

# Knowledge Tier
[T1] Kunsch (1989) optimal block size for stationary observations.
"""
function default_block_size(n::Int)
    n > 0 || throw(ArgumentError("n must be > 0, got $n"))
    return max(1, floor(Int, n^(1/3)))
end

# =============================================================================
# Gate: Shuffled Target Test
# =============================================================================

using StatsBase: mean
using ..TemporalValidation: GateResult, GateStatus, HALT, PASS, SKIP
using ..TemporalValidation: SHUFFLED_TARGET_THRESHOLD, DEFAULT_N_SHUFFLES

# Import MLJBase at the top level (not function-level)
import MLJBase

"""
    gate_shuffled_target(model, X, y; kwargs...) -> GateResult

Shuffled target test: definitive leakage detection.

If a model performs better on real target than shuffled target,
features may contain information about target ordering (leakage).

A model should NOT beat shuffled baseline — the temporal relationship
between X and y should be destroyed by shuffling.

# Arguments
- `model`: MLJ model (not machine) — will be fit fresh for each evaluation
- `X`: Features (NamedTuple, Table, or any MLJ-compatible format)
- `y::AbstractVector`: Target vector

# Keyword Arguments
- `n_shuffles::Int = DEFAULT_N_SHUFFLES`: Number of shuffled targets to average over
- `threshold::Float64 = SHUFFLED_TARGET_THRESHOLD`: Maximum allowed improvement ratio (0.05)
- `n_cv_splits::Int = 3`: Number of walk-forward CV splits
- `test_size::Int = 10`: Test size for each CV fold
- `permutation::Symbol = :block`: Permutation strategy (`:block` or `:iid`)
- `block_size::Union{Int, Symbol} = :auto`: Block size for block permutation
- `rng::AbstractRNG = Random.default_rng()`: Random number generator

# Returns
`GateResult` with:
- `HALT` if model significantly beats shuffled baseline (leakage detected)
- `PASS` if improvement is within acceptable bounds
- `SKIP` if evaluation fails or data is insufficient

# Algorithm
1. Compute out-of-sample MAE on real targets using walk-forward CV
2. For each shuffle:
   - Permute y using block or iid permutation
   - Compute out-of-sample MAE on shuffled targets
3. Compute improvement_ratio = 1 - (mae_real / mean(mae_shuffled))
4. HALT if improvement_ratio > threshold

# Notes

**Power Analysis Warning** [T3]: With n_shuffles=5, the minimum achievable
p-value is 1/6 ≈ 0.167, which is ABOVE the 0.05 threshold. This means the
test can only detect *very strong* leakage. For rigorous testing, use
n_shuffles >= 19 for p < 0.05.

The block permutation (default) preserves local autocorrelation structure,
which is important for time series with persistence. IID permutation
may produce false positives on persistent series.

Uses `WalkForwardCV` internally with `gap=1, horizon=1` (minimum allowed).
This differs from Python implementation (gap=0) but maintains our safety
invariant while still being appropriate for leakage detection.

# Knowledge Tier
[T1] Kunsch (1989) block permutation preserves autocorrelation structure.
[T2] Shuffled target as definitive leakage test (myga-forecasting-v2 validation).
[T3] 5% improvement threshold is standard but arbitrary.

# Example
```julia
using TemporalValidation
using MLJDecisionTreeInterface

model = DecisionTreeRegressor(max_depth=3)
X = (feature1 = randn(100), feature2 = randn(100))
y = randn(100)

result = gate_shuffled_target(model, X, y; n_shuffles=10)
if result.status == HALT
    @warn "Leakage detected: \$(result.message)"
end
```

# See Also
- `gate_synthetic_ar1`: Test against theoretical AR(1) bounds
- `gate_suspicious_improvement`: Check for implausible improvement ratios
- `block_permute`: The permutation function used internally
"""
function gate_shuffled_target(
    model,
    X,
    y::AbstractVector;
    n_shuffles::Int = DEFAULT_N_SHUFFLES,
    threshold::Float64 = SHUFFLED_TARGET_THRESHOLD,
    n_cv_splits::Int = 3,
    test_size::Int = 10,
    permutation::Symbol = :block,
    block_size::Union{Int, Symbol} = :auto,
    rng::AbstractRNG = Random.default_rng()
)::GateResult
    n = length(y)

    # =========================================================================
    # Input Validation
    # =========================================================================

    # Check for NaN values
    if any(isnan, y)
        throw(ArgumentError(
            "y contains NaN values. Clean data before processing."
        ))
    end

    # Validate permutation type
    if permutation ∉ (:block, :iid)
        throw(ArgumentError(
            "permutation must be :block or :iid, got :$permutation"
        ))
    end

    # Validate n_shuffles
    if n_shuffles < 1
        throw(ArgumentError("n_shuffles must be >= 1, got $n_shuffles"))
    end

    # Validate n_cv_splits
    if n_cv_splits < 1
        throw(ArgumentError("n_cv_splits must be >= 1, got $n_cv_splits"))
    end

    # Compute block size for block permutation
    computed_block_size = if permutation == :block
        if block_size === :auto
            default_block_size(n)
        elseif block_size isa Int
            block_size < 1 && throw(ArgumentError("block_size must be >= 1, got $block_size"))
            block_size
        else
            throw(ArgumentError("block_size must be :auto or an Int, got $block_size"))
        end
    else
        1  # Not used for IID, but set for details
    end

    # =========================================================================
    # Check minimum data requirements
    # =========================================================================

    # WalkForwardCV needs: gap + test_size + 1 (at least 1 training point)
    # With gap=1, test_size=test_size, we need: 1 + test_size + 1 = test_size + 2
    min_required = 1 + test_size + 1  # gap=1, test_size, 1 training point
    if n < min_required
        return GateResult(
            name = :shuffled_target,
            status = SKIP,
            message = "Insufficient data: n=$n < $min_required required",
            details = Dict{Symbol, Any}(
                :n => n,
                :min_required => min_required,
                :n_cv_splits => n_cv_splits,
                :test_size => test_size
            )
        )
    end

    # =========================================================================
    # Set up walk-forward CV
    # =========================================================================

    # Use gap=1, horizon=1 (minimum allowed by WalkForwardCV)
    # This maintains our safety invariant while still working for leakage detection
    # Access WalkForwardCV from parent module (Gates is inside TemporalValidation)
    ParentModule = parentmodule(@__MODULE__)
    cv = ParentModule.WalkForwardCV(
        n_splits = n_cv_splits,
        horizon = 1,
        gap = 1,
        test_size = test_size
    )

    # =========================================================================
    # Compute MAE on real targets
    # =========================================================================

    local mae_real::Float64
    try
        result = MLJBase.evaluate(model, X, y, resampling=cv, measure=MLJBase.mae, verbosity=0)
        mae_real = result.measurement[1]
    catch e
        return GateResult(
            name = :shuffled_target,
            status = SKIP,
            message = "Evaluation failed: $(typeof(e))",
            details = Dict{Symbol, Any}(
                :error => string(e),
                :n => n,
                :n_cv_splits => n_cv_splits
            )
        )
    end

    # =========================================================================
    # Compute MAE on shuffled targets
    # =========================================================================

    shuffled_maes = Float64[]

    for _ in 1:n_shuffles
        # Apply permutation strategy
        y_shuffled = if permutation == :block
            block_permute(y, computed_block_size, rng)
        else
            shuffle(rng, collect(y))
        end

        try
            result = MLJBase.evaluate(model, X, y_shuffled, resampling=cv, measure=MLJBase.mae, verbosity=0)
            push!(shuffled_maes, result.measurement[1])
        catch e
            # If a shuffle fails, continue with others
            @warn "Shuffle evaluation failed: $e"
        end
    end

    # Need at least one successful shuffle
    if isempty(shuffled_maes)
        return GateResult(
            name = :shuffled_target,
            status = SKIP,
            message = "All shuffle evaluations failed",
            details = Dict{Symbol, Any}(
                :n_shuffles => n_shuffles,
                :mae_real => mae_real
            )
        )
    end

    mae_shuffled_avg = mean(shuffled_maes)

    # =========================================================================
    # Compute improvement ratio
    # =========================================================================

    # Improvement ratio: positive = model beats shuffled (suspicious)
    improvement_ratio = if mae_shuffled_avg > 0
        1.0 - (mae_real / mae_shuffled_avg)
    else
        @warn "Shuffled target MAE is zero - this is unusual"
        0.0
    end

    # =========================================================================
    # Build result
    # =========================================================================

    details = Dict{Symbol, Any}(
        :mae_real => mae_real,
        :mae_shuffled_avg => mae_shuffled_avg,
        :mae_shuffled_all => shuffled_maes,
        :n_shuffles => n_shuffles,
        :n_shuffles_successful => length(shuffled_maes),
        :n_cv_splits => n_cv_splits,
        :test_size => test_size,
        :permutation => permutation,
        :block_size => computed_block_size,
        :min_pvalue => 1.0 / (n_shuffles + 1)
    )

    if improvement_ratio > threshold
        return GateResult(
            name = :shuffled_target,
            status = HALT,
            message = "Model beats shuffled by $(round(improvement_ratio * 100, digits=1))% (max: $(round(threshold * 100, digits=0))%)",
            metric_value = improvement_ratio,
            threshold = threshold,
            details = details,
            recommendation = "Check for data leakage. Model should NOT beat shuffled target."
        )
    end

    return GateResult(
        name = :shuffled_target,
        status = PASS,
        message = "Model improvement $(round(improvement_ratio * 100, digits=1))% is acceptable",
        metric_value = improvement_ratio,
        threshold = threshold,
        details = details
    )
end
