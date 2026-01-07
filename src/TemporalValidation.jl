"""
    TemporalValidation

Rigorous validation for time-series ML: Leakage detection, statistical testing,
and temporal safety.

# Overview

TemporalValidation.jl provides:

1. **Validation Gates Framework** - HALT/PASS/WARN/SKIP leakage detection
2. **Walk-Forward Cross-Validation** - With strict gap enforcement
3. **Statistical Tests** - Diebold-Mariano, Pesaran-Timmermann
4. **Move-Conditional Metrics** - For high-persistence series

# Quick Start

```julia
using TemporalValidation

# Create walk-forward CV with gap enforcement
cv = WalkForwardCV(n_splits=5, gap=4, horizon=4)

# Run validation gates
result = gate_suspicious_improvement(baseline_mae, model_mae)
if result.status == HALT
    @warn result.message
end
```

# Knowledge Tier System

All thresholds are tagged by confidence level:
- [T1]: Academically validated with full citation
- [T2]: Empirical finding from prior work
- [T3]: Assumption needing justification

See `SPECIFICATION.md` for frozen parameters.
"""
module TemporalValidation

using Random
using LinearAlgebra
using Distributions
using StatsBase
using MLJBase: MLJBase, ResamplingStrategy

# Core types
include("types.jl")

# Gates submodule
include("gates/Gates.jl")
using .Gates

# Statistical tests submodule
include("statistical_tests/StatisticalTests.jl")
using .StatisticalTests

# Metrics submodule
include("metrics/Metrics.jl")
using .Metrics

# Regimes submodule
include("regimes/Regimes.jl")
using .Regimes

# Export core types
export GateStatus, HALT, WARN, PASS, SKIP
export GateResult, ValidationReport
export status, halt_gates, warn_gates, pass_gates, skip_gates, passed
export SplitInfo, train_size, test_size

# Export gates
export gate_suspicious_improvement
export gate_shuffled_target
export gate_synthetic_ar1
export run_gates, run_standard_gates
export block_permute, default_block_size

# Export statistical tests types and functions
export DMTestResult, PTTestResult, MultiModelComparisonResult
export dm_test, pt_test, compare_multiple_models
export compute_hac_variance
export bartlett_kernel, default_bandwidth
export significant_at_05, significant_at_01, skill
export n_comparisons, n_significant, get_pairwise

# Export core metrics (from Metrics submodule)
export compute_mae, compute_mse, compute_rmse
export compute_mape, compute_smape, compute_bias
export compute_mase, compute_mrae, compute_theils_u
export compute_r_squared, compute_forecast_correlation
export compute_naive_error

# Export move-conditional (from Metrics submodule)
export MoveDirection, UP, DOWN, FLAT
export MoveConditionalResult
export compute_move_threshold, classify_moves
export compute_move_conditional_metrics
export is_reliable, n_total, n_moves, move_fraction
export compute_direction_accuracy
export compute_move_only_mae, compute_persistence_mae

# Export regimes (from Regimes submodule)
export classify_volatility_regime, classify_direction_regime
export get_combined_regimes, get_regime_counts, mask_low_n_regimes
export StratifiedMetricsResult, compute_stratified_metrics

# Export frozen thresholds
export SUSPICIOUS_IMPROVEMENT_HALT, SUSPICIOUS_IMPROVEMENT_WARN
export SHUFFLED_TARGET_THRESHOLD, DEFAULT_N_SHUFFLES
export AR1_PHI, AR1_TOLERANCE, THEORETICAL_AR1_MAE_FACTOR
export DM_TEST_MIN_SAMPLES, PT_TEST_MIN_SAMPLES
export DEFAULT_ALPHA
export CONFORMAL_CALIBRATION_FRACTION, ADAPTIVE_CONFORMAL_GAMMA
export MOVE_THRESHOLD_PERCENTILE
export VOLATILITY_WINDOW, VOLATILITY_LOW_PERCENTILE, VOLATILITY_HIGH_PERCENTILE

# =============================================================================
# Walk-Forward Cross-Validation
# =============================================================================

"""
    WindowType

Type of window for walk-forward cross-validation.

- `Expanding`: Training window grows with each split
- `Sliding`: Training window maintains fixed size
"""
@enum WindowType begin
    Expanding
    Sliding
end

"""
    WalkForwardCV

Walk-forward cross-validation with strict gap enforcement.

CRITICAL: The gap parameter must be >= horizon to prevent target leakage.
This is enforced at construction time.

# Fields
- `n_splits::Int`: Number of train/test splits
- `horizon::Int`: Forecast horizon (how far ahead we predict)
- `gap::Int`: Gap between train end and test start (must be >= horizon)
- `window::WindowType`: Expanding or Sliding window
- `window_size::Union{Int, Nothing}`: Size of sliding window (required if Sliding)
- `test_size::Int`: Number of test observations per split

# Constructor
```julia
WalkForwardCV(;
    n_splits::Int = 5,
    horizon::Int = 1,
    gap::Int = horizon,
    window::WindowType = Expanding,
    window_size::Union{Int, Nothing} = nothing,
    test_size::Int = 1
)
```

# Example
```julia
# h=4 forecast with 4-period gap (no leakage)
cv = WalkForwardCV(n_splits=10, horizon=4, gap=4)

# Sliding window of 100 observations
cv = WalkForwardCV(
    n_splits=20,
    horizon=1,
    gap=1,
    window=Sliding,
    window_size=100
)
```

# Gap Enforcement [T2]
The formula `gap >= horizon` ensures no overlap between the latest training
observation and the test target. For h-step forecasts, we need h periods
between train end and test start.

# Knowledge Tier
[T1] Walk-forward validation is the standard for time series (Tashman 2000).
[T2] Gap enforcement prevents target leakage (temporalcv implementation).
"""
struct WalkForwardCV <: ResamplingStrategy
    n_splits::Int
    horizon::Int
    gap::Int
    window::WindowType
    window_size::Union{Int, Nothing}
    test_size::Int

    function WalkForwardCV(;
        n_splits::Int = 5,
        horizon::Int = 1,
        gap::Union{Int, Nothing} = nothing,
        window::WindowType = Expanding,
        window_size::Union{Int, Nothing} = nothing,
        test_size::Int = 1
    )
        # Default gap to horizon if not specified
        actual_gap = isnothing(gap) ? horizon : gap

        # Validation
        n_splits >= 1 || throw(ArgumentError("n_splits must be >= 1, got $n_splits"))
        horizon >= 1 || throw(ArgumentError("horizon must be >= 1, got $horizon"))
        actual_gap >= 0 || throw(ArgumentError("gap must be >= 0, got $actual_gap"))
        test_size >= 1 || throw(ArgumentError("test_size must be >= 1, got $test_size"))

        # Critical: gap must be >= horizon to prevent leakage
        if actual_gap < horizon
            throw(ArgumentError(
                "gap ($actual_gap) must be >= horizon ($horizon) to prevent target leakage. " *
                "This is a critical temporal safety requirement."
            ))
        end

        # Sliding window requires window_size
        if window == Sliding && isnothing(window_size)
            throw(ArgumentError("window_size is required when window=Sliding"))
        end

        if !isnothing(window_size) && window_size < 1
            throw(ArgumentError("window_size must be >= 1, got $window_size"))
        end

        new(n_splits, horizon, actual_gap, window, window_size, test_size)
    end
end

export WindowType, Expanding, Sliding
export WalkForwardCV

"""
    get_splits(cv::WalkForwardCV, n::Int) -> Vector{SplitInfo}

Generate train/test split indices for walk-forward cross-validation.

# Arguments
- `cv::WalkForwardCV`: Cross-validation configuration
- `n::Int`: Total number of observations

# Returns
Vector of `SplitInfo` objects describing each split.

# Example
```julia
cv = WalkForwardCV(n_splits=5, horizon=1, gap=1)
splits = get_splits(cv, 100)
for s in splits
    println("Train: \$(s.train_start):\$(s.train_end), Test: \$(s.test_start):\$(s.test_end)")
end
```
"""
function get_splits(cv::WalkForwardCV, n::Int)
    n >= 1 || throw(ArgumentError("n must be >= 1, got $n"))

    # Minimum required: enough for at least one split
    min_required = cv.gap + cv.test_size + 1  # At least 1 training point
    if n < min_required
        throw(ArgumentError(
            "Insufficient data: n=$n but need at least $min_required " *
            "(gap=$(cv.gap) + test_size=$(cv.test_size) + 1 training point)"
        ))
    end

    splits = SplitInfo[]

    # Calculate step size between splits
    # Reserve space for last test set
    available_for_splits = n - cv.gap - cv.test_size

    if cv.window == Expanding
        # For expanding window, start with minimum training size and grow
        min_train_size = max(1, available_for_splits ÷ cv.n_splits)

        for i in 0:(cv.n_splits - 1)
            # Training window grows
            train_start = 1
            train_end = min_train_size + i * ((available_for_splits - min_train_size) ÷ max(1, cv.n_splits - 1))

            # Ensure we don't exceed available data
            if train_end + cv.gap + cv.test_size > n
                break
            end

            test_start = train_end + cv.gap + 1
            test_end = test_start + cv.test_size - 1

            push!(splits, SplitInfo(i, train_start, train_end, test_start, test_end, cv.gap))
        end
    else  # Sliding
        window_size = cv.window_size

        # Calculate how many complete splits we can make
        n_possible = (n - window_size - cv.gap - cv.test_size) ÷ 1 + 1
        n_actual = min(cv.n_splits, n_possible)

        step = max(1, (n - window_size - cv.gap - cv.test_size) ÷ max(1, n_actual - 1))

        for i in 0:(n_actual - 1)
            train_start = 1 + i * step
            train_end = train_start + window_size - 1

            if train_end + cv.gap + cv.test_size > n
                break
            end

            test_start = train_end + cv.gap + 1
            test_end = test_start + cv.test_size - 1

            push!(splits, SplitInfo(i, train_start, train_end, test_start, test_end, cv.gap))
        end
    end

    if isempty(splits)
        throw(ArgumentError(
            "Could not generate any splits with n=$n, n_splits=$(cv.n_splits), " *
            "gap=$(cv.gap), test_size=$(cv.test_size), window=$(cv.window)"
        ))
    end

    return splits
end

export get_splits

# =============================================================================
# MLJBase Integration
# =============================================================================

"""
    MLJBase.train_test_pairs(cv::WalkForwardCV, rows)

Generate train/test index pairs for MLJ.jl integration.

This method enables `WalkForwardCV` to be used with `MLJ.evaluate!`:

```julia
using MLJ
cv = WalkForwardCV(n_splits=5, horizon=4, gap=4)
evaluate!(mach, resampling=cv, measure=rmse)
```

# Arguments
- `cv::WalkForwardCV`: Walk-forward CV configuration
- `rows`: Row indices to split (typically `1:n`)

# Returns
`Vector{Tuple{Vector{Int}, Vector{Int}}}` where each tuple is (train_indices, test_indices).

# Knowledge Tier
[T1] MLJBase.ResamplingStrategy interface (MLJ.jl documentation).
"""
function MLJBase.train_test_pairs(
    cv::WalkForwardCV,
    rows
)
    # Convert rows to vector if needed
    rows_vec = collect(rows)
    n = length(rows_vec)

    # Generate splits using our internal method
    splits = get_splits(cv, n)

    # Convert SplitInfo to (train, test) tuples with actual row indices
    pairs = Vector{Tuple{Vector{Int}, Vector{Int}}}()

    for split in splits
        train_indices = rows_vec[split.train_start:split.train_end]
        test_indices = rows_vec[split.test_start:split.test_end]
        push!(pairs, (train_indices, test_indices))
    end

    return pairs
end

# Pretty printing
function Base.show(io::IO, cv::WalkForwardCV)
    window_str = cv.window == Expanding ? "expanding" : "sliding($(cv.window_size))"
    print(io, "WalkForwardCV(n=$(cv.n_splits), h=$(cv.horizon), gap=$(cv.gap), $window_str)")
end

function Base.show(io::IO, ::MIME"text/plain", cv::WalkForwardCV)
    println(io, "WalkForwardCV")
    println(io, "  n_splits: $(cv.n_splits)")
    println(io, "  horizon: $(cv.horizon)")
    println(io, "  gap: $(cv.gap)")
    println(io, "  window: $(cv.window)")
    if cv.window == Sliding
        println(io, "  window_size: $(cv.window_size)")
    end
    print(io, "  test_size: $(cv.test_size)")
end

end # module
