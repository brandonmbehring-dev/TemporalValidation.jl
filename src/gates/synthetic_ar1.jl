# synthetic_ar1.jl - Synthetic AR(1) gate for theoretical bound verification
#
# Test model on synthetic AR(1) process where optimal forecast is phi * y_{t-1}.
# Model MAE should not significantly beat theoretical optimum.
#
# For AR(1): y_t = phi * y_{t-1} + sigma * epsilon_t
# Theoretical optimal 1-step MAE = sigma * sqrt(2/pi) ≈ 0.798 * sigma
#
# Knowledge Tiers:
#   [T1] Optimal MAE for N(0,σ) = σ√(2/π) - standard half-normal mean
#   [T2] Synthetic AR(1) test catches lookahead bias
#   [T3] Tolerance factor 1.5 allows for finite-sample variation

using Random
using StatsBase: mean
using ..TemporalValidation: GateResult, GateStatus, HALT, PASS, SKIP
using ..TemporalValidation: AR1_PHI, AR1_TOLERANCE, THEORETICAL_AR1_MAE_FACTOR

# Import MLJBase at the top level
import MLJBase

"""
    gate_synthetic_ar1(model; kwargs...) -> GateResult

Synthetic AR(1) test: theoretical bound verification.

Test model on synthetic AR(1) process where optimal forecast is `phi * y_{t-1}`.
Model MAE should not significantly beat theoretical optimum.

If a model significantly beats the theoretical optimum on AR(1) data,
it's likely exploiting lookahead bias or has implementation bugs.

# Arguments
- `model`: MLJ model (not machine) — will be fit for evaluation

# Keyword Arguments
- `phi::Float64 = AR1_PHI`: AR(1) coefficient (0.95), must be in (-1, 1)
- `sigma::Float64 = 1.0`: Innovation standard deviation
- `n_samples::Int = 500`: Number of samples to generate
- `n_lags::Int = 5`: Number of lagged features to create
- `tolerance::Float64 = AR1_TOLERANCE`: Tolerance factor (1.5)
- `n_cv_splits::Int = 3`: Number of walk-forward CV splits
- `test_size::Int = 20`: Test size for each CV fold
- `rng::AbstractRNG = Random.default_rng()`: Random number generator

# Returns
`GateResult` with:
- `HALT` if model beats theoretical bound by too much (ratio < 1/tolerance)
- `PASS` if model MAE is within acceptable bounds
- `SKIP` if evaluation fails

# Theoretical Background

For AR(1) process: `y_t = phi * y_{t-1} + sigma * epsilon_t`

The optimal 1-step forecast is `phi * y_{t-1}`, giving forecast error
`sigma * epsilon_t ~ N(0, sigma)`.

The MAE of `N(0, sigma)` is `sigma * sqrt(2/pi) ≈ 0.798 * sigma` [T1].

Any model achieving MAE significantly below this is likely cheating.

# Knowledge Tier
[T1] Optimal MAE = σ√(2/π) from half-normal distribution mean.
[T2] Synthetic AR(1) test catches lookahead bias (myga-forecasting-v2).
[T3] Tolerance factor 1.5 allows for finite-sample variation.

# Example
```julia
using TemporalValidation
using MLJDecisionTreeInterface

model = DecisionTreeRegressor(max_depth=3)
result = gate_synthetic_ar1(model; n_samples=300, phi=0.9)

if result.status == HALT
    @warn "Model beats theoretical bounds: \$(result.message)"
end
```

# See Also
- `gate_shuffled_target`: Definitive leakage test via permutation
- `gate_suspicious_improvement`: Check for implausible improvement ratios
"""
function gate_synthetic_ar1(
    model;
    phi::Float64 = AR1_PHI,
    sigma::Float64 = 1.0,
    n_samples::Int = 500,
    n_lags::Int = 5,
    tolerance::Float64 = AR1_TOLERANCE,
    n_cv_splits::Int = 3,
    test_size::Int = 20,
    rng::AbstractRNG = Random.default_rng()
)::GateResult

    # =========================================================================
    # Input Validation
    # =========================================================================

    # Validate phi for stationarity
    if !(-1 < phi < 1)
        throw(ArgumentError(
            "phi must be in (-1, 1) for stationarity. Got phi=$phi. " *
            "Values outside this range produce non-stationary or explosive series."
        ))
    end

    # Validate n_samples > n_lags
    if n_samples <= n_lags
        throw(ArgumentError(
            "n_samples must be > n_lags to have data for prediction. " *
            "Got n_samples=$n_samples, n_lags=$n_lags."
        ))
    end

    # Validate sigma
    if sigma <= 0
        throw(ArgumentError("sigma must be > 0, got $sigma"))
    end

    # Validate tolerance
    if tolerance <= 1
        throw(ArgumentError("tolerance must be > 1, got $tolerance"))
    end

    # =========================================================================
    # Generate AR(1) process
    # =========================================================================

    # Generate full series (n_samples + n_lags to have enough for lagged features)
    y_full = zeros(n_samples + n_lags)

    # Stationary initialization: y_0 ~ N(0, sigma / sqrt(1 - phi^2))
    y_full[1] = randn(rng) * sigma / sqrt(1 - phi^2)

    # Generate AR(1) process
    for t in 2:length(y_full)
        y_full[t] = phi * y_full[t-1] + sigma * randn(rng)
    end

    # =========================================================================
    # Create lagged features (proper temporal alignment)
    # =========================================================================

    # Target: y_t (from index n_lags+1 onwards)
    y = y_full[n_lags+1:end]
    n = length(y)

    # Features: lag1 = y_{t-1}, lag2 = y_{t-2}, ..., lagL = y_{t-L}
    # Create as NamedTuple for MLJ compatibility
    lag_columns = Dict{Symbol, Vector{Float64}}()
    for lag in 1:n_lags
        col_name = Symbol("lag$lag")
        # lag1: y_full[n_lags:end-1] (y_{t-1})
        # lag2: y_full[n_lags-1:end-2] (y_{t-2})
        # etc.
        lag_columns[col_name] = y_full[n_lags+1-lag:end-lag]
    end
    X = NamedTuple(lag_columns)

    # =========================================================================
    # Check data requirements for CV
    # =========================================================================

    min_required = 1 + test_size + 1  # gap=1, test_size, 1 training point
    if n < min_required
        return GateResult(
            name = :synthetic_ar1,
            status = SKIP,
            message = "Insufficient generated data: n=$n < $min_required required",
            details = Dict{Symbol, Any}(
                :n => n,
                :n_samples => n_samples,
                :n_lags => n_lags,
                :min_required => min_required
            )
        )
    end

    # =========================================================================
    # Set up walk-forward CV and evaluate
    # =========================================================================

    # Access WalkForwardCV from parent module (Gates is inside TemporalValidation)
    ParentModule = parentmodule(@__MODULE__)
    cv = ParentModule.WalkForwardCV(
        n_splits = n_cv_splits,
        horizon = 1,
        gap = 1,
        test_size = test_size
    )

    local model_mae::Float64
    try
        result = MLJBase.evaluate(model, X, y, resampling=cv, measure=MLJBase.mae, verbosity=0)
        model_mae = result.measurement[1]
    catch e
        return GateResult(
            name = :synthetic_ar1,
            status = SKIP,
            message = "Evaluation failed: $(typeof(e))",
            details = Dict{Symbol, Any}(
                :error => string(e),
                :phi => phi,
                :sigma => sigma,
                :n_samples => n_samples
            )
        )
    end

    # =========================================================================
    # Compute theoretical bound and ratio
    # =========================================================================

    # Theoretical optimal MAE for AR(1) 1-step forecast
    # Optimal predictor is phi * y_{t-1}, error is sigma * epsilon ~ N(0, sigma)
    # MAE of N(0, sigma) = sigma * sqrt(2/pi)
    theoretical_mae = sigma * THEORETICAL_AR1_MAE_FACTOR

    ratio = model_mae / theoretical_mae

    # =========================================================================
    # Build result
    # =========================================================================

    details = Dict{Symbol, Any}(
        :model_mae => model_mae,
        :theoretical_mae => theoretical_mae,
        :ratio => ratio,
        :phi => phi,
        :sigma => sigma,
        :n_samples => n_samples,
        :n_lags => n_lags,
        :tolerance => tolerance,
        :halt_threshold => 1.0 / tolerance,
        :n_cv_splits => n_cv_splits,
        :test_size => test_size
    )

    # HALT if model beats theoretical optimum by too much
    if ratio < 1.0 / tolerance
        return GateResult(
            name = :synthetic_ar1,
            status = HALT,
            message = "Model MAE $(round(model_mae, digits=4)) << theoretical $(round(theoretical_mae, digits=4)) (ratio=$(round(ratio, digits=2)))",
            metric_value = ratio,
            threshold = 1.0 / tolerance,
            details = details,
            recommendation = "Model beats theoretical optimum. Check for lookahead bias."
        )
    end

    return GateResult(
        name = :synthetic_ar1,
        status = PASS,
        message = "Model MAE ratio $(round(ratio, digits=2)) is within bounds",
        metric_value = ratio,
        threshold = 1.0 / tolerance,
        details = details
    )
end
