# runner.jl - Gate aggregation and orchestration
#
# Provides run_gates() for aggregating gate results into a ValidationReport,
# and run_standard_gates() for running the standard validation suite.
#
# Knowledge Tiers:
#   [T2] Gate framework from myga-forecasting-v2 postmortem analysis
#   [T2] "External-first" validation ordering (synthetic → shuffled → internal)

using Random
using ..TemporalValidation: GateResult, ValidationReport, GateStatus
using ..TemporalValidation: HALT, WARN, PASS, SKIP

"""
    run_gates(gates::Vector{GateResult}) -> ValidationReport

Aggregate gate results into a validation report.

This is the simple form that takes pre-computed gate results
and aggregates them. The `ValidationReport` handles status aggregation:
- HALT if any HALT
- WARN if any WARN (but no HALT)
- SKIP if any SKIP (but no HALT/WARN)
- PASS otherwise

# Arguments
- `gates::Vector{GateResult}`: Pre-computed gate results

# Returns
`ValidationReport` containing all gate results with aggregated status.

# Example
```julia
using TemporalValidation

results = [
    gate_suspicious_improvement(0.10, 0.15),
    gate_shuffled_target(model, X, y),
]
report = run_gates(results)

if report.status == HALT
    println("Validation failed: ", first(halt_gates(report)).message)
end
```

# See Also
- `ValidationReport`: The aggregated report type
- `status`: Get overall status from report
- `halt_gates`, `warn_gates`, `pass_gates`, `skip_gates`: Filter by status
"""
function run_gates(gates::Vector{GateResult})::ValidationReport
    return ValidationReport(gates)
end

# Convenience: accept tuple or varargs
run_gates(gates::Tuple) = run_gates(collect(GateResult, gates))
run_gates(gates::GateResult...) = run_gates(collect(gates))


"""
    run_standard_gates(model, X, y; kwargs...) -> ValidationReport

Run the standard validation gate suite.

Executes the recommended gate sequence in priority order:
1. `gate_shuffled_target()` - External validation (leakage detection)
2. `gate_synthetic_ar1()` - External validation (theoretical bounds)
3. `gate_suspicious_improvement()` - Internal validation (if metrics provided)

# Arguments
- `model`: MLJ model for gates requiring model evaluation
- `X`: Feature data
- `y::AbstractVector`: Target vector

# Keyword Arguments
- `model_metric::Union{Real, Nothing} = nothing`: Model's error metric for suspicious improvement
- `baseline_metric::Union{Real, Nothing} = nothing`: Baseline error metric
- `n_shuffles::Int = 5`: Number of shuffles for shuffled target test
- `n_cv_splits::Int = 3`: Number of CV splits for model evaluation
- `test_size::Int = 10`: Test size for CV folds
- `run_synthetic_ar1::Bool = true`: Whether to run synthetic AR(1) test
- `rng::AbstractRNG = Random.default_rng()`: Random number generator

# Returns
`ValidationReport` with results from all executed gates.

# Gate Execution Order

The gate order follows the "external-first" principle [T2]:

1. **External validation** runs first (shuffled target, synthetic AR(1))
   - Catches gross errors before trusting internal metrics
   - Independent of model's reported performance

2. **Internal validation** runs second (suspicious improvement)
   - Only meaningful after external validation passes
   - Compares model to baseline

# Example
```julia
using TemporalValidation
using MLJDecisionTreeInterface

model = DecisionTreeRegressor(max_depth=3)
X = (x1 = randn(200), x2 = randn(200))
y = randn(200)

# Without baseline comparison
report = run_standard_gates(model, X, y)

# With baseline comparison (recommended)
baseline_mae = 1.0  # e.g., persistence MAE
model_mae = 0.85    # your model's MAE
report = run_standard_gates(
    model, X, y;
    model_metric=model_mae,
    baseline_metric=baseline_mae
)

if status(report) == HALT
    println(report)  # Will show which gates failed
end
```

# Knowledge Tier
[T2] "External-first" validation ordering from myga-forecasting-v2 postmortem.

# See Also
- `gate_shuffled_target`: Definitive leakage test
- `gate_synthetic_ar1`: Theoretical bound verification
- `gate_suspicious_improvement`: Internal validation
"""
function run_standard_gates(
    model,
    X,
    y::AbstractVector;
    model_metric::Union{Real, Nothing} = nothing,
    baseline_metric::Union{Real, Nothing} = nothing,
    n_shuffles::Int = 5,
    n_cv_splits::Int = 3,
    test_size::Int = 10,
    run_synthetic_ar1::Bool = true,
    rng::AbstractRNG = Random.default_rng()
)::ValidationReport
    gates = GateResult[]

    # =========================================================================
    # Stage 1: External Validation (run first per "external-first" principle)
    # =========================================================================

    # Gate 1: Shuffled target test - definitive leakage detection
    push!(gates, gate_shuffled_target(
        model, X, y;
        n_shuffles = n_shuffles,
        n_cv_splits = n_cv_splits,
        test_size = test_size,
        rng = rng
    ))

    # Gate 2: Synthetic AR(1) test - theoretical bounds
    if run_synthetic_ar1
        push!(gates, gate_synthetic_ar1(
            model;
            n_cv_splits = n_cv_splits,
            test_size = test_size,
            rng = rng
        ))
    end

    # =========================================================================
    # Stage 2: Internal Validation
    # =========================================================================

    # Gate 3: Suspicious improvement (only if metrics provided)
    if !isnothing(model_metric) && !isnothing(baseline_metric)
        push!(gates, gate_suspicious_improvement(
            Float64(model_metric),
            Float64(baseline_metric)
        ))
    end

    return run_gates(gates)
end
