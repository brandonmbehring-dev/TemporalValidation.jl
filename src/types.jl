# types.jl - Core types for TemporalValidation.jl
#
# Knowledge Tiers (from SPECIFICATION.md):
#   [T1] = Academically validated with citation
#   [T2] = Empirical finding from prior work
#   [T3] = Assumption needing justification

"""
    GateStatus

Validation gate status indicating the severity of the result.

- `HALT`: Critical failure - stop and investigate
- `WARN`: Caution - continue but verify
- `PASS`: Validation passed
- `SKIP`: Insufficient data to run gate

# Knowledge Tier
[T2] Gate framework from myga-forecasting-v2 postmortem analysis.
"""
@enum GateStatus begin
    HALT  # Critical failure - stop and investigate
    WARN  # Caution - continue but verify
    PASS  # Validation passed
    SKIP  # Insufficient data to run gate
end

"""
    GateResult

Result from a validation gate.

# Fields
- `name::Symbol`: Gate identifier (e.g., `:shuffled_target`, `:synthetic_ar1`)
- `status::GateStatus`: HALT, WARN, PASS, or SKIP
- `message::String`: Human-readable description of result
- `metric_value::Union{Float64, Nothing}`: Primary metric for this gate
- `threshold::Union{Float64, Nothing}`: Threshold used for decision
- `details::Dict{Symbol, Any}`: Additional metrics and diagnostics
- `recommendation::String`: What to do if not PASS

# Example
```julia
result = GateResult(
    name = :suspicious_improvement,
    status = HALT,
    message = "Improvement of 35% exceeds 20% threshold",
    metric_value = 0.35,
    threshold = 0.20,
    details = Dict(:baseline_mae => 0.15, :model_mae => 0.098),
    recommendation = "Check for data leakage in feature engineering"
)
```

# Knowledge Tier
[T2] Gate result structure from temporalcv Python implementation.
"""
struct GateResult
    name::Symbol
    status::GateStatus
    message::String
    metric_value::Union{Float64, Nothing}
    threshold::Union{Float64, Nothing}
    details::Dict{Symbol, Any}
    recommendation::String
end

# Convenience constructor with defaults
function GateResult(;
    name::Symbol,
    status::GateStatus,
    message::String,
    metric_value::Union{Float64, Nothing} = nothing,
    threshold::Union{Float64, Nothing} = nothing,
    details::AbstractDict = Dict{Symbol, Any}(),
    recommendation::String = ""
)
    # Convert details to Dict{Symbol, Any} to allow flexible input types
    details_converted = Dict{Symbol, Any}(k => v for (k, v) in details)
    GateResult(name, status, message, metric_value, threshold, details_converted, recommendation)
end

"""
    ValidationReport

Complete validation report across all gates.

# Fields
- `gates::Vector{GateResult}`: Results from all gates run

# Properties
- `status`: Overall status (HALT if any HALT, WARN if any WARN, else PASS)
- `halt_gates`: Gates that returned HALT
- `warn_gates`: Gates that returned WARN
- `pass_gates`: Gates that returned PASS
- `skip_gates`: Gates that returned SKIP

# Example
```julia
report = ValidationReport([result1, result2, result3])
if report.status == HALT
    println("Validation failed: ", first(report.halt_gates).message)
end
```

# Knowledge Tier
[T2] Validation report aggregation from temporalcv Python implementation.
"""
struct ValidationReport
    gates::Vector{GateResult}
end

# Empty constructor
ValidationReport() = ValidationReport(GateResult[])

"""
    status(report::ValidationReport) -> GateStatus

Overall status: HALT if any HALT, WARN if any WARN, SKIP if any SKIP, else PASS.
"""
function status(report::ValidationReport)
    if isempty(report.gates)
        return PASS
    end

    statuses = [g.status for g in report.gates]

    if HALT in statuses
        return HALT
    elseif WARN in statuses
        return WARN
    elseif SKIP in statuses
        return SKIP
    else
        return PASS
    end
end

"""
    halt_gates(report::ValidationReport) -> Vector{GateResult}

Return all gates with HALT status.
"""
halt_gates(report::ValidationReport) = filter(g -> g.status == HALT, report.gates)

"""
    warn_gates(report::ValidationReport) -> Vector{GateResult}

Return all gates with WARN status.
"""
warn_gates(report::ValidationReport) = filter(g -> g.status == WARN, report.gates)

"""
    pass_gates(report::ValidationReport) -> Vector{GateResult}

Return all gates with PASS status.
"""
pass_gates(report::ValidationReport) = filter(g -> g.status == PASS, report.gates)

"""
    skip_gates(report::ValidationReport) -> Vector{GateResult}

Return all gates with SKIP status.
"""
skip_gates(report::ValidationReport) = filter(g -> g.status == SKIP, report.gates)

"""
    passed(report::ValidationReport) -> Bool

Return true if no HALT conditions exist (PASS, WARN, or SKIP only).
"""
passed(report::ValidationReport) = status(report) != HALT

# Pretty printing
function Base.show(io::IO, g::GateResult)
    status_str = string(g.status)
    print(io, "GateResult(:$(g.name), $status_str)")
end

function Base.show(io::IO, ::MIME"text/plain", g::GateResult)
    println(io, "GateResult")
    println(io, "  name: :$(g.name)")
    println(io, "  status: $(g.status)")
    println(io, "  message: \"$(g.message)\"")
    if !isnothing(g.metric_value)
        println(io, "  metric_value: $(g.metric_value)")
    end
    if !isnothing(g.threshold)
        println(io, "  threshold: $(g.threshold)")
    end
    if !isempty(g.details)
        println(io, "  details: $(g.details)")
    end
    if !isempty(g.recommendation)
        print(io, "  recommendation: \"$(g.recommendation)\"")
    end
end

function Base.show(io::IO, r::ValidationReport)
    n = length(r.gates)
    s = status(r)
    print(io, "ValidationReport($n gates, status=$s)")
end

function Base.show(io::IO, ::MIME"text/plain", r::ValidationReport)
    println(io, "ValidationReport")
    println(io, "  Overall status: $(status(r))")
    println(io, "  Gates run: $(length(r.gates))")

    halts = halt_gates(r)
    warns = warn_gates(r)
    passes = pass_gates(r)
    skips = skip_gates(r)

    if !isempty(halts)
        println(io, "  HALT ($(length(halts))): ", join([":$(g.name)" for g in halts], ", "))
    end
    if !isempty(warns)
        println(io, "  WARN ($(length(warns))): ", join([":$(g.name)" for g in warns], ", "))
    end
    if !isempty(passes)
        println(io, "  PASS ($(length(passes))): ", join([":$(g.name)" for g in passes], ", "))
    end
    if !isempty(skips)
        print(io, "  SKIP ($(length(skips))): ", join([":$(g.name)" for g in skips], ", "))
    end
end


# =============================================================================
# Cross-Validation Types
# =============================================================================

"""
    SplitInfo

Metadata for a single CV split.

Useful for debugging and visualizing the split structure.

# Fields
- `split_idx::Int`: Zero-based split index
- `train_start::Int`: First training index (inclusive)
- `train_end::Int`: Last training index (inclusive)
- `test_start::Int`: First test index (inclusive)
- `test_end::Int`: Last test index (inclusive)
- `gap::Int`: Gap between train_end and test_start

# Knowledge Tier
[T2] Split metadata from temporalcv Python implementation.
"""
struct SplitInfo
    split_idx::Int
    train_start::Int
    train_end::Int
    test_start::Int
    test_end::Int
    gap::Int
end

"""
    train_size(info::SplitInfo) -> Int

Number of training observations.
"""
train_size(info::SplitInfo) = info.train_end - info.train_start + 1

"""
    test_size(info::SplitInfo) -> Int

Number of test observations.
"""
test_size(info::SplitInfo) = info.test_end - info.test_start + 1

function Base.show(io::IO, s::SplitInfo)
    print(io, "SplitInfo(split=$(s.split_idx), train=$(s.train_start):$(s.train_end), test=$(s.test_start):$(s.test_end), gap=$(s.gap))")
end


# =============================================================================
# Frozen Thresholds from SPECIFICATION.md
# =============================================================================

"""
    SUSPICIOUS_IMPROVEMENT_HALT

Threshold for HALT: >20% improvement over baseline is "too good to be true".

# Knowledge Tier
[T3] Empirical heuristic from myga-forecasting-v2 postmortem.
"""
const SUSPICIOUS_IMPROVEMENT_HALT = 0.20

"""
    SUSPICIOUS_IMPROVEMENT_WARN

Threshold for WARN: 10-20% improvement = proceed with caution.

# Knowledge Tier
[T3] Empirical heuristic from myga-forecasting-v2 postmortem.
"""
const SUSPICIOUS_IMPROVEMENT_WARN = 0.10

"""
    SHUFFLED_TARGET_THRESHOLD

Standard p-value threshold for shuffled target test.

# Knowledge Tier
[T3] Standard significance level for hypothesis testing.
"""
const SHUFFLED_TARGET_THRESHOLD = 0.05

"""
    DEFAULT_N_SHUFFLES

Default number of shuffles for shuffled target test.
Balance between statistical power and runtime.

# Knowledge Tier
[T3] Speed/power tradeoff choice (100 too slow for typical use).
"""
const DEFAULT_N_SHUFFLES = 5

"""
    AR1_PHI

High persistence coefficient for synthetic AR(1) test.

# Knowledge Tier
[T2] Typical of financial/economic data.
"""
const AR1_PHI = 0.95

"""
    AR1_TOLERANCE

Tolerance factor for finite-sample variation in AR(1) bounds.

# Knowledge Tier
[T3] Finite-sample allowance.
"""
const AR1_TOLERANCE = 1.5

"""
    THEORETICAL_AR1_MAE_FACTOR

Theoretical MAE for AR(1) process = sigma * sqrt(2/pi) ≈ 0.798 * sigma.
Derived from the half-normal distribution mean.

# Knowledge Tier
[T1] Mathematical derivation (half-normal mean).
"""
const THEORETICAL_AR1_MAE_FACTOR = sqrt(2 / π)  # ≈ 0.7978845608

"""
    DM_TEST_MIN_SAMPLES

Minimum samples required for DM test (CLT requirement).

# Knowledge Tier
[T2] CLT requirement for asymptotic normality + Harvey adjustment.
"""
const DM_TEST_MIN_SAMPLES = 30

"""
    PT_TEST_MIN_SAMPLES

Minimum samples required for PT test (variance estimation stability).

# Knowledge Tier
[T2] Variance estimation stability.
"""
const PT_TEST_MIN_SAMPLES = 20

"""
    DEFAULT_ALPHA

Default significance level / miscoverage rate.

# Knowledge Tier
[T1] Standard statistical convention.
"""
const DEFAULT_ALPHA = 0.05

"""
    CONFORMAL_CALIBRATION_FRACTION

Default fraction of data for conformal calibration.

# Knowledge Tier
[T2] Standard split conformal practice.
"""
const CONFORMAL_CALIBRATION_FRACTION = 0.30

"""
    ADAPTIVE_CONFORMAL_GAMMA

Learning rate for adaptive conformal inference.

# Reference
Gibbs & Candes (2021). "Adaptive conformal inference under distribution shift." NeurIPS.

# Knowledge Tier
[T1] From Gibbs & Candes (2021).
"""
const ADAPTIVE_CONFORMAL_GAMMA = 0.1

"""
    MOVE_THRESHOLD_PERCENTILE

Percentile of |actuals| defining "significant" move (70th percentile ≈ 30% moves, 70% flat).

# Knowledge Tier
[T2] From myga-forecasting-v2 Phase 11 analysis.
"""
const MOVE_THRESHOLD_PERCENTILE = 70.0

"""
    VOLATILITY_WINDOW

Window for rolling volatility computation (weeks).

# Knowledge Tier
[T3] ~13 weeks for quarterly volatility.
"""
const VOLATILITY_WINDOW = 13

"""
    VOLATILITY_LOW_PERCENTILE

Low tercile boundary for volatility classification.

# Knowledge Tier
[T3] Tercile split.
"""
const VOLATILITY_LOW_PERCENTILE = 33.0

"""
    VOLATILITY_HIGH_PERCENTILE

High tercile boundary for volatility classification.

# Knowledge Tier
[T3] Tercile split.
"""
const VOLATILITY_HIGH_PERCENTILE = 67.0
