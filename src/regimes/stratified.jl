# stratified.jl - Stratified metrics by regime
#
# Compute MAE and RMSE stratified by regime for conditional performance analysis.
#
# Knowledge Tiers:
#   [T1] Standard stratified analysis approach
#   [T3] Minimum n=10 per regime for reliability

using StatsBase: mean

"""
    StratifiedMetricsResult

Metrics stratified by regime.

Provides overall metrics and per-regime breakdown for MAE, RMSE, and sample counts.

# Fields
- `overall_mae::Float64`: Mean Absolute Error across all samples
- `overall_rmse::Float64`: Root Mean Squared Error across all samples
- `n_total::Int`: Total number of samples
- `by_regime::Dict{Symbol, Dict{Symbol, Any}}`: Per-regime metrics. Each key is a
  regime label, value is Dict with :mae, :rmse, :n, :pct
- `masked_regimes::Vector{Symbol}`: Regimes with n < min_n that were excluded

# Example
```julia
result = compute_stratified_metrics(preds, actuals, regimes)
println("Overall MAE: \$(result.overall_mae)")
for (regime, metrics) in result.by_regime
    println("\$regime: MAE=\$(metrics[:mae]), n=\$(metrics[:n])")
end
```
"""
struct StratifiedMetricsResult
    overall_mae::Float64
    overall_rmse::Float64
    n_total::Int
    by_regime::Dict{Symbol, Dict{Symbol, Any}}
    masked_regimes::Vector{Symbol}
end

# Pretty printing
function Base.show(io::IO, r::StratifiedMetricsResult)
    n_regimes = length(r.by_regime)
    n_masked = length(r.masked_regimes)
    print(io, "StratifiedMetricsResult(n=$(r.n_total), regimes=$n_regimes, masked=$n_masked)")
end

function Base.show(io::IO, ::MIME"text/plain", r::StratifiedMetricsResult)
    println(io, "StratifiedMetricsResult")
    println(io, "  overall_mae: $(round(r.overall_mae, digits=4))")
    println(io, "  overall_rmse: $(round(r.overall_rmse, digits=4))")
    println(io, "  n_total: $(r.n_total)")
    println(io, "  regimes: $(length(r.by_regime))")
    if !isempty(r.masked_regimes)
        println(io, "  masked: $(r.masked_regimes)")
    end

    # Sort by n descending
    sorted_regimes = sort(collect(r.by_regime); by=x -> x[2][:n], rev=true)

    for (regime, metrics) in sorted_regimes
        mae = round(metrics[:mae], digits=4)
        rmse = round(metrics[:rmse], digits=4)
        n = metrics[:n]
        pct = round(metrics[:pct], digits=1)
        println(io, "    $regime: MAE=$mae, RMSE=$rmse, n=$n ($pct%)")
    end
end


"""
    Base.summary(r::StratifiedMetricsResult) -> String

Generate human-readable summary of stratified metrics.

# Example
```julia
result = compute_stratified_metrics(preds, actuals, regimes)
println(summary(result))
```
"""
function Base.summary(r::StratifiedMetricsResult)::String
    lines = String[]

    push!(lines, "Overall: MAE=$(round(r.overall_mae, digits=4)), " *
                 "RMSE=$(round(r.overall_rmse, digits=4)), n=$(r.n_total)")
    push!(lines, "")
    push!(lines, "By Regime:")

    # Sort by n descending
    sorted_regimes = sort(collect(r.by_regime); by=x -> x[2][:n], rev=true)

    for (regime, metrics) in sorted_regimes
        mae = round(metrics[:mae], digits=4)
        rmse = round(metrics[:rmse], digits=4)
        n = metrics[:n]
        pct = round(metrics[:pct], digits=1)
        push!(lines, "  $regime: MAE=$mae, RMSE=$rmse, n=$n ($pct%)")
    end

    if !isempty(r.masked_regimes)
        push!(lines, "")
        push!(lines, "Masked (n < min_n): $(join(r.masked_regimes, ", "))")
    end

    return join(lines, "\n")
end


"""
    compute_stratified_metrics(predictions, actuals, regimes; min_n=10) -> StratifiedMetricsResult

Compute MAE and RMSE stratified by regime.

Provides automatic per-regime breakdown of prediction errors, essential for
understanding model performance across different market conditions.

# Arguments
- `predictions::AbstractVector{<:Real}`: Model predictions
- `actuals::AbstractVector{<:Real}`: Actual values
- `regimes::AbstractVector{Symbol}`: Regime labels for each point
  (e.g., :LOW, :MED, :HIGH or :UP, :DOWN, :FLAT)
- `min_n::Int=10`: Minimum samples required per regime [T3].
  Regimes with fewer samples are masked and reported in `masked_regimes`.

# Returns
`StratifiedMetricsResult` with overall metrics, per-regime breakdown, and masked regimes.

# Notes
This function is designed for post-hoc analysis of walk-forward results.
The `min_n` threshold helps avoid drawing conclusions from statistically
unreliable subsets. Default of 10 is a conservative rule of thumb [T3].

# Knowledge Tier
[T1] Standard stratified analysis.
[T3] Minimum n=10 per regime (rule of thumb).

# Example
```julia
# Classify regimes
vol_regimes = classify_volatility_regime(actuals; basis=:changes)

# Compute stratified metrics
result = compute_stratified_metrics(predictions, actuals, vol_regimes)
println(summary(result))
```

Output:
```
Overall: MAE=0.0234, RMSE=0.0312, n=200

By Regime:
  LOW: MAE=0.0156, RMSE=0.0201, n=67 (33.5%)
  MED: MAE=0.0245, RMSE=0.0298, n=66 (33.0%)
  HIGH: MAE=0.0301, RMSE=0.0437, n=67 (33.5%)
```

# See Also
- `classify_volatility_regime`: Classify by volatility level.
- `classify_direction_regime`: Classify by direction.
- `mask_low_n_regimes`: Mask regimes with insufficient samples.
"""
function compute_stratified_metrics(
    predictions::AbstractVector{<:Real},
    actuals::AbstractVector{<:Real},
    regimes::AbstractVector{Symbol};
    min_n::Int = 10
)::StratifiedMetricsResult

    # Validation
    if isempty(predictions) || isempty(actuals)
        throw(ArgumentError("predictions and actuals cannot be empty"))
    end

    if length(predictions) != length(actuals)
        throw(ArgumentError(
            "predictions and actuals must have same length. " *
            "Got $(length(predictions)) and $(length(actuals))."
        ))
    end

    if length(predictions) != length(regimes)
        throw(ArgumentError(
            "regimes must have same length as predictions. " *
            "Got $(length(regimes)) regimes for $(length(predictions)) predictions."
        ))
    end

    preds = Float64.(predictions)
    acts = Float64.(actuals)

    # Compute overall metrics
    errors = preds .- acts
    overall_mae = mean(abs.(errors))
    overall_rmse = sqrt(mean(errors .^ 2))
    n_total = length(preds)

    # Get unique regimes and counts
    unique_regimes = unique(regimes)
    regime_counts = get_regime_counts(regimes)

    # Identify masked regimes
    masked_regimes = Symbol[r for (r, c) in regime_counts if c < min_n]

    # Compute per-regime metrics
    by_regime = Dict{Symbol, Dict{Symbol, Any}}()

    for regime in unique_regimes
        n_regime = regime_counts[regime]

        if n_regime < min_n
            # Skip masked regimes
            continue
        end

        mask = regimes .== regime
        regime_errors = errors[mask]

        mae = mean(abs.(regime_errors))
        rmse = sqrt(mean(regime_errors .^ 2))
        pct = 100.0 * n_regime / n_total

        by_regime[regime] = Dict{Symbol, Any}(
            :mae => mae,
            :rmse => rmse,
            :n => n_regime,
            :pct => pct
        )
    end

    return StratifiedMetricsResult(
        overall_mae,
        overall_rmse,
        n_total,
        by_regime,
        masked_regimes
    )
end
