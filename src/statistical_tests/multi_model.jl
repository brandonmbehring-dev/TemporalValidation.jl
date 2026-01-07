# multi_model.jl - Multi-model comparison using pairwise DM tests
#
# Implements comparison of multiple forecast models with Bonferroni correction
# for multiple comparisons.
#
# Knowledge Tiers:
#   [T1] Bonferroni correction for family-wise error rate control.
#   [T1] Diebold-Mariano test for pairwise comparisons.

using StatsBase: mean

"""
    MultiModelComparisonResult

Result from multi-model comparison using pairwise DM tests with Bonferroni correction.

# Fields
- `pairwise_results::Dict{Tuple{Symbol, Symbol}, DMTestResult}`: (model_a, model_b) => DM result.
  Tests are ordered so model_a has lower mean loss (tests if A is better).
- `best_model::Symbol`: Model with lowest mean loss.
- `bonferroni_alpha::Float64`: Corrected significance level (α / n_comparisons).
- `original_alpha::Float64`: Original significance level before Bonferroni correction.
- `model_rankings::Vector{Tuple{Symbol, Float64}}`: Models sorted by mean loss (ascending).
- `significant_pairs::Vector{Tuple{Symbol, Symbol}}`: Pairs where model_a is significantly
  better than model_b at corrected α level.

# Knowledge Tier
[T1] Bonferroni correction: α_corrected = α / n_comparisons.

# Example
```julia
result = compare_multiple_models(errors_dict; h=2)
println(summary(result))
println("Best model: \$(result.best_model)")
for (a, b) in result.significant_pairs
    println("\$a significantly better than \$b")
end
```
"""
struct MultiModelComparisonResult
    pairwise_results::Dict{Tuple{Symbol, Symbol}, DMTestResult}
    best_model::Symbol
    bonferroni_alpha::Float64
    original_alpha::Float64
    model_rankings::Vector{Tuple{Symbol, Float64}}
    significant_pairs::Vector{Tuple{Symbol, Symbol}}
end

"""
    n_comparisons(r::MultiModelComparisonResult) -> Int

Number of pairwise comparisons performed.
For k models: k(k-1)/2 comparisons.
"""
n_comparisons(r::MultiModelComparisonResult) = length(r.pairwise_results)

"""
    n_significant(r::MultiModelComparisonResult) -> Int

Number of significant differences at Bonferroni-corrected level.
"""
n_significant(r::MultiModelComparisonResult) = length(r.significant_pairs)

# Pretty printing
function Base.show(io::IO, r::MultiModelComparisonResult)
    n_models = length(r.model_rankings)
    print(io, "MultiModelComparisonResult($n_models models, $(n_comparisons(r)) pairs, $(n_significant(r)) significant)")
end

function Base.show(io::IO, ::MIME"text/plain", r::MultiModelComparisonResult)
    println(io, "MultiModelComparisonResult")
    println(io, "  Models: $(length(r.model_rankings))")
    println(io, "  Comparisons: $(n_comparisons(r))")
    println(io, "  Significant pairs: $(n_significant(r))")
    println(io, "  Bonferroni α: $(round(r.bonferroni_alpha, digits=4))")
    println(io, "  Best model: :$(r.best_model)")
    println(io, "  Rankings:")
    for (rank, (name, loss)) in enumerate(r.model_rankings)
        marker = name == r.best_model ? " <- best" : ""
        println(io, "    $rank. :$name (loss=$(round(loss, digits=6)))$marker")
    end
end


"""
    Base.summary(r::MultiModelComparisonResult) -> String

Generate human-readable summary of comparison results.

# Example
```julia
result = compare_multiple_models(errors)
println(summary(result))
```
"""
function Base.summary(r::MultiModelComparisonResult)::String
    lines = String[]

    push!(lines, "Multi-Model Comparison ($(length(r.model_rankings)) models, $(n_comparisons(r)) pairs)")
    push!(lines, "Bonferroni-corrected α = $(round(r.bonferroni_alpha, digits=4)) (original α = $(r.original_alpha))")
    push!(lines, "")
    push!(lines, "Model Rankings (by mean loss):")

    for (rank, (name, loss)) in enumerate(r.model_rankings)
        marker = name == r.best_model ? " <- best" : ""
        push!(lines, "  $rank. :$name: $(round(loss, digits=6))$marker")
    end

    push!(lines, "")

    if !isempty(r.significant_pairs)
        push!(lines, "Significant differences ($(n_significant(r))):")
        for (model_a, model_b) in r.significant_pairs
            result = r.pairwise_results[(model_a, model_b)]
            push!(lines, "  :$model_a > :$model_b: p=$(round(result.pvalue, digits=4))")
        end
    else
        push!(lines, "No significant differences at corrected α level.")
    end

    return join(lines, "\n")
end


"""
    get_pairwise(r::MultiModelComparisonResult, model_a::Symbol, model_b::Symbol) -> Union{DMTestResult, Nothing}

Get DM test result for a specific pair of models.

Lookup is order-independent: `get_pairwise(r, :A, :B)` and `get_pairwise(r, :B, :A)`
will both return the same result (or `nothing` if pair not found).

# Example
```julia
dm_result = get_pairwise(result, :ridge, :baseline)
if !isnothing(dm_result) && significant_at_05(dm_result)
    println("Ridge significantly different from baseline")
end
```
"""
function get_pairwise(r::MultiModelComparisonResult, model_a::Symbol, model_b::Symbol)::Union{DMTestResult, Nothing}
    if haskey(r.pairwise_results, (model_a, model_b))
        return r.pairwise_results[(model_a, model_b)]
    elseif haskey(r.pairwise_results, (model_b, model_a))
        return r.pairwise_results[(model_b, model_a)]
    end
    return nothing
end


"""
    compare_multiple_models(errors_dict; kwargs...) -> MultiModelComparisonResult

Compare multiple models using pairwise DM tests with Bonferroni correction.

Performs all pairwise comparisons and applies Bonferroni correction to control
family-wise error rate.

# Arguments
- `errors_dict::Dict{Symbol, <:AbstractVector{<:Real}}`: Mapping from model name to error array.
  All arrays must have the same length.

# Keyword Arguments
- `h::Int = 1`: Forecast horizon for DM test HAC bandwidth.
- `alpha::Float64 = 0.05`: Significance level (before Bonferroni correction).
- `loss::Symbol = :squared`: Loss function for DM test (`:squared` or `:absolute`).
- `harvey_correction::Bool = true`: Apply Harvey et al. (1997) small-sample adjustment.

# Returns
`MultiModelComparisonResult` with comprehensive comparison results including rankings
and significant pairs.

# Notes

## Bonferroni Correction [T1]
For k models, there are k(k-1)/2 pairwise comparisons:
- 2 models: 1 comparison
- 3 models: 3 comparisons
- 5 models: 10 comparisons
- 10 models: 45 comparisons

The corrected significance level is: α_corrected = α / n_comparisons

## Pairwise Test Direction
Each pairwise test is ordered so the lower-loss model is `errors_1` (model_a),
testing whether model_a is significantly better than model_b. The `alternative=:less`
is used, meaning we test H1: model_a has lower loss.

## Alternatives to Bonferroni
- Holm-Bonferroni (more powerful, sequential)
- Benjamini-Hochberg FDR (controls false discovery rate)
- Model Confidence Set (Hansen 2011)

Bonferroni is the most conservative and widely accepted, hence used here.

# Knowledge Tier
[T1] Bonferroni correction for multiple comparisons.
[T1] Diebold-Mariano test for pairwise forecast comparison.

# Example
```julia
using TemporalValidation

errors = Dict(
    :ridge => model_ridge_errors,
    :lasso => model_lasso_errors,
    :baseline => baseline_errors
)

result = compare_multiple_models(errors; h=2)
println(summary(result))

# Access specific comparison
dm_result = get_pairwise(result, :ridge, :baseline)
```

# See Also
- `dm_test`: Pairwise comparison between two models.
- `summary`: Generate human-readable summary.
- `get_pairwise`: Access specific pairwise result.
"""
function compare_multiple_models(
    errors_dict::Dict{Symbol, <:AbstractVector{<:Real}};
    h::Int = 1,
    alpha::Float64 = 0.05,
    loss::Symbol = :squared,
    harvey_correction::Bool = true
)::MultiModelComparisonResult

    model_names = collect(keys(errors_dict))
    n_models = length(model_names)

    # =========================================================================
    # Input Validation
    # =========================================================================

    if n_models < 2
        throw(ArgumentError(
            "Need at least 2 models to compare. Got $n_models. " *
            "Use dm_test() for single pairwise comparison."
        ))
    end

    # Validate all arrays have same length
    lengths = [length(errors_dict[name]) for name in model_names]
    if length(unique(lengths)) > 1
        length_info = join(["$name=$(lengths[i])" for (i, name) in enumerate(model_names)], ", ")
        throw(ArgumentError("All error arrays must have same length. Got: $length_info"))
    end

    # Validate alpha
    if alpha <= 0 || alpha >= 1
        throw(ArgumentError("alpha must be in (0, 1), got $alpha"))
    end

    # =========================================================================
    # Compute Mean Losses
    # =========================================================================

    mean_losses = Dict{Symbol, Float64}()
    for (name, errors) in errors_dict
        mean_losses[name] = if loss == :squared
            mean(errors .^ 2)
        else  # :absolute
            mean(abs.(errors))
        end
    end

    # Rank models by mean loss (lower is better)
    # Convert Pairs to Tuples for type stability
    sorted_pairs = sort(collect(mean_losses), by=x -> x[2])
    model_rankings = [(p.first, p.second) for p in sorted_pairs]
    best_model = model_rankings[1][1]

    # =========================================================================
    # Bonferroni Correction
    # =========================================================================

    n_comparisons_val = n_models * (n_models - 1) ÷ 2
    bonferroni_alpha = alpha / n_comparisons_val

    # =========================================================================
    # Run All Pairwise DM Tests
    # =========================================================================

    pairwise_results = Dict{Tuple{Symbol, Symbol}, DMTestResult}()
    significant_pairs = Tuple{Symbol, Symbol}[]

    for i in 1:n_models
        for j in (i+1):n_models
            name_a, name_b = model_names[i], model_names[j]

            # Order so lower-loss model is first (tests if A is better)
            if mean_losses[name_a] < mean_losses[name_b]
                better, worse = name_a, name_b
            else
                better, worse = name_b, name_a
            end

            # Run DM test with alternative=:less (testing if better has lower loss)
            result = dm_test(
                errors_dict[better],
                errors_dict[worse];
                h = h,
                loss = loss,
                alternative = :less,
                harvey_correction = harvey_correction
            )

            pairwise_results[(better, worse)] = result

            # Check significance at Bonferroni-corrected level
            if result.pvalue < bonferroni_alpha
                push!(significant_pairs, (better, worse))
            end
        end
    end

    return MultiModelComparisonResult(
        pairwise_results,
        best_model,
        bonferroni_alpha,
        alpha,
        model_rankings,
        significant_pairs
    )
end
