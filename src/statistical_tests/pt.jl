# pt.jl - Pesaran-Timmermann test for directional accuracy
#
# Implements the PT test (Pesaran & Timmermann 1992) for testing whether
# directional predictions are significantly better than random guessing.
#
# Knowledge Tiers:
#   [T1] Pesaran & Timmermann (1992). 2-class variance formula (equation 8).
#   [T3] 3-class mode is ad-hoc extension (not published, exploratory use only).

using StatsBase: mean
using Distributions: Normal, cdf

"""
    PTTestResult

Result from Pesaran-Timmermann directional accuracy test.

# Fields
- `statistic::Float64`: PT test statistic (z-score, asymptotically N(0,1) under H0)
- `pvalue::Float64`: P-value (one-sided: testing if better than random)
- `accuracy::Float64`: Observed directional accuracy
- `expected::Float64`: Expected accuracy under null hypothesis (independence)
- `n::Int`: Number of observations used (may differ from input length in 2-class)
- `n_classes::Int`: Number of direction classes (2 or 3)

# Interpretation
- `accuracy > expected` suggests directional skill
- `skill = accuracy - expected` measures the skill above random
- One-sided test: significant pvalue means accuracy > random at that level

# Knowledge Tier
[T1] Pesaran & Timmermann (1992). A simple nonparametric test of predictive
     performance. Journal of Business & Economic Statistics, 10(4), 461-465.
[T3] 3-class mode uses ad-hoc variance approximation.
"""
struct PTTestResult
    statistic::Float64
    pvalue::Float64
    accuracy::Float64
    expected::Float64
    n::Int
    n_classes::Int
end

# Convenience predicates
"""
    significant_at_05(r::PTTestResult) -> Bool

Is directional accuracy significantly better than random at α = 0.05?
"""
significant_at_05(r::PTTestResult) = r.pvalue < 0.05

"""
    significant_at_01(r::PTTestResult) -> Bool

Is directional accuracy significantly better than random at α = 0.01?
"""
significant_at_01(r::PTTestResult) = r.pvalue < 0.01

"""
    skill(r::PTTestResult) -> Float64

Directional skill = accuracy - expected.
Positive skill means predicting direction better than random.
"""
skill(r::PTTestResult) = r.accuracy - r.expected

# Pretty printing
function Base.show(io::IO, r::PTTestResult)
    sig = if r.pvalue < 0.01
        "***"
    elseif r.pvalue < 0.05
        "**"
    elseif r.pvalue < 0.10
        "*"
    else
        ""
    end
    acc_pct = round(r.accuracy * 100, digits=1)
    exp_pct = round(r.expected * 100, digits=1)
    print(io, "PT: $(acc_pct)% vs $(exp_pct)% expected (z=$(round(r.statistic, digits=3)), p=$(round(r.pvalue, digits=4)))$sig")
end

function Base.show(io::IO, ::MIME"text/plain", r::PTTestResult)
    println(io, "PTTestResult")
    println(io, "  statistic: $(round(r.statistic, digits=4))")
    println(io, "  pvalue: $(round(r.pvalue, digits=6))")
    println(io, "  accuracy: $(round(r.accuracy * 100, digits=2))%")
    println(io, "  expected: $(round(r.expected * 100, digits=2))%")
    println(io, "  skill: $(round(skill(r) * 100, digits=2))%")
    println(io, "  n: $(r.n)")
    print(io, "  n_classes: $(r.n_classes)")
end


"""
    pt_test(actual, predicted; move_threshold=nothing, min_samples=20) -> PTTestResult

Pesaran-Timmermann test for directional accuracy.

Tests whether the model's ability to predict direction (sign) is
significantly better than random guessing.

# Arguments
- `actual::AbstractVector{<:Real}`: Actual values (typically changes/returns)
- `predicted::AbstractVector{<:Real}`: Predicted values (typically changes/returns)

# Keyword Arguments
- `move_threshold::Union{Float64, Nothing} = nothing`: If provided, uses 3-class mode:
    - UP: value > threshold
    - DOWN: value < -threshold
    - FLAT: |value| <= threshold

  If `nothing`, uses 2-class (positive/negative sign) comparison.
  Using a threshold is recommended when comparing against a persistence baseline
  (which predicts 0 = FLAT).

- `min_samples::Int = 20`: Minimum number of samples required (PT_TEST_MIN_SAMPLES)

# Returns
`PTTestResult` with test statistic, p-value, accuracy, and expected accuracy.

# Hypotheses
- H0: Direction predictions are independent of actual directions (no skill)
- H1: Direction predictions have skill (one-sided test, better than random)

# Modes

## 2-Class Mode [T1]
When `move_threshold=nothing`:
- Compares sign of actual vs predicted
- Excludes zero values (undefined direction)
- Uses exact variance formula from PT (1992) equation 8
- Academically validated

## 3-Class Mode [T3]
When `move_threshold` is provided:
- Classifies into UP (>threshold), DOWN (<-threshold), FLAT (within)
- Includes all samples
- Uses ad-hoc variance approximation (×4 heuristic factor)
- **WARNING**: Not published, use for exploratory analysis only

# Notes
The test accounts for marginal probabilities of directions in both actual
and predicted series, providing a proper baseline comparison.

For h > 1 step forecasts, the current variance formula does NOT include
HAC correction, so p-values may be overly optimistic. Consider using
DM test which includes proper HAC adjustment for multi-step testing.

# Knowledge Tier
[T1] Pesaran & Timmermann (1992). 2-class variance formula (equation 8).
[T3] 3-class mode is ad-hoc extension (not published).

# Example
```julia
using TemporalValidation

# Test with 3-class (UP/DOWN/FLAT)
actual_changes = randn(100)
pred_changes = 0.3 .* actual_changes .+ 0.7 .* randn(100)

result = pt_test(actual_changes, pred_changes; move_threshold=0.5)
println("Accuracy: \$(round(result.accuracy * 100, digits=1))%")
println("Skill: \$(round(skill(result) * 100, digits=1))%")

if significant_at_05(result)
    println("Direction accuracy significantly better than random")
end
```

# See Also
- `dm_test`: Complementary test for predictive accuracy (magnitude).
- `significant_at_05`: Check significance at α = 0.05.
- `skill`: Compute directional skill (accuracy - expected).
"""
function pt_test(
    actual::AbstractVector{<:Real},
    predicted::AbstractVector{<:Real};
    move_threshold::Union{Float64, Nothing} = nothing,
    min_samples::Int = 20  # PT_TEST_MIN_SAMPLES from types.jl
)::PTTestResult

    # =========================================================================
    # Input Validation
    # =========================================================================

    # Check for NaN values
    if any(isnan, actual)
        throw(ArgumentError(
            "actual contains NaN values. Clean data before processing. " *
            "Use filter(!isnan, actual) to remove missing values."
        ))
    end
    if any(isnan, predicted)
        throw(ArgumentError(
            "predicted contains NaN values. Clean data before processing. " *
            "Use filter(!isnan, predicted) to remove missing values."
        ))
    end

    # Check equal lengths
    if length(actual) != length(predicted)
        throw(ArgumentError(
            "Arrays must have same length. " *
            "Got actual=$(length(actual)), predicted=$(length(predicted))"
        ))
    end

    n = length(actual)

    # Minimum sample size check
    if n < min_samples
        throw(ArgumentError(
            "Insufficient samples for PT test. " *
            "Need >= $min_samples, got $n"
        ))
    end

    # Validate move_threshold if provided
    if !isnothing(move_threshold) && move_threshold < 0
        throw(ArgumentError(
            "move_threshold must be >= 0, got $move_threshold"
        ))
    end

    # =========================================================================
    # Dispatch to appropriate mode
    # =========================================================================

    if isnothing(move_threshold)
        return _pt_test_2class(actual, predicted)
    else
        return _pt_test_3class(actual, predicted, move_threshold)
    end
end


"""
    _pt_test_2class(actual, predicted) -> PTTestResult

Internal: 2-class PT test using sign comparison.

Excludes zero values (undefined direction).
Uses exact variance formula from PT (1992) equation 8.

# Knowledge Tier
[T1] Pesaran & Timmermann (1992). 2-class variance formula.
"""
function _pt_test_2class(
    actual::AbstractVector{<:Real},
    predicted::AbstractVector{<:Real}
)::PTTestResult

    # Sign classification
    actual_sign = sign.(actual)
    pred_sign = sign.(predicted)

    # Filter out zeros (undefined direction)
    nonzero_mask = actual_sign .!= 0
    n_effective = sum(nonzero_mask)

    # Handle degenerate case: all zeros
    if n_effective == 0
        @warn "PT test (2-class): no non-zero actual values. " *
              "All actual values may be zero. Returning pvalue=1.0. " *
              "Consider using 3-class mode with move_threshold parameter."
        return PTTestResult(NaN, 1.0, 0.0, 0.5, 0, 2)
    end

    # Compute accuracy on non-zero subset
    correct = actual_sign[nonzero_mask] .== pred_sign[nonzero_mask]
    p_hat = mean(correct)

    # Marginal probabilities (on non-zero subset)
    # p_y_pos = P(actual > 0), p_x_pos = P(predicted > 0)
    p_y_pos = mean(actual[nonzero_mask] .> 0)
    p_x_pos = mean(predicted[nonzero_mask] .> 0)

    # Expected accuracy under independence (null hypothesis)
    # p* = P(same sign) = P(both +) + P(both -)
    #    = p_y_pos × p_x_pos + (1 - p_y_pos) × (1 - p_x_pos)
    p_star = p_y_pos * p_x_pos + (1 - p_y_pos) * (1 - p_x_pos)

    # =========================================================================
    # Variance Estimates [T1] PT (1992) equation 8
    # =========================================================================

    # Variance of p̂ under the null
    var_p_hat = p_star * (1 - p_star) / n_effective

    # Variance of p* (the random baseline) - accounts for estimation uncertainty
    # From PT (1992) equation 8:
    # Var(p*) = term1 + term2 + term3
    term1 = (2*p_y_pos - 1)^2 * p_x_pos * (1 - p_x_pos) / n_effective
    term2 = (2*p_x_pos - 1)^2 * p_y_pos * (1 - p_y_pos) / n_effective
    term3 = 4 * p_y_pos * p_x_pos * (1 - p_y_pos) * (1 - p_x_pos) / n_effective
    var_p_star = term1 + term2 + term3

    # Total variance under null
    var_total = var_p_hat + var_p_star

    # Handle degenerate variance
    if var_total <= 0
        @warn "PT test (2-class): non-positive total variance (var_total=$(round(var_total, sigdigits=3))). " *
              "This can occur with degenerate probability estimates. " *
              "Returning pvalue=1.0 (cannot reject null). " *
              "Check that predictions have variance."
        return PTTestResult(NaN, 1.0, p_hat, p_star, n_effective, 2)
    end

    # =========================================================================
    # PT Statistic and P-value
    # =========================================================================

    # z-score
    pt_stat = (p_hat - p_star) / sqrt(var_total)

    # One-sided p-value (testing if better than random)
    # P-value = P(Z > pt_stat) = 1 - Φ(pt_stat)
    pvalue = 1 - cdf(Normal(), pt_stat)

    return PTTestResult(pt_stat, pvalue, p_hat, p_star, n_effective, 2)
end


"""
    _pt_test_3class(actual, predicted, threshold) -> PTTestResult

Internal: 3-class PT test using UP/DOWN/FLAT classification.

Includes all samples.
Uses ad-hoc variance approximation (×4 heuristic factor).

# Warning
This is an ad-hoc extension [T3]. Use 2-class mode for rigorous testing.

# Knowledge Tier
[T3] 3-class variance is ad-hoc approximation, not published.
"""
function _pt_test_3class(
    actual::AbstractVector{<:Real},
    predicted::AbstractVector{<:Real},
    threshold::Float64
)::PTTestResult

    n = length(actual)

    # 3-class classification function
    # UP = 1, DOWN = -1, FLAT = 0
    function classify(v::Real)::Int8
        if v > threshold
            return Int8(1)   # UP
        elseif v < -threshold
            return Int8(-1)  # DOWN
        else
            return Int8(0)   # FLAT
        end
    end

    # Classify all values
    actual_class = classify.(actual)
    pred_class = classify.(predicted)

    # Accuracy on all samples
    correct = actual_class .== pred_class
    p_hat = mean(correct)

    # Marginal probabilities for each class
    # p_y[c] = P(actual_class == c), p_x[c] = P(pred_class == c)
    p_y = Dict{Int8, Float64}(
        Int8(1) => mean(actual_class .== 1),
        Int8(-1) => mean(actual_class .== -1),
        Int8(0) => mean(actual_class .== 0)
    )
    p_x = Dict{Int8, Float64}(
        Int8(1) => mean(pred_class .== 1),
        Int8(-1) => mean(pred_class .== -1),
        Int8(0) => mean(pred_class .== 0)
    )

    # Expected accuracy under independence
    # p* = Σ_c P(actual=c) × P(pred=c)
    p_star = p_y[Int8(1)]*p_x[Int8(1)] + p_y[Int8(-1)]*p_x[Int8(-1)] + p_y[Int8(0)]*p_x[Int8(0)]

    # =========================================================================
    # Variance Estimates [T3] Ad-hoc approximation
    # =========================================================================

    # Note: The ×4 factor is a heuristic approximation for 3-class.
    # This has NOT been validated against published extensions of PT (1992).
    # Use for exploratory analysis only.
    var_p_hat = p_star * (1 - p_star) / n
    var_p_star = p_star * (1 - p_star) / n * 4  # [T3] heuristic factor

    var_total = var_p_hat + var_p_star

    # Handle degenerate variance
    if var_total <= 0
        @warn "PT test (3-class): non-positive total variance. " *
              "Returning pvalue=1.0 (cannot reject null). " *
              "Check that predictions have variance across all classes."
        return PTTestResult(NaN, 1.0, p_hat, p_star, n, 3)
    end

    # =========================================================================
    # PT Statistic and P-value
    # =========================================================================

    pt_stat = (p_hat - p_star) / sqrt(var_total)
    pvalue = 1 - cdf(Normal(), pt_stat)

    return PTTestResult(pt_stat, pvalue, p_hat, p_star, n, 3)
end
