# dm.jl - Diebold-Mariano test for equal predictive accuracy
#
# Implements the DM test (Diebold & Mariano 1995) with Harvey et al. (1997)
# small-sample correction and HAC variance estimation.
#
# Knowledge Tiers:
#   [T1] Diebold & Mariano (1995). Comparing predictive accuracy.
#   [T1] Harvey et al. (1997). Small-sample adjustment.
#   [T1] Newey & West (1987). HAC variance estimation.

using StatsBase: mean
using Distributions: TDist, Normal, cdf

# Import from parent module (will be set up in StatisticalTests.jl)
# using ..TemporalValidation: DM_TEST_MIN_SAMPLES

"""
    DMTestResult

Result from Diebold-Mariano test for equal predictive accuracy.

# Fields
- `statistic::Float64`: DM test statistic (asymptotically N(0,1) under H0)
- `pvalue::Float64`: P-value for the test
- `h::Int`: Forecast horizon used
- `n::Int`: Number of observations
- `loss::Symbol`: Loss function used (`:squared` or `:absolute`)
- `alternative::Symbol`: Alternative hypothesis (`:two_sided`, `:less`, `:greater`)
- `harvey_adjusted::Bool`: Whether Harvey et al. (1997) adjustment was applied
- `mean_loss_diff::Float64`: Mean loss differential (positive = model 1 has higher loss)

# Interpretation
- Positive statistic: Model 1 has higher average loss (worse)
- Negative statistic: Model 1 has lower average loss (better)
- `alternative=:less` tests H1: Model 1 is better (lower loss)

# Knowledge Tier
[T1] Diebold & Mariano (1995). Comparing predictive accuracy.
     Journal of Business & Economic Statistics, 13(3), 253-263.
[T1] Harvey et al. (1997). Testing the equality of prediction mean squared errors.
     International Journal of Forecasting, 13(2), 281-291.
"""
struct DMTestResult
    statistic::Float64
    pvalue::Float64
    h::Int
    n::Int
    loss::Symbol
    alternative::Symbol
    harvey_adjusted::Bool
    mean_loss_diff::Float64
end

# Convenience predicates
"""
    significant_at_05(r::DMTestResult) -> Bool

Is the test result significant at α = 0.05?
"""
significant_at_05(r::DMTestResult) = r.pvalue < 0.05

"""
    significant_at_01(r::DMTestResult) -> Bool

Is the test result significant at α = 0.01?
"""
significant_at_01(r::DMTestResult) = r.pvalue < 0.01

# Pretty printing
function Base.show(io::IO, r::DMTestResult)
    sig = if r.pvalue < 0.01
        "***"
    elseif r.pvalue < 0.05
        "**"
    elseif r.pvalue < 0.10
        "*"
    else
        ""
    end
    print(io, "DM(h=$(r.h)): $(round(r.statistic, digits=3)) (p=$(round(r.pvalue, digits=4)))$sig")
end

function Base.show(io::IO, ::MIME"text/plain", r::DMTestResult)
    println(io, "DMTestResult")
    println(io, "  statistic: $(round(r.statistic, digits=4))")
    println(io, "  pvalue: $(round(r.pvalue, digits=6))")
    println(io, "  h: $(r.h)")
    println(io, "  n: $(r.n)")
    println(io, "  loss: :$(r.loss)")
    println(io, "  alternative: :$(r.alternative)")
    println(io, "  harvey_adjusted: $(r.harvey_adjusted)")
    print(io, "  mean_loss_diff: $(round(r.mean_loss_diff, digits=6))")
end


"""
    dm_test(errors_1, errors_2; kwargs...) -> DMTestResult

Diebold-Mariano test for equal predictive accuracy.

Tests H0: E[dₜ] = 0 where dₜ = L(e₁ₜ) - L(e₂ₜ) is the loss differential.

# Arguments
- `errors_1::AbstractVector{<:Real}`: Forecast errors from model 1 (actual - prediction)
- `errors_2::AbstractVector{<:Real}`: Forecast errors from model 2 (baseline)

# Keyword Arguments
- `h::Int = 1`: Forecast horizon. Used for HAC bandwidth (h-1) and Harvey adjustment.
- `loss::Symbol = :squared`: Loss function (`:squared` or `:absolute`)
- `alternative::Symbol = :two_sided`: Alternative hypothesis:
    - `:two_sided`: Models have different accuracy (H1: E[dₜ] ≠ 0)
    - `:less`: Model 1 more accurate (H1: E[dₜ] < 0, i.e., lower loss)
    - `:greater`: Model 2 more accurate (H1: E[dₜ] > 0, i.e., model 1 has higher loss)
- `harvey_correction::Bool = true`: Apply Harvey et al. (1997) small-sample adjustment.
  Recommended for n < 100 or h > 1.

# Returns
`DMTestResult` containing test statistic, p-value, and diagnostics.

# Notes

## Harvey Adjustment [T1]
The Harvey et al. (1997) small-sample adjustment corrects for bias in the
variance estimate when n is small or h > 1:

    DM_adj = DM × √((n + 1 - 2h + h(h-1)/n) / n)

When Harvey correction is applied, the t-distribution (df = n-1) is used
instead of the normal distribution.

## HAC Variance [T1]
For h-step forecasts, errors are MA(h-1) and thus autocorrelated up to lag h-1.
The HAC bandwidth is set to max(0, h-1) to account for this.

## Important Limitations (Diebold 2015 retrospective)
1. **Designed for forecasts, not models**: The DM test compares forecasts
   under a fixed DGP, not models (nested model comparison requires Clark-West 2007).
2. **Negative variance**: HAC can produce negative variance with small samples
   or high autocorrelation. Returns pvalue=1.0 with warning in this case.
3. **Size distortions**: Even with Harvey adjustment, the test may have
   incorrect size for n < 50.

# Knowledge Tier
[T1] Diebold & Mariano (1995). Comparing predictive accuracy.
[T1] Harvey et al. (1997). Testing the equality of prediction MSEs.
[T1] Newey & West (1987). HAC covariance estimation.

# Example
```julia
using TemporalValidation

# Compare model to baseline
model_errors = randn(100) .* 0.8
baseline_errors = randn(100) .* 1.0

result = dm_test(model_errors, baseline_errors; h=2, alternative=:less)
if significant_at_05(result)
    println("Model significantly better than baseline")
end
```

# See Also
- `pt_test`: Complementary test for directional accuracy.
- `compute_hac_variance`: HAC variance estimator used internally.
- `compare_multiple_models`: Compare more than two models.
"""
function dm_test(
    errors_1::AbstractVector{<:Real},
    errors_2::AbstractVector{<:Real};
    h::Int = 1,
    loss::Symbol = :squared,
    alternative::Symbol = :two_sided,
    harvey_correction::Bool = true,
    min_samples::Int = 30  # DM_TEST_MIN_SAMPLES from types.jl
)::DMTestResult

    # =========================================================================
    # Input Validation
    # =========================================================================

    # Check for NaN values
    if any(isnan, errors_1)
        throw(ArgumentError(
            "errors_1 contains NaN values. Clean data before processing. " *
            "Use filter(!isnan, errors_1) to remove missing values."
        ))
    end
    if any(isnan, errors_2)
        throw(ArgumentError(
            "errors_2 contains NaN values. Clean data before processing. " *
            "Use filter(!isnan, errors_2) to remove missing values."
        ))
    end

    # Check equal lengths
    if length(errors_1) != length(errors_2)
        throw(ArgumentError(
            "Error arrays must have same length. " *
            "Got $(length(errors_1)) and $(length(errors_2))"
        ))
    end

    n = length(errors_1)

    # Minimum sample size check
    if n < min_samples
        throw(ArgumentError(
            "Insufficient samples for reliable DM test. " *
            "Need >= $min_samples, got $n. " *
            "For n < $min_samples, consider bootstrap-based tests or qualitative comparison."
        ))
    end

    # Validate horizon
    if h < 1
        throw(ArgumentError("Horizon h must be >= 1, got $h"))
    end

    # Validate loss function
    if loss ∉ (:squared, :absolute)
        throw(ArgumentError(
            "loss must be :squared or :absolute, got :$loss"
        ))
    end

    # Validate alternative
    if alternative ∉ (:two_sided, :less, :greater)
        throw(ArgumentError(
            "alternative must be :two_sided, :less, or :greater, got :$alternative"
        ))
    end

    # =========================================================================
    # Compute Loss Differential
    # =========================================================================

    loss_1 = if loss == :squared
        errors_1 .^ 2
    else  # :absolute
        abs.(errors_1)
    end

    loss_2 = if loss == :squared
        errors_2 .^ 2
    else
        abs.(errors_2)
    end

    # Positive d_t means model 1 has higher loss (worse)
    d = loss_1 .- loss_2
    d_bar = mean(d)

    # =========================================================================
    # HAC Variance Estimation
    # =========================================================================

    # For h-step forecasts, errors are MA(h-1)
    # Use bandwidth = h-1 to capture autocorrelation structure
    bandwidth = max(0, h - 1)

    # Warn if bandwidth is large relative to sample size
    if bandwidth > n / 4
        @warn "DM test bandwidth ($bandwidth) exceeds n/4 ($(n÷4)). " *
              "HAC variance estimation may be unreliable with long forecast horizons " *
              "relative to sample size. Consider: (1) increasing sample size, " *
              "(2) using bootstrap-based tests, (3) reducing forecast horizon."
    end

    var_d = compute_hac_variance(d; bandwidth=bandwidth)

    # =========================================================================
    # Handle Degenerate Cases
    # =========================================================================

    if var_d <= 0
        @warn "DM test variance is non-positive (var_d=$(round(var_d, sigdigits=3))). " *
              "This can occur when loss differences are constant or nearly constant. " *
              "Returning pvalue=1.0 (cannot reject null). " *
              "Consider: (1) checking for identical predictions, " *
              "(2) using bootstrap-based tests for small samples."
        return DMTestResult(
            NaN, 1.0, h, n, loss, alternative, harvey_correction, d_bar
        )
    end

    # =========================================================================
    # DM Statistic
    # =========================================================================

    dm_stat = d_bar / sqrt(var_d)

    # =========================================================================
    # Harvey Adjustment [T1]
    # =========================================================================

    if harvey_correction
        # Harvey et al. (1997) small-sample adjustment
        # DM_adj = DM × √((n + 1 - 2h + h(h-1)/n) / n)
        adjustment = sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
        dm_stat = dm_stat * adjustment
    end

    # =========================================================================
    # P-value Computation
    # =========================================================================

    # Distribution choice:
    # - With Harvey correction: t-distribution (df = n-1) for small-sample inference
    # - Without Harvey correction: Normal distribution for asymptotic inference
    if harvey_correction
        dist = TDist(n - 1)
    else
        dist = Normal()
    end

    pvalue = if alternative == :two_sided
        # Two-sided: P(|Z| > |dm_stat|) = 2 × P(Z > |dm_stat|)
        2 * (1 - cdf(dist, abs(dm_stat)))
    elseif alternative == :less
        # H1: Model 1 better (lower loss) => d_bar < 0 => dm_stat < 0
        # P-value = P(Z < dm_stat)
        cdf(dist, dm_stat)
    else  # :greater
        # H1: Model 2 better => d_bar > 0 => dm_stat > 0
        # P-value = P(Z > dm_stat) = 1 - P(Z < dm_stat)
        1 - cdf(dist, dm_stat)
    end

    return DMTestResult(
        dm_stat, pvalue, h, n, loss, alternative, harvey_correction, d_bar
    )
end
