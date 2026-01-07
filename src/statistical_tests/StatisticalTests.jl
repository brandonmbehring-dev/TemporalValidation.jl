# StatisticalTests.jl - Submodule for statistical testing in forecast evaluation
#
# Provides statistical tests for comparing forecast accuracy and directional skill.
#
# Knowledge Tiers:
#   [T1] DM, PT (2-class), HAC variance are academically validated
#   [T3] PT 3-class mode is ad-hoc extension (exploratory use only)

"""
    StatisticalTests

Statistical tests for forecast evaluation.

This submodule provides:
- **Diebold-Mariano test** (`dm_test`): Compare predictive accuracy of two forecasts [T1]
- **Pesaran-Timmermann test** (`pt_test`): Test directional accuracy [T1/T3]
- **Multi-model comparison** (`compare_multiple_models`): Pairwise DM with Bonferroni [T1]
- **HAC variance** (`compute_hac_variance`): Newey-West estimator with Bartlett kernel [T1]

# Knowledge Tier
[T1] DM test: Diebold & Mariano (1995). Comparing predictive accuracy.
[T1] Harvey adjustment: Harvey et al. (1997). Small-sample correction.
[T1] PT test (2-class): Pesaran & Timmermann (1992). Directional accuracy.
[T1] HAC variance: Newey & West (1987). HAC covariance estimation.
[T1] Bandwidth selection: Andrews (1991). Automatic bandwidth.
[T3] PT test (3-class): Ad-hoc extension, not published.

# Example
```julia
using TemporalValidation

# Compare two models
model_errors = randn(100) .* 0.8
baseline_errors = randn(100) .* 1.0
dm_result = dm_test(model_errors, baseline_errors; h=2, alternative=:less)

if significant_at_05(dm_result)
    println("Model significantly better than baseline")
end

# Test directional accuracy
actual = randn(100)
predicted = 0.5 .* actual .+ 0.5 .* randn(100)
pt_result = pt_test(actual, predicted)
println("Direction skill: \$(round(skill(pt_result) * 100, digits=1))%")

# Compare multiple models
errors = Dict(:ridge => e1, :lasso => e2, :baseline => e3)
multi_result = compare_multiple_models(errors)
println(summary(multi_result))
```

# See Also
- `DM_TEST_MIN_SAMPLES`, `PT_TEST_MIN_SAMPLES`: Frozen thresholds from SPECIFICATION.md
"""
module StatisticalTests

using StatsBase: mean
using Distributions: TDist, Normal, cdf

# Import frozen constants from parent module
using ..TemporalValidation: DM_TEST_MIN_SAMPLES, PT_TEST_MIN_SAMPLES, DEFAULT_ALPHA

# Include implementation files (order matters: dependencies first)
include("hac.jl")      # HAC variance estimation (foundation)
include("dm.jl")       # DM test (depends on hac.jl)
include("pt.jl")       # PT test (independent)
include("multi_model.jl")  # Multi-model comparison (depends on dm.jl)

# =========================================================================
# Exports
# =========================================================================

# Result types
export DMTestResult
export PTTestResult
export MultiModelComparisonResult

# Core functions
export dm_test
export pt_test
export compare_multiple_models
export compute_hac_variance

# Helper functions
export bartlett_kernel
export default_bandwidth
export get_pairwise

# Convenience predicates (shared across result types)
export significant_at_05
export significant_at_01
export skill
export n_comparisons
export n_significant

end # module StatisticalTests
