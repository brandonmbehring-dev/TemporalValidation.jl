# hac.jl - HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimation
#
# Implements Newey-West (1987) estimator with Bartlett kernel for robust variance
# estimation in the presence of serial correlation.
#
# Knowledge Tiers:
#   [T1] Newey & West (1987). HAC covariance matrix estimation.
#   [T1] Andrews (1991). Automatic bandwidth selection.
#   [T1] Bartlett kernel is equivalent to Parzen kernel with linear decay.

using StatsBase: mean

"""
    bartlett_kernel(j::Int, bandwidth::Int) -> Float64

Bartlett kernel weight for lag j.

# Formula
    weight = 1 - |j| / (bandwidth + 1)  for |j| <= bandwidth
    weight = 0                          for |j| > bandwidth

# Arguments
- `j::Int`: Lag index (non-negative)
- `bandwidth::Int`: Kernel bandwidth (must be >= 0)

# Returns
Kernel weight in [0, 1].

# Knowledge Tier
[T1] Newey & West (1987). The Bartlett kernel is triangular, declining linearly
     from 1 at lag 0 to 0 at lag bandwidth+1.

# Example
```julia
bartlett_kernel(0, 5)  # => 1.0
bartlett_kernel(3, 5)  # => 0.5
bartlett_kernel(6, 5)  # => 0.0
```
"""
function bartlett_kernel(j::Int, bandwidth::Int)::Float64
    abs(j) <= bandwidth ? 1.0 - abs(j) / (bandwidth + 1) : 0.0
end


"""
    default_bandwidth(n::Int) -> Int

Automatic bandwidth selection using Andrews (1991) rule.

# Formula
    bandwidth = floor(4 × (n / 100)^(2/9))

This is the data-dependent optimal bandwidth for the Bartlett kernel under
an AR(1) error structure.

# Arguments
- `n::Int`: Number of observations

# Returns
Recommended bandwidth (minimum 1).

# Knowledge Tier
[T1] Andrews (1991). Heteroskedasticity and autocorrelation consistent
     covariance matrix estimation. Econometrica, 59(3), 817-858.

# Example
```julia
default_bandwidth(100)  # => 4
default_bandwidth(500)  # => 6
default_bandwidth(50)   # => 3
```
"""
function default_bandwidth(n::Int)::Int
    n > 0 || throw(ArgumentError("n must be > 0, got $n"))
    max(1, floor(Int, 4 * (n / 100)^(2/9)))
end


"""
    compute_hac_variance(d::AbstractVector{<:Real}; bandwidth::Union{Int, Nothing}=nothing) -> Float64

Compute HAC (Heteroskedasticity and Autocorrelation Consistent) variance estimate.

Uses Newey-West estimator with Bartlett kernel.

# Arguments
- `d`: Series of values (typically loss differential for DM test)

# Keyword Arguments
- `bandwidth::Union{Int, Nothing} = nothing`: Kernel bandwidth. If `nothing`,
  uses automatic selection via `default_bandwidth(n)`.

# Returns
HAC variance estimate (variance of the mean).

# Formula
The HAC variance of the mean is computed as:

    V̂ = (1/n) × [γ̂₀ + 2 × Σⱼ₌₁ᴮ w(j) × γ̂ⱼ]

where:
- γ̂ⱼ = (1/n) Σₜ₌ⱼ₊₁ⁿ (dₜ - d̄)(dₜ₋ⱼ - d̄) is the sample autocovariance at lag j
- w(j) = bartlett_kernel(j, bandwidth) is the kernel weight
- B = bandwidth

# Notes
- For h-step forecasts in DM test, bandwidth = h-1 is appropriate since
  errors are MA(h-1) and thus autocorrelated up to lag h-1.
- The automatic bandwidth is a general-purpose choice when h is unknown.
- Negative variance can occur with small samples or high autocorrelation.
  Callers should handle this case (e.g., return pvalue=1.0).

# Complexity
O(n × bandwidth)

# Knowledge Tier
[T1] Newey & West (1987). A simple, positive semi-definite, heteroskedasticity
     and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703-708.
[T1] Andrews (1991). Automatic bandwidth selection.

# Example
```julia
d = randn(100)  # Loss differential series
var_d = compute_hac_variance(d)

# With explicit bandwidth for h=2 step forecasts
var_d = compute_hac_variance(d; bandwidth=1)
```

# See Also
- `dm_test`: Primary consumer of HAC variance estimation.
- `default_bandwidth`: Automatic bandwidth selection rule.
"""
function compute_hac_variance(
    d::AbstractVector{<:Real};
    bandwidth::Union{Int, Nothing} = nothing
)::Float64
    n = length(d)
    n > 0 || throw(ArgumentError("d cannot be empty"))

    # Demean the series
    d_demeaned = d .- mean(d)

    # Automatic bandwidth if not specified
    bw = isnothing(bandwidth) ? default_bandwidth(n) : bandwidth
    bw >= 0 || throw(ArgumentError("bandwidth must be >= 0, got $bw"))

    # Compute autocovariances up to lag bw
    # γ̂ⱼ = (1/n) Σₜ₌ⱼ₊₁ⁿ (dₜ - d̄)(dₜ₋ⱼ - d̄)
    gamma = zeros(bw + 1)
    for j in 0:bw
        if j == 0
            # Variance at lag 0
            gamma[j+1] = mean(d_demeaned .^ 2)
        else
            # Autocovariance at lag j
            # Note: Julia is 1-indexed, so d_demeaned[j+1:end] corresponds to dₜ for t>j
            # and d_demeaned[1:end-j] corresponds to dₜ₋ⱼ
            gamma[j+1] = mean(d_demeaned[j+1:end] .* d_demeaned[1:end-j])
        end
    end

    # Apply Bartlett kernel weights
    # V̂ = γ̂₀ + 2 × Σⱼ₌₁ᴮ w(j) × γ̂ⱼ
    variance = gamma[1]  # γ̂₀
    for j in 1:bw
        weight = bartlett_kernel(j, bw)
        variance += 2 * weight * gamma[j+1]
    end

    # Return variance of the mean: V̂(d̄) = V̂ / n
    return variance / n
end
