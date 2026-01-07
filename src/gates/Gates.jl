# Gates.jl - Validation gates submodule for TemporalValidation.jl
#
# This module provides leakage detection gates that return HALT/WARN/PASS/SKIP
# status codes to prevent overly optimistic model evaluation.
#
# Knowledge Tiers (from SPECIFICATION.md):
#   [T1] = Academically validated with citation
#   [T2] = Empirical finding from prior work
#   [T3] = Assumption needing justification

"""
    Gates

Validation gates for detecting data leakage and suspicious model performance.

# Available Gates

- `gate_suspicious_improvement()`: Detect implausibly large improvements over baseline
- `gate_shuffled_target()`: Definitive leakage detection via shuffled targets
- `gate_synthetic_ar1()`: Theoretical bound verification using synthetic AR(1) data

# Orchestration

- `run_gates()`: Aggregate gate results into ValidationReport
- `run_standard_gates()`: Run the standard gate suite (shuffled target + synthetic AR(1) + suspicious improvement)

# Helper Functions

- `block_permute()`: Block-wise permutation for shuffled target test [T1 Kunsch 1989]
- `default_block_size()`: Compute optimal block size

# Knowledge Tier
[T2] Gate framework from myga-forecasting-v2 postmortem analysis.
[T2] "External-first" validation ordering (synthetic → shuffled → internal).
"""
module Gates

using Random
using ..TemporalValidation: GateStatus, GateResult, ValidationReport, HALT, WARN, PASS, SKIP
using ..TemporalValidation: SUSPICIOUS_IMPROVEMENT_HALT, SUSPICIOUS_IMPROVEMENT_WARN

# Gate implementations
include("suspicious.jl")
include("shuffled_target.jl")
include("synthetic_ar1.jl")
include("runner.jl")

# Exports - Gates
export gate_suspicious_improvement
export gate_shuffled_target
export gate_synthetic_ar1

# Exports - Orchestration
export run_gates, run_standard_gates

# Exports - Helpers
export block_permute, default_block_size

end # module Gates
