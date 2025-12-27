# TemporalValidation

Rigorous validation for time-series ML: Leakage detection, statistical testing, and temporal safety.

## Development

### Build & Test

```bash
# Run tests
julia --project=. -e "using Pkg; Pkg.test()"

# REPL development
julia --project=.
```

### Architecture Notes

**Module Structure**:
```
src/
├── TemporalValidation.jl  # Main module, exports
├── types.jl               # Core types (ValidationStatus, etc.)
├── gates/                 # Leakage detection gates
├── statistical_tests/     # Diebold-Mariano, Pesaran-Timmermann
├── metrics/               # Move-conditional metrics
└── regimes/               # Market regime detection
```

**Key Components**:

1. **Validation Gates** (HALT/PASS/WARN/SKIP)
   - `gate_suspicious_improvement`: Flags unrealistic accuracy gains
   - `gate_temporal_leakage`: Detects future information leaks
   - Status enum enforces handling of each case

2. **Walk-Forward CV**
   - Strict gap enforcement between train/test
   - Horizon parameter for multi-step forecasts
   - MLJBase integration for standard interface

3. **Statistical Tests**
   - Diebold-Mariano: Predictive accuracy comparison
   - Pesaran-Timmermann: Directional accuracy test

4. **Knowledge Tier System**
   - [T1]: Academically validated (full citation)
   - [T2]: Empirical finding (prior work)
   - [T3]: Assumption (needs justification)
   - All thresholds are tagged with tier

### Testing Patterns

**Frozen specification**: See `SPECIFICATION.md` for locked parameters
- Thresholds must not change without updating spec
- T1 thresholds require academic citation

**Test structure**:
- Each gate has dedicated test file
- Statistical tests use known distributions (validated against R/scipy)
- Reproducibility via StableRNGs

### Common Pitfalls

1. **Gap enforcement**: Always use `gap` parameter to prevent train/test overlap
2. **High-persistence series**: Use move-conditional metrics, not standard accuracy
3. **Multiple comparisons**: Apply appropriate correction when testing many models

## Contributing

**Adding a new gate**:
1. Define in `src/gates/`
2. Return `ValidationResult` with status + message
3. Tag threshold with knowledge tier ([T1]/[T2]/[T3])
4. Add test with edge cases

**Changing thresholds**:
1. Update `SPECIFICATION.md` first
2. Justify tier assignment
3. Bump version if T1/T2 threshold changes

---

**Hub**: @~/Claude/lever_of_archimedes/
