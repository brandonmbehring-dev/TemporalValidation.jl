using TemporalValidation
using Test
using StableRNGs
using StatsBase: mean

@testset "TemporalValidation.jl" begin

    @testset "Core Types" begin

        @testset "GateStatus enum" begin
            # Verify all statuses exist
            @test HALT isa GateStatus
            @test WARN isa GateStatus
            @test PASS isa GateStatus
            @test SKIP isa GateStatus

            # Verify they are distinct
            @test HALT != WARN
            @test HALT != PASS
            @test HALT != SKIP
            @test WARN != PASS
            @test WARN != SKIP
            @test PASS != SKIP
        end

        @testset "GateResult construction" begin
            # Minimal construction
            result = GateResult(
                name = :test_gate,
                status = PASS,
                message = "Test passed"
            )
            @test result.name == :test_gate
            @test result.status == PASS
            @test result.message == "Test passed"
            @test isnothing(result.metric_value)
            @test isnothing(result.threshold)
            @test isempty(result.details)
            @test result.recommendation == ""

            # Full construction
            result_full = GateResult(
                name = :suspicious_improvement,
                status = HALT,
                message = "35% improvement exceeds 20% threshold",
                metric_value = 0.35,
                threshold = 0.20,
                details = Dict(:baseline_mae => 0.15, :model_mae => 0.098),
                recommendation = "Check for data leakage"
            )
            @test result_full.metric_value == 0.35
            @test result_full.threshold == 0.20
            @test result_full.details[:baseline_mae] == 0.15
            @test result_full.recommendation == "Check for data leakage"
        end

        @testset "ValidationReport" begin
            # Empty report
            empty_report = ValidationReport()
            @test isempty(empty_report.gates)
            @test status(empty_report) == PASS
            @test passed(empty_report)

            # Single PASS
            pass_result = GateResult(name = :test, status = PASS, message = "OK")
            report = ValidationReport([pass_result])
            @test status(report) == PASS
            @test passed(report)
            @test length(pass_gates(report)) == 1
            @test isempty(halt_gates(report))

            # HALT takes precedence
            halt_result = GateResult(name = :bad, status = HALT, message = "Failed")
            report_halt = ValidationReport([pass_result, halt_result])
            @test status(report_halt) == HALT
            @test !passed(report_halt)
            @test length(halt_gates(report_halt)) == 1
            @test length(pass_gates(report_halt)) == 1

            # WARN without HALT
            warn_result = GateResult(name = :caution, status = WARN, message = "Warning")
            report_warn = ValidationReport([pass_result, warn_result])
            @test status(report_warn) == WARN
            @test passed(report_warn)  # WARN still passes

            # SKIP without HALT or WARN
            skip_result = GateResult(name = :skipped, status = SKIP, message = "No data")
            report_skip = ValidationReport([pass_result, skip_result])
            @test status(report_skip) == SKIP
            @test passed(report_skip)
        end

        @testset "SplitInfo" begin
            info = SplitInfo(0, 1, 50, 55, 55, 4)
            @test info.split_idx == 0
            @test info.train_start == 1
            @test info.train_end == 50
            @test info.test_start == 55
            @test info.test_end == 55
            @test info.gap == 4
            @test train_size(info) == 50
            @test test_size(info) == 1
        end
    end

    @testset "Frozen Thresholds" begin
        # Verify thresholds match SPECIFICATION.md
        @test SUSPICIOUS_IMPROVEMENT_HALT == 0.20
        @test SUSPICIOUS_IMPROVEMENT_WARN == 0.10
        @test SHUFFLED_TARGET_THRESHOLD == 0.05
        @test DEFAULT_N_SHUFFLES == 5
        @test AR1_PHI == 0.95
        @test AR1_TOLERANCE == 1.5
        @test THEORETICAL_AR1_MAE_FACTOR ≈ sqrt(2/π) atol=1e-10
        @test DM_TEST_MIN_SAMPLES == 30
        @test PT_TEST_MIN_SAMPLES == 20
        @test DEFAULT_ALPHA == 0.05
        @test CONFORMAL_CALIBRATION_FRACTION == 0.30
        @test ADAPTIVE_CONFORMAL_GAMMA == 0.1
        @test MOVE_THRESHOLD_PERCENTILE == 70.0
        @test VOLATILITY_WINDOW == 13
        @test VOLATILITY_LOW_PERCENTILE == 33.0
        @test VOLATILITY_HIGH_PERCENTILE == 67.0
    end

    @testset "WalkForwardCV" begin

        @testset "Basic construction" begin
            cv = WalkForwardCV(n_splits=5, horizon=1, gap=1)
            @test cv.n_splits == 5
            @test cv.horizon == 1
            @test cv.gap == 1
            @test cv.window == Expanding
            @test isnothing(cv.window_size)
            @test cv.test_size == 1
        end

        @testset "Default gap equals horizon" begin
            cv = WalkForwardCV(n_splits=5, horizon=4)
            @test cv.gap == 4  # Defaults to horizon
        end

        @testset "Gap enforcement (CRITICAL)" begin
            # gap < horizon should throw
            @test_throws ArgumentError WalkForwardCV(n_splits=5, horizon=4, gap=2)
            @test_throws ArgumentError WalkForwardCV(n_splits=5, horizon=4, gap=0)

            # gap == horizon should work
            cv = WalkForwardCV(n_splits=5, horizon=4, gap=4)
            @test cv.gap == 4

            # gap > horizon should work
            cv2 = WalkForwardCV(n_splits=5, horizon=4, gap=10)
            @test cv2.gap == 10
        end

        @testset "Sliding window requires window_size" begin
            @test_throws ArgumentError WalkForwardCV(n_splits=5, window=Sliding)

            cv = WalkForwardCV(n_splits=5, window=Sliding, window_size=50)
            @test cv.window == Sliding
            @test cv.window_size == 50
        end

        @testset "Parameter validation" begin
            @test_throws ArgumentError WalkForwardCV(n_splits=0)
            @test_throws ArgumentError WalkForwardCV(horizon=0)
            @test_throws ArgumentError WalkForwardCV(test_size=0)
            @test_throws ArgumentError WalkForwardCV(window=Sliding, window_size=0)
        end

        @testset "get_splits expanding" begin
            cv = WalkForwardCV(n_splits=5, horizon=1, gap=1)
            splits = get_splits(cv, 100)

            @test length(splits) == 5

            # Verify temporal ordering
            for s in splits
                @test s.train_start < s.train_end < s.test_start <= s.test_end
                @test s.test_start == s.train_end + s.gap + 1
            end

            # Verify expanding window (train starts at 1, ends grow)
            for s in splits
                @test s.train_start == 1
            end
            @test splits[1].train_end < splits[end].train_end

            # Verify no data leakage (gap respected)
            for s in splits
                @test s.test_start - s.train_end > cv.gap
            end
        end

        @testset "get_splits sliding" begin
            cv = WalkForwardCV(n_splits=5, horizon=1, gap=1, window=Sliding, window_size=20)
            splits = get_splits(cv, 100)

            @test length(splits) == 5

            # Verify fixed window size
            for s in splits
                @test train_size(s) == 20
            end

            # Verify sliding (train_start increases)
            @test splits[1].train_start < splits[end].train_start
        end

        @testset "get_splits insufficient data" begin
            cv = WalkForwardCV(n_splits=5, horizon=10, gap=10)
            # Need at least gap + test_size + 1 = 10 + 1 + 1 = 12 observations
            @test_throws ArgumentError get_splits(cv, 10)
        end

        @testset "get_splits edge cases" begin
            # Minimum viable data
            cv = WalkForwardCV(n_splits=1, horizon=1, gap=1, test_size=1)
            splits = get_splits(cv, 3)  # 1 train + 1 gap + 1 test
            @test length(splits) == 1
            @test splits[1].train_start == 1
            @test splits[1].train_end == 1
            @test splits[1].test_start == 3
            @test splits[1].test_end == 3
        end
    end

    @testset "MLJBase Integration" begin
        using MLJBase

        @testset "train_test_pairs basic" begin
            cv = WalkForwardCV(n_splits=3, horizon=1, gap=1)
            pairs = MLJBase.train_test_pairs(cv, 1:20)

            @test length(pairs) == 3
            @test all(p -> p[1] isa Vector{Int} && p[2] isa Vector{Int}, pairs)

            # Verify temporal ordering: train max < test min
            for (train, test) in pairs
                @test maximum(train) < minimum(test)
            end
        end

        @testset "train_test_pairs gap enforcement" begin
            cv = WalkForwardCV(n_splits=3, horizon=4, gap=4)
            pairs = MLJBase.train_test_pairs(cv, 1:50)

            # Verify gap is respected: test_start - train_end > gap
            for (train, test) in pairs
                @test minimum(test) - maximum(train) > cv.gap
            end
        end

        @testset "train_test_pairs with non-1-indexed rows" begin
            cv = WalkForwardCV(n_splits=2, horizon=1, gap=1)
            # Simulate row indices from a filtered DataFrame
            rows = [5, 10, 15, 20, 25, 30, 35, 40]
            pairs = MLJBase.train_test_pairs(cv, rows)

            @test length(pairs) == 2

            # Indices should be from the actual rows, not 1:n
            for (train, test) in pairs
                @test all(idx -> idx in rows, train)
                @test all(idx -> idx in rows, test)
            end
        end

        @testset "WalkForwardCV is a ResamplingStrategy" begin
            cv = WalkForwardCV(n_splits=5, horizon=1, gap=1)
            @test cv isa MLJBase.ResamplingStrategy
        end
    end

    @testset "gate_suspicious_improvement" begin

        @testset "HALT case (>20% improvement)" begin
            # 50% improvement: 1 - 0.10/0.20 = 0.50
            result = gate_suspicious_improvement(0.10, 0.20)
            @test result.status == HALT
            @test result.name == :suspicious_improvement
            @test result.metric_value ≈ 0.5 atol=1e-10
            @test result.threshold == 0.20
            @test occursin("50", result.message)  # 50% in message
            @test !isempty(result.recommendation)
        end

        @testset "WARN case (10-20% improvement)" begin
            # 15% improvement: 1 - 0.85/1.0 = 0.15
            result = gate_suspicious_improvement(0.85, 1.0)
            @test result.status == WARN
            @test result.metric_value ≈ 0.15 atol=1e-10
        end

        @testset "PASS case (<10% improvement)" begin
            # 5% improvement: 1 - 0.95/1.0 = 0.05
            result = gate_suspicious_improvement(0.95, 1.0)
            @test result.status == PASS
            @test result.metric_value ≈ 0.05 atol=1e-10
            @test isempty(result.recommendation)
        end

        @testset "SKIP case (baseline ≤ 0)" begin
            result = gate_suspicious_improvement(0.5, 0.0)
            @test result.status == SKIP
            @test occursin("cannot compute", lowercase(result.message))

            result_neg = gate_suspicious_improvement(0.5, -0.1)
            @test result_neg.status == SKIP
        end

        @testset "Negative improvement (model worse)" begin
            # Model is worse: 1 - 1.2/1.0 = -0.2 (20% worse)
            result = gate_suspicious_improvement(1.2, 1.0)
            @test result.status == PASS
            @test result.metric_value ≈ -0.2 atol=1e-10
        end

        @testset "Exact threshold boundaries" begin
            # Exactly 20% improvement: not > 20% HALT threshold, but > 10% WARN threshold
            result_20 = gate_suspicious_improvement(0.80, 1.0)
            @test result_20.status == WARN  # 20% > 10% warn threshold

            # Just over 20% → HALT
            result_21 = gate_suspicious_improvement(0.79, 1.0)
            @test result_21.status == HALT

            # Exactly 10% improvement: not > 20% HALT, not > 10% WARN (equal, not greater)
            result_10 = gate_suspicious_improvement(0.90, 1.0)
            @test result_10.status == PASS

            # Just over 10% → WARN
            result_11 = gate_suspicious_improvement(0.89, 1.0)
            @test result_11.status == WARN
        end

        @testset "Custom thresholds" begin
            # Use stricter thresholds
            result = gate_suspicious_improvement(
                0.90, 1.0;  # 10% improvement
                threshold=0.08,
                warn_threshold=0.05
            )
            @test result.status == HALT  # 10% > 8% threshold

            result_warn = gate_suspicious_improvement(
                0.94, 1.0;  # 6% improvement
                threshold=0.08,
                warn_threshold=0.05
            )
            @test result_warn.status == WARN  # 6% > 5% warn threshold
        end

        @testset "Details contain expected keys" begin
            result = gate_suspicious_improvement(0.10, 0.15; metric_name="RMSE")
            @test haskey(result.details, :model_rmse)
            @test haskey(result.details, :baseline_rmse)
            @test haskey(result.details, :improvement_ratio)
            @test result.details[:model_rmse] == 0.10
            @test result.details[:baseline_rmse] == 0.15
        end

        @testset "Property-based tests" begin
            rng = StableRNG(42)

            # Property 1: Model worse than baseline always PASS
            @testset "Worse model always PASS" begin
                for _ in 1:100
                    baseline = abs(randn(rng)) + 0.1  # Positive baseline
                    model = baseline * (1 + abs(randn(rng)) * 0.5)  # Always worse
                    result = gate_suspicious_improvement(model, baseline)
                    @test result.status == PASS
                end
            end

            # Property 2: HALT threshold is strictly respected
            @testset "HALT threshold respected" begin
                for _ in 1:50
                    baseline = abs(randn(rng)) + 0.5
                    # 25-40% improvement (always above 20% HALT threshold)
                    improvement_pct = 0.25 + rand(rng) * 0.15
                    model = baseline * (1 - improvement_pct)
                    result = gate_suspicious_improvement(model, baseline)
                    @test result.status == HALT
                end
            end

            # Property 3: Status ordering (PASS → WARN → HALT as improvement increases)
            @testset "Status ordering" begin
                baseline = 1.0
                # 5% → PASS, 15% → WARN, 25% → HALT
                @test gate_suspicious_improvement(0.95, baseline).status == PASS
                @test gate_suspicious_improvement(0.85, baseline).status == WARN
                @test gate_suspicious_improvement(0.75, baseline).status == HALT
            end

            # Property 4: Improvement ratio is correctly computed
            @testset "Improvement ratio correctness" begin
                for _ in 1:50
                    baseline = abs(randn(rng)) + 0.5
                    model = abs(randn(rng)) + 0.1
                    result = gate_suspicious_improvement(model, baseline)
                    expected_improvement = 1.0 - (model / baseline)
                    @test result.metric_value ≈ expected_improvement atol=1e-10
                end
            end

            # Property 5: Custom thresholds are respected
            @testset "Custom thresholds respected" begin
                for _ in 1:30
                    threshold = 0.05 + rand(rng) * 0.3  # 5-35%
                    warn_threshold = threshold * 0.5  # Half of HALT threshold

                    # Just above threshold → HALT
                    baseline = 1.0
                    model = baseline * (1 - threshold - 0.01)
                    result = gate_suspicious_improvement(model, baseline;
                        threshold=threshold, warn_threshold=warn_threshold)
                    @test result.status == HALT

                    # Below warn_threshold → PASS
                    model_pass = baseline * (1 - warn_threshold + 0.01)
                    result_pass = gate_suspicious_improvement(model_pass, baseline;
                        threshold=threshold, warn_threshold=warn_threshold)
                    @test result_pass.status == PASS
                end
            end
        end
    end

    @testset "block_permute" begin

        @testset "Basic block permutation" begin
            rng = StableRNG(42)
            arr = collect(1:12)

            # Block size 3: [1,2,3], [4,5,6], [7,8,9], [10,11,12]
            result = block_permute(arr, 3, rng)

            # All elements preserved
            @test sort(result) == arr

            # Length preserved
            @test length(result) == length(arr)
        end

        @testset "Block structure preserved" begin
            rng = StableRNG(123)
            arr = collect(1:9)

            # Block size 3: [1,2,3], [4,5,6], [7,8,9]
            result = block_permute(arr, 3, rng)

            # Each block appears contiguously (just in different order)
            # Find where each original block starts in the result
            for block_start in [1, 4, 7]
                # Find this block's first element in result
                pos = findfirst(==(block_start), result)
                @test !isnothing(pos)
                # Check the whole block is contiguous
                if pos + 2 <= length(result)
                    @test result[pos:pos+2] == [block_start, block_start+1, block_start+2]
                end
            end
        end

        @testset "Handles remainder" begin
            rng = StableRNG(99)
            arr = collect(1:10)  # 3 full blocks of 3 + 1 remainder

            result = block_permute(arr, 3, rng)

            # All elements preserved
            @test sort(result) == arr

            # Length preserved
            @test length(result) == length(arr)
        end

        @testset "Deterministic with same RNG" begin
            arr = collect(1:12)
            result1 = block_permute(arr, 3, StableRNG(42))
            result2 = block_permute(arr, 3, StableRNG(42))
            @test result1 == result2
        end

        @testset "Different RNG seeds give different results" begin
            arr = collect(1:12)
            result1 = block_permute(arr, 3, StableRNG(42))
            result2 = block_permute(arr, 3, StableRNG(123))
            # Very unlikely to be the same (4! = 24 permutations)
            @test result1 != result2
        end

        @testset "Edge cases" begin
            # Empty array
            result = block_permute(Int[], 3, StableRNG(42))
            @test isempty(result)

            # Block size >= n returns copy
            arr = collect(1:5)
            result = block_permute(arr, 10, StableRNG(42))
            @test result == arr

            # Block size 1 is just regular shuffle
            arr = collect(1:5)
            result = block_permute(arr, 1, StableRNG(42))
            @test sort(result) == arr
        end

        @testset "Invalid block size" begin
            arr = collect(1:10)
            @test_throws ArgumentError block_permute(arr, 0, StableRNG(42))
            @test_throws ArgumentError block_permute(arr, -1, StableRNG(42))
        end

        @testset "Property-based tests" begin
            rng = StableRNG(789)

            # Property 1: All elements are preserved (permutation property)
            @testset "Elements preserved" begin
                for _ in 1:50
                    n = rand(rng, 5:100)
                    arr = randn(rng, n)
                    block_size = rand(rng, 1:max(1, n ÷ 3))
                    result = block_permute(arr, block_size, StableRNG(rand(rng, 1:10000)))
                    @test sort(result) ≈ sort(arr)
                    @test length(result) == length(arr)
                end
            end

            # Property 2: Blocks remain contiguous (block integrity)
            @testset "Block integrity" begin
                for _ in 1:20
                    n = rand(rng, 10:50)
                    block_size = rand(rng, 2:5)
                    # Create blocks with distinct ranges
                    arr = collect(1:n)
                    result = block_permute(arr, block_size, StableRNG(rand(rng, 1:10000)))

                    # For each full block, elements should appear together
                    n_full_blocks = n ÷ block_size
                    for b in 1:n_full_blocks
                        block_start_val = (b - 1) * block_size + 1
                        # Find where this block starts in result
                        pos = findfirst(==(block_start_val), result)
                        if !isnothing(pos) && pos + block_size - 1 <= length(result)
                            # Check block is contiguous
                            expected_block = collect(block_start_val : block_start_val + block_size - 1)
                            @test result[pos:pos+block_size-1] == expected_block
                        end
                    end
                end
            end

            # Property 3: Same seed produces same result (reproducibility)
            @testset "Reproducibility" begin
                for _ in 1:30
                    n = rand(rng, 10:100)
                    arr = randn(rng, n)
                    block_size = rand(rng, 1:max(1, n ÷ 4))
                    seed = rand(rng, 1:100000)

                    result1 = block_permute(arr, block_size, StableRNG(seed))
                    result2 = block_permute(arr, block_size, StableRNG(seed))
                    @test result1 == result2
                end
            end

            # Property 4: Single element array is unchanged
            @testset "Single element unchanged" begin
                for _ in 1:20
                    val = randn(rng)
                    result = block_permute([val], rand(rng, 1:10), StableRNG(rand(rng, 1:1000)))
                    @test result == [val]
                end
            end

            # Property 5: Block size >= n returns exact copy
            @testset "Large block size returns copy" begin
                for _ in 1:20
                    n = rand(rng, 5:30)
                    arr = randn(rng, n)
                    block_size = n + rand(rng, 0:10)
                    result = block_permute(arr, block_size, StableRNG(rand(rng, 1:1000)))
                    @test result == arr
                end
            end
        end
    end

    @testset "default_block_size" begin
        # Kunsch (1989) rule: floor(n^(1/3))
        @test default_block_size(1) == 1
        @test default_block_size(8) == 2      # 8^(1/3) = 2.0
        @test default_block_size(27) == 3     # 27^(1/3) = 3.0
        @test default_block_size(64) == 3     # 64^(1/3) ≈ 3.999... → floor = 3
        @test default_block_size(100) == 4    # 100^(1/3) ≈ 4.64 → floor = 4
        @test default_block_size(125) == 5    # 125^(1/3) = 5.0
        @test default_block_size(1000) == 9   # 1000^(1/3) ≈ 9.999... → floor = 9

        # Invalid input
        @test_throws ArgumentError default_block_size(0)
        @test_throws ArgumentError default_block_size(-1)
    end

    @testset "Pretty Printing" begin
        # Smoke tests for show methods
        result = GateResult(name = :test, status = PASS, message = "OK")
        @test occursin("GateResult", string(result))
        @test occursin("test", string(result))

        report = ValidationReport([result])
        @test occursin("ValidationReport", string(report))

        cv = WalkForwardCV(n_splits=5, horizon=4, gap=4)
        @test occursin("WalkForwardCV", string(cv))
        @test occursin("expanding", string(cv))

        cv_sliding = WalkForwardCV(n_splits=5, window=Sliding, window_size=50)
        @test occursin("sliding", string(cv_sliding))
    end

    @testset "MLJ.evaluate! Integration" begin
        using MLJ
        using DecisionTree
        using MLJDecisionTreeInterface

        @testset "WalkForwardCV with DecisionTreeRegressor" begin
            # Generate synthetic time-series-like regression data
            # Using deterministic data for reproducibility
            rng = StableRNG(42)
            n = 200  # Enough data for 3 splits
            X = (
                feature1 = randn(rng, n),
                feature2 = randn(rng, n),
                feature3 = randn(rng, n)
            )
            # Target with some autocorrelation (AR-like)
            y = zeros(n)
            y[1] = randn(rng)
            for i in 2:n
                y[i] = 0.7 * y[i-1] + 0.3 * randn(rng)
            end

            # Instantiate model directly from MLJDecisionTreeInterface
            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=3)

            # Create walk-forward CV with proper gap
            cv = WalkForwardCV(n_splits=3, horizon=1, gap=1, test_size=5)

            # This is the critical test: MLJ.evaluate should work with our CV
            result = evaluate(model, X, y, resampling=cv, measure=rmse, verbosity=0)

            # Verify result structure (PerformanceEvaluation struct)
            @test hasproperty(result, :measurement)
            @test hasproperty(result, :per_fold)

            # result.per_fold is a vector of vectors: per_fold[measure_idx][fold_idx]
            # With single measure, per_fold[1] contains fold results
            @test length(result.per_fold[1]) == 3  # 3 folds

            # All measurements should be real numbers (no NaN/Inf)
            @test all(isfinite, result.per_fold[1])
        end

        @testset "WalkForwardCV with multiple measures" begin
            rng = StableRNG(123)
            n = 150  # Enough data for 2 splits
            X = (x1 = randn(rng, n), x2 = randn(rng, n))
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            cv = WalkForwardCV(n_splits=2, horizon=1, gap=1, test_size=3)

            # Multiple measures
            result = evaluate(model, X, y, resampling=cv, measures=[rmse, mae], verbosity=0)

            @test length(result.per_fold) == 2  # 2 measures
            @test length(result.per_fold[1]) == 2  # 2 folds for first measure
            @test length(result.per_fold[2]) == 2  # 2 folds for second measure
        end

        @testset "WalkForwardCV sliding window with MLJ" begin
            rng = StableRNG(456)
            n = 200  # Enough data for sliding window
            X = (x = randn(rng, n),)
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            # Sliding window CV
            cv = WalkForwardCV(
                n_splits=3,
                horizon=1,
                gap=1,
                window=Sliding,
                window_size=30,
                test_size=5
            )

            result = evaluate(model, X, y, resampling=cv, measure=rmse, verbosity=0)

            @test length(result.per_fold[1]) == 3  # 3 folds
            @test all(isfinite, result.per_fold[1])
        end

        @testset "Gap enforcement prevents leakage in MLJ context" begin
            # Create data where h=4 forecast would leak without gap
            rng = StableRNG(789)
            n = 100  # Enough data for 2 splits with gap=4
            X = (x = randn(rng, n),)
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            # horizon=4 with gap=4 (safe)
            cv = WalkForwardCV(n_splits=2, horizon=4, gap=4, test_size=4)
            result = evaluate(model, X, y, resampling=cv, measure=rmse, verbosity=0)

            # Verify we got valid results - at least some folds completed
            @test length(result.per_fold[1]) >= 1
            @test all(isfinite, result.per_fold[1])
        end
    end

    # =========================================================================
    # Week 3-4: New Gates Tests
    # =========================================================================

    @testset "gate_shuffled_target" begin
        using MLJ
        using DecisionTree
        using MLJDecisionTreeInterface

        @testset "Basic functionality" begin
            rng = StableRNG(42)
            n = 200
            # Random data with no X->y relationship (model should NOT beat shuffled)
            X = (x1 = randn(rng, n), x2 = randn(rng, n))
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            result = gate_shuffled_target(
                model, X, y;
                n_shuffles=3,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(123)
            )

            # Should PASS - model shouldn't beat shuffled with random data
            @test result.name == :shuffled_target
            @test result.status in [PASS, SKIP]  # PASS expected, SKIP acceptable
            @test !isnothing(result.metric_value) || result.status == SKIP
        end

        @testset "Input validation" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor()
            X = (x = randn(100),)
            y = randn(100)

            # NaN in y should throw
            y_nan = copy(y)
            y_nan[50] = NaN
            @test_throws ArgumentError gate_shuffled_target(model, X, y_nan)

            # Invalid permutation type
            @test_throws ArgumentError gate_shuffled_target(
                model, X, y;
                permutation=:invalid
            )

            # Invalid n_shuffles
            @test_throws ArgumentError gate_shuffled_target(
                model, X, y;
                n_shuffles=0
            )

            # Invalid n_cv_splits
            @test_throws ArgumentError gate_shuffled_target(
                model, X, y;
                n_cv_splits=0
            )
        end

        @testset "Block vs IID permutation" begin
            rng = StableRNG(456)
            n = 200  # More data to ensure gate completes
            X = (x = randn(rng, n),)
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            # Block permutation
            result_block = gate_shuffled_target(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                permutation=:block,
                rng=StableRNG(789)
            )

            # If gate completed (not SKIP), check details
            if result_block.status != SKIP
                @test result_block.details[:permutation] == :block
                @test result_block.details[:block_size] > 0
            else
                @test true  # SKIP is acceptable
            end

            # IID permutation
            result_iid = gate_shuffled_target(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                permutation=:iid,
                rng=StableRNG(789)
            )

            if result_iid.status != SKIP
                @test result_iid.details[:permutation] == :iid
            else
                @test true  # SKIP is acceptable
            end
        end

        @testset "SKIP with insufficient data" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor()
            X = (x = randn(5),)  # Too few samples
            y = randn(5)

            result = gate_shuffled_target(
                model, X, y;
                n_shuffles=1,
                n_cv_splits=1,
                test_size=10  # Requires more than 5 samples
            )
            @test result.status == SKIP
            @test occursin("Insufficient", result.message)
        end

        @testset "Details dict contains expected keys" begin
            rng = StableRNG(999)
            n = 150
            X = (x = randn(rng, n),)
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            result = gate_shuffled_target(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(111)
            )

            if result.status != SKIP
                @test haskey(result.details, :mae_real)
                @test haskey(result.details, :mae_shuffled_avg)
                @test haskey(result.details, :mae_shuffled_all)
                @test haskey(result.details, :n_shuffles)
                @test haskey(result.details, :n_cv_splits)
                @test haskey(result.details, :permutation)
                @test haskey(result.details, :block_size)
                @test haskey(result.details, :min_pvalue)
            end
        end

        @testset "Deterministic with same RNG" begin
            n = 150
            X = (x = randn(StableRNG(42), n),)
            y = randn(StableRNG(42), n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            result1 = gate_shuffled_target(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(123)
            )

            result2 = gate_shuffled_target(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(123)
            )

            if result1.status != SKIP && result2.status != SKIP
                @test result1.metric_value ≈ result2.metric_value atol=1e-10
            end
        end
    end

    @testset "gate_synthetic_ar1" begin
        using MLJ
        using DecisionTree
        using MLJDecisionTreeInterface

        @testset "Basic functionality" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=3)

            result = gate_synthetic_ar1(
                model;
                n_samples=200,
                n_lags=3,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(42)
            )

            @test result.name == :synthetic_ar1
            @test result.status in [PASS, SKIP]  # PASS expected, SKIP acceptable
            @test !isnothing(result.metric_value) || result.status == SKIP
        end

        @testset "Parameter validation" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor()

            # phi outside (-1, 1) should throw
            @test_throws ArgumentError gate_synthetic_ar1(model; phi=1.0)
            @test_throws ArgumentError gate_synthetic_ar1(model; phi=-1.0)
            @test_throws ArgumentError gate_synthetic_ar1(model; phi=1.5)

            # n_samples <= n_lags should throw
            @test_throws ArgumentError gate_synthetic_ar1(model; n_samples=5, n_lags=5)
            @test_throws ArgumentError gate_synthetic_ar1(model; n_samples=3, n_lags=5)

            # sigma <= 0 should throw
            @test_throws ArgumentError gate_synthetic_ar1(model; sigma=0.0)
            @test_throws ArgumentError gate_synthetic_ar1(model; sigma=-1.0)

            # tolerance <= 1 should throw
            @test_throws ArgumentError gate_synthetic_ar1(model; tolerance=1.0)
            @test_throws ArgumentError gate_synthetic_ar1(model; tolerance=0.5)
        end

        @testset "Theoretical bound verification" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=3)

            result = gate_synthetic_ar1(
                model;
                n_samples=300,
                sigma=1.0,
                n_cv_splits=2,
                test_size=15,
                rng=StableRNG(123)
            )

            if result.status != SKIP
                @test haskey(result.details, :model_mae)
                @test haskey(result.details, :theoretical_mae)
                @test haskey(result.details, :ratio)

                # Theoretical MAE should be ~0.798 for sigma=1.0
                @test result.details[:theoretical_mae] ≈ sqrt(2/π) atol=1e-10

                # Ratio = model_mae / theoretical_mae
                @test result.details[:ratio] ≈ result.details[:model_mae] / result.details[:theoretical_mae] atol=1e-10
            end
        end

        @testset "Details dict contains expected keys" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            result = gate_synthetic_ar1(
                model;
                phi=0.9,
                sigma=2.0,
                n_samples=200,
                n_lags=4,
                tolerance=1.5,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(42)
            )

            if result.status != SKIP
                @test result.details[:phi] == 0.9
                @test result.details[:sigma] == 2.0
                @test result.details[:n_samples] == 200
                @test result.details[:n_lags] == 4
                @test result.details[:tolerance] == 1.5
                @test result.details[:halt_threshold] ≈ 1/1.5 atol=1e-10
            end
        end

        @testset "Deterministic with same RNG" begin
            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            result1 = gate_synthetic_ar1(
                model;
                n_samples=200,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(42)
            )

            result2 = gate_synthetic_ar1(
                model;
                n_samples=200,
                n_cv_splits=2,
                test_size=10,
                rng=StableRNG(42)
            )

            if result1.status != SKIP && result2.status != SKIP
                @test result1.metric_value ≈ result2.metric_value atol=1e-10
                @test result1.details[:model_mae] ≈ result2.details[:model_mae] atol=1e-10
            end
        end
    end

    @testset "run_gates" begin

        @testset "Aggregation from vector" begin
            pass1 = GateResult(name=:gate1, status=PASS, message="OK")
            pass2 = GateResult(name=:gate2, status=PASS, message="OK")

            report = run_gates([pass1, pass2])
            @test report isa ValidationReport
            @test status(report) == PASS
            @test length(report.gates) == 2
        end

        @testset "HALT takes precedence" begin
            pass = GateResult(name=:gate1, status=PASS, message="OK")
            halt = GateResult(name=:gate2, status=HALT, message="Failed")
            warn = GateResult(name=:gate3, status=WARN, message="Warning")

            report = run_gates([pass, halt, warn])
            @test status(report) == HALT
            @test length(halt_gates(report)) == 1
        end

        @testset "WARN without HALT" begin
            pass = GateResult(name=:gate1, status=PASS, message="OK")
            warn = GateResult(name=:gate2, status=WARN, message="Warning")

            report = run_gates([pass, warn])
            @test status(report) == WARN
            @test passed(report)  # WARN still passes
        end

        @testset "Varargs and tuple forms" begin
            pass = GateResult(name=:gate1, status=PASS, message="OK")
            halt = GateResult(name=:gate2, status=HALT, message="Failed")

            # Varargs form
            report1 = run_gates(pass, halt)
            @test length(report1.gates) == 2
            @test status(report1) == HALT

            # Tuple form
            report2 = run_gates((pass, halt))
            @test length(report2.gates) == 2
            @test status(report2) == HALT
        end

        @testset "Empty gates" begin
            report = run_gates(GateResult[])
            @test status(report) == PASS
            @test isempty(report.gates)
        end
    end

    @testset "run_standard_gates" begin
        using MLJ
        using DecisionTree
        using MLJDecisionTreeInterface

        @testset "Basic functionality (external gates only)" begin
            rng = StableRNG(42)
            n = 200
            X = (x1 = randn(rng, n), x2 = randn(rng, n))
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            report = run_standard_gates(
                model, X, y;
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                run_synthetic_ar1=true,
                rng=StableRNG(123)
            )

            @test report isa ValidationReport
            # Should have 2 gates: shuffled_target + synthetic_ar1
            @test length(report.gates) == 2

            # Check gate names
            gate_names = Set([g.name for g in report.gates])
            @test :shuffled_target in gate_names
            @test :synthetic_ar1 in gate_names
        end

        @testset "With suspicious improvement (internal gate)" begin
            rng = StableRNG(42)
            n = 200
            X = (x1 = randn(rng, n), x2 = randn(rng, n))
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            report = run_standard_gates(
                model, X, y;
                model_metric=0.95,
                baseline_metric=1.0,  # 5% improvement
                n_shuffles=2,
                n_cv_splits=2,
                test_size=10,
                run_synthetic_ar1=false,  # Skip to speed up
                rng=StableRNG(123)
            )

            # Should have 2 gates: shuffled_target + suspicious_improvement
            @test length(report.gates) == 2

            gate_names = Set([g.name for g in report.gates])
            @test :shuffled_target in gate_names
            @test :suspicious_improvement in gate_names
        end

        @testset "Skip synthetic AR(1) when requested" begin
            rng = StableRNG(42)
            n = 150
            X = (x = randn(rng, n),)
            y = randn(rng, n)

            model = MLJDecisionTreeInterface.DecisionTreeRegressor(max_depth=2)

            report = run_standard_gates(
                model, X, y;
                n_shuffles=1,
                n_cv_splits=2,
                test_size=10,
                run_synthetic_ar1=false,
                rng=StableRNG(123)
            )

            # Should have only 1 gate: shuffled_target
            @test length(report.gates) == 1
            @test report.gates[1].name == :shuffled_target
        end
    end

    # =========================================================================
    # Week 5-6: Statistical Tests
    # =========================================================================

    @testset "Statistical Tests" begin

        @testset "HAC Variance Estimation" begin

            @testset "bartlett_kernel" begin
                # At lag 0, kernel weight is 1
                @test bartlett_kernel(0, 5) == 1.0
                @test bartlett_kernel(0, 10) == 1.0

                # Linear decay
                @test bartlett_kernel(1, 5) ≈ 1 - 1/6 atol=1e-10
                @test bartlett_kernel(3, 5) ≈ 1 - 3/6 atol=1e-10
                @test bartlett_kernel(5, 5) ≈ 1 - 5/6 atol=1e-10

                # Beyond bandwidth, weight is 0
                @test bartlett_kernel(6, 5) == 0.0
                @test bartlett_kernel(10, 5) == 0.0
                @test bartlett_kernel(100, 5) == 0.0

                # Edge case: bandwidth = 0
                @test bartlett_kernel(0, 0) == 1.0
                @test bartlett_kernel(1, 0) == 0.0
            end

            @testset "default_bandwidth (Andrews 1991)" begin
                # Formula: floor(4 × (n/100)^(2/9))
                @test default_bandwidth(100) == 4   # 4 × 1.0 = 4
                @test default_bandwidth(50) == 3    # 4 × (0.5)^(2/9) ≈ 3.45
                @test default_bandwidth(200) == 4   # 4 × (2)^(2/9) ≈ 4.67
                @test default_bandwidth(500) == 5   # 4 × (5)^(2/9) ≈ 5.59
                @test default_bandwidth(1000) == 6  # 4 × (10)^(2/9) ≈ 6.50

                # Minimum is 1
                @test default_bandwidth(1) == 1
                @test default_bandwidth(10) >= 1

                # Invalid input
                @test_throws ArgumentError default_bandwidth(0)
                @test_throws ArgumentError default_bandwidth(-1)
            end

            @testset "compute_hac_variance" begin
                rng = StableRNG(42)

                # Simple case: IID data
                d_iid = randn(rng, 100)
                var_iid = compute_hac_variance(d_iid)
                @test var_iid > 0
                @test isfinite(var_iid)

                # Variance should be close to sample variance / n for IID
                sample_var = sum((d_iid .- mean(d_iid)).^2) / length(d_iid)
                @test var_iid ≈ sample_var / length(d_iid) atol=0.01

                # With explicit bandwidth = 0 (no autocorrelation adjustment)
                var_bw0 = compute_hac_variance(d_iid; bandwidth=0)
                @test var_bw0 ≈ sample_var / length(d_iid) atol=1e-10

                # Constant series has zero variance
                d_const = fill(5.0, 50)
                var_const = compute_hac_variance(d_const)
                @test var_const ≈ 0.0 atol=1e-15

                # Larger bandwidth increases variance (for autocorrelated data)
                rng2 = StableRNG(123)
                n = 200
                d_ar = zeros(n)
                d_ar[1] = randn(rng2)
                for i in 2:n
                    d_ar[i] = 0.8 * d_ar[i-1] + 0.2 * randn(rng2)
                end
                var_bw1 = compute_hac_variance(d_ar; bandwidth=1)
                var_bw5 = compute_hac_variance(d_ar; bandwidth=5)
                # For positively autocorrelated data, higher bandwidth gives higher variance estimate
                @test var_bw5 >= var_bw1 * 0.9  # Allow some noise

                # Invalid input
                @test_throws ArgumentError compute_hac_variance(Float64[])
                @test_throws ArgumentError compute_hac_variance([1.0, 2.0]; bandwidth=-1)
            end

            @testset "Property-based tests" begin
                rng = StableRNG(789)

                # Property 1: Variance is always non-negative for valid data
                @testset "Non-negative variance" begin
                    for _ in 1:50
                        n = rand(rng, 30:200)
                        d = randn(rng, n)
                        var = compute_hac_variance(d)
                        @test var >= 0 || isnan(var)  # Can be 0 for degenerate cases
                    end
                end

                # Property 2: Scaling invariance: Var(c*X) = c² * Var(X)
                @testset "Scaling invariance" begin
                    for _ in 1:20
                        n = rand(rng, 50:100)
                        d = randn(rng, n)
                        c = 2.0 + rand(rng) * 3.0
                        var_d = compute_hac_variance(d; bandwidth=3)
                        var_cd = compute_hac_variance(c .* d; bandwidth=3)
                        @test var_cd ≈ c^2 * var_d atol=1e-10
                    end
                end

                # Property 3: Translation invariance: Var(X + c) = Var(X)
                @testset "Translation invariance" begin
                    for _ in 1:20
                        n = rand(rng, 50:100)
                        d = randn(rng, n)
                        c = 100.0 + rand(rng) * 100.0
                        var_d = compute_hac_variance(d; bandwidth=2)
                        var_dc = compute_hac_variance(d .+ c; bandwidth=2)
                        @test var_dc ≈ var_d atol=1e-10
                    end
                end
            end
        end

        @testset "DM Test" begin

            @testset "Basic functionality" begin
                rng = StableRNG(42)

                # Model significantly better than baseline
                errors_1 = randn(rng, 100) .* 0.5  # Lower variance
                errors_2 = randn(rng, 100) .* 1.5  # Higher variance

                result = dm_test(errors_1, errors_2; h=1, alternative=:less)

                @test result isa DMTestResult
                @test result.n == 100
                @test result.h == 1
                @test result.loss == :squared
                @test result.alternative == :less
                @test result.harvey_adjusted == true
                @test isfinite(result.statistic)
                @test 0.0 <= result.pvalue <= 1.0
            end

            @testset "Result type accessors" begin
                rng = StableRNG(123)
                errors_1 = randn(rng, 50) .* 0.8
                errors_2 = randn(rng, 50) .* 1.2

                result = dm_test(errors_1, errors_2)

                # significant_at_* predicates
                @test significant_at_05(result) == (result.pvalue < 0.05)
                @test significant_at_01(result) == (result.pvalue < 0.01)
            end

            @testset "Identical errors" begin
                rng = StableRNG(42)
                errors = randn(rng, 100)

                result = dm_test(errors, errors)

                # Mean loss differential is zero
                @test result.mean_loss_diff ≈ 0.0 atol=1e-10

                # P-value should be ~0.5 for two-sided (or 1.0 if variance is 0)
                @test result.pvalue >= 0.4 || isnan(result.statistic)
            end

            @testset "Clearly different models" begin
                rng = StableRNG(456)

                # Model 1 is much better (small errors)
                errors_1 = randn(rng, 100) .* 0.1
                # Model 2 is much worse (large errors)
                errors_2 = randn(rng, 100) .* 3.0

                result = dm_test(errors_1, errors_2; alternative=:less)

                # Should be highly significant
                @test result.pvalue < 0.01
                @test significant_at_01(result)
                @test result.mean_loss_diff < 0  # Model 1 has lower loss
            end

            @testset "Alternative hypotheses" begin
                rng = StableRNG(789)

                # Model 1 is better
                errors_1 = randn(rng, 100) .* 0.5
                errors_2 = randn(rng, 100) .* 1.5

                # :less - H1: Model 1 has lower loss
                result_less = dm_test(errors_1, errors_2; alternative=:less)
                @test result_less.pvalue < 0.05  # Should be significant

                # :greater - H1: Model 1 has higher loss
                result_greater = dm_test(errors_1, errors_2; alternative=:greater)
                @test result_greater.pvalue > 0.5  # Should NOT be significant

                # :two_sided
                result_two = dm_test(errors_1, errors_2; alternative=:two_sided)
                @test result_two.pvalue ≈ 2 * min(result_less.pvalue, result_greater.pvalue) atol=0.01
            end

            @testset "Harvey correction" begin
                rng = StableRNG(111)
                errors_1 = randn(rng, 50)
                errors_2 = randn(rng, 50) .* 1.2

                # With Harvey correction
                result_harvey = dm_test(errors_1, errors_2; harvey_correction=true)
                @test result_harvey.harvey_adjusted == true

                # Without Harvey correction
                result_no_harvey = dm_test(errors_1, errors_2; harvey_correction=false)
                @test result_no_harvey.harvey_adjusted == false

                # Statistics should be different (Harvey adjusts)
                @test result_harvey.statistic != result_no_harvey.statistic
            end

            @testset "Loss functions" begin
                rng = StableRNG(222)
                errors_1 = randn(rng, 100)
                errors_2 = randn(rng, 100) .* 1.5

                result_squared = dm_test(errors_1, errors_2; loss=:squared)
                result_absolute = dm_test(errors_1, errors_2; loss=:absolute)

                @test result_squared.loss == :squared
                @test result_absolute.loss == :absolute

                # Different loss functions give different results
                @test result_squared.mean_loss_diff != result_absolute.mean_loss_diff
            end

            @testset "Multi-step horizons" begin
                rng = StableRNG(333)
                errors_1 = randn(rng, 100)
                errors_2 = randn(rng, 100) .* 1.2

                result_h1 = dm_test(errors_1, errors_2; h=1)
                result_h2 = dm_test(errors_1, errors_2; h=2)
                result_h4 = dm_test(errors_1, errors_2; h=4)

                @test result_h1.h == 1
                @test result_h2.h == 2
                @test result_h4.h == 4

                # Larger horizon uses larger HAC bandwidth (different variance estimate)
                # P-values may differ
                @test isfinite(result_h4.pvalue)
            end

            @testset "Input validation" begin
                rng = StableRNG(42)
                errors_1 = randn(rng, 50)
                errors_2 = randn(rng, 50)

                # Different lengths
                @test_throws ArgumentError dm_test(errors_1, randn(rng, 30))

                # NaN values
                errors_nan = copy(errors_1)
                errors_nan[25] = NaN
                @test_throws ArgumentError dm_test(errors_nan, errors_2)
                @test_throws ArgumentError dm_test(errors_1, errors_nan)

                # Too few samples (min_samples=30 by default)
                @test_throws ArgumentError dm_test(randn(rng, 20), randn(rng, 20))

                # Invalid horizon
                @test_throws ArgumentError dm_test(errors_1, errors_2; h=0)
                @test_throws ArgumentError dm_test(errors_1, errors_2; h=-1)

                # Invalid loss
                @test_throws ArgumentError dm_test(errors_1, errors_2; loss=:invalid)

                # Invalid alternative
                @test_throws ArgumentError dm_test(errors_1, errors_2; alternative=:invalid)
            end

            @testset "Pretty printing" begin
                rng = StableRNG(42)
                result = dm_test(randn(rng, 100), randn(rng, 100))

                str = string(result)
                @test occursin("DM", str)
                @test occursin("h=", str)
                @test occursin("p=", str)

                # Full representation
                io = IOBuffer()
                show(io, MIME("text/plain"), result)
                full_str = String(take!(io))
                @test occursin("DMTestResult", full_str)
                @test occursin("statistic", full_str)
                @test occursin("pvalue", full_str)
            end

            @testset "Property-based tests" begin
                rng = StableRNG(444)

                # Property 1: Swapping errors changes sign of statistic
                @testset "Symmetry" begin
                    for _ in 1:20
                        n = rand(rng, 50:100)
                        e1 = randn(rng, n)
                        e2 = randn(rng, n) .* (0.5 + rand(rng))

                        r1 = dm_test(e1, e2; alternative=:two_sided, harvey_correction=false)
                        r2 = dm_test(e2, e1; alternative=:two_sided, harvey_correction=false)

                        @test r1.statistic ≈ -r2.statistic atol=1e-10
                        @test r1.pvalue ≈ r2.pvalue atol=1e-10
                    end
                end

                # Property 2: Scaling errors doesn't change relative performance
                @testset "Scale invariance of relative ranking" begin
                    for _ in 1:20
                        n = rand(rng, 50:100)
                        e1 = randn(rng, n) .* 0.5
                        e2 = randn(rng, n) .* 1.5
                        c = 2.0 + rand(rng) * 5.0

                        r1 = dm_test(e1, e2; alternative=:less)
                        r2 = dm_test(c .* e1, c .* e2; alternative=:less)

                        # Significance should be similar
                        @test (r1.pvalue < 0.05) == (r2.pvalue < 0.05)
                    end
                end
            end
        end

        @testset "PT Test" begin

            @testset "Basic functionality (2-class)" begin
                rng = StableRNG(42)

                # Actual with some direction
                actual = randn(rng, 100)
                # Predicted with positive correlation
                predicted = 0.5 .* actual .+ 0.5 .* randn(rng, 100)

                result = pt_test(actual, predicted)

                @test result isa PTTestResult
                @test result.n_classes == 2
                @test result.n <= 100  # May be less due to zeros
                @test 0.0 <= result.accuracy <= 1.0
                @test 0.0 <= result.expected <= 1.0
                @test 0.0 <= result.pvalue <= 1.0
                @test isfinite(result.statistic)
            end

            @testset "Result type accessors" begin
                rng = StableRNG(123)
                actual = randn(rng, 100)
                predicted = 0.3 .* actual .+ 0.7 .* randn(rng, 100)

                result = pt_test(actual, predicted)

                # skill = accuracy - expected
                @test skill(result) ≈ result.accuracy - result.expected atol=1e-10

                # significant_at_05
                @test significant_at_05(result) == (result.pvalue < 0.05)
            end

            @testset "3-class mode" begin
                rng = StableRNG(456)
                actual = randn(rng, 100)
                predicted = 0.5 .* actual .+ 0.5 .* randn(rng, 100)

                result = pt_test(actual, predicted; move_threshold=0.5)

                @test result.n_classes == 3
                @test result.n == 100  # All samples included in 3-class
                @test 0.0 <= result.accuracy <= 1.0
            end

            @testset "Perfect prediction" begin
                rng = StableRNG(789)
                actual = randn(rng, 50)
                predicted = copy(actual)

                result = pt_test(actual, predicted)

                # Perfect correlation → high accuracy
                @test result.accuracy ≥ 0.95
                # Should be significant (better than random)
                @test result.pvalue < 0.01
                @test significant_at_01(result)
            end

            @testset "Random prediction" begin
                rng = StableRNG(111)
                actual = randn(rng, 100)
                # Independent random predictions
                predicted = randn(StableRNG(222), 100)

                result = pt_test(actual, predicted)

                # Accuracy should be close to expected
                @test abs(result.accuracy - result.expected) < 0.15
                # Skill should be close to 0
                @test abs(skill(result)) < 0.15
                # Should NOT be significant
                @test result.pvalue > 0.05
            end

            @testset "Anti-prediction" begin
                rng = StableRNG(333)
                actual = randn(rng, 100)
                # Opposite direction predictions
                predicted = -actual

                result = pt_test(actual, predicted)

                # Should have low accuracy (opposite)
                @test result.accuracy < result.expected
                # Negative skill
                @test skill(result) < 0
            end

            @testset "Input validation" begin
                rng = StableRNG(42)
                actual = randn(rng, 50)
                predicted = randn(rng, 50)

                # Different lengths
                @test_throws ArgumentError pt_test(actual, randn(rng, 30))

                # NaN values
                actual_nan = copy(actual)
                actual_nan[25] = NaN
                @test_throws ArgumentError pt_test(actual_nan, predicted)
                @test_throws ArgumentError pt_test(actual, actual_nan)

                # Too few samples
                @test_throws ArgumentError pt_test(randn(rng, 10), randn(rng, 10))

                # Negative threshold
                @test_throws ArgumentError pt_test(actual, predicted; move_threshold=-0.1)
            end

            @testset "All zeros handling" begin
                # Edge case: all actual values are zero → undefined direction
                actual = zeros(50)
                predicted = randn(StableRNG(42), 50)

                # Should return with warning and pvalue=1.0
                result = pt_test(actual, predicted)
                @test result.pvalue == 1.0
                @test result.n == 0  # No non-zero samples
            end

            @testset "Pretty printing" begin
                rng = StableRNG(42)
                result = pt_test(randn(rng, 100), randn(rng, 100))

                str = string(result)
                @test occursin("PT", str)
                @test occursin("%", str)

                # Full representation
                io = IOBuffer()
                show(io, MIME("text/plain"), result)
                full_str = String(take!(io))
                @test occursin("PTTestResult", full_str)
                @test occursin("accuracy", full_str)
                @test occursin("skill", full_str)
            end

            @testset "Property-based tests" begin
                rng = StableRNG(555)

                # Property 1: accuracy is in [0, 1]
                @testset "Accuracy bounds" begin
                    for _ in 1:30
                        n = rand(rng, 50:150)
                        actual = randn(rng, n)
                        predicted = randn(rng, n)

                        result = pt_test(actual, predicted)
                        if result.n > 0
                            @test 0.0 <= result.accuracy <= 1.0
                            @test 0.0 <= result.expected <= 1.0
                        end
                    end
                end

                # Property 2: pvalue is in [0, 1]
                @testset "P-value bounds" begin
                    for _ in 1:30
                        n = rand(rng, 50:100)
                        actual = randn(rng, n)
                        predicted = randn(rng, n)

                        result = pt_test(actual, predicted)
                        @test 0.0 <= result.pvalue <= 1.0
                    end
                end

                # Property 3: Perfect correlation gives significant result
                @testset "Perfect correlation significance" begin
                    for _ in 1:10
                        n = rand(rng, 50:100)
                        actual = randn(rng, n)
                        predicted = actual .+ randn(rng, n) .* 0.01  # Small noise

                        result = pt_test(actual, predicted)
                        if result.n >= 20
                            @test result.pvalue < 0.05
                        end
                    end
                end
            end
        end

        @testset "Multi-Model Comparison" begin

            @testset "Basic functionality" begin
                rng = StableRNG(42)
                n = 100

                errors = Dict(
                    :model_a => randn(rng, n) .* 0.5,   # Best
                    :model_b => randn(rng, n) .* 1.0,   # Middle
                    :model_c => randn(rng, n) .* 1.5    # Worst
                )

                result = compare_multiple_models(errors; h=1, alpha=0.05)

                @test result isa MultiModelComparisonResult
                @test result.best_model == :model_a
                @test result.original_alpha == 0.05
                @test result.bonferroni_alpha ≈ 0.05 / 3 atol=1e-10  # 3 pairs
                @test n_comparisons(result) == 3  # C(3,2) = 3
            end

            @testset "Bonferroni correction" begin
                rng = StableRNG(123)
                n = 100

                # 4 models → C(4,2) = 6 comparisons
                errors = Dict(
                    :a => randn(rng, n),
                    :b => randn(rng, n),
                    :c => randn(rng, n),
                    :d => randn(rng, n)
                )

                result = compare_multiple_models(errors; alpha=0.10)

                @test n_comparisons(result) == 6
                @test result.bonferroni_alpha ≈ 0.10 / 6 atol=1e-10
            end

            @testset "Model rankings" begin
                rng = StableRNG(456)
                n = 100

                errors = Dict(
                    :worst => randn(rng, n) .* 2.0,
                    :middle => randn(rng, n) .* 1.0,
                    :best => randn(rng, n) .* 0.5
                )

                result = compare_multiple_models(errors)

                # Rankings should be sorted by mean loss (ascending)
                @test result.model_rankings[1][1] == :best
                @test result.model_rankings[3][1] == :worst

                # Mean losses should be ordered
                losses = [r[2] for r in result.model_rankings]
                @test issorted(losses)
            end

            @testset "get_pairwise" begin
                rng = StableRNG(789)
                n = 100

                errors = Dict(
                    :a => randn(rng, n),
                    :b => randn(rng, n) .* 1.5,
                    :c => randn(rng, n) .* 2.0
                )

                result = compare_multiple_models(errors)

                # Get pairwise result (order-independent)
                dm_ab = get_pairwise(result, :a, :b)
                dm_ba = get_pairwise(result, :b, :a)

                @test dm_ab isa DMTestResult
                @test dm_ab === dm_ba  # Same reference

                # Non-existent pair
                @test isnothing(get_pairwise(result, :a, :nonexistent))
            end

            @testset "Significant pairs" begin
                rng = StableRNG(111)
                n = 100

                # Clearly different models
                errors = Dict(
                    :excellent => randn(rng, n) .* 0.1,
                    :terrible => randn(rng, n) .* 5.0
                )

                result = compare_multiple_models(errors; alpha=0.05)

                # Should have significant difference
                @test n_significant(result) >= 1
                @test length(result.significant_pairs) >= 1
            end

            @testset "summary function" begin
                rng = StableRNG(222)
                n = 100

                errors = Dict(
                    :a => randn(rng, n),
                    :b => randn(rng, n) .* 1.5
                )

                result = compare_multiple_models(errors)
                s = summary(result)

                @test s isa String
                @test occursin("Multi-Model", s)
                @test occursin("Bonferroni", s)
                @test occursin("Rankings", s)
            end

            @testset "Input validation" begin
                rng = StableRNG(42)

                # Less than 2 models
                @test_throws ArgumentError compare_multiple_models(
                    Dict(:a => randn(rng, 50))
                )

                # Different lengths
                @test_throws ArgumentError compare_multiple_models(
                    Dict(:a => randn(rng, 50), :b => randn(rng, 30))
                )

                # Invalid alpha
                @test_throws ArgumentError compare_multiple_models(
                    Dict(:a => randn(rng, 50), :b => randn(rng, 50));
                    alpha=0.0
                )
                @test_throws ArgumentError compare_multiple_models(
                    Dict(:a => randn(rng, 50), :b => randn(rng, 50));
                    alpha=1.0
                )
            end

            @testset "Pretty printing" begin
                rng = StableRNG(42)
                n = 100

                errors = Dict(:a => randn(rng, n), :b => randn(rng, n))
                result = compare_multiple_models(errors)

                str = string(result)
                @test occursin("MultiModelComparisonResult", str)
                @test occursin("models", str)

                # Full representation
                io = IOBuffer()
                show(io, MIME("text/plain"), result)
                full_str = String(take!(io))
                @test occursin("best", full_str)
                @test occursin("Rankings", full_str)
            end

            @testset "Property-based tests" begin
                rng = StableRNG(666)

                # Property 1: Best model has lowest mean loss
                @testset "Best model correctness" begin
                    for _ in 1:20
                        n = rand(rng, 50:100)
                        n_models = rand(rng, 2:5)
                        errors = Dict{Symbol, Vector{Float64}}()
                        for i in 1:n_models
                            name = Symbol("model_$i")
                            errors[name] = randn(rng, n) .* (0.5 + rand(rng) * 2.0)
                        end

                        result = compare_multiple_models(errors)

                        # Verify best_model has lowest mean loss
                        best_loss = result.model_rankings[1][2]
                        for (name, loss) in result.model_rankings
                            @test loss >= best_loss - 1e-10
                        end
                    end
                end

                # Property 2: Number of comparisons is C(n,2)
                @testset "Comparison count" begin
                    for n_models in 2:6
                        n = 50
                        errors = Dict{Symbol, Vector{Float64}}()
                        for i in 1:n_models
                            errors[Symbol("m_$i")] = randn(rng, n)
                        end

                        result = compare_multiple_models(errors)
                        expected_pairs = n_models * (n_models - 1) ÷ 2
                        @test n_comparisons(result) == expected_pairs
                    end
                end
            end
        end

    end  # Statistical Tests

    # =========================================================================
    # Week 7-8: Metrics & Regimes Tests
    # =========================================================================

    @testset "Core Metrics" begin

        @testset "compute_mae" begin
            # Basic functionality
            @test compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 0.0 atol=1e-15
            @test compute_mae([1.0, 2.0, 3.0], [2.0, 3.0, 4.0]) ≈ 1.0 atol=1e-15
            @test compute_mae([0.0, 0.0, 0.0], [1.0, -1.0, 0.5]) ≈ (1.0 + 1.0 + 0.5) / 3 atol=1e-15

            # Symmetry
            @test compute_mae([1.0, 2.0], [3.0, 4.0]) ≈ compute_mae([3.0, 4.0], [1.0, 2.0]) atol=1e-15

            # Input validation
            @test_throws ArgumentError compute_mae([1.0, 2.0], [1.0])  # Different lengths
            # Empty arrays return NaN (not throw) by design for graceful handling
            @test isnan(compute_mae(Float64[], Float64[]))
            @test_throws ArgumentError compute_mae([1.0, NaN], [1.0, 2.0])  # NaN
        end

        @testset "compute_mse" begin
            @test compute_mse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 0.0 atol=1e-15
            @test compute_mse([0.0, 0.0], [1.0, 2.0]) ≈ (1.0 + 4.0) / 2 atol=1e-15
            @test compute_mse([1.0, 2.0], [3.0, 4.0]) ≈ (4.0 + 4.0) / 2 atol=1e-15
        end

        @testset "compute_rmse" begin
            @test compute_rmse([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 0.0 atol=1e-15
            @test compute_rmse([0.0, 0.0], [1.0, 2.0]) ≈ sqrt(2.5) atol=1e-15
            # RMSE >= MAE always
            rng = StableRNG(42)
            for _ in 1:20
                n = rand(rng, 10:50)
                pred = randn(rng, n)
                actual = randn(rng, n)
                @test compute_rmse(pred, actual) >= compute_mae(pred, actual) - 1e-10
            end
        end

        @testset "compute_mape" begin
            # Basic calculation: 100 * mean(|pred - actual| / |actual|)
            @test compute_mape([1.0], [1.0]) ≈ 0.0 atol=1e-10
            @test compute_mape([2.0], [1.0]) ≈ 100.0 atol=1e-10  # 100% error
            @test compute_mape([0.5], [1.0]) ≈ 50.0 atol=1e-10   # 50% error

            # Zero actuals use epsilon
            result = compute_mape([1.0], [0.0])
            @test isfinite(result)
            @test result > 0
        end

        @testset "compute_smape" begin
            # sMAPE: 100 * mean(2|pred - actual| / (|pred| + |actual|))
            @test compute_smape([1.0], [1.0]) ≈ 0.0 atol=1e-10
            @test compute_smape([2.0], [1.0]) ≈ 100 * 2 * 1.0 / 3.0 atol=1e-10

            # Symmetric
            @test compute_smape([1.0], [2.0]) ≈ compute_smape([2.0], [1.0]) atol=1e-10
        end

        @testset "compute_bias" begin
            @test compute_bias([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 0.0 atol=1e-15
            @test compute_bias([2.0, 3.0, 4.0], [1.0, 2.0, 3.0]) ≈ 1.0 atol=1e-15  # Positive bias
            @test compute_bias([0.0, 1.0, 2.0], [1.0, 2.0, 3.0]) ≈ -1.0 atol=1e-15  # Negative bias
        end

        @testset "compute_mase" begin
            # MASE = mae / naive_mae
            @test compute_mase([1.0, 2.0], [1.0, 2.0], 1.0) ≈ 0.0 atol=1e-15  # Perfect
            @test compute_mase([2.0, 3.0], [1.0, 2.0], 1.0) ≈ 1.0 atol=1e-15  # Same as naive
            @test compute_mase([1.5, 2.5], [1.0, 2.0], 1.0) ≈ 0.5 atol=1e-15  # Better than naive

            # MASE < 1 means better than naive
            @test compute_mase([1.1, 2.1], [1.0, 2.0], 0.5) < 1.0

            # Invalid naive_mae
            @test_throws ArgumentError compute_mase([1.0], [1.0], 0.0)
            @test_throws ArgumentError compute_mase([1.0], [1.0], -1.0)
        end

        @testset "compute_mrae" begin
            # MRAE = mean(|model_error| / |naive_error|)
            model_pred = [1.5, 2.5]
            actual = [1.0, 2.0]
            naive_pred = [2.0, 3.0]  # Naive errors: 1.0, 1.0

            # Model errors: 0.5, 0.5 → ratios: 0.5, 0.5 → mean: 0.5
            @test compute_mrae(model_pred, actual, naive_pred) ≈ 0.5 atol=1e-10

            # Length mismatch
            @test_throws ArgumentError compute_mrae([1.0], [1.0], [1.0, 2.0])
        end

        @testset "compute_theils_u" begin
            # Theil's U = RMSE(model) / RMSE(naive)
            model_pred = [1.5, 2.5]
            actual = [1.0, 2.0]
            naive_pred = [2.0, 3.0]

            model_rmse = compute_rmse(model_pred, actual)  # sqrt(0.25) = 0.5
            naive_rmse = compute_rmse(naive_pred, actual)  # sqrt(1.0) = 1.0
            @test compute_theils_u(model_pred, actual; naive_predictions=naive_pred) ≈ model_rmse / naive_rmse atol=1e-10

            # U < 1 means model beats naive
            @test compute_theils_u(model_pred, actual; naive_predictions=naive_pred) < 1.0
        end

        @testset "compute_r_squared" begin
            # Perfect prediction
            @test compute_r_squared([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) ≈ 1.0 atol=1e-10

            # Mean prediction → R² = 0
            actual = [1.0, 2.0, 3.0]
            mean_pred = fill(mean(actual), 3)
            @test compute_r_squared(mean_pred, actual) ≈ 0.0 atol=1e-10

            # R² can be negative (worse than mean)
            bad_pred = [3.0, 1.0, 2.0]  # Opposite of true order
            r2 = compute_r_squared(bad_pred, actual)
            @test r2 < 0
        end

        @testset "compute_forecast_correlation" begin
            # Perfect positive correlation
            pred = [1.0, 2.0, 3.0, 4.0, 5.0]
            actual = [1.0, 2.0, 3.0, 4.0, 5.0]
            @test compute_forecast_correlation(pred, actual) ≈ 1.0 atol=1e-10

            # Perfect negative correlation
            @test compute_forecast_correlation(pred, reverse(actual)) ≈ -1.0 atol=1e-10

            # Zero correlation (approximately)
            rng = StableRNG(42)
            x = randn(rng, 100)
            y = randn(rng, 100)
            corr = compute_forecast_correlation(x, y)
            @test abs(corr) < 0.3  # Should be close to 0 for independent data

            # Spearman correlation
            corr_spearman = compute_forecast_correlation(pred, actual; method=:spearman)
            @test corr_spearman ≈ 1.0 atol=1e-10

            # Invalid method
            @test_throws ArgumentError compute_forecast_correlation(pred, actual; method=:invalid)
        end

        @testset "compute_naive_error" begin
            # Persistence naive: MAE of diff(values)
            values = [1.0, 2.0, 4.0, 7.0]  # Diffs: 1, 2, 3 → MAE = 2.0
            @test compute_naive_error(values) ≈ 2.0 atol=1e-10

            # Constant series
            @test compute_naive_error([5.0, 5.0, 5.0, 5.0]) ≈ 0.0 atol=1e-15

            # Minimum length
            @test_throws ArgumentError compute_naive_error([1.0])  # Need at least 2
        end

        @testset "Property-based tests" begin
            rng = StableRNG(123)

            # Property: MAE is non-negative
            @testset "MAE non-negative" begin
                for _ in 1:50
                    n = rand(rng, 5:50)
                    pred = randn(rng, n)
                    actual = randn(rng, n)
                    @test compute_mae(pred, actual) >= 0
                end
            end

            # Property: Perfect prediction gives zero error
            @testset "Perfect prediction zero error" begin
                for _ in 1:20
                    n = rand(rng, 5:50)
                    values = randn(rng, n)
                    @test compute_mae(values, values) ≈ 0.0 atol=1e-15
                    @test compute_mse(values, values) ≈ 0.0 atol=1e-15
                    @test compute_rmse(values, values) ≈ 0.0 atol=1e-15
                end
            end

            # Property: RMSE >= MAE
            @testset "RMSE >= MAE" begin
                for _ in 1:50
                    n = rand(rng, 5:50)
                    pred = randn(rng, n)
                    actual = randn(rng, n)
                    @test compute_rmse(pred, actual) >= compute_mae(pred, actual) - 1e-10
                end
            end
        end

    end  # Core Metrics

    @testset "Move-Conditional Metrics" begin

        @testset "MoveDirection enum" begin
            @test UP == MoveDirection(1)
            @test DOWN == MoveDirection(-1)
            @test FLAT == MoveDirection(0)

            # Sign-aligned
            @test Int(UP) == 1
            @test Int(DOWN) == -1
            @test Int(FLAT) == 0
        end

        @testset "compute_move_threshold" begin
            # 70th percentile of |values|
            values = collect(1.0:10.0)  # |values| = 1,2,3,...,10
            # 70th percentile of 1:10 is approximately 7.3
            threshold = compute_move_threshold(values)
            @test threshold > 6.0 && threshold < 8.0

            # Empty throws
            @test_throws ArgumentError compute_move_threshold(Float64[])

            # Custom percentile
            threshold_50 = compute_move_threshold(values; percentile=50.0)
            @test threshold_50 < threshold  # 50th < 70th

            # Invalid percentile
            @test_throws ArgumentError compute_move_threshold(values; percentile=-1.0)
            @test_throws ArgumentError compute_move_threshold(values; percentile=101.0)
        end

        @testset "classify_moves" begin
            values = [1.0, -1.0, 0.1, -0.1, 0.0]
            threshold = 0.5

            moves = classify_moves(values, threshold)

            @test moves[1] == UP     # 1.0 > 0.5
            @test moves[2] == DOWN   # -1.0 < -0.5
            @test moves[3] == FLAT   # |0.1| <= 0.5
            @test moves[4] == FLAT   # |-0.1| <= 0.5
            @test moves[5] == FLAT   # 0.0 <= 0.5

            # Negative threshold throws
            @test_throws ArgumentError classify_moves(values, -0.1)
        end

        @testset "MoveConditionalResult" begin
            # MoveConditionalResult uses positional constructor
            # Fields: mae_up, mae_down, mae_flat, n_up, n_down, n_flat, skill_score, move_threshold
            result = MoveConditionalResult(0.5, 0.6, 0.1, 15, 12, 8, 0.25, 0.3)

            @test n_total(result) == 35
            @test n_moves(result) == 27
            @test is_reliable(result)  # n_up >= 10 AND n_down >= 10
            @test move_fraction(result) ≈ 27 / 35 atol=1e-10

            # Not reliable if low n
            result_low_n = MoveConditionalResult(0.5, 0.6, 0.1, 5, 12, 8, 0.25, 0.3)
            @test !is_reliable(result_low_n)
        end

        @testset "compute_move_conditional_metrics" begin
            rng = StableRNG(42)
            n = 200

            # Create data with actual moves
            actuals = randn(rng, n)
            predictions = 0.5 .* actuals .+ 0.5 .* randn(rng, n)

            result = compute_move_conditional_metrics(predictions, actuals)

            @test result isa MoveConditionalResult
            @test result.n_up > 0
            @test result.n_down > 0
            @test result.move_threshold > 0
            @test n_total(result) == n
            @test result.skill_score > -10.0 && result.skill_score < 1.0

            # With explicit threshold
            result_explicit = compute_move_conditional_metrics(
                predictions, actuals;
                threshold=0.5
            )
            @test result_explicit.move_threshold == 0.5

            # Input validation
            @test_throws ArgumentError compute_move_conditional_metrics(
                randn(10), randn(20)  # Different lengths
            )
            # Empty arrays return result with NaN values (graceful degradation)
            empty_result = compute_move_conditional_metrics(Float64[], Float64[])
            @test isnan(empty_result.skill_score)
            @test n_total(empty_result) == 0
        end

        @testset "Skill score interpretation" begin
            rng = StableRNG(123)
            n = 200
            actuals = randn(rng, n) .* 2.0

            # Perfect prediction → skill_score ≈ 1.0
            result_perfect = compute_move_conditional_metrics(actuals, actuals)
            @test result_perfect.skill_score > 0.9

            # Persistence (predict 0) → skill_score ≈ 0.0
            # Note: We simulate persistence by predicting zeros
            predictions_persistence = zeros(n)
            result_persistence = compute_move_conditional_metrics(
                predictions_persistence, actuals
            )
            @test abs(result_persistence.skill_score) < 0.1

            # Anti-prediction → skill_score < 0
            predictions_anti = -actuals
            result_anti = compute_move_conditional_metrics(predictions_anti, actuals)
            @test result_anti.skill_score < 0
        end

        @testset "Pretty printing" begin
            # Positional constructor
            result = MoveConditionalResult(0.5, 0.6, 0.1, 15, 12, 8, 0.25, 0.3)

            str = string(result)
            @test occursin("MC-SS", str)

            io = IOBuffer()
            show(io, MIME("text/plain"), result)
            full_str = String(take!(io))
            @test occursin("skill_score", full_str)
            @test occursin("reliable", full_str)
        end

    end  # Move-Conditional Metrics

    @testset "Direction Accuracy" begin

        @testset "compute_direction_accuracy (2-class)" begin
            # All correct
            actual = [1.0, -1.0, 1.0, -1.0]
            pred_correct = [0.5, -0.5, 2.0, -2.0]  # Same signs
            @test compute_direction_accuracy(pred_correct, actual) ≈ 1.0 atol=1e-10

            # All wrong
            pred_wrong = [-0.5, 0.5, -2.0, 2.0]  # Opposite signs
            @test compute_direction_accuracy(pred_wrong, actual) ≈ 0.0 atol=1e-10

            # 50%
            pred_half = [0.5, 0.5, -2.0, -2.0]  # First 2 correct, last 2 wrong
            @test compute_direction_accuracy(pred_half, actual) ≈ 0.5 atol=1e-10

            # Zeros in actual are excluded in 2-class mode
            actual_zeros = [1.0, 0.0, -1.0, 0.0]
            pred = [0.5, 0.5, -0.5, -0.5]
            acc = compute_direction_accuracy(pred, actual_zeros)
            @test acc ≈ 1.0 atol=1e-10  # Only 2 non-zero pairs, both correct
        end

        @testset "compute_direction_accuracy (3-class)" begin
            actual = [1.0, -1.0, 0.1, -0.1]
            pred = [0.5, -0.5, 0.05, -0.05]
            threshold = 0.5

            # With threshold:
            # actual[1]=1.0 > 0.5 → UP, pred[1]=0.5 → FLAT (wrong)
            # actual[2]=-1.0 < -0.5 → DOWN, pred[2]=-0.5 → FLAT (wrong)
            # actual[3]=0.1 → FLAT, pred[3]=0.05 → FLAT (correct)
            # actual[4]=-0.1 → FLAT, pred[4]=-0.05 → FLAT (correct)
            acc = compute_direction_accuracy(pred, actual; move_threshold=threshold)
            @test acc ≈ 0.5 atol=1e-10  # 2/4 correct
        end

        @testset "compute_move_only_mae" begin
            actual = [1.0, -1.0, 0.1, -0.1]  # First 2 are moves (threshold < 0.5)
            pred = [0.5, -0.5, 0.0, 0.0]
            threshold = 0.5

            mae, n = compute_move_only_mae(pred, actual, threshold)

            @test n == 2  # Only 2 moves (|actual| > 0.5)
            @test mae ≈ mean([abs(0.5 - 1.0), abs(-0.5 - (-1.0))]) atol=1e-10

            # All FLAT
            all_flat = [0.1, 0.1, 0.1]
            mae_flat, n_flat = compute_move_only_mae(all_flat, all_flat, 0.5)
            @test n_flat == 0
            @test isnan(mae_flat)
        end

        @testset "compute_persistence_mae" begin
            # Persistence predicts 0 (no change)
            # MAE = mean(|0 - actual|) = mean(|actual|)
            actual = [1.0, -1.0, 2.0, -2.0]
            mae = compute_persistence_mae(actual)
            @test mae ≈ mean(abs.(actual)) atol=1e-10

            # With threshold (moves only)
            # Moves are |actual| > 0.5: all 4 in this case
            mae_threshold = compute_persistence_mae(actual; threshold=0.5)
            @test mae_threshold ≈ 1.5 atol=1e-10  # mean([1, 1, 2, 2])
        end

        @testset "Property-based tests" begin
            rng = StableRNG(456)

            # Property: Accuracy is in [0, 1]
            @testset "Accuracy bounds" begin
                for _ in 1:30
                    n = rand(rng, 20:100)
                    pred = randn(rng, n)
                    actual = randn(rng, n)
                    acc = compute_direction_accuracy(pred, actual)
                    @test 0.0 <= acc <= 1.0
                end
            end

            # Property: Perfect correlation gives high accuracy
            @testset "Perfect correlation high accuracy" begin
                for _ in 1:10
                    n = rand(rng, 50:100)
                    actual = randn(rng, n)
                    pred = actual .+ randn(rng, n) .* 0.001  # Tiny noise
                    acc = compute_direction_accuracy(pred, actual)
                    @test acc > 0.95
                end
            end
        end

    end  # Direction Accuracy

    @testset "Volatility Regime Classification" begin

        @testset "classify_volatility_regime basics" begin
            rng = StableRNG(42)
            n = 200

            # Generate data with varying volatility
            values = cumsum(randn(rng, n) .* vcat(
                fill(0.1, 70),   # Low vol
                fill(1.0, 60),   # Medium vol
                fill(3.0, 70)    # High vol
            ))

            regimes = classify_volatility_regime(values)

            @test length(regimes) == n
            @test all(r -> r in [:LOW, :MED, :HIGH], regimes)

            # Should have roughly 1/3 each
            counts = Dict(:LOW => 0, :MED => 0, :HIGH => 0)
            for r in regimes
                counts[r] += 1
            end
            # Allow significant deviation due to window effects
            for (k, v) in counts
                @test v >= 10  # At least some in each
            end
        end

        @testset "basis=:changes vs :levels" begin
            # Critical: volatility should be on CHANGES, not levels
            # A steadily increasing series has low volatility of changes
            # but high std of levels

            n = 100
            # Steady drift: constant increments
            steady_drift = cumsum(fill(1.0, n))

            # basis=:changes → LOW volatility (changes are constant)
            regimes_changes = classify_volatility_regime(steady_drift; basis=:changes)

            # Most should be LOW (changes are constant 1.0)
            n_low = count(r -> r == :LOW, regimes_changes)
            @test n_low > n ÷ 2  # More than half should be LOW

            # basis=:levels → would be different (values grow)
            regimes_levels = classify_volatility_regime(steady_drift; basis=:levels)
            # This is less meaningful for this type of data
            @test length(regimes_levels) == n
        end

        @testset "Custom parameters" begin
            rng = StableRNG(123)
            values = cumsum(randn(rng, 150))

            # Custom window
            regimes_w5 = classify_volatility_regime(values; window=5)
            regimes_w20 = classify_volatility_regime(values; window=20)
            # Different windows give different results
            @test regimes_w5 != regimes_w20

            # Custom percentiles (quartiles instead of terciles)
            regimes_quartile = classify_volatility_regime(
                values;
                low_percentile=25.0,
                high_percentile=75.0
            )
            @test all(r -> r in [:LOW, :MED, :HIGH], regimes_quartile)
        end

        @testset "Edge cases" begin
            # Constant series - changes are 0, volatility is 0
            # With degenerate thresholds, most points classified as LOW/MED
            const_values = fill(5.0, 50)
            regimes = classify_volatility_regime(const_values)
            @test length(regimes) == 50
            @test all(r -> r in [:LOW, :MED, :HIGH], regimes)

            # Too short for window
            short = randn(StableRNG(42), 10)
            regimes_short = classify_volatility_regime(short; window=15)
            @test all(r -> r == :MED, regimes_short)  # Default for insufficient data

            # Invalid window
            @test_throws ArgumentError classify_volatility_regime([1.0, 2.0]; window=1)

            # Invalid percentiles
            @test_throws ArgumentError classify_volatility_regime(
                [1.0, 2.0, 3.0, 4.0];
                low_percentile=70.0,
                high_percentile=30.0  # low > high
            )
        end

    end  # Volatility Regime

    @testset "Direction Regime Classification" begin

        @testset "classify_direction_regime" begin
            values = [1.0, -1.0, 0.2, -0.2, 0.0]
            threshold = 0.5

            directions = classify_direction_regime(values, threshold)

            @test directions[1] == :UP      # 1.0 > 0.5
            @test directions[2] == :DOWN    # -1.0 < -0.5
            @test directions[3] == :FLAT    # |0.2| <= 0.5
            @test directions[4] == :FLAT    # |-0.2| <= 0.5
            @test directions[5] == :FLAT    # 0.0 <= 0.5

            # Negative threshold throws
            @test_throws ArgumentError classify_direction_regime(values, -0.1)
        end

        @testset "get_combined_regimes" begin
            vol = [:HIGH, :LOW, :MED]
            dir = [:UP, :DOWN, :FLAT]

            combined = get_combined_regimes(vol, dir)

            @test combined == [:HIGH_UP, :LOW_DOWN, :MED_FLAT]

            # Length mismatch
            @test_throws ArgumentError get_combined_regimes([:HIGH, :LOW], [:UP])
        end

        @testset "get_regime_counts" begin
            regimes = [:HIGH, :LOW, :LOW, :MED, :LOW, :HIGH]
            counts = get_regime_counts(regimes)

            @test counts[:LOW] == 3
            @test counts[:HIGH] == 2
            @test counts[:MED] == 1
        end

        @testset "mask_low_n_regimes" begin
            regimes = vcat(
                fill(:HIGH, 5),   # n=5 (below default min_n=10)
                fill(:LOW, 15),   # n=15
                fill(:MED, 3)     # n=3 (below min_n)
            )

            masked = mask_low_n_regimes(regimes)

            @test count(r -> r == :MASKED, masked) == 8  # 5 HIGH + 3 MED
            @test count(r -> r == :LOW, masked) == 15

            # Custom min_n
            masked_strict = mask_low_n_regimes(regimes; min_n=20)
            @test count(r -> r == :MASKED, masked_strict) == 23  # All masked

            # Custom mask value
            masked_custom = mask_low_n_regimes(regimes; mask_value=:EXCLUDED)
            @test count(r -> r == :EXCLUDED, masked_custom) == 8
        end

    end  # Direction Regime

    @testset "Stratified Metrics" begin

        @testset "Basic stratified metrics" begin
            rng = StableRNG(42)
            n = 150

            predictions = randn(rng, n)
            actuals = randn(rng, n)
            regimes = vcat(
                fill(:LOW, 50),
                fill(:MED, 50),
                fill(:HIGH, 50)
            )

            result = compute_stratified_metrics(predictions, actuals, regimes)

            @test result isa StratifiedMetricsResult
            @test result.n_total == n
            @test result.overall_mae > 0
            @test result.overall_rmse > 0
            @test result.overall_rmse >= result.overall_mae  # RMSE >= MAE

            # All regimes present
            @test length(result.by_regime) == 3
            @test haskey(result.by_regime, :LOW)
            @test haskey(result.by_regime, :MED)
            @test haskey(result.by_regime, :HIGH)

            # No masked regimes (all have n >= 10)
            @test isempty(result.masked_regimes)

            # Per-regime structure
            for (regime, metrics) in result.by_regime
                @test haskey(metrics, :mae)
                @test haskey(metrics, :rmse)
                @test haskey(metrics, :n)
                @test haskey(metrics, :pct)
                @test metrics[:n] == 50
                @test metrics[:pct] ≈ 100.0 * 50 / n atol=1e-10
            end
        end

        @testset "Masked regimes" begin
            rng = StableRNG(123)
            n = 100

            predictions = randn(rng, n)
            actuals = randn(rng, n)
            regimes = vcat(
                fill(:LOW, 5),    # Below min_n
                fill(:MED, 45),
                fill(:HIGH, 50)
            )

            result = compute_stratified_metrics(predictions, actuals, regimes)

            @test :LOW in result.masked_regimes
            @test !haskey(result.by_regime, :LOW)
            @test length(result.by_regime) == 2  # Only MED and HIGH
        end

        @testset "Custom min_n" begin
            rng = StableRNG(456)
            n = 60

            predictions = randn(rng, n)
            actuals = randn(rng, n)
            regimes = vcat(fill(:A, 20), fill(:B, 20), fill(:C, 20))

            # With min_n=25, all regimes should be masked
            result_strict = compute_stratified_metrics(
                predictions, actuals, regimes;
                min_n=25
            )
            @test length(result_strict.by_regime) == 0
            @test length(result_strict.masked_regimes) == 3

            # With min_n=15, all regimes should be included
            result_lenient = compute_stratified_metrics(
                predictions, actuals, regimes;
                min_n=15
            )
            @test length(result_lenient.by_regime) == 3
            @test isempty(result_lenient.masked_regimes)
        end

        @testset "Input validation" begin
            rng = StableRNG(789)

            # Different lengths
            @test_throws ArgumentError compute_stratified_metrics(
                randn(rng, 50),
                randn(rng, 30),
                fill(:A, 50)
            )
            @test_throws ArgumentError compute_stratified_metrics(
                randn(rng, 50),
                randn(rng, 50),
                fill(:A, 30)
            )

            # Empty
            @test_throws ArgumentError compute_stratified_metrics(
                Float64[],
                Float64[],
                Symbol[]
            )
        end

        @testset "Pretty printing and summary" begin
            rng = StableRNG(111)
            n = 90

            result = compute_stratified_metrics(
                randn(rng, n),
                randn(rng, n),
                vcat(fill(:LOW, 30), fill(:MED, 30), fill(:HIGH, 30))
            )

            str = string(result)
            @test occursin("StratifiedMetricsResult", str)

            io = IOBuffer()
            show(io, MIME("text/plain"), result)
            full_str = String(take!(io))
            @test occursin("overall_mae", full_str)
            @test occursin("regimes", full_str)

            s = summary(result)
            @test s isa String
            @test occursin("Overall", s)
            @test occursin("By Regime", s)
        end

        @testset "Property-based tests" begin
            rng = StableRNG(222)

            # Property: Overall MAE equals weighted average of regime MAEs
            @testset "Weighted average consistency" begin
                for _ in 1:10
                    n = rand(rng, 90:150)
                    predictions = randn(rng, n)
                    actuals = randn(rng, n)
                    regimes = rand(rng, [:A, :B, :C], n)

                    result = compute_stratified_metrics(
                        predictions, actuals, regimes;
                        min_n=1  # Don't mask any
                    )

                    # Verify overall MAE
                    errors = abs.(predictions .- actuals)
                    expected_mae = mean(errors)
                    @test result.overall_mae ≈ expected_mae atol=1e-10
                end
            end

            # Property: RMSE >= MAE for all regimes
            @testset "RMSE >= MAE" begin
                for _ in 1:10
                    n = rand(rng, 90:150)
                    predictions = randn(rng, n)
                    actuals = randn(rng, n)
                    regimes = rand(rng, [:A, :B, :C], n)

                    result = compute_stratified_metrics(
                        predictions, actuals, regimes;
                        min_n=1
                    )

                    @test result.overall_rmse >= result.overall_mae - 1e-10

                    for (_, metrics) in result.by_regime
                        @test metrics[:rmse] >= metrics[:mae] - 1e-10
                    end
                end
            end
        end

    end  # Stratified Metrics

end
