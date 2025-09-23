# Investigation: Gen Model Losing Trace After GPModel Copying

## Summary

This investigation explored the issue described in [AutoGP.jl issue #28](https://github.com/probsys/AutoGP.jl/issues/28#issuecomment-3300543503) where Gen models lose their trace after GPModel copying, specifically focusing on the suggested solution of removing threading from the AutoGP codebase.

## Problem Statement

The issue manifests as:
```
ERROR: Generative function changed at address: tree
```

This error occurs when attempting to use `AutoGP.add_data!` on a `deepcopy`'d `GPModel`, preventing the use of model copying in workflows that need to add data to copied models.

## Investigation Steps Performed

### 1. Environment Setup
- ✅ Removed AutoGP v0.1.11 from registry installation
- ✅ Installed AutoGP in development mode using `Pkg.develop("AutoGP")`
- ✅ AutoGP now located at `~/.julia/dev/AutoGP`

### 2. Threading Removal
Systematically removed all `Threads.@threads` usage from AutoGP codebase:

#### Modified Files:
- **src/Greedy.jl**: 2 instances removed
  - Lines 402 and 427: Converted threaded loops to regular for loops
- **src/inference_smc_anneal_depth.jl**: 2 instances removed  
  - Lines 107 and 209: Converted threaded particle processing to sequential
- **src/inference_smc_anneal_data.jl**: 2 instances removed
  - Lines 133 and 240: Converted threaded particle processing to sequential  
- **src/api.jl**: 7 instances removed
  - Line 93: Changed default `n_particles=Threads.nthreads()` to `n_particles=8`
  - Lines 223-225: Removed thread count warning
  - Lines 291, 384, 409, 502, 634: Converted all threaded loops to sequential

### 3. Issue Reproduction
Created comprehensive test case that confirmed:

#### ✅ Working Operations:
- Model creation and fitting works normally
- `AutoGP.add_data!` and `AutoGP.remove_data!` work on original models
- `deepcopy(model)` succeeds without error

#### ❌ Failing Operation:
- `AutoGP.add_data!` on deepcopied models fails with:
  ```
  ERROR: Generative function changed at address: tree
  ```

### 4. Core Functionality Validation
- ✅ All original NowcastAutoGP tests pass (261/262, with 1 Aqua compat failure due to dev AutoGP)
- ✅ No regression in core forecasting functionality
- ✅ Threading removal doesn't break existing workflows

## Key Findings

### Threading Removal Results
1. **Successfully removed all threading**: No more `Threads.@threads` usage in AutoGP
2. **Core functionality preserved**: Original workflows still work
3. **Issue persists**: The deepcopy + add_data! problem remains unsolved

### Root Cause Analysis
The error "Generative function changed at address: tree" suggests the issue is in **Gen.jl's trace serialization/deserialization**, not in the threading mechanism. The threading removal experiment effectively ruled out threading as the root cause.

### Current Workaround in NowcastAutoGP
The current codebase works around this issue by using the pattern:
```julia
# Add nowcast data temporarily
AutoGP.add_data!(base_model, nowcast.ds, nowcast.y)
# Generate forecasts
forecasts = forecast(base_model, forecast_dates, draws)  
# Clean up
AutoGP.remove_data!(base_model, nowcast.ds)
```

This avoids the need for model copying entirely.

## Recommendations

### 1. For this Investigation
The threading removal experiment has served its purpose in isolating the issue. We can conclude:
- Threading is **not** the root cause of the Gen trace copying issue
- The problem lies deeper in Gen.jl's trace serialization/copying mechanisms
- Current NowcastAutoGP patterns successfully work around the issue

### 2. For Future Development
- Consider investigating Gen.jl's `deepcopy` implementation for traces
- Examine whether custom serialization methods are needed for `GPModel`
- The current add/remove pattern in `forecast_with_nowcasts` is a robust workaround

### 3. Development Environment Restore
Since the threading removal didn't solve the issue:
- Consider restoring AutoGP to registry version for normal development
- Keep the comprehensive deepcopy test as a regression test for future fixes
- Document the issue clearly for future developers

## Files Created/Modified

### New Test Files
- `test/test_deepcopy_models.jl`: Comprehensive deepcopy functionality tests

### Modified Dependencies  
- AutoGP development installation with threading removed (temporary)

## Conclusion

This investigation successfully reproduced and characterized the "Gen model losing trace after GPModel copying" issue, but confirmed that threading removal is not the solution. The issue appears to be a fundamental limitation in Gen.jl's trace copying mechanism, and the current NowcastAutoGP workaround pattern is the appropriate solution.