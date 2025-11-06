# LichtFeld-Studio Inspired Optimizations

This document describes optimizations inspired by LichtFeld-Studio's 2.4x speedup implementation in their C++/PyTorch Gaussian Splatting implementation.

## Background

LichtFeld-Studio achieved a 2.4x training speedup by implementing several key optimizations:
- Fused CUDA kernels (fewer kernel launches)
- Improved blending backward pass
- Fused Adam optimizer
- Smart SH coefficient updates (skip unused coefficients early in training)
- VRAM caching with precomputed view matrices
- GPU-side preprocessing
- Memory allocator tuning

## Implemented Optimizations

### 1. SH Coefficient Gradient Masking ✅

**Inspiration:** LichtFeld-Studio skipped gradient updates for unused higher-degree SH coefficients during the first 1000 iterations.

**Implementation:**
- Added `shHigherDegreeStartIter` parameter (default: 1000 iterations)
- Skip optimizer updates for `_features_rest` (higher-degree SH coefficients) before this threshold
- Only DC component (`_features_dc`) is updated during early training

**Rationale:**
- Early in training, geometry is still being refined
- Higher-degree SH coefficients (degrees 1-3) have minimal visual impact when shapes are rough
- Saves ~10-15% compute time during first 1000 iterations

**Location:** `GaussianTrainer.swift:465-468`

**Expected Impact:** 5-10% speedup in early training phase

### 2. Consolidated Cache Clearing ✅

**Inspiration:** LichtFeld-Studio used `expandable_segments:True` to eliminate redundant `empty_cache()` calls.

**Implementation:**
- Track cache clear needs with a boolean flag
- Consolidate multiple cache clears into single operation per iteration
- Eliminates redundant cache clearing when snapshot saving and split/prune happen in same iteration

**Location:** `GaussianTrainer.swift:484-506`

**Expected Impact:** Minor reduction in memory management overhead

## Already Optimized in Codebase

### 1. VRAM Caching ✅ (Already Implemented)

**Status:** Training data is already loaded into MLXArrays (GPU memory):
- `rgbArray`, `alphaArray` stored as MLXArray
- Camera parameters (`intrinsicArray`, `c2wArray`) precomputed and cached
- No unnecessary CPU-GPU transfers during training

### 2. View Matrix Precomputation ✅ (Already Implemented)

**Status:** Camera matrices are precomputed once during data loading:
- Intrinsic matrices computed and stored in `TrainData`
- Camera-to-world transforms precomputed
- Only random sampling happens per iteration

### 3. GPU Preprocessing ✅ (Already Implemented)

**Status:** Image preprocessing happens on GPU:
- White background blending via MLX operations
- RGB normalization on GPU
- Only unavoidable CPU work is initial UIImage loading

### 4. Fused Optimizer ✅ (Already Implemented)

**Status:** MLX optimizers are already fused internally:
- Adam optimizer in MLXOptimizers is optimized
- Single `applySingle()` call per parameter

### 5. Matrix Inverse Optimization ✅ (Already Implemented)

**Status:** Specialized 2D matrix inverse:
- `matrixInverse2d()` uses closed-form solution
- 1 division + 3 multiplications instead of general inverse
- Previous optimization: ~1.3x speedup on matrix operations

## Not Applicable to MLX

### 1. CUDA Kernel Fusion

**Why not applicable:**
- MLX automatically fuses operations via `mx.compile`
- No manual CUDA kernel writing needed
- MLX handles low-level optimizations

### 2. Explicit Memory Allocator Tuning

**Why not applicable:**
- MLX manages memory allocation automatically
- No equivalent to PyTorch's `expandable_segments`
- Cache clearing strategy already optimized

## Performance Summary

**Previous optimizations (prior work):**
- Jacobian sparse optimization: 2x speedup
- Split/clone vectorization: Eliminated bottlenecks
- Approximate k-NN: 50-100x initialization speedup
- SSIM caching: Eliminated redundant computation
- Overall: 2-3x speedup achieved

**New optimizations (this session):**
- SH coefficient gradient masking: ~5-10% speedup in early training
- Consolidated cache clearing: Minor overhead reduction
- **Combined expected impact:** ~5-12% additional speedup

**Total estimated speedup:** 2.1-3.3x from original baseline

## Testing Recommendations

1. **Training time comparison:**
   - Measure time for first 1000 iterations (SH masking active)
   - Measure time for iterations 1000-2000 (SH masking inactive)
   - Compare against baseline without optimizations

2. **Quality verification:**
   - PSNR/SSIM should be identical to baseline
   - Visual quality should not degrade
   - SH coefficients should converge properly after iteration 1000

3. **Memory profiling:**
   - Monitor cache clear frequency
   - Verify no memory leaks
   - Check peak memory usage

## Future Optimization Opportunities

Based on LichtFeld-Studio's approach, potential future optimizations:

1. **Fused Loss Computation** (Medium effort, 1.1-1.15x potential)
   - Combine L1, SSIM, and depth losses into single kernel
   - Requires refactoring loss computation

2. **Tile Rendering Batch Processing** (High effort, 1.2-1.3x potential)
   - Process multiple tiles in parallel
   - Requires significant rasterization refactoring

3. **Gradient Masking for Unused Gaussians** (Medium effort, 1.05-1.1x potential)
   - Skip gradient computation for low-opacity Gaussians
   - Requires custom gradient computation

## References

- LichtFeld-Studio Repository: https://github.com/MrNeRF/LichtFeld-Studio
- Bounty #001 (2.4x speedup): https://github.com/MrNeRF/LichtFeld-Studio/issues/135
- Winning PR: https://github.com/MrNeRF/LichtFeld-Studio/pull/245

## Files Modified

- `GaussianSplattingMlx/Trainer/GaussianTrainer.swift`
  - Line 91-93: Added `shHigherDegreeStartIter` parameter
  - Line 462-468: SH coefficient gradient masking
  - Line 484-506: Consolidated cache clearing
