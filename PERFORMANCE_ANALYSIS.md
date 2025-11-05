# Performance Analysis Report: GaussianSplattingMlx

**Analysis Date:** 2025-11-05
**Codebase:** GaussianSplattingMlx - 3D Gaussian Splatting for Apple Silicon/MLX

---

## Executive Summary

This analysis identifies 10+ performance optimization opportunities across initialization, training, and rendering pipelines. Implementing these optimizations could yield:

- **Initialization: 50-100x faster** (via O(N²) → O(N log N) nearest neighbor)
- **Training iteration: 2-3x faster** (via vectorization and sync reduction)
- **Split/prune operations: 20-50x faster** (via GPU-CPU sync elimination)
- **Training quality: 30-50% improvement** (via optimizer state preservation)

---

## 🔴 CRITICAL ISSUES (High Impact)

### 1. O(N²) Complexity in Nearest Neighbor Search
**Location:** `GaussianSplattingMlx/Trainer/GaussianModel.swift:11-31`

**Problem:**
```swift
for i in stride(from: 0, to: iterationSize, by: chunkSize) {
    let batchX1 = X1[i..<i + chunkSize]
    let diff = batchX1 - X2  // [chunk, N, 3] - broadcasts to ALL pairwise distances!
    let sq = MLX.square(diff)
    let dist2 = -1 * MLX.sum(sq, axes: [-1])  // O(N²) operation
```

**Impact:**
- For 100K points: 10 billion distance comparisons
- Primary initialization bottleneck
- Complexity: O(N²) space and time

**Solution:** Implement approximate nearest neighbor:
- Spatial hashing / KD-tree: O(N log N)
- Voxel grid approach
- FAISS or MLX-optimized k-NN

**Estimated Speedup:** 50-100x for initialization

---

### 2. GPU-CPU Synchronization in Split Operation
**Location:** `GaussianSplattingMlx/Trainer/GaussianTrainer.swift:250-255`

**Problem:**
```swift
for i in 0..<indices.shape[0] {
    let idx = indices[i].item(Int.self)  // ← GPU→CPU transfer EACH iteration!
    if idx < totalPoints {
        keepMask[idx] = MLXArray(false)
    }
}
```

**Impact:**
- Breaks GPU compute graph
- Forces synchronization for every split Gaussian
- Can be thousands of sync points per operation
- Serial execution on CPU

**Solution:** Vectorize using MLX scatter operations:
```swift
// Vectorized version
let validMask = indices .< totalPoints
let validIndices = indices[validMask]
keepMask = MLX.scatterUpdate(keepMask, validIndices, MLXArray(false))
```

**Estimated Speedup:** 10-50x for split/prune operations

---

### 3. Sequential Tile Rendering (No Parallelization)
**Location:** `GaussianSplattingMlx/Trainer/GaussianRenderer.swift:241-270`

**Problem:**
```swift
for h in stride(from: 0, to: camera.imageHeight, by: TILE_SIZE.h) {
    for w in stride(from: 0, to: camera.imageWidth, by: TILE_SIZE.w) {
        let (tile_color, tile_depth, acc_alpha) = renderTile(...)
        render_color[...] = tile_color  // Sequential processing
```

**Impact:**
- For 800x600 image with 16x16 tiles = 1,900 sequential renders
- Each tile independently sorts all visible Gaussians
- No parallel tile processing
- Multiple array slice assignments create copies

**Solution:**
- Use Metal compute shader for parallel tile processing
- Batch process all tiles in single MLX operation
- Global depth sort instead of per-tile sorting
- Single output buffer assignment

**Estimated Speedup:** 5-20x for rendering

---

### 4. 6x Full Array Concatenations in Split
**Location:** `GaussianSplattingMlx/Trainer/GaussianTrainer.swift:267-272`

**Problem:**
```swift
let newXYZAll = MLX.concatenated([keptXYZ, newXYZ1, newXYZ2], axis: 0)
let newFeaturesDCAll = MLX.concatenated([keptFeaturesDC, selectedFeaturesDC, selectedFeaturesDC], axis: 0)
let newFeaturesRestAll = MLX.concatenated([keptFeaturesRest, selectedFeaturesRest, selectedFeaturesRest], axis: 0)
let newScalesAll = MLX.concatenated([keptScales, newScales, newScales], axis: 0)
let newRotationAll = MLX.concatenated([keptRotation, selectedRotation, selectedRotation], axis: 0)
let newOpacityAll = MLX.concatenated([keptOpacity, selectedOpacity, selectedOpacity], axis: 0)
```

**Impact:**
- 6 full memory copies of potentially 100K+ Gaussian parameters
- For 100K Gaussians with 1K splits: ~600MB of data copied
- Creates memory fragmentation

**Solution:**
- Pre-allocate output arrays
- Use scatter/gather operations
- Single concatenation of all parameters

**Estimated Speedup:** 2-3x for split operation

---

### 5. Optimizer State Loss Every 100 Iterations
**Location:** `GaussianSplattingMlx/Trainer/GaussianTrainer.swift:474-476`

**Problem:**
```swift
if iteration % self.split_and_prune_per_iteration == 0 {
    self.split_and_prune(params: params, states: states, iteration: iteration)
    params = model.getParams()
    states = params.map {
        optimizer.newState(parameter: $0)  // ← Discards ALL momentum/variance!
    }
```

**Impact:**
- Adam optimizer loses first and second moment estimates
- Training convergence significantly degraded
- Effectively resets optimization progress every 100 iterations
- Wastes 99 iterations of momentum accumulation

**Solution:**
- Track parameter indices through split/prune operations
- Transfer optimizer states for kept Gaussians
- Initialize new Gaussian states from parent states (for splits)
- Interpolate states for cloned Gaussians

**Estimated Impact:** 30-50% improvement in training quality and convergence speed

---

## 🟠 HIGH PRIORITY ISSUES (Medium-High Impact)

### 6. Multiple eval() Synchronization Points
**Locations:** `GaussianTrainer.swift:420, 432, 456, 461`

**Problem:**
```swift
eval(loss[0], grads)      // Line 420
eval(image)               // Line 432
eval(newParam)            // Line 456 - inside loop (6x per iteration!)
eval(optimizer)           // Line 461 - unnecessary?
```

**Impact:**
- Forces GPU synchronization multiple times per iteration
- Inside parameter update loop = 6 sync points per iteration
- `eval(optimizer)` likely unnecessary

**Solution:**
```swift
// Batch eval calls
eval(loss[0], grads)  // Keep for gradient computation

// Move out of loop, single eval
eval(params)  // After all updates

// Only eval image when needed for delegate
if iteration % 20 == 0 {
    eval(image)
}
```

**Estimated Speedup:** 1.3-1.5x per training iteration

---

### 7. Spherical Harmonics Redundant Computations
**Location:** `GaussianSplattingMlx/Trainer/ShUtils.swift:42-95`

**Problem:**
```swift
if deg > 1 {
    let xx = x * x
    let yy = y * y
    let zz = z * z
    let xy = x * y
    let yz = y * z
    let xz = x * z
    if deg > 2 {
        // Computes: 3*xx, 3*yy, 4*zz multiple times
        result = (... + C3[2] * y * (4 * zz - xx - yy) * sh[.ellipsis, 11]
                     + C3[4] * x * (4 * zz - xx - yy) * sh[.ellipsis, 13] ...)
```

**Impact:**
- Repeated multiplications (e.g., `4 * zz` computed twice)
- Deeply nested branches cause branch misprediction
- Called for every visible Gaussian per frame

**Solution:**
```swift
// Precompute all intermediate terms once
let x2 = x * x, y2 = y * y, z2 = z * z
let xy = x * y, yz = y * z, xz = x * z
let x2_y2 = x2 - y2
let zz2 = 2 * z2
let zz4_x2_y2 = 4 * z2 - x2 - y2  // Shared term
let xx_3yy = x2 - 3 * y2
let yy_3xx = 3 * x2 - y2
// Then use precomputed values
```

**Estimated Speedup:** 1.2-1.3x for color computation

---

### 8. SSIM Window Recreation Every Iteration
**Location:** `GaussianSplattingMlx/Trainer/SsimUtils.swift:10-26`

**Problem:**
```swift
func ssim(img1: MLXArray, img2: MLXArray, windowSize: Int = 11, ...) -> MLXArray {
    let window = createWindow(windowSize: windowSize, channel: channel)  // ← Created every call!
    // Window computation involves:
    // - Gaussian generation
    // - matmul
    // - reshape
    // - broadcast
```

**Impact:**
- Called every training iteration (30K+ times)
- Window is constant (11x11, channel=3)
- Redundant Gaussian computation and matmul

**Solution:**
```swift
// In GaussianRenderer init:
let cachedSsimWindow = createWindow(windowSize: 11, channel: 3)

// In ssim function:
func ssim(img1: MLXArray, img2: MLXArray, window: MLXArray, ...) -> MLXArray {
    // Use provided window
```

**Estimated Speedup:** 1.05-1.1x per training iteration

---

### 9. Manual 2D Matrix Inverse
**Location:** `GaussianSplattingMlx/Trainer/GaussianSplattingMlxUtil.swift:203-219`

**Problem:**
```swift
func matrixInverse2d(_ m: MLXArray) -> MLXArray {
    let det = a * d - b * c
    inv[.ellipsis, 0, 0] = d / det    // 4 separate divisions
    inv[.ellipsis, 0, 1] = -b / det   // Each creates new array
    inv[.ellipsis, 1, 0] = -c / det
    inv[.ellipsis, 1, 1] = a / det
```

**Impact:**
- Called for every visible Gaussian per frame (renderTile → getSortedValues)
- 4 divisions instead of 1 (4x more operations)
- Each division creates intermediate array

**Solution:**
```swift
func matrixInverse2d(_ m: MLXArray) -> MLXArray {
    let det = a * d - b * c
    let invDet = 1.0 / det  // Single division
    inv[.ellipsis, 0, 0] = d * invDet    // Multiplications are cheaper
    inv[.ellipsis, 0, 1] = -b * invDet
    inv[.ellipsis, 1, 0] = -c * invDet
    inv[.ellipsis, 1, 1] = a * invDet
```

**Estimated Speedup:** 1.2-1.4x for rendering (called frequently)

---

### 10. Manual Jacobian Construction with Multiple Matmuls
**Location:** `GaussianSplattingMlx/Trainer/GaussianSplattingMlxUtil.swift:106-114`

**Problem:**
```swift
let J = MLX.zeros([mean3d.shape[0], 3, 3])  // Full sparse matrix allocation
J[.ellipsis, 0, 0] = 1 / tz * focal_x
J[.ellipsis, 0, 2] = -tx / (tz * tz) * focal_x
J[.ellipsis, 1, 1] = 1 / tz * focal_y
J[.ellipsis, 1, 2] = -ty / (tz * tz) * focal_y
// Then: J.matmul(W).matmul(cov3d).matmul(W.T).matmul(J.T)
// = 4 matrix multiplications
```

**Impact:**
- Allocates full N×3×3 matrix (mostly zeros)
- 4 sequential matmul operations per Gaussian
- Called every forward pass for all visible Gaussians

**Solution:**
- Exploit sparsity: J has only 4 non-zero entries
- Manually compute only non-zero contributions to final result
- Reduces from 4 matmuls to ~10 element-wise operations

**Estimated Speedup:** 1.3-1.5x for covariance computation

---

## 🟡 MEDIUM PRIORITY ISSUES

### 11. conditionToIndices Implementation
**Location:** `GaussianSplattingMlx/Trainer/GaussianSplattingMlxUtil.swift:220-228`

Uses `MLX.where` + `sorted` + `argMax` which is indirect. Could use:
```swift
MLX.where(condition).nonzero()  // If available in MLX
```

### 12. Image Pixel Normalization Loop
If ColmapDataLoader is used, per-pixel normalization should be vectorized with SIMD or MLX operations.

### 13. PLY Export Iteration
Per-point iteration in checkpoint export. Lower priority (I/O bound), but could batch MLX→binary conversion.

---

## 📊 Impact Summary

| Issue | Current | Optimized | Speedup | Priority |
|-------|---------|-----------|---------|----------|
| #1: distTopK | O(N²) | O(N log N) | 50-100x | P1 |
| #2: Split loop .item() | Serial CPU | Parallel GPU | 10-50x | P1 |
| #3: Tile rendering | Sequential | Parallel | 5-20x | P2 |
| #4: 6x concatenation | 6 copies | In-place | 2-3x | P2 |
| #5: Optimizer reset | Every 100 iter | Preserve | Quality +30-50% | **P0** |
| #6: Multiple eval() | 8+ per iter | 2-3 per iter | 1.3-1.5x | P1 |
| #7: SH recomputation | Redundant | Cached | 1.2-1.3x | P2 |
| #8: SSIM window | Every call | Cached | 1.05-1.1x | P2 |
| #9: Matrix inverse | 4 divs | 1 div | 1.2-1.4x | P1 |
| #10: Jacobian matmuls | 4 matmuls | Direct | 1.3-1.5x | P2 |

**Overall Potential:**
- **Initialization: 50-100x faster**
- **Training iteration: 2-3x faster**
- **Split/prune: 20-50x faster**
- **Training quality: +30-50%**

---

## 🎯 Recommended Implementation Order

### Phase 1: Quick Wins (1-2 hours)
1. ✅ **Fix #9 (Matrix inverse)** - 1 line change
2. ✅ **Fix #6 (Batch eval())** - 3 line change
3. ✅ **Fix #8 (Cache SSIM window)** - 5 lines + init parameter
4. ✅ **Fix #7 (Precompute SH terms)** - 10 lines

**Expected: 1.5-2x training speedup immediately**

### Phase 2: Critical Fixes (1-2 days)
5. ⭐ **Fix #5 (Optimizer state preservation)** - HIGH IMPACT on quality
6. ⭐ **Fix #2 (Vectorize split loop)** - Replace .item() loop with scatter

**Expected: Major quality improvement + 10-50x split/prune speedup**

### Phase 3: Algorithmic Improvements (3-5 days)
7. **Fix #1 (Approximate k-NN)** - Implement KD-tree or spatial hashing
8. **Fix #4 (Pre-allocate in split)** - Reduce memory copies

**Expected: 50-100x init speedup + 2-3x split speedup**

### Phase 4: Architectural Changes (1-2 weeks)
9. **Fix #3 (Parallel tile rendering)** - Metal compute shader
10. **Fix #10 (Optimize Jacobian)** - Manual sparse computation

**Expected: 5-20x rendering speedup**

---

## 🔧 Implementation Notes

### Testing Strategy
1. Benchmark current performance:
   - Initialization time
   - Per-iteration time
   - Split/prune time
   - Final PSNR/SSIM metrics

2. Implement optimizations incrementally
3. Validate correctness (compare outputs with original)
4. Measure performance improvement
5. Run full training to verify convergence

### Correctness Validation
- Use small test scenes (garden, lego)
- Compare rendered images pixel-by-pixel (< 0.01 diff)
- Verify gradient magnitudes match
- Check optimizer state shapes after split/prune

### Performance Measurement
```swift
let start = Date()
// ... operation ...
let elapsed = Date().timeIntervalSince(start)
```

Add timers to:
- `distTopK()` (initialization)
- `split_and_prune()` (densification)
- `forward()` (rendering)
- Training iteration loop

---

## 🚀 Next Steps

1. **Validate findings**: Run profiling with Instruments to confirm hotspots
2. **Prioritize**: Decide between quick wins vs. high-impact quality fix (#5)
3. **Implement Phase 1**: Start with trivial optimizations
4. **Measure**: Benchmark before/after for each change
5. **Iterate**: Move to Phase 2 based on results

---

## 📚 References

- Original Gaussian Splatting paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- MLX documentation: https://ml-explore.github.io/mlx/
- Apple Metal Performance Shaders: https://developer.apple.com/metal/

---

**Analysis Performed By:** Claude Code
**Date:** 2025-11-05
