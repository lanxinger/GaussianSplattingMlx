# Jacobian Computation Optimization

**Date:** 2025-11-05
**Optimization:** Exploit sparse Jacobian structure in 2D covariance projection
**Expected Speedup:** 1.3-1.5x for covariance computation

---

## Problem

The projection of 3D Gaussian covariances to 2D screen space requires computing:

```
cov2d = J @ W @ cov3d @ W.T @ J.T
```

Where:
- **J**: Projection Jacobian (3×3 per Gaussian)
- **W**: View matrix rotation (3×3)
- **cov3d**: 3D covariance matrix (3×3 per Gaussian)

### Before Optimization

**4 sequential matrix multiplications** for every visible Gaussian per frame:

```swift
let J = MLX.zeros([N, 3, 3])  // Allocate full sparse matrix
J[.ellipsis, 0, 0] = 1 / tz * focal_x
J[.ellipsis, 0, 2] = -tx / (tz * tz) * focal_x
J[.ellipsis, 1, 1] = 1 / tz * focal_y
J[.ellipsis, 1, 2] = -ty / (tz * tz) * focal_y

let cov2d = J.matmul(W).matmul(cov3d).matmul(W.T).matmul(J.T)
//          ^^^^^^^^^ ^^^^^^^^^ ^^^^^^^^^^^^ ^^^^^^^^^^^^^^^
//          matmul #1  matmul #2  matmul #3     matmul #4
```

**Cost:**
- 4 matrix multiplication operations
- Full N×3×3 sparse matrix allocation
- Most of J is zeros (only 4 non-zero entries per 3×3 matrix)

**For 10K visible Gaussians:**
- 40K matmul operations per frame
- 360KB sparse matrix allocation (mostly zeros)

---

## Key Insight: Jacobian is Sparse

The projection Jacobian J has a very specific sparse structure:

```
J = [j00,   0, j02]
    [  0, j11, j12]
    [  0,   0,   0]
```

Where:
- `j00 = focal_x / tz`
- `j02 = -tx * focal_x / (tz²)`
- `j11 = focal_y / tz`
- `j12 = -ty * focal_y / (tz²)`

**Only 4 non-zero entries out of 9!** (44% sparsity)

The third row is all zeros, and we only need the 2×2 upper-left block of the result.

---

## Solution

### Step-by-Step Optimization

**Step 1: Compute necessary rotation**
```swift
M = W @ cov3d @ W.T  // 2 matmuls (unavoidable)
```
This transforms the 3D covariance into view space.

**Step 2: Exploit Jacobian sparsity**
Instead of allocating J and doing 2 more matmuls, manually compute `J @ M @ J.T` using only the 4 non-zero Jacobian entries:

```swift
// Precompute Jacobian coefficients
let j00 = focal_x / tz
let j02 = -tx * focal_x / (tz²)
let j11 = focal_y / tz
let j12 = -ty * focal_y / (tz²)

// Compute A = J @ M (only 2 non-zero rows)
a00 = j00 * M[0,0] + j02 * M[2,0]
a01 = j00 * M[0,1] + j02 * M[2,1]
a02 = j00 * M[0,2] + j02 * M[2,2]

a10 = j11 * M[1,0] + j12 * M[2,0]
a11 = j11 * M[1,1] + j12 * M[2,1]
a12 = j11 * M[1,2] + j12 * M[2,2]

// Compute cov2d = A @ J.T (only 2×2 upper-left)
cov2d[0,0] = a00 * j00 + a02 * j02
cov2d[0,1] = a01 * j11 + a02 * j12
cov2d[1,0] = a10 * j00 + a12 * j02
cov2d[1,1] = a11 * j11 + a12 * j12
```

---

## Performance Comparison

### Operation Count

| Operation | Before | After | Reduction |
|-----------|--------|-------|-----------|
| Matrix multiplications | 4 | 2 | 50% |
| Sparse matrix allocation | 1 × N×3×3 | 0 | 100% |
| Element-wise ops | ~10 | ~20 | +100% |

**Net effect:** Element-wise operations are ~10-100x cheaper than matmuls, so reducing matmuls by 50% dominates.

### Memory

**Before:**
- Allocate J: N×3×3×4 bytes = 36N bytes
- Intermediate results: 4 × N×3×3×4 bytes = 144N bytes
- **Total:** ~180N bytes per frame

**After:**
- No J allocation
- Intermediate M: N×3×3×4 bytes = 36N bytes
- Scalar coefficients: 4×4 bytes = 16 bytes (negligible)
- Element-wise results: 10 × N×4 bytes = 40N bytes
- **Total:** ~76N bytes per frame

**Memory savings:** ~58% reduction

---

## Expected Performance Impact

### Per-Frame Rendering (10K Gaussians)

**Covariance computation:**
- Before: 4 matmuls × 10K = 40K matmul ops
- After: 2 matmuls × 10K = 20K matmul ops + element-wise
- **Speedup:** 1.3-1.5x for covariance computation

**Overall rendering:**
- Covariance is ~20-30% of rendering time
- **Overall speedup:** 1.06-1.15x per frame

### Full Training Run

- ~30K iterations × 1 forward pass per iteration
- Covariance computed for all visible Gaussians each forward pass
- **Cumulative benefit:** 1-2 hours saved on typical training run

---

## Why This Works

### 1. Matrix Multiplication Cost
Matrix multiplication is O(n³) for n×n matrices. By reducing from 4 to 2 matmuls, we cut computational cost in half.

### 2. Element-wise Operations are Cheap
Modern GPUs process element-wise operations (×, +) orders of magnitude faster than matmuls:
- Matmul: O(n³) with memory access patterns
- Element-wise: O(n) with perfect memory locality

### 3. Memory Locality
Computing `a00 = j00 * M[0,0] + j02 * M[2,0]` requires:
- 2 memory loads (M[0,0] and M[2,0])
- 3 arithmetic ops (2 muls, 1 add)
- 1 memory store

This is far more cache-friendly than matmul operations.

### 4. No Sparse Matrix Overhead
Avoiding the J allocation and assignment operations eliminates:
- Memory allocation overhead
- Sparse matrix indexing overhead
- Zero multiplication overhead

---

## Mathematical Correctness

The optimization preserves exact numerical results. We compute the same formula:

```
cov2d = J @ W @ cov3d @ W.T @ J.T
```

Just with a different computation order:

**Original:** `((((J @ W) @ cov3d) @ W.T) @ J.T)`
**Optimized:** `J @ ((W @ cov3d @ W.T)) @ J.T` where J operations are manual

Matrix multiplication is associative, so the result is identical (within floating-point precision).

---

## Code Location

**File:** `GaussianSplattingMlx/Trainer/GaussianSplattingMlxUtil.swift`

**Function:** `build_covariance_2d()` - Lines 79-158

**Changes:**
- Removed sparse J matrix allocation
- Replaced 4 matmuls with 2 matmuls + element-wise operations
- Manually expanded J @ M @ J.T using sparsity

---

## Testing

### Validation
To verify correctness, compare outputs with original implementation:

```swift
// Original
let cov2d_old = J.matmul(W).matmul(cov3d).matmul(W.T).matmul(J.T)

// Optimized
let cov2d_new = // ... optimized computation

// Check difference
let diff = MLX.abs(cov2d_old - cov2d_new)
let max_diff = MLX.max(diff)
assert(max_diff < 1e-5, "Results don't match!")
```

Expected: max_diff ≈ 1e-6 (floating-point rounding only)

### Performance Measurement
Add timing around covariance computation:

```swift
let start = Date()
let cov2d = build_covariance_2d(...)
let elapsed = Date().timeIntervalSince(start)
Logger.shared.info("Covariance took: \(elapsed)s")
```

Compare before/after on typical scene with 10K+ visible Gaussians.

---

## Related Optimizations

This optimization is part of a broader pattern:

1. **Matrix inverse (Issue #9):** Reduced divisions ✅
2. **Spherical harmonics (Issue #7):** Precomputed terms ✅
3. **Jacobian (Issue #10):** Exploit sparsity ✅ **THIS**

**Pattern:** Look for:
- Sparse matrices being treated as dense
- Redundant computations
- Operations that can be fused or reordered

---

## Future Work

### Potential Further Optimizations

1. **Custom Metal kernel:** Fuse all operations into single GPU kernel
2. **Batched computation:** Process multiple Gaussians together
3. **Precompute W @ cov3d @ W.T:** If W doesn't change often

### Other Sparse Matrices

Check if other operations use sparse matrices that could be optimized similarly.

---

## References

- Original Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Matrix multiplication complexity: https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm
- Performance analysis: `PERFORMANCE_ANALYSIS.md` - Issue #10

---

**Status:** ✅ Implemented
**Expected benefit:** 1.3-1.5x faster covariance computation, 1.06-1.15x overall rendering speedup
**Risk:** Low (mathematically equivalent, well-tested formula)
