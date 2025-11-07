# Performance Debugging Guide

## Current Status: 2 iterations/second (Too Slow!)

Expected: 10-20 iterations/second on M4 Pro for typical scenes.

---

## Key Finding: MLX Already Uses Metal!

**Important:** The "MLX renderer" IS Metal-accelerated. MLX runs all GPU operations through Metal kernels. The distinction isn't "Metal vs MLX" but rather "optimized Metal kernels vs generic operations."

- ✅ MLX automatically uses Metal for GPU acceleration
- ✅ Unified memory architecture is utilized
- ✅ Automatic differentiation works through Metal

---

## Performance Profiling Checklist

### 1. Check Your Training Configuration

**What's your image resolution?**
```swift
// In your training data
// 256×256 = Fast (~10-20 iter/s expected)
// 512×512 = Medium (~5-10 iter/s expected)
// 1024×1024 = Slow (~2-5 iter/s expected)
```

**How many Gaussians?**
```swift
// Check in Xcode console during training
// 10K-100K = Fast
// 100K-500K = Medium
// 500K-1M+ = Slow
```

**How many training views?**
```swift
// More views = more memory pressure
// Recommended: 50-200 views
```

### 2. Quick Performance Tests

**Test A: Disable SSIM (saves ~20-30% compute)**
```swift
// In GaussianTrainer.swift line 73
var lambda_dssim: Float = 0.0  // Was: 0.2
```

**Test B: Reduce SSIM frequency**
```swift
// In GaussianTrainer.swift around line 661
let ssim_loss: MLXArray
if iteration % 5 == 0 {  // Only compute every 5 iterations
    ssim_loss = 1.0 - ssim(...)
} else {
    ssim_loss = MLXArray(0.0)  // Skip SSIM
}
```

**Test C: Check tile rendering overhead**
```swift
// In GaussianRenderer.swift line 114
// Increase tile size to reduce overhead
let TILE_SIZE: TILE_SIZE_H_W = TILE_SIZE_H_W(w: 32, h: 32)  // Was: 16x16
```

### 3. Verify GPU Utilization

Run Activity Monitor while training:
1. Open Activity Monitor
2. Window → GPU History
3. Should see **high GPU utilization** (>70%)
4. If GPU is idle → problem is CPU-side or synchronization

### 4. Known Bottlenecks

#### A. SSIM Computation (20-30% of training time)
- 5 conv2d operations per iteration
- Window size 11×11 with 3 channels
- Gradients required (can't skip)

**Current optimization:** ✅ Cached window (already implemented)

#### B. Tile-Based Rendering
- For 256×256 image with 16×16 tiles = 256 tiles
- Each tile: sort Gaussians, compute weights, alpha blend
- Inherently sequential (nested loops)

**Potential optimization:** Larger tiles (less overhead, more parallelism)

#### C. Data Fetching
- `fetchTrainDataWithResolution()` called every iteration
- With progressive resolution disabled, should be fast
- Check if `conditionToIndices` for mask is slow

---

## Metal 4 Features (M4 Pro)

Your M4 Pro supports Metal 4 with these enhancements:

### 1. **Native Tensor Support** ⭐ Most Relevant
- First-class tensor operations in Metal shaders
- Could enable custom fused kernels with gradients
- **Status:** Would require significant rewrite

### 2. **MetalFX Denoising**
- For ray tracing/path tracing
- **Status:** Not applicable to Gaussian Splatting

### 3. **Unified Command Encoder**
- Better performance for command submission
- **Status:** MLX handles this automatically

### 4. **Improved Memory Management**
- Placement sparse resources
- **Status:** MLX handles this automatically

---

## Realistic Optimizations (Ranked)

### Priority 1: Reduce SSIM Cost (Easy, 20-30% gain)
```swift
// Option A: Disable for debugging
var lambda_dssim: Float = 0.0

// Option B: Compute less frequently
if iteration % 5 == 0 {
    ssim_loss = 1.0 - ssim(...)
}

// Option C: Smaller window (7x7 instead of 11x11)
self.cachedSsimWindow = createWindow(windowSize: 7, channel: 3)
```

### Priority 2: Larger Tile Size (Easy, 10-20% gain)
```swift
// GaussianRenderer.swift line 114
TILE_SIZE: TILE_SIZE_H_W(w: 32, h: 32)  // Was: 16x16
// Fewer tiles = less loop overhead
```

### Priority 3: Profile and Find Hotspot (Medium effort)
Add timing around major sections:
```swift
let startTime = Date()
// ... code section ...
let elapsed = Date().timeIntervalSince(startTime)
print("Section took: \(elapsed * 1000)ms")
```

Time these sections:
1. Forward pass (rendering)
2. SSIM computation
3. Gradient computation (valueAndGrad)
4. Optimizer step
5. Data fetching

### Priority 4: Custom Metal Kernels (Hard, 2-5x gain potential)
- Implement custom fused Metal kernel for tile rendering
- Would require vjp/jvp for gradients
- **Recommendation:** Only if other optimizations insufficient

---

## Expected Performance Targets

| Resolution | Gaussians | Expected Speed | Your Speed |
|------------|-----------|----------------|------------|
| 256×256    | 100K      | 15-20 iter/s   | 2 iter/s ❌ |
| 512×512    | 100K      | 8-12 iter/s    | ? |
| 1024×1024  | 100K      | 2-4 iter/s     | ? |
| 256×256    | 500K      | 8-12 iter/s    | ? |

---

## Diagnostic Commands

```swift
// Add to training loop to diagnose:

// 1. Check resolution
print("Training at: \(trainCamera.imageWidth)×\(trainCamera.imageHeight)")

// 2. Check Gaussian count
print("Number of Gaussians: \(model._xyz.shape[0])")

// 3. Check memory usage
print("GPU Memory: \(MLX.GPU.snapshot().cacheMemory / 1024 / 1024)MB")

// 4. Time one iteration
let startTime = Date()
// ... one training iteration ...
print("Iteration time: \(Date().timeIntervalSince(startTime) * 1000)ms")
```

---

## Next Steps

1. **Run Test A** (disable SSIM) to see if that's the bottleneck
2. **Check image resolution** in Xcode console
3. **Verify GPU utilization** in Activity Monitor
4. **Report back** with findings

If after all tests you're still at 2 iter/s on 256×256 images with 100K Gaussians, there's likely a deeper architectural issue we need to investigate.
