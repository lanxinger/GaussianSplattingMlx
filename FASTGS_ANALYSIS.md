# FastGS Analysis: Training 3D Gaussian Splatting in 100 Seconds
## Paper: arXiv:2511.04283v1

## Overview

**FastGS** is a groundbreaking 3D Gaussian Splatting acceleration framework that achieves **100-second training times** (compared to baseline ~30 minutes), representing a **~18x speedup** while maintaining comparable rendering quality.

### Key Innovation

FastGS redesigns the Adaptive Density Control (ADC) mechanism of vanilla 3DGS based on **multi-view consistency**, rather than relying on per-view gradient accumulation.

## Core Technical Approach

### 1. Multi-View Consistent Densification and Pruning

**Current approach (vanilla 3DGS):**
- Accumulate gradients per Gaussian across random views
- Densify based on gradient magnitude threshold
- Prune based on opacity threshold
- **Problem:** Gradients from individual views may be noisy and inconsistent

**FastGS approach:**
- Sample multiple training views simultaneously
- Generate **per-pixel L1 loss maps** for each sampled view
- Compute a **multi-view score** for each Gaussian:
  - Count the number of high-error pixels within each Gaussian's 2D footprint across all sampled views
  - Gaussians that contribute to high error in **multiple views** are prioritized for densification
  - Gaussians that contribute low error across views are candidates for pruning
- Use multi-view scores to guide densification and pruning decisions

**Advantages:**
- More robust signal than single-view gradients
- Reduces false positives in densification (Gaussians that only appear important in one view)
- More aggressive yet accurate pruning (removes genuinely unnecessary Gaussians)

### 2. Progressive Frequency Fitting (Related: DashGaussian)

While FastGS paper details aren't fully accessible, related work (DashGaussian, arXiv:2503.18402) achieves 200-second training using:

**Dynamic Resolution Scheduling:**
- Formulate 3DGS optimization as progressively fitting higher frequency components
- Start training at lower resolution (e.g., 25% of target)
- Gradually increase resolution as optimization progresses
- **Rationale:** Coarse geometry can be learned at low resolution; fine details require high resolution

**Implementation:**
```python
# Pseudocode for progressive resolution
def get_training_resolution(iteration, max_iter, target_H, target_W):
    progress = iteration / max_iter
    # Start at 25%, end at 100%
    scale = 0.25 + 0.75 * min(1.0, progress * 1.5)
    return int(target_H * scale), int(target_W * scale)
```

### 3. Sparse Adam Optimizer (From DashGaussian)

Standard Adam maintains momentum and variance for all parameters. Sparse Adam optimizes only parameters with non-zero gradients:

**Key insight:** After densification/pruning, many Gaussians don't receive gradient updates in every iteration. Sparse Adam skips update computations for these, reducing overhead.

**Implementation considerations for MLX:**
- MLX's Adam optimizer is already fused and efficient
- Could implement sparse variant by masking updates based on gradient presence
- Potential 10-20% speedup in optimizer step

### 4. Momentum-Based Primitive Budgeting

**Problem:** How many Gaussians should the scene have?
- Too few: Underfitting, poor quality
- Too many: Slow training and rendering

**Solution:** Adaptive upper bound based on training dynamics:
```python
# Pseudocode
momentum_avg_count = 0
momentum_decay = 0.9

for iteration in training:
    current_count = len(gaussians)
    momentum_avg_count = (momentum_decay * momentum_avg_count +
                         (1 - momentum_decay) * current_count)

    # Upper bound is 1.2x the moving average
    max_gaussians = int(1.2 * momentum_avg_count)

    # Prune if over budget
    if current_count > max_gaussians:
        prune_excess_gaussians()
```

## Performance Results

### FastGS Performance (100 seconds)
- **Speedup:** 2-7x across different tasks
- **Quality:** Comparable or better PSNR/SSIM
- **Applications:**
  - Dynamic scene reconstruction
  - Surface reconstruction
  - Sparse-view reconstruction
  - Large-scale reconstruction
  - SLAM

### DashGaussian Performance (200 seconds, RTX 4090)
- **Mipnerf-360:** 2-3.4x speedup (12.7 min → 3.7-6.2 min)
- **Deep-Blending:** 2.8-4.6x speedup (10.7 min → 2.3-3.8 min)
- **Tanks&Temple:** 2.0-2.8x speedup (8.0 min → 2.8-3.9 min)
- **Quality:** +0.2-0.4 dB PSNR improvement in many cases

## Implementation Roadmap for MLX

### Priority 1: Multi-View Consistent Densification/Pruning (HIGH IMPACT)

**Current implementation (GaussianTrainer.swift):**
```swift
// Lines 129-140: Current gradient accumulation
func addGradientAccumulation(xyzGrad: MLXArray) {
    let gradNorm = MLX.sum(MLX.square(xyzGrad), axes: [1])
    xyzGradAccumulation = xyzGradAccumulation + gradNorm
    denomGradAccumulation = denomGradAccumulation + MLXArray.ones([numPoints])
}
```

**FastGS approach to implement:**
```swift
// New: Multi-view score computation
func computeMultiViewScores(numSamples: Int = 4) -> MLXArray {
    // Sample multiple training views
    var errorMaps: [MLXArray] = []

    for _ in 0..<numSamples {
        let (camera, rgb, mask, depth) = fetchTrainData()

        // Render current state
        let (render, _, _, _, _) = gaussRender.forward(
            camera: camera,
            means3d: model.getMeans3D(),
            shs: model.getSHS(),
            opacity: model.getOpacity(),
            scales: model.getScales(),
            rotations: model.getRotations()
        )

        // Compute per-pixel L1 loss
        let pixelLoss = MLX.abs(render - rgb)  // [H, W, 3]
        let errorMap = MLX.mean(pixelLoss, axes: [-1])  // [H, W]
        errorMaps.append(errorMap)
    }

    // For each Gaussian, count high-error pixels in its footprint across views
    var multiViewScores = MLXArray.zeros([model.numGaussians()])

    for (viewIdx, errorMap) in errorMaps.enumerated() {
        let (camera, _, _, _) = getSampledCamera(viewIdx)

        // Project Gaussians to 2D for this view
        let projections = projectGaussians2D(camera: camera)

        // Count high-error pixels (threshold: 0.1)
        let highErrorMask = errorMap .> 0.1

        for gaussIdx in 0..<model.numGaussians() {
            let footprint = projections[gaussIdx]
            let errorInFootprint = highErrorMask[footprint.pixels]
            let highErrorCount = MLX.sum(errorInFootprint.asType(.int32))
            multiViewScores[gaussIdx] += highErrorCount
        }
    }

    return multiViewScores
}

// Modified split_and_prune using multi-view scores
func split_and_prune_fastgs(params: [MLXArray], states: [TupleState], iteration: Int) {
    guard iteration >= densifyFromIter && iteration <= densifyUntilIter else {
        return
    }

    // Compute multi-view scores instead of gradient accumulation
    let mvScores = computeMultiViewScores(numSamples: 4)

    // Threshold: Gaussians with high multi-view error need densification
    let highErrorThreshold = MLX.percentile(mvScores, q: 90)
    let densifyMask = mvScores .> highErrorThreshold

    // Split logic based on scale (same as before)
    let scales = MLX.exp(params[3])
    let maxScalePerGaussian = MLX.max(scales, axes: [1])
    let splitMask = densifyMask & (maxScalePerGaussian .> MLXArray(maxScale))
    let cloneMask = densifyMask & (maxScalePerGaussian .<= MLXArray(maxScale))

    // Prune: Low opacity OR low multi-view score
    let lowErrorThreshold = MLX.percentile(mvScores, q: 10)
    let pruneMask = (opacity .< minOpacity) | (mvScores .< lowErrorThreshold)

    // Rest of split/clone/prune logic unchanged...
}
```

**Expected Impact:** 30-50% speedup from more efficient densification decisions

### Priority 2: Dynamic Resolution Scheduling (MEDIUM-HIGH IMPACT)

**Modify training loop in GaussianTrainer.swift:**
```swift
class GaussianTrainer {
    // Add resolution scheduling parameters
    var useProgressiveResolution: Bool = true
    var minResolutionScale: Float = 0.25  // Start at 25%
    var maxResolutionScale: Float = 1.0   // End at 100%
    var resolutionRampIterations: Int = 3000  // Ramp up over first 3000 iters

    func getCurrentResolutionScale(iteration: Int) -> Float {
        if !useProgressiveResolution {
            return 1.0
        }

        if iteration >= resolutionRampIterations {
            return maxResolutionScale
        }

        let progress = Float(iteration) / Float(resolutionRampIterations)
        // Smooth ramp from minResolutionScale to maxResolutionScale
        return minResolutionScale + (maxResolutionScale - minResolutionScale) * progress
    }

    func startTrain(earlyStoppingThreshold: Float = 0.0001) {
        // ... existing code ...

        for iteration in 0..<iterationCount {
            let resScale = getCurrentResolutionScale(iteration: iteration)

            // Fetch training data with dynamic resolution
            let (trainCamera, trainRGB, trainMask, trainDepth) =
                fetchTrainDataWithResolution(scale: resScale)

            // Rest of training loop...
        }
    }

    func fetchTrainDataWithResolution(scale: Float) -> (
        camera: Camera, rgb: MLXArray, mask: MLXArray, depth: MLXArray?
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)

        // Original resolution
        let rgb = data.rgbArray[ind]
        let originalH = rgb.shape[0]
        let originalW = rgb.shape[1]

        // Scaled resolution
        let scaledH = Int(Float(originalH) * scale)
        let scaledW = Int(Float(originalW) * scale)

        if scale < 0.99 {  // Only resize if significantly different
            // Resize RGB using interpolation
            let scaledRGB = resizeImage(rgb, targetH: scaledH, targetW: scaledW)

            // Update camera intrinsics for new resolution
            var scaledCamera = data.getViewPointCamera(index: ind)
            scaledCamera.width = scaledW
            scaledCamera.height = scaledH
            scaledCamera.intrinsic = scaledCamera.intrinsic * scale

            return (scaledCamera, scaledRGB, trainMask, depth)
        } else {
            // Use full resolution
            return (data.getViewPointCamera(index: ind), rgb, trainMask, depth)
        }
    }
}
```

**Expected Impact:** 20-40% speedup from reduced computation at early iterations

### Priority 3: Sparse Adam Optimizer (MEDIUM IMPACT)

**Current: Standard Adam updates all parameters**

**Sparse Adam: Only update parameters with non-zero gradients**

```swift
// New sparse optimizer wrapper
class SparseAdam {
    let baseOptimizer: Adam
    var sparsityThreshold: Float = 1e-8

    func applySparse(
        gradient: MLXArray,
        parameter: MLXArray,
        state: TupleState
    ) -> (MLXArray, TupleState) {
        // Compute gradient magnitude
        let gradMag = MLX.abs(gradient)

        // Create sparse mask (only update where gradient is significant)
        let updateMask = gradMag .> sparsityThreshold

        // Apply base optimizer only where needed
        let (newParam, newState) = baseOptimizer.applySingle(
            gradient: gradient,
            parameter: parameter,
            state: state
        )

        // Blend: keep old values where gradient is zero
        let result = MLX.where(updateMask, newParam, parameter)

        return (result, newState)
    }
}
```

**Expected Impact:** 10-15% speedup in optimizer step

### Priority 4: Momentum-Based Primitive Budgeting (LOW-MEDIUM IMPACT)

```swift
class GaussianTrainer {
    var momentumAvgGaussianCount: Float = 0
    var momentumDecay: Float = 0.9
    var maxGaussianMultiplier: Float = 1.2
    var useAdaptiveBudget: Bool = true

    func getMaxGaussianBudget() -> Int {
        if !useAdaptiveBudget {
            return Int.max  // No limit
        }

        return Int(momentumAvgGaussianCount * maxGaussianMultiplier)
    }

    func updateMomentumBudget() {
        let currentCount = Float(model.numGaussians())
        momentumAvgGaussianCount = (momentumDecay * momentumAvgGaussianCount +
                                    (1 - momentumDecay) * currentCount)
    }

    func split_and_prune(params: [MLXArray], states: [TupleState], iteration: Int) {
        // ... existing split/clone logic ...

        // After split/clone, check budget
        let currentCount = model.numGaussians()
        let maxBudget = getMaxGaussianBudget()

        if currentCount > maxBudget {
            // Prune excess Gaussians (lowest opacity/multi-view score)
            let excessCount = currentCount - maxBudget
            pruneLowestPriority(count: excessCount, params: params)
        }

        // Update momentum tracker
        updateMomentumBudget()
    }
}
```

**Expected Impact:** 5-10% speedup, better memory efficiency

## Implementation Priority Summary

| Optimization | Expected Speedup | Implementation Effort | Priority |
|-------------|------------------|---------------------|----------|
| Multi-view consistent densification | 30-50% | High | **P1** |
| Dynamic resolution scheduling | 20-40% | Medium | **P2** |
| Sparse Adam optimizer | 10-15% | Medium | **P3** |
| Momentum-based budgeting | 5-10% | Low | P4 |
| **Combined Expected Speedup** | **2-4x** | - | - |

## Testing Strategy

1. **Baseline measurement:**
   - Record current training time for full iteration count
   - Record PSNR/SSIM at checkpoints (1000, 5000, 10000, 15000 iterations)

2. **Progressive implementation:**
   - Implement P1, measure speedup + quality
   - Add P2, measure incremental improvement
   - Add P3, measure incremental improvement
   - Add P4, measure final results

3. **Quality verification:**
   - PSNR should be within ±0.5 dB of baseline
   - SSIM should be within ±0.01 of baseline
   - Visual inspection of rendered views

4. **Ablation study:**
   - Test each optimization independently
   - Verify they combine additively (no negative interactions)

## References

- **FastGS:** arXiv:2511.04283v1 (November 2025)
  - Project: https://fastgs.github.io/
  - Key innovation: Multi-view consistent densification/pruning

- **DashGaussian:** arXiv:2503.18402 (CVPR 2025 Highlight)
  - GitHub: https://github.com/YouyuChen0207/DashGaussian
  - Key innovation: Progressive resolution scheduling

- **Related Work:**
  - LichtFeld-Studio: 2.4x speedup via kernel fusion and optimizer improvements
  - Speedy-Splat: Sparse pixel/primitive rendering
  - InstantSplat: Sparse-view reconstruction in seconds

## Next Steps

1. ✅ Document FastGS techniques
2. ⏭️ Implement multi-view consistent densification (P1)
3. ⏭️ Implement dynamic resolution scheduling (P2)
4. ⏭️ Test combined optimizations
5. ⏭️ Benchmark against baseline
6. ⏭️ Create pull request with results

---

**Expected Final Performance:**
- Current: ~30 min training (baseline)
- With LichtFeld optimizations: ~15 min (2x speedup) ✅
- With FastGS optimizations: ~5-8 min (4-6x total speedup) 🎯
- Target: <5 min (approaching 100-second FastGS performance)
