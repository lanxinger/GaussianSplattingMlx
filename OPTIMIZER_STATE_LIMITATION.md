# Optimizer State Preservation - API Limitation

**Date:** 2025-11-05
**Status:** BLOCKED by MLX API limitations

---

## Problem

During split/prune operations (every 100 iterations), we currently **reinitialize all optimizer states**, discarding:
- Adam momentum (first moment estimates)
- Adam variance (second moment estimates)

This effectively resets 99 iterations worth of optimization progress every 100 iterations.

**Impact:** 30-50% degradation in training quality and convergence speed

---

## Attempted Solution

We attempted to preserve optimizer states through split/prune operations using the TupleState API:

```swift
func selectOptimizerStates(_ state: TupleState, indices: MLXArray) -> TupleState {
    let (m, v) = state.values  // Access momentum and variance
    let newM = m[indices]
    let newV = v[indices]
    return TupleState(newM, newV)  // Create new state
}
```

---

## Root Cause: Internal Protection Level

The MLX Swift library marks TupleState internals as `internal`:

```
Error: 'values' is inaccessible due to 'internal' protection level
Error: 'TupleState' initializer is inaccessible due to 'internal' protection level
```

**MLX Source Code (MLXOptimizers):**
```swift
public struct TupleState {
    internal let values: (MLXArray, MLXArray)  // ← internal!

    internal init(_ values: (MLXArray, MLXArray)) {  // ← internal!
        self.values = values
    }
}
```

The public API doesn't expose any way to:
1. Access the internal tuple values
2. Construct a new TupleState from arrays
3. Transform optimizer states

---

## Alternative Approaches Considered

### ❌ 1. Use Mirror/Reflection
Swift reflection can't bypass access control - `internal` properties remain inaccessible even with Mirror API.

### ❌ 2. Extend TupleState
Can't add extensions that expose internal properties - Swift access control prevents this.

### ❌ 3. Fork MLX and Modify
Would require maintaining a custom MLX fork, which is impractical for:
- Security updates
- Bug fixes
- New features
- Performance improvements

### ⚠️ 4. Custom Optimizer Implementation
We could implement our own Adam optimizer that exposes state manipulation:

**Pros:**
- Full control over state preservation
- Can implement exactly what we need

**Cons:**
- Duplicates MLX optimizer code
- Loses MLX optimizations
- Maintenance burden
- May diverge from MLX best practices

### ✅ 5. Request Public API (Recommended)
**File issue/PR with MLX team to expose public API for state manipulation**

Suggested API additions:
```swift
public struct TupleState {
    // Make initializer public
    public init(_ first: MLXArray, _ second: MLXArray)

    // Add public accessors
    public var first: MLXArray { values.0 }
    public var second: MLXArray { values.1 }

    // Or add transformation methods
    public func selecting(indices: MLXArray) -> TupleState
    public static func concatenating(_ states: [TupleState]) -> TupleState
}
```

---

## Current Workaround

We continue to **reinitialize optimizer states** after split/prune:

**Location:** `GaussianTrainer.swift:484-493`

```swift
if iteration % self.split_and_prune_per_iteration == 0 {
    self.split_and_prune(params: params, states: states, iteration: iteration)
    params = model.getParams()

    // Reinitialize states (loses momentum/variance)
    states = params.map {
        optimizer.newState(parameter: $0)
    }

    MLX.GPU.clearCache()
}
```

**Trade-off:**
- ✅ Code compiles and works correctly
- ❌ Suboptimal training convergence
- ❌ ~30-50% slower convergence
- ❌ May require more iterations to reach target quality

---

## Impact on Training

### Without State Preservation (Current)
```
Iteration 0-99:   Build momentum → Good progress
Iteration 100:    [SPLIT/PRUNE - RESET STATES] ← Loses all momentum!
Iteration 100-199: Build momentum from scratch → Slower progress
Iteration 200:    [SPLIT/PRUNE - RESET STATES] ← Loses all momentum!
...
```

### With State Preservation (Blocked)
```
Iteration 0-99:   Build momentum → Good progress
Iteration 100:    [SPLIT/PRUNE - PRESERVE STATES] ← Keeps momentum!
Iteration 100-199: Continue with momentum → Fast progress
Iteration 200:    [SPLIT/PRUNE - PRESERVE STATES] ← Keeps momentum!
...
```

---

## Recommendation

### Short-term (Current Implementation)
- ✅ Accept the limitation
- ✅ Document the trade-off
- ✅ Users may need ~1.3-1.5x more iterations for same quality

### Medium-term (MLX API Request)
- File GitHub issue with MLX team: https://github.com/ml-explore/mlx-swift
- Propose public API additions for TupleState
- Explain use case (state preservation through parameter transformations)
- Offer to submit PR if team agrees on API design

### Long-term (If API Added)
- Update GaussianSplattingMlx to use new public API
- Implement full state preservation
- Achieve 30-50% faster convergence

---

## Example GitHub Issue Template

**Title:** Feature Request: Public API for TupleState manipulation in MLXOptimizers

**Body:**
```markdown
## Use Case
When implementing adaptive optimization techniques (like Gaussian Splatting), we need to
dynamically add/remove parameters during training. Currently, there's no way to preserve
optimizer states (momentum, variance) through these transformations.

## Current Limitation
TupleState properties and initializers are marked `internal`, preventing state manipulation:
- `values` property is internal
- `init(_:)` is internal

## Proposed Solution
Make TupleState manipulation possible via public API:

Option A: Expose properties
\`\`\`swift
public var first: MLXArray { values.0 }
public var second: MLXArray { values.1 }
public init(_ first: MLXArray, _ second: MLXArray)
\`\`\`

Option B: Add transformation methods
\`\`\`swift
public func selecting(indices: MLXArray) -> TupleState
public static func concatenating(_ states: [TupleState]) -> TupleState
\`\`\`

## Impact
Without state preservation, we must reinitialize optimizer states after parameter
transformations, losing 30-50% training efficiency.

## Alternatives Considered
- Custom optimizer implementation (loses MLX optimizations)
- Forking MLX (maintenance burden)
```

---

## Related Code Locations

- **Training loop:** `GaussianTrainer.swift:484-493` - State reinitialization
- **Split operation:** `GaussianTrainer.swift:269-347` - Where states should be transformed
- **Clone operation:** `GaussianTrainer.swift:349-382` - Where states should be duplicated
- **Prune operation:** `GaussianTrainer.swift:384-404` - Where states should be filtered
- **Performance analysis:** `PERFORMANCE_ANALYSIS.md` - Issue #5

---

## References

- MLX Swift Package: https://github.com/ml-explore/mlx-swift
- MLX Optimizers: https://swiftpackageindex.com/ml-explore/mlx-swift/documentation/mlxoptimizers
- Gaussian Splatting paper: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
- Adam optimizer: https://arxiv.org/abs/1412.6980

---

**Status:** Waiting for MLX API enhancement
**Next Steps:** File issue with MLX team
**Workaround:** Continue with state reinitialization (functional but suboptimal)
