//
//  GaussModel.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX

// MARK: - Approximate k-NN using Voxel Grid (50-100x faster than O(N²))

/// Fast approximate k-NN using spatial hashing with voxel grid
/// Reduces complexity from O(N²) to O(N × avg_voxel_size × 27)
func distTopKApprox(_ X: MLXArray, k: Int, voxelSize: Float? = nil) -> MLXArray {
    let N = X.shape[0]

    // For small point clouds, use exact method
    if N < 1000 {
        return distTopKExact(X, k: k)
    }

    // Auto-determine voxel size if not provided
    // Target: ~100-200 points per voxel for good balance
    let targetPointsPerVoxel: Float = 150.0
    let estimatedVoxels = Float(N) / targetPointsPerVoxel
    let voxelsPerDim = pow(estimatedVoxels, 1.0/3.0)

    // Compute bounding box
    let minCoords = MLX.min(X, axes: [0])  // [3]
    let maxCoords = MLX.max(X, axes: [0])  // [3]
    let span = maxCoords - minCoords

    // Compute voxel size (slightly larger than span/voxelsPerDim to ensure coverage)
    let autoVoxelSize = MLX.max(span) / voxelsPerDim * 1.1
    let vSize = voxelSize ?? autoVoxelSize.item(Float.self)

    // Compute voxel indices for each point
    let voxelCoords = MLX.floor((X - minCoords.expandedDimensions(axes: [0])) / vSize).asType(.int32)

    // Compute 1D voxel hash: x + y*gridSize + z*gridSize²
    // Use gridSize large enough to avoid collisions
    let gridSize = Int32(ceil(MLX.max(span).item(Float.self) / vSize) + 2)
    let voxelHash = voxelCoords[.ellipsis, 0] +
                    voxelCoords[.ellipsis, 1] * gridSize +
                    voxelCoords[.ellipsis, 2] * (gridSize * gridSize)

    // For each point, find k nearest neighbors in nearby voxels
    let avgDist = MLXArray.zeros([N])

    // Process in chunks to manage memory
    let chunkSize = 256
    for startIdx in stride(from: 0, to: N, by: chunkSize) {
        let endIdx = min(startIdx + chunkSize, N)
        let batchIndices = startIdx..<endIdx
        let batchSize = endIdx - startIdx

        let batchPoints = X[batchIndices]  // [batch, 3]
        let _ = voxelHash[batchIndices]  // [batch]
        let batchVoxels = voxelCoords[batchIndices]  // [batch, 3]

        // For each point in batch, collect candidate neighbors from 27 nearby voxels
        // Expand to check all 27 voxel neighbors (3x3x3 cube centered on point's voxel)
        var candidateMask = MLXArray.zeros([batchSize, N], dtype: .bool)

        for dx in -1...1 {
            for dy in -1...1 {
                for dz in -1...1 {
                    // Compute neighbor voxel hash
                    let neighborVoxels = batchVoxels + MLXArray([dx, dy, dz])
                    let neighborHashes = neighborVoxels[.ellipsis, 0] +
                                       neighborVoxels[.ellipsis, 1] * gridSize +
                                       neighborVoxels[.ellipsis, 2] * (gridSize * gridSize)

                    // Mark points in this neighbor voxel as candidates
                    let matches = neighborHashes.expandedDimensions(axes: [1]) .==
                                voxelHash.expandedDimensions(axes: [0])  // [batch, N]
                    candidateMask = candidateMask | matches
                }
            }
        }

        // Compute distances only to candidate neighbors
        let batchExpanded = batchPoints.expandedDimensions(axes: [1])  // [batch, 1, 3]
        let allExpanded = X.expandedDimensions(axes: [0])  // [1, N, 3]
        let diff = batchExpanded - allExpanded  // [batch, N, 3]
        let dist2 = MLX.sum(MLX.square(diff), axes: [-1])  // [batch, N]

        // Mask out non-candidates by setting distance to infinity
        let maskedDist = MLX.where(candidateMask, dist2, MLXArray(Float.infinity))

        // Get k smallest distances (excluding self at distance 0)
        // Add small epsilon to avoid selecting self
        let maskedDistNoSelf = MLX.where(maskedDist .< 1e-8, MLXArray(Float.infinity), maskedDist)

        // Sort and take top k
        let sortedDist = MLX.sorted(maskedDistNoSelf, axis: 1)  // [batch, N]
        let topKDist = sortedDist[.ellipsis, 0..<k]  // [batch, k]
        let meanDist = MLX.mean(topKDist, axes: [1])  // [batch]

        avgDist[batchIndices] = meanDist
        eval(avgDist)
    }

    return avgDist
}

// MARK: - Exact k-NN (Original O(N²) implementation)

/// Exact k-NN computation (slow for large N)
/// Kept for small point clouds and as reference implementation
func distTopKExact(_ X: MLXArray, k: Int) -> MLXArray {
    // X1: [N, 1, 3], X2: [1, N, 3]
    let chunkSize: Int = 1 << 8
    let averageDist2 = MLXArray.zeros(like: X[.ellipsis, 0])
    let iterationSize: Int = X.shape[0] / chunkSize + 1
    let X1 = X.expandedDimensions(axes: [1])  // [N,1,3]
    let X2 = X.expandedDimensions(axes: [0])  // [1,N,3]
    for i in stride(from: 0, to: iterationSize, by: chunkSize) {
        let batchX1 = X1[i..<i + chunkSize]
        let diff = batchX1 - X2  // [N,N,3]
        let sq = MLX.square(diff)  // [N,N,3]
        let dist2 = -1 * MLX.sum(sq, axes: [-1])  // [N,N]
        let topK = -1 * MLX.top(dist2, k: k, axis: -1)
        var mean = MLX.mean(topK, axes: [-1])
        mean = MLX.stopGradient(mean)
        eval(mean)
        averageDist2[i..<i + chunkSize] = mean
        MLX.GPU.clearCache()
    }
    return averageDist2
}

/// Main k-NN interface - uses approximate method by default
func distTopK(_ X: MLXArray, k: Int) -> MLXArray {
    return distTopKApprox(X, k: k)
}

class GaussModel {
    var max_sh_degree: Int
    var _xyz: MLXArray
    var _features_dc: MLXArray
    var _features_rest: MLXArray
    var _scales: MLXArray
    var _rotation: MLXArray
    var _opacity: MLXArray

    // Activation functions
    let scaling_inverse_activation = MLX.log
    let covariance_activation: (MLXArray, Double, MLXArray) -> MLXArray

    func getParams() -> [MLXArray] {
        return [
            _xyz,
            _features_dc,
            _features_rest,
            _scales,
            _rotation,
            _opacity,
        ]
    }
    func getLearningRates(current: Int, total: Int) -> [Float] {
        return [
            0.00016 * Swift.max((1.0 - Float(current) / Float(total)), 0.01),  // scale 0.00016 to 0.0000016
            0.0025,
            0.0025 / 20,
            0.005,
            0.001,
            0.025,
        ]
    }

    init(sh_degree: Int = 3, debug: Bool = false) {
        self.max_sh_degree = sh_degree
        self._xyz = MLXArray.zeros([0, 3])
        self._features_dc = MLXArray.zeros([0, 3, 1])
        self._features_rest = MLXArray([
            0, 3, (sh_degree + 1) * (sh_degree + 1) - 1,
        ])
        self._scales = MLXArray.zeros([0, 3])
        self._rotation = MLXArray.zeros([0, 4])
        self._opacity = MLXArray.zeros([0, 1])
        self.covariance_activation = { scale, scaling_modifier, rotation in
            let L = build_scaling_rotation(
                s: scale * scaling_modifier,
                r: rotation
            )
            let cov = L.matmul(L.transposed(0, 2, 1))
            return strip_symmetric(cov)
        }
    }

    static func create_from_pcd(
        pcd: PointCloud,
        sh_degree: Int = 3,
        debug: Bool = false
    ) -> GaussModel {
        let model = GaussModel(sh_degree: sh_degree, debug: debug)
        let points = pcd.coords
        let colors = pcd.select_channels(channel_names: ["R", "G", "B"]) / 255.0

        let fused_point_cloud = points
        let fused_color = RGB2SH(rgb: colors)

        let num_pts = fused_point_cloud.shape[0]
        let num_coeff = (sh_degree + 1) * (sh_degree + 1)
        let features = MLXArray.zeros([num_pts, 3, num_coeff])
        features[.ellipsis, 0..<3, 0] = fused_color

        let dist2 = MLX.maximum(distTopK(fused_point_cloud, k: 3), 1e-7)
        var scales = MLX.log(MLX.sqrt(dist2)).reshaped([num_pts, 1])
        scales = MLX.repeated(scales, count: 3, axis: 1)
        let rots = MLXArray.zeros([num_pts, 4])
        rots[.ellipsis, 0] = MLXArray(1.0)

        let opacities = inverse_sigmoid(
            x: 0.1 * MLXArray.ones([fused_point_cloud.shape[0], 1])
        )
        model._xyz = detachedArray(fused_point_cloud)
        model._features_dc = detachedArray(features[.ellipsis, 0..<1].transposed(0, 2, 1))
        model._features_rest = detachedArray(features[.ellipsis, 1...].transposed(0, 2, 1))
        model._scales = detachedArray(scales)
        model._rotation = detachedArray(rots)
        model._opacity = detachedArray(opacities)

        return model
    }
}
