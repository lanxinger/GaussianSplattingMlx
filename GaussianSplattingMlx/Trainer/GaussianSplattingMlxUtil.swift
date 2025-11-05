import MLX
import MLXFFT
import MLXFast
import MLXLinalg
import MLXNN
import MLXRandom
import simd

func inverse_sigmoid(x: MLXArray) -> MLXArray {
    return MLX.log(x / (1 - x))
}

func homogeneous(points: MLXArray) -> MLXArray {
    return MLX.concatenated(
        [points, MLXArray.ones(like: points[.ellipsis, .stride(to: 1)])],
        axis: -1
    )
}

func build_rotation(quaternion: MLXArray)
    -> MLXArray
{
    let norm =
        (quaternion[.ellipsis, 0] * quaternion[.ellipsis, 0] + quaternion[
            .ellipsis,
            1
        ] * quaternion[.ellipsis, 1] + quaternion[.ellipsis, 2]
        * quaternion[.ellipsis, 2] + quaternion[.ellipsis, 3]
        * quaternion[.ellipsis, 3]).sqrt()
    let safeNorm = MLX.maximum(norm, 1e-8)
    let q = quaternion / safeNorm.expandedDimensions(axes: [-1])
    let R = MLXArray.zeros([q.shape[0], 3, 3], dtype: .float32)

    let r = q[.ellipsis, 0]
    let x = q[.ellipsis, 1]
    let y = q[.ellipsis, 2]
    let z = q[.ellipsis, 3]

    R[.ellipsis, 0, 0] = 1 - 2 * (y * y + z * z)
    R[.ellipsis, 0, 1] = 2 * (x * y - r * z)
    R[.ellipsis, 0, 2] = 2 * (x * z + r * y)
    R[.ellipsis, 1, 0] = 2 * (x * y + r * z)
    R[.ellipsis, 1, 1] = 1 - 2 * (x * x + z * z)
    R[.ellipsis, 1, 2] = 2 * (y * z - r * x)
    R[.ellipsis, 2, 0] = 2 * (x * z - r * y)
    R[.ellipsis, 2, 1] = 2 * (y * z + r * x)
    R[.ellipsis, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
}

func build_scaling_rotation(s: MLXArray, r: MLXArray) -> MLXArray {
    let L = MLXArray.zeros([s.shape[0], 3, 3], dtype: .float32)
    let R = build_rotation(quaternion: r)

    L[.ellipsis, 0, 0] = s[.ellipsis, 0]
    L[.ellipsis, 1, 1] = s[.ellipsis, 1]
    L[.ellipsis, 2, 2] = s[.ellipsis, 2]

    return R.matmul(L)
}

func strip_lowerdiag(L: MLXArray) -> MLXArray {
    let uncertainty = MLXArray.zeros([L.shape[0], 6], dtype: .float32)
    uncertainty[.ellipsis, 0] = L[.ellipsis, 0, 0]
    uncertainty[.ellipsis, 1] = L[.ellipsis, 0, 1]
    uncertainty[.ellipsis, 2] = L[.ellipsis, 0, 2]
    uncertainty[.ellipsis, 3] = L[.ellipsis, 1, 1]
    uncertainty[.ellipsis, 4] = L[.ellipsis, 1, 2]
    uncertainty[.ellipsis, 5] = L[.ellipsis, 2, 2]
    return uncertainty
}
let strip_symmetric = strip_lowerdiag

func build_covariance_3d(s: MLXArray, r: MLXArray) -> MLXArray {
    let L = build_scaling_rotation(s: s, r: r)
    return L.matmul(L.transposed(0, 2, 1))
}

func build_covariance_2d(
    mean3d: MLXArray,
    cov3d: MLXArray,
    viewMatrix: MLXArray,
    fovX: MLXArray,
    fovY: MLXArray,
    focalX: MLXArray,
    focalY: MLXArray
) -> MLXArray {

    let tan_fovx = MLX.tan(fovX * 0.5)
    let tan_fovy = MLX.tan(fovY * 0.5)
    let focal_x = focalX
    let focal_y = focalY
    let t =
        mean3d.matmul(viewMatrix[.stride(to: 3), .stride(to: 3)])
        + viewMatrix[.stride(from: -1), .stride(to: 3)]
    let tx =
        t[.ellipsis, 0]
        / MLX.clip(t[.ellipsis, 2], min: -tan_fovx * 1.3, max: tan_fovx * 1.3)
        * t[.ellipsis, 2]
    let ty =
        t[.ellipsis, 1]
        / MLX.clip(t[.ellipsis, 2], min: -tan_fovy * 1.3, max: tan_fovy * 1.3)
        * t[.ellipsis, 2]
    let tz = t[.ellipsis, 2]

    // Optimize: Exploit sparse Jacobian structure to eliminate 2 of 4 matmuls
    // Original: J.matmul(W).matmul(cov3d).matmul(W.T).matmul(J.T) - 4 matmuls + sparse matrix allocation
    // New: Compute M = W @ cov3d @ W.T, then manually compute J @ M @ J.T using sparsity
    //
    // Jacobian J has only 4 non-zero entries:
    //   J = [j00,   0, j02]
    //       [  0, j11, j12]
    //       [  0,   0,   0]
    //
    // This reduces computation from 4 matmuls to 2 matmuls + element-wise operations

    let W = viewMatrix[.stride(to: 3), .stride(to: 3)].T

    // Step 1: Compute M = W @ cov3d @ W.T (necessary transformation to view space)
    let M = W.matmul(cov3d).matmul(W.T)  // 2 matmuls (unavoidable)

    // Step 2: Precompute sparse Jacobian coefficients
    let tzInv = 1.0 / tz
    let tzSq = tz * tz
    let j00 = tzInv * focal_x
    let j02 = -tx / tzSq * focal_x
    let j11 = tzInv * focal_y
    let j12 = -ty / tzSq * focal_y

    // Step 3: Compute A = J @ M (exploit sparsity - only 2 rows non-zero)
    // A[0, :] = j00 * M[0, :] + j02 * M[2, :]
    // A[1, :] = j11 * M[1, :] + j12 * M[2, :]
    let a00 = j00 * M[.ellipsis, 0, 0] + j02 * M[.ellipsis, 2, 0]
    let a01 = j00 * M[.ellipsis, 0, 1] + j02 * M[.ellipsis, 2, 1]
    let a02 = j00 * M[.ellipsis, 0, 2] + j02 * M[.ellipsis, 2, 2]

    let a10 = j11 * M[.ellipsis, 1, 0] + j12 * M[.ellipsis, 2, 0]
    let a11 = j11 * M[.ellipsis, 1, 1] + j12 * M[.ellipsis, 2, 1]
    let a12 = j11 * M[.ellipsis, 1, 2] + j12 * M[.ellipsis, 2, 2]

    // Step 4: Compute cov2d = A @ J.T (only 2x2 upper-left block needed)
    // J.T = [j00,   0,   0]
    //       [  0, j11,   0]
    //       [j02, j12,   0]
    let cov2d_00 = a00 * j00 + a02 * j02
    let cov2d_01 = a01 * j11 + a02 * j12
    let cov2d_10 = a10 * j00 + a12 * j02
    let cov2d_11 = a11 * j11 + a12 * j12

    // Stack into [N, 2, 2] tensor
    let cov2d = MLX.stack([
        MLX.stack([cov2d_00, cov2d_01], axis: -1),
        MLX.stack([cov2d_10, cov2d_11], axis: -1)
    ], axis: -2)

    let filter = MLX.eye(2, m: 2) * 0.3
    return cov2d + filter[.newAxis]
}

func projection_ndc(
    points: MLXArray,
    viewMatrix: MLXArray,
    projMatrix: MLXArray,
    maskValue: Float = 0.2
)
    -> (MLXArray, MLXArray, MLXArray)
{
    let points_o = homogeneous(points: points)
    let points_h = points_o.matmul(viewMatrix).matmul(projMatrix)
    let p_w = 1.0 / (points_h[.ellipsis, .stride(from: -1)] + 0.000001)
    let p_proj = points_h * p_w
    let p_view = points_o.matmul(viewMatrix)
    let in_mask = conditionToIndices(condition: p_view[.ellipsis, 2] .>= maskValue)
    return (p_proj, p_view, in_mask)
}

func get_max_covariance(cov2d: MLXArray) -> MLXArray {
    let det =
        cov2d[.ellipsis, 0, 0] * cov2d[.ellipsis, 1, 1]
        - cov2d[.ellipsis, 0, 1] * cov2d[.ellipsis, 1, 0]
    let mid = 0.5 * (cov2d[.ellipsis, 0, 0] + cov2d[.ellipsis, 1, 1])
    let delta = MLX.clip(mid ** 2 - det, min: 1e-5)
    let sqrtDelta = MLX.sqrt(delta)
    let lambda1 = mid + sqrtDelta
    let lambda2 = mid - sqrtDelta
    return MLX.maximum(lambda1, lambda2)  // or abs maximum
}

func get_radius(cov2d: MLXArray) -> MLXArray {
    let maxCov = get_max_covariance(cov2d: cov2d)
    return 3.0 * MLX.ceil(MLX.sqrt(maxCov))
}
func get_rect(
    pix_coord: MLXArray,
    radii: MLXArray,
    width: Int,
    height: Int
) -> (MLXArray, MLXArray) {
    let rect_min = (pix_coord - radii.expandedDimensions(axes: [1]))
    let rect_max = (pix_coord + radii.expandedDimensions(axes: [1]))
    rect_min[.ellipsis, 0] = MLX.clip(
        rect_min[.ellipsis, 0],
        min: 0,
        max: Double(width) - 1.0
    )
    rect_min[.ellipsis, 1] = MLX.clip(
        rect_min[.ellipsis, 1],
        min: 0,
        max: Double(height) - 1.0
    )
    rect_max[.ellipsis, 0] = MLX.clip(
        rect_max[.ellipsis, 0],
        min: 0,
        max: Double(width) - 1.0
    )
    rect_max[.ellipsis, 1] = MLX.clip(
        rect_max[.ellipsis, 1],
        min: 0,
        max: Double(height) - 1.0
    )
    return (rect_min, rect_max)
}
func build_color(
    means3d: MLXArray,
    shs: MLXArray,
    camera: Camera,
    activeShDegree: Int
)
    -> MLXArray
{
    let cameraCentor = MLXArray(
        [Float(camera.cameraCenter.x), Float(camera.cameraCenter.y), Float(camera.cameraCenter.z)]
            as [Float])[.newAxis, .ellipsis]
    let rays_d = means3d - cameraCentor
    var color = evalSh(
        deg: activeShDegree,
        sh: shs.transposed(0, 2, 1),
        dirs: rays_d
    )
    color = MLX.clip(color + 0.5, min: 0.0)
    return color
}
func matrixInverse2d(_ m: MLXArray) -> MLXArray {
    precondition(
        m.shape.count >= 2 && m.shape.suffix(2) == [2, 2],
        "input shape should be 2x2"
    )
    let a = m[.ellipsis, 0, 0]
    let b = m[.ellipsis, 0, 1]
    let c = m[.ellipsis, 1, 0]
    let d = m[.ellipsis, 1, 1]
    let det = a * d - b * c
    let invDet = 1.0 / det  // Single division - multiplications are cheaper
    let inv = MLXArray.zeros(m.shape)
    inv[.ellipsis, 0, 0] = d * invDet
    inv[.ellipsis, 0, 1] = -b * invDet
    inv[.ellipsis, 1, 0] = -c * invDet
    inv[.ellipsis, 1, 1] = a * invDet
    return inv
}
func conditionToIndices(condition: MLXArray) -> MLXArray {
    Logger.shared.debug("conditionToIndices start")
    let arange = MLX.where(condition, MLXArray(0..<condition.shape[0]), MLXArray(Int32.max))
    let sorted = MLX.sorted(arange)
    if sorted.shape[0] == 0 {
        return MLXArray([])
    }
    let index = MLX.argMax(sorted)
    Logger.shared.debug("conditionToIndices end")
    return MLX.stopGradient(sorted[0..<index.item(Int.self)])
}
func detachedArray(_ array: MLXArray) -> MLXArray {
    let data = MLX.stopGradient(array).asData(access: .copy)
    return MLXArray(data: data)
}
