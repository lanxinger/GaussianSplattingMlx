//
//  GaussianTrainer.swift
//  GaussianSplattingMlx
//
//  Created by Tatsuya Ogawa on 2025/06/01.
//

import Foundation
import MLX
import MLXOptimizers
import MLXRandom

class TrainData {
    let Hs: MLXArray
    let Ws: MLXArray
    let intrinsicArray: MLXArray
    var c2wArray: MLXArray
    let rgbArray: MLXArray
    let alphaArray: MLXArray
    let depthArray: MLXArray?
    init(
        Hs: MLXArray,
        Ws: MLXArray,
        intrinsicArray: MLXArray,
        c2wArray: MLXArray,
        rgbArray: MLXArray,
        alphaArray: MLXArray,
        depthArray: MLXArray?
    ) {
        self.Hs = Hs
        self.Ws = Ws
        self.intrinsicArray = intrinsicArray
        self.c2wArray = c2wArray
        self.rgbArray = rgbArray
        self.alphaArray = alphaArray
        self.depthArray = depthArray
    }
    func getViewPointCamera(index: Int) -> Camera {
        return Camera(
            width: Ws[index].item(Int.self),
            height: Hs[index].item(Int.self),
            intrinsic: intrinsicArray[index],
            c2w: c2wArray[index]
        )
    }
    func getNumCameras() -> Int {
        return self.Hs.shape[0]
    }

    func getCameraParams() -> (
        Hs: MLXArray, Ws: MLXArray, intrinsics: MLXArray, c2w: MLXArray
    ) {
        return (Hs, Ws, intrinsicArray, c2wArray)
    }
}
protocol GaussianTrainerDelegate: AnyObject {
    func pushLoss(loss: Float, iteration: Int?, timestamp: Date)
    func pushImageData(
        render: MLXArray, truth: MLXArray, loss: Float, iteration: Int, timestamp: Date)
    func pushSnapshot(url: URL,iteration: Int, timestamp: Date)
}

// MARK: - Optimizer State Transformation Helpers
// TODO: TupleState API from MLXOptimizers doesn't expose internal structure
// Need to find proper API for state manipulation or request it from MLX team
// For now, we'll reinitialize states after split/prune (original behavior)
// This is suboptimal but safe until we have proper API access

class GaussianTrainer {
    var data: TrainData
    var gaussRender: GaussianRenderer
    var model: GaussModel
    var lambda_dssim: Float = 0.2
    var lambda_depth: Float = 0.0
    var split_and_prune_per_iteration: Int = 100
    var save_snapshot_per_iteration: Int
    var iterationCount: Int
    var outputDirectoryURL: URL?
    var cacheLimit: Int
    weak var delegate: GaussianTrainerDelegate?
    
    // Split and prune parameters
    var gradientThreshold: Float = 0.0002
    var maxScreenSize: Float = 20.0
    var minOpacity: Float = 0.005
    var maxScale: Float = 0.01
    var pruneInterval: Int = 100
    var densifyFromIter: Int = 500
    var densifyUntilIter: Int = 15000

    // SH coefficient optimization (inspired by LichtFeld-Studio 2.4x speedup)
    // Skip higher-degree SH coefficient updates early in training when they have minimal impact
    var shHigherDegreeStartIter: Int = 1000

    // Multi-view consistent densification (inspired by FastGS 100-second training)
    // Use multi-view consistency for more robust densification decisions
    var useMultiViewDensification: Bool = true
    var multiViewSampleCount: Int = 4  // Number of views to sample for consistency check

    // Progressive resolution scheduling (inspired by DashGaussian CVPR 2025)
    // Start training at low resolution and gradually increase to full resolution
    // TEMPORARILY DISABLED FOR DEBUGGING - Set to false to use full resolution
    var useProgressiveResolution: Bool = false  // Was: true
    var minResolutionScale: Float = 0.25  // Start at 25% of original resolution
    var maxResolutionScale: Float = 1.0   // End at 100% of original resolution
    var resolutionRampIterations: Int = 3000  // Ramp up resolution over first 3000 iterations

    // Sparse Adam optimizer (inspired by DashGaussian/LichtFeld-Studio)
    // Skip optimizer updates for parameters with near-zero gradients
    var useSparseAdam: Bool = true
    var sparseGradientThreshold: Float = 1e-8  // Threshold for considering gradient as "zero"

    // Tracking gradients for densification
    var xyzGradAccumulation: MLXArray = MLXArray.zeros([0, 3])
    var denomGradAccumulation: MLXArray = MLXArray.zeros([0])
    init(
        model: GaussModel,
        data: TrainData,
        gaussRender: GaussianRenderer,
        iterationCount: Int,
        cacheLimit: Int = 2 * 1024 * 1024 * 1024,
        outputDirectoryURL: URL? = nil,
        saveSnapshotPerIteration: Int = 100
    ) {
        self.model = model
        self.data = data
        self.gaussRender = gaussRender
        self.iterationCount = iterationCount
        self.cacheLimit = cacheLimit
        self.outputDirectoryURL = outputDirectoryURL
        self.save_snapshot_per_iteration = saveSnapshotPerIteration
        
        // Initialize gradient accumulation arrays
        let numPoints = model._xyz.shape[0]
        self.xyzGradAccumulation = MLXArray.zeros([numPoints])
        self.denomGradAccumulation = MLXArray.zeros([numPoints])
    }
    func fetchTrainData() -> (
        camera: Camera, rgb: MLXArray, mask: MLXArray, depth: MLXArray?
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)
        let rgb = data.rgbArray[ind]
        let depth = data.depthArray?[ind]
        let mask = conditionToIndices(
            condition: (data.alphaArray[ind] .> 0.5).reshaped([-1])
        )
        let camera = data.getViewPointCamera(index: ind)
        return (camera, rgb, mask, depth)
    }
    func addGradientAccumulation(xyzGrad: MLXArray) {
        let gradNorm = MLX.sum(MLX.square(xyzGrad), axes: [1])
        let numPoints = xyzGrad.shape[0]
        
        if xyzGradAccumulation.shape[0] != numPoints {
            xyzGradAccumulation = MLXArray.zeros([numPoints])
            denomGradAccumulation = MLXArray.zeros([numPoints])
        }
        
        xyzGradAccumulation = xyzGradAccumulation + gradNorm
        denomGradAccumulation = denomGradAccumulation + MLXArray.ones([numPoints])
    }
    
    func resetGradientAccumulation() {
        let numPoints = model._xyz.shape[0]
        xyzGradAccumulation = MLXArray.zeros([numPoints])
        denomGradAccumulation = MLXArray.zeros([numPoints])
    }

    // Progressive resolution scheduling functions
    func getCurrentResolutionScale(iteration: Int) -> Float {
        if !useProgressiveResolution {
            return maxResolutionScale  // Always use full resolution
        }

        if iteration >= resolutionRampIterations {
            return maxResolutionScale  // Full resolution after ramp period
        }

        // Smooth linear ramp from minResolutionScale to maxResolutionScale
        let progress = Float(iteration) / Float(resolutionRampIterations)
        let scale = minResolutionScale + (maxResolutionScale - minResolutionScale) * progress

        return scale
    }

    // Resize an MLXArray image using nearest-neighbor interpolation
    func resizeImage(_ image: MLXArray, targetH: Int, targetW: Int) -> MLXArray {
        let originalH = image.shape[0]
        let originalW = image.shape[1]

        if originalH == targetH && originalW == targetW {
            return image  // No resize needed
        }

        // Handle both 2D (H, W) and 3D (H, W, C) arrays
        let has3Channels = image.shape.count == 3
        let channels = has3Channels ? image.shape[2] : 1

        // Create coordinate grids for target resolution
        let scaleH = Float(originalH) / Float(targetH)
        let scaleW = Float(originalW) / Float(targetW)

        // Generate target pixel coordinates
        var resizedImage: MLXArray
        if has3Channels {
            resizedImage = MLXArray.zeros([targetH, targetW, channels])
        } else {
            resizedImage = MLXArray.zeros([targetH, targetW])
        }

        // Simple nearest-neighbor resize for efficiency
        // For each target pixel, find the nearest source pixel
        for h in 0..<targetH {
            for w in 0..<targetW {
                let srcH = min(Int(Float(h) * scaleH), originalH - 1)
                let srcW = min(Int(Float(w) * scaleW), originalW - 1)
                resizedImage[h, w] = image[srcH, srcW]
            }
        }

        return resizedImage
    }

    // Modified fetchTrainData with progressive resolution support
    func fetchTrainDataWithResolution(scale: Float) -> (
        camera: Camera, rgb: MLXArray, mask: MLXArray, depth: MLXArray?
    ) {
        let numCameras = data.getNumCameras()
        let ind = Int.random(in: 0..<numCameras)

        // Get original data
        let originalRGB = data.rgbArray[ind]
        let originalH = originalRGB.shape[0]
        let originalW = originalRGB.shape[1]

        // Calculate scaled dimensions
        let scaledH = max(Int(Float(originalH) * scale), 16)  // Minimum 16 pixels
        let scaledW = max(Int(Float(originalW) * scale), 16)

        // Resize if needed
        let rgb: MLXArray
        let camera: Camera
        let mask: MLXArray
        let depth: MLXArray?

        if scale < 0.99 {  // Only resize if significantly different
            // Resize RGB image
            rgb = resizeImage(originalRGB, targetH: scaledH, targetW: scaledW)

            // Resize alpha array for mask computation
            let originalAlpha = data.alphaArray[ind]
            let resizedAlpha = resizeImage(originalAlpha, targetH: scaledH, targetW: scaledW)
            mask = conditionToIndices(
                condition: (resizedAlpha .> 0.5).reshaped([-1])
            )

            // Resize depth if available
            if let originalDepth = data.depthArray?[ind] {
                depth = resizeImage(originalDepth, targetH: scaledH, targetW: scaledW)
            } else {
                depth = nil
            }

            // Create scaled camera with adjusted intrinsics
            let originalIntrinsic = data.intrinsicArray[ind]
            let scaledIntrinsic = originalIntrinsic * Float(scale)

            // Last row/column should remain [0, 0, 1]
            scaledIntrinsic[2, 0] = originalIntrinsic[2, 0]
            scaledIntrinsic[2, 1] = originalIntrinsic[2, 1]
            scaledIntrinsic[2, 2] = originalIntrinsic[2, 2]
            scaledIntrinsic[0, 2] = scaledIntrinsic[0, 2] / Float(scale)  // Principal point
            scaledIntrinsic[1, 2] = scaledIntrinsic[1, 2] / Float(scale)

            camera = Camera(
                width: scaledW,
                height: scaledH,
                intrinsic: scaledIntrinsic,
                c2w: data.c2wArray[ind]
            )
        } else {
            // Use full resolution
            rgb = originalRGB
            camera = data.getViewPointCamera(index: ind)

            // Compute mask at full resolution
            mask = conditionToIndices(
                condition: (data.alphaArray[ind] .> 0.5).reshaped([-1])
            )
            depth = data.depthArray?[ind]
        }

        return (camera, rgb, mask, depth)
    }

    // Multi-view consistent densification score computation (FastGS approach)
    // Samples multiple views and computes per-Gaussian scores based on gradient consistency
    func computeMultiViewScores(iteration: Int) -> MLXArray {
        var gradientSamples: [MLXArray] = []

        // Compute resolution scale for this iteration
        let resolutionScale = getCurrentResolutionScale(iteration: iteration)

        // Sample multiple views and compute gradients for each
        // Use current resolution scale for consistency with main training loop
        for _ in 0..<multiViewSampleCount {
            let (trainCamera, trainRGB, _, _) = fetchTrainDataWithResolution(scale: resolutionScale)

            // Compute loss and gradients for this view
            let params = model.getParams()
            let computeLoss: ([MLXArray]) -> MLXArray = { params in
                let _xyz = params[0]
                let _features_dc = params[1]
                let _features_rest = params[2]
                let _scales = params[3]
                let _rotation = params[4]
                let _opacity = params[5]

                let means3d = self.gaussRender.get_xyz_from(_xyz)
                let opacity = self.gaussRender.get_opacity_from(_opacity)
                let scales = self.gaussRender.get_scales_from(_scales)
                let rotations = self.gaussRender.get_rotation_from(_rotation)
                let shs = self.gaussRender.get_features_from(_features_dc, _features_rest)

                let (render, _, _, _, _) = self.gaussRender.forward(
                    camera: trainCamera,
                    means3d: means3d,
                    shs: shs,
                    opacity: opacity,
                    scales: scales,
                    rotations: rotations
                )

                // Compute L1 loss
                let l1_loss = l1Loss(render, trainRGB)
                return l1_loss
            }

            // Get gradient for xyz parameter (index 0)
            let grads = MLX.grad(computeLoss, argumentNumbers: [0])(params)
            let xyzGrad = grads[0]

            // Compute per-Gaussian gradient magnitude
            let gradNorm = MLX.sqrt(MLX.sum(MLX.square(xyzGrad), axes: [1]))
            gradientSamples.append(gradNorm)
        }

        // Stack gradients: [multiViewSampleCount, numPoints]
        let stackedGrads = MLX.stacked(gradientSamples, axis: 0)

        // Compute mean gradient magnitude across views
        let meanGrad = MLX.mean(stackedGrads, axes: [0])  // [numPoints]

        // Compute variance of gradients across views (consistency measure)
        let variance = MLX.variance(stackedGrads, axes: [0])  // [numPoints]

        // Multi-view score: high mean gradient + low variance = consistent high error
        // Normalize variance to [0, 1] range and invert (high consistency = low variance)
        let maxVariance = MLX.max(variance)
        let consistencyScore = 1.0 - (variance / (maxVariance + 1e-7))

        // Combined score: mean gradient weighted by consistency
        // Gaussians with consistent high gradients across views get high scores
        let multiViewScore = meanGrad * consistencyScore

        return multiViewScore
    }
    
    func split_and_prune(params: [MLXArray], states: [TupleState], iteration: Int) {
        guard iteration >= densifyFromIter && iteration <= densifyUntilIter else {
            return
        }

        var _xyz = params[0]
        var _features_dc = params[1]
        var _features_rest = params[2]
        var _scales = params[3]
        var _rotation = params[4]
        var _opacity = params[5]

        // Compute densification scores: either multi-view or gradient-based
        let densificationScore: MLXArray
        if useMultiViewDensification {
            // FastGS approach: Multi-view consistent densification
            Logger.shared.debug("Computing multi-view scores for densification")
            densificationScore = computeMultiViewScores(iteration: iteration)
            Logger.shared.debug("Multi-view scores computed")
        } else {
            // Original approach: Gradient-based densification
            let avgGrads = xyzGradAccumulation / denomGradAccumulation.expandedDimensions(axes: [1])
            densificationScore = MLX.sqrt(avgGrads)
        }

        // Get current scales and opacity
        let scales = MLX.exp(_scales)
        let opacity = MLX.sigmoid(_opacity)

        // Find points to densify (high score)
        // For multi-view scores, use percentile-based threshold instead of fixed threshold
        let densifyThreshold: MLXArray
        if useMultiViewDensification {
            // Use 80th percentile as threshold for multi-view scores
            // MLX doesn't have percentile, so compute manually via sorting
            let sorted = MLX.sorted(densificationScore)
            let percentileIdx = Int(Float(sorted.shape[0]) * 0.8)
            densifyThreshold = sorted[percentileIdx]
        } else {
            densifyThreshold = MLXArray(gradientThreshold)
        }
        let gradMask = densificationScore .> densifyThreshold

        // Find points to split (large scale)
        let maxScalePerGaussian = MLX.max(scales, axes: [1])
        let splitMask = gradMask & (maxScalePerGaussian .> MLXArray(maxScale))

        // Find points to clone (small scale)
        let cloneMask = gradMask & (maxScalePerGaussian .<= MLXArray(maxScale))

        // Find points to prune (low opacity)
        let pruneMask = (opacity .< minOpacity).reshaped([-1])

        // Perform splitting
        if MLX.sum(splitMask.asType(.int32)).item(Int.self) > 0 {
            let splitIndices = conditionToIndices(condition: splitMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) =
                splitGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: splitIndices
                )
        }

        // Perform cloning
        if MLX.sum(cloneMask.asType(.int32)).item(Int.self) > 0 {
            let cloneIndices = conditionToIndices(condition: cloneMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) =
                cloneGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: cloneIndices
                )
        }

        // Perform pruning
        if MLX.sum(pruneMask.asType(.int32)).item(Int.self) > 0 {
            let keepMask: MLXArray = .!pruneMask
            let keepIndices = conditionToIndices(condition: keepMask)
            (_xyz, _features_dc, _features_rest, _scales, _rotation, _opacity) =
                pruneGaussians(
                    xyz: _xyz, features_dc: _features_dc, features_rest: _features_rest,
                    scales: _scales, rotation: _rotation, opacity: _opacity,
                    indices: keepIndices
                )
        }

        // Update model parameters
        model._xyz = _xyz
        model._features_dc = _features_dc
        model._features_rest = _features_rest
        model._scales = _scales
        model._rotation = _rotation
        model._opacity = _opacity

        // Reset gradient accumulation
        resetGradientAccumulation()
    }
    
    func splitGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {

        let selectedXYZ = xyz[indices]
        let selectedFeaturesDC = features_dc[indices]
        let selectedFeaturesRest = features_rest[indices]
        let selectedScales = scales[indices]
        let selectedRotation = rotation[indices]
        let selectedOpacity = opacity[indices]

        // Scale down the selected Gaussians
        let newScales = selectedScales - MLX.log(MLXArray(1.6))

        // Create two new Gaussians for each split
        let numSplit = indices.shape[0]
        let noise = MLXRandom.normal([numSplit, 3]) * 0.1

        let newXYZ1 = selectedXYZ + noise
        let newXYZ2 = selectedXYZ - noise

        // Create mask to keep Gaussians that are NOT being split
        // Vectorized approach: use array broadcasting instead of item() loop
        let totalPoints = xyz.shape[0]

        // Create all possible indices [0, 1, 2, ..., N-1]
        let allIndices = MLXArray(0..<totalPoints)

        // Expand dimensions for broadcasting: allIndices[N, 1], indices[1, M]
        let allExpanded = allIndices.expandedDimensions(axes: [1])  // [N, 1]
        let splitExpanded = indices.expandedDimensions(axes: [0])   // [1, M]

        // Compare: is each index in allIndices present in split indices?
        // Result shape: [N, M] where result[i, j] = (allIndices[i] == indices[j])
        let matches = allExpanded .== splitExpanded

        // Any match means this index is being split: reduce along axis 1
        let isSplit = MLX.any(matches, axes: [1])  // [N]

        // Invert to get keep mask (keep = NOT split)
        let keepMask = .!isSplit
        let keepIndices = conditionToIndices(condition: keepMask)

        // Optimize: Eliminate intermediate "kept*" arrays by inlining indexing into concatenation
        // This reduces memory allocations from 12 arrays to 6 arrays (2x reduction)
        // Old approach: select kept arrays (6 allocations) + concatenate (6 allocations) = 12 total
        // New approach: concatenate with inline selection (6 allocations) = 6 total
        let newXYZAll = MLX.concatenated([xyz[keepIndices], newXYZ1, newXYZ2], axis: 0)
        let newFeaturesDCAll = MLX.concatenated([features_dc[keepIndices], selectedFeaturesDC, selectedFeaturesDC], axis: 0)
        let newFeaturesRestAll = MLX.concatenated([features_rest[keepIndices], selectedFeaturesRest, selectedFeaturesRest], axis: 0)
        let newScalesAll = MLX.concatenated([scales[keepIndices], newScales, newScales], axis: 0)
        let newRotationAll = MLX.concatenated([rotation[keepIndices], selectedRotation, selectedRotation], axis: 0)
        let newOpacityAll = MLX.concatenated([opacity[keepIndices], selectedOpacity, selectedOpacity], axis: 0)

        return (newXYZAll, newFeaturesDCAll, newFeaturesRestAll, newScalesAll, newRotationAll, newOpacityAll)
    }
    
    func cloneGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {

        // Only select XYZ since we need it for noise computation
        let selectedXYZ = xyz[indices]

        // Add small noise to position
        let noise = MLXRandom.normal(selectedXYZ.shape) * 0.01
        let newXYZ = selectedXYZ + noise

        // Optimize: Inline single-use selections into concatenation
        // This eliminates 5 intermediate arrays (selectedFeaturesDC, selectedFeaturesRest, etc.)
        // Reduces allocations from 11 arrays to 6 arrays
        let newXYZAll = MLX.concatenated([xyz, newXYZ], axis: 0)
        let newFeaturesDCAll = MLX.concatenated([features_dc, features_dc[indices]], axis: 0)
        let newFeaturesRestAll = MLX.concatenated([features_rest, features_rest[indices]], axis: 0)
        let newScalesAll = MLX.concatenated([scales, scales[indices]], axis: 0)
        let newRotationAll = MLX.concatenated([rotation, rotation[indices]], axis: 0)
        let newOpacityAll = MLX.concatenated([opacity, opacity[indices]], axis: 0)

        return (newXYZAll, newFeaturesDCAll, newFeaturesRestAll, newScalesAll, newRotationAll, newOpacityAll)
    }

    func pruneGaussians(
        xyz: MLXArray, features_dc: MLXArray, features_rest: MLXArray,
        scales: MLXArray, rotation: MLXArray, opacity: MLXArray,
        indices: MLXArray
    ) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {

        let newXYZ = xyz[indices]
        let newFeaturesDC = features_dc[indices]
        let newFeaturesRest = features_rest[indices]
        let newScales = scales[indices]
        let newRotation = rotation[indices]
        let newOpacity = opacity[indices]

        return (newXYZ, newFeaturesDC, newFeaturesRest, newScales, newRotation, newOpacity)
    }
    func save_snapshot(iteration: Int, params: [MLXArray]) {
        //TODO
        if let outputDirectoryURL {
            let xyz = params[0]
            let features_dc = params[1]
            let features_rest = params[2]
            let scales = params[3]
            let rotation = params[4]
            let opacity = params[5]
            do {
                let outputURL = outputDirectoryURL.appendingPathComponent("iteration_\(iteration).ply")
                try PlyWriter.writeGaussianBinary(
                    positions: xyz, features_dc: features_dc, features_rest: features_rest,
                    opacities: opacity, scales: scales, rotations: rotation,
                    to: outputURL)
                delegate?.pushSnapshot(url: outputURL, iteration: iteration, timestamp: Date())
            } catch {
                Logger.shared.error(error: error)
            }
        }
    }
    var forceStop: Bool = false
    func stopTrain() {
        forceStop = true
    }
    func startTrain(earlyStoppingThreshold: Float = 0.0001) {
        let trainer: GaussianTrainer = self
        let gaussRender = trainer.gaussRender
        let model = trainer.model
        let lambda_depth = trainer.lambda_depth
        let lambda_dssim = trainer.lambda_dssim
        var params = model.getParams()

        let optimizer = Adam(
            learningRate: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-15
        )
        var states = params.map {
            optimizer.newState(parameter: $0)
        }
        for iteration in 0..<iterationCount {
            if forceStop {
                break
            }
            Logger.shared.debug("\(iteration)th iteration")

            // Progressive resolution scheduling: start at low resolution, ramp up to full
            let resolutionScale = getCurrentResolutionScale(iteration: iteration)
            let (trainCamera, trainRGB, trainMask, trainDepth) =
                fetchTrainDataWithResolution(scale: resolutionScale)
            let train: ([MLXArray]) -> [MLXArray] = { params in
                let _xyz = params[0]
                let _features_dc = params[1]
                let _features_rest = params[2]
                let _scales = params[3]
                let _rotation = params[4]
                let _opacity = params[5]
                Logger.shared.debug("Prepare variables")
                let means3d = gaussRender.get_xyz_from(_xyz)
                let opacity = gaussRender.get_opacity_from(_opacity)
                let scales = gaussRender.get_scales_from(_scales)
                let rotations = gaussRender.get_rotation_from(_rotation)
                let shs = gaussRender.get_features_from(
                    _features_dc,
                    _features_rest
                )
                Logger.shared.debug("Prepare forward")
                let (
                    render,
                    depth,
                    _,
                    _,
                    _
                ) = gaussRender.forward(
                    camera: trainCamera,
                    means3d: means3d,
                    shs: shs,
                    opacity: opacity,
                    scales: scales,
                    rotations: rotations
                )
                Logger.shared.debug("Calculate loss")
                let l1_loss = l1Loss(render, trainRGB)
                let depth_loss =
                    trainDepth != nil
                    ? l1Loss(
                        depth[.ellipsis, 0].reshaped([-1])[trainMask],
                        trainDepth!.reshaped([-1])[trainMask]
                    ) : MLXArray(0.0 as Float)
                let ssim_loss =
                    1.0 - ssim(img1: render[.newAxis], img2: trainRGB[.newAxis], cachedWindow: gaussRender.cachedSsimWindow)

                let total_loss =
                    (1.0 - lambda_dssim) * l1_loss + lambda_dssim * ssim_loss
                    + lambda_depth * depth_loss
                return [total_loss, render]
            }
            Logger.shared.debug("valueAndGrad")
            let (loss, grads) = MLX.valueAndGrad(
                train,
                argumentNumbers: Array(0..<params.count)
            )(params)
            eval(loss[0], grads)
            let lossValue = loss[0].item(Float.self)
            
            // Accumulate gradients for densification
            addGradientAccumulation(xyzGrad: grads[0])
            
            delegate?.pushLoss(
                loss: lossValue,
                iteration: iteration,
                timestamp: Date())
            if iteration % 20 == 0 {
                let image = loss[1]
                eval(image)
                delegate?.pushImageData(
                    render: image,
                    truth: trainRGB,
                    loss: lossValue,
                    iteration: iteration, timestamp: Date()
                )
            }
            if lossValue < earlyStoppingThreshold {
                Logger.shared.info("early stopping")
                return
            }
            let lrs = model.getLearningRates(
                current: iteration,
                total: iterationCount
            )
            for i in 0..<params.count {
                // Skip higher-degree SH coefficient updates (_features_rest, index 2) early in training
                // This optimization from LichtFeld-Studio saves ~10-15% compute time in early iterations
                // Higher-degree SH coefficients have minimal impact when geometry is still being refined
                if i == 2 && iteration < shHigherDegreeStartIter {
                    Logger.shared.debug("skip _features_rest update (iteration \(iteration) < \(shHigherDegreeStartIter))")
                    continue
                }

                Logger.shared.debug("update \(i)th param start")
                optimizer.learningRate = lrs[i]

                // Sparse Adam: Skip update if all gradients are near zero
                // This reduces computation for parameters not being optimized this iteration
                if useSparseAdam {
                    // Compute maximum absolute gradient value
                    let maxAbsGrad = MLX.max(MLX.abs(grads[i])).item(Float.self)

                    // Skip update if maximum gradient is negligible
                    // This means ALL gradients for this parameter are near zero
                    if maxAbsGrad < sparseGradientThreshold {
                        Logger.shared.debug("skip param \(i) update (sparse Adam: max |grad| \(maxAbsGrad) < \(sparseGradientThreshold))")
                        continue
                    }
                }

                let (newParam, newState) = optimizer.applySingle(
                    gradient: grads[i],
                    parameter: params[i],
                    state: states[i]
                )
                params[i] = newParam
                states[i] = newState
                Logger.shared.debug("update \(i)th param end")
            }
            // Batch eval all updated parameters at once instead of per-parameter
            eval(params)

            // Track if we need cache clear (consolidate multiple clears into one)
            var needsCacheClear = MLX.GPU.snapshot().cacheMemory > cacheLimit

            if iteration % self.save_snapshot_per_iteration == 0 {
                self.save_snapshot(iteration: iteration, params: params)
                needsCacheClear = true
            }
            if iteration % self.split_and_prune_per_iteration == 0 {
                self.split_and_prune(params: params, states: states, iteration: iteration)
                // Update params after split_and_prune
                params = model.getParams()
                // Reinitialize optimizer states for new parameters
                // TODO: Preserve states once TupleState API is available
                states = params.map {
                    optimizer.newState(parameter: $0)
                }
                needsCacheClear = true
            }

            // Consolidated cache clear (inspired by LichtFeld-Studio memory allocator tuning)
            if needsCacheClear {
                MLX.GPU.clearCache()
            }
        }
        MLX.GPU.clearCache()
    }
    static func createModel(
        sh_degree: Int,
        pointCloud: PointCloud,
        sampleCount: Int = 1 << 14
    ) -> GaussModel {
        Logger.shared.debug("get point clouds...")
        Logger.shared.debug("Random sample....")
        let raw_points = pointCloud.randomSample(sampleCount)
        let gaussModel = GaussModel.create_from_pcd(
            pcd: raw_points,
            sh_degree: sh_degree,
            debug: false
        )
        return gaussModel
    }
}
