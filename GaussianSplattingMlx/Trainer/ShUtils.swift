import Foundation
import MLX

let C0: Float = 0.28209479177387814
let C1: Float = 0.4886025119029199
let C2: [Float] = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396,
]
let C3: [Float] = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435,
]
let C4: [Float] = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

func RGB2SH(rgb: MLXArray) -> MLXArray {
    return (rgb - 0.5) / C0
}

func SH2RGB(sh: MLXArray) -> MLXArray {
    return sh * C0 + 0.5
}

func evalSh(deg: Int, sh: MLXArray, dirs: MLXArray) -> MLXArray {
    let coeff = pow(Double(deg) + 1, 2)
    if sh.shape[sh.shape.count - 1] < Int(coeff) {
        fatalError("coeff:\(coeff) is too large")
    }
    var result = C0 * sh[.ellipsis, 0]
    if deg > 0 {
        let x = dirs[.ellipsis, .stride(from: 0, to: 1)]
        let y = dirs[.ellipsis, .stride(from: 1, to: 2)]
        let z = dirs[.ellipsis, .stride(from: 2, to: 3)]
        result =
            (result - C1 * y * sh[.ellipsis, 1] + C1 * z * sh[.ellipsis, 2] - C1
                * x * sh[.ellipsis, 3])

        if deg > 1 {
            // Precompute all squared and product terms
            let xx = x * x
            let yy = y * y
            let zz = z * z
            let xy = x * y
            let yz = y * z
            let xz = x * z

            // Precompute common intermediate terms
            let xx_yy = xx - yy  // Used in deg 1, 2, 3
            let zz2 = 2.0 * zz   // Used multiple times
            let zz2_xx_yy = zz2 - xx - yy  // Used in deg 1

            result =
                (result + C2[0] * xy * sh[.ellipsis, 4] + C2[1] * yz
                    * sh[.ellipsis, 5] + C2[2] * zz2_xx_yy
                    * sh[.ellipsis, 6] + C2[3] * xz * sh[.ellipsis, 7] + C2[4]
                    * xx_yy * sh[.ellipsis, 8])
            if deg > 2 {
                // Precompute more intermediate terms for deg 2
                let xx3 = 3 * xx
                let yy3 = 3 * yy
                let zz4_xx_yy = 4 * zz - xx - yy  // Shared term used twice
                let xx3_yy = xx3 - yy
                let xx_yy3 = xx - yy3

                result =
                    (result + C3[0] * y * xx3_yy * sh[.ellipsis, 9] + C3[1]
                        * xy * z * sh[.ellipsis, 10] + C3[2] * y
                        * zz4_xx_yy * sh[.ellipsis, 11] + C3[3] * z
                        * (zz2 - xx3 - yy3) * sh[.ellipsis, 12] + C3[4]
                        * x * zz4_xx_yy * sh[.ellipsis, 13] + C3[5] * z
                        * xx_yy * sh[.ellipsis, 14] + C3[6] * x
                        * xx_yy3 * sh[.ellipsis, 15])
                if deg > 3 {
                    // Precompute intermediate terms for deg 3
                    let zz7 = 7 * zz
                    let zz7_1 = zz7 - 1
                    let zz7_3 = zz7 - 3

                    result =
                        (result + C4[0] * xy * xx_yy * sh[.ellipsis, 16]
                            + C4[1] * yz * xx3_yy * sh[.ellipsis, 17]
                            + C4[2] * xy * zz7_1 * sh[.ellipsis, 18]
                            + C4[3] * yz * zz7_3 * sh[.ellipsis, 19]
                            + C4[4] * (zz * (35 * zz - 30) + 3)
                            * sh[.ellipsis, 20] + C4[5] * xz * zz7_3
                            * sh[.ellipsis, 21] + C4[6] * xx_yy
                            * zz7_1 * sh[.ellipsis, 22] + C4[7] * xz
                            * xx_yy3 * sh[.ellipsis, 23] + C4[8]
                            * (xx * xx_yy3 - yy * xx3_yy)
                            * sh[.ellipsis, 24])
                }
            }
        }
    }
    return result
}
