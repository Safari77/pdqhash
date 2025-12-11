///! Compute PDQ hash of an image.

pub use image;

// Coeffs for Luma conversion (Rec. 601)
const LUMA_FROM_R_COEFF: f32 = 0.299;
const LUMA_FROM_G_COEFF: f32 = 0.587;
const LUMA_FROM_B_COEFF: f32 = 0.114;

// External DCT module (assumed to exist in your project structure)
mod dct;

const MIN_HASHABLE_DIM: u32 = 5;
const PDQ_NUM_JAROSZ_XY_PASSES: usize = 2;
const DOWNSAMPLE_DIMS: u32 = 512;
const BUFFER_W_H: usize = 64;
const DCT_OUTPUT_W_H: usize = 16;
const DCT_OUTPUT_MATRIX_SIZE: usize = DCT_OUTPUT_W_H * DCT_OUTPUT_W_H;
const HASH_LENGTH: usize = DCT_OUTPUT_MATRIX_SIZE / 8;

/// Represents the 16x16 DCT coefficients in the frequency domain.
/// This intermediate state is required to generate accurate Dihedral hashes.
#[derive(Clone, Debug)]
pub struct PdqFeatures {
    /// 16x16 frequency domain coefficients
    coefficients: [f32; DCT_OUTPUT_MATRIX_SIZE],
}

impl PdqFeatures {
    /// Create features from a 64x64 spatial domain buffer.
    fn new(buffer64x64: &[[f32; BUFFER_W_H]; BUFFER_W_H]) -> Self {
        let coefficients = dct64_to_16(buffer64x64);
        Self { coefficients }
    }

    /// Converts the frequency features into the final binary PDQ hash.
    /// This performs the median quantization step.
    pub fn to_hash(&self) -> [u8; HASH_LENGTH] {
        let mut sorted = self.coefficients;
        // Handle NaNs safely
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Less));

        // The median is the element at rank 128 (0-based index)
        let median = sorted[DCT_OUTPUT_MATRIX_SIZE / 2];
        let mut hash = [0; HASH_LENGTH];

        for i in 0..HASH_LENGTH {
            let mut byte = 0;
            for j in 0..8 {
                let val = self.coefficients[i * 8 + j];
                // PDQ standard: Bit is 1 if value is strictly GREATER than median
                if val > median {
                    byte |= 1 << j;
                }
            }
            hash[HASH_LENGTH - i - 1] = byte;
        }
        hash
    }
}

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  Public API
//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// Returns PDQ hash and quality of an image.
/// Returns None if image is too small.
pub fn generate_pdq(image: &image::DynamicImage) -> Option<([u8; HASH_LENGTH], f32)> {
    if image.width() < MIN_HASHABLE_DIM || image.height() < MIN_HASHABLE_DIM {
        return None;
    }

    // Downsample if necessary for performance
    let out = if image.width() > DOWNSAMPLE_DIMS || image.height() > DOWNSAMPLE_DIMS {
        generate_pdq_full_size_internal(&image.thumbnail_exact(
            DOWNSAMPLE_DIMS.min(image.width()),
            DOWNSAMPLE_DIMS.min(image.height()),
        ))
    } else {
        generate_pdq_full_size_internal(&image)
    };
    Some((out.0.to_hash(), out.1))
}

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
//  Internal Logic
//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

/// Internal helper that returns the PdqFeatures struct instead of just the bits
fn generate_pdq_full_size_internal(image: &image::DynamicImage) -> (PdqFeatures, f32) {
    let (num_cols, num_rows, mut luma_buffer) = to_luma_image(image);

    // Calculate Window sizes for Jarosz Filter
    let window_size_along_rows = (num_cols + 2 * BUFFER_W_H - 1) / (2 * BUFFER_W_H);
    let window_size_along_cols = (num_rows + 2 * BUFFER_W_H - 1) / (2 * BUFFER_W_H);

    jarosz_filter_float(
        &mut luma_buffer,
        num_rows,
        num_cols,
        window_size_along_rows,
        window_size_along_cols,
        PDQ_NUM_JAROSZ_XY_PASSES,
    );

    let buffer64x64 =
        decimate_float::<BUFFER_W_H, BUFFER_W_H>(&luma_buffer, num_rows, num_cols);

    // Compute quality on the spatial 64x64 buffer
    let quality = pdq_image_domain_quality_metric(&buffer64x64);

    // Convert to frequency domain features
    let features = PdqFeatures::new(&buffer64x64);

    (features, quality)
}

fn to_luma_image(image: &image::DynamicImage) -> (usize, usize, Vec<f32>) {
    match image {
        image::DynamicImage::ImageLuma8(img) => {
            (
                img.width() as usize,
                img.height() as usize,
                img.pixels().map(|p| p.0[0] as f32).collect()
            )
        },
        _ => {
            let rgb = image.to_rgb8();
            let width = rgb.width() as usize;
            let height = rgb.height() as usize;
            let data: Vec<f32> = rgb.pixels()
                .map(|p| {
                    p.0[0] as f32 * LUMA_FROM_R_COEFF
                    + p.0[1] as f32 * LUMA_FROM_G_COEFF
                    + p.0[2] as f32 * LUMA_FROM_B_COEFF
                })
                .collect();
            (width, height, data)
        }
    }
}

/// Perform a discrete cosine transform from a 64x64 matrix and compute only a 16x16 corner of it.
///
fn dct64_to_16<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    input: &[[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) -> [f32; DCT_OUTPUT_MATRIX_SIZE] {
    let mut intermediate_matrix = [[0.0; OUT_NUM_COLS]; DCT_OUTPUT_W_H];

    // Pass 1: Rows
    for i in 0..DCT_OUTPUT_W_H {
        for j in 0..OUT_NUM_COLS {
            let mut sumk = 0.0;
            for k in 0..BUFFER_W_H {
                sumk += f32::from_bits(dct::DCT_MATRIX[i][k]) * input[k][j];
            }
            intermediate_matrix[i][j] = sumk;
        }
    }

    let mut output = [0.0; DCT_OUTPUT_MATRIX_SIZE];

    // Pass 2: Columns
    for i in 0..DCT_OUTPUT_W_H {
        for j in 0..DCT_OUTPUT_W_H {
            let mut sumk = 0.0;
            for k in 0..BUFFER_W_H {
                sumk += intermediate_matrix[i][k] * f32::from_bits(dct::DCT_MATRIX[j][k]);
            }
            output[i * DCT_OUTPUT_W_H + j] = sumk;
        }
    }
    output
}

// ----------------------------------------------------------------
// Filter and Decimation Logic (Preserved from original with optimization)
// ----------------------------------------------------------------

fn transpose(input: &[f32], output: &mut [f32], width: usize, height: usize) {
    for y in 0..height {
        for x in 0..width {
            output[x * height + y] = input[y * width + x];
        }
    }
}

#[inline(always)]
fn box_one_d_float(
    invec: &[f32],
    in_start_offset: usize,
    outvec: &mut [f32],
    vector_length: usize,
    full_window_size: usize,
) {
    let half_window_size = (full_window_size + 2) / 2;
    let oi_off = half_window_size - 1;
    let li_off = full_window_size - half_window_size + 1;

    let mut sum = 0.0;
    let mut current_window_size = 0.0;

    let phase_1_end = in_start_offset + oi_off;
    for ri in in_start_offset..phase_1_end {
        sum += invec[ri];
        current_window_size += 1.0;
    }

    let phase_2_end = in_start_offset + full_window_size;
    for ri in phase_1_end..phase_2_end {
        let oi = ri - oi_off;
        sum += invec[ri];
        current_window_size += 1.0;
        outvec[oi] = sum / current_window_size;
    }

    let phase_3_end = in_start_offset + vector_length;
    for ri in phase_2_end..phase_3_end {
        let oi = ri - oi_off;
        let li = oi - li_off;
        sum += invec[ri];
        sum -= invec[li];
        outvec[oi] = sum / current_window_size;
    }

    let phase_4_start = in_start_offset + vector_length - half_window_size + 1;
    for oi in phase_4_start..phase_3_end {
        let li = oi - li_off;
        sum -= invec[li];
        current_window_size -= 1.0;
        outvec[oi] = sum / current_window_size;
    }
}

fn box_along_rows_float(
    input: &[f32],
    output: &mut [f32],
    n_rows: usize,
    n_cols: usize,
    window_size: usize,
) {
    for i in 0..n_rows {
        box_one_d_float(input, i * n_cols, output, n_cols, window_size);
    }
}

fn jarosz_filter_float(
    buffer: &mut [f32],
    num_rows: usize,
    num_cols: usize,
    window_size_along_rows: usize,
    window_size_along_cols: usize,
    nreps: usize,
) {
    let mut temp_buf = vec![0.0; buffer.len()];

    for _ in 0..nreps {
        box_along_rows_float(buffer, &mut temp_buf, num_rows, num_cols, window_size_along_rows);
        transpose(&temp_buf, buffer, num_cols, num_rows);
        box_along_rows_float(buffer, &mut temp_buf, num_cols, num_rows, window_size_along_cols);
        transpose(&temp_buf, buffer, num_rows, num_cols);
    }
}

fn decimate_float<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    input: &[f32],
    in_num_rows: usize,
    in_num_cols: usize,
) -> [[f32; OUT_NUM_COLS]; OUT_NUM_ROWS] {
    let mut output = [[0.0; OUT_NUM_COLS]; OUT_NUM_ROWS];
    for outi in 0..OUT_NUM_ROWS {
        let ini = ((outi * 2 + 1) * in_num_rows) / (OUT_NUM_ROWS * 2);
        for outj in 0..OUT_NUM_COLS {
            let inj = ((outj * 2 + 1) * in_num_cols) / (OUT_NUM_COLS * 2);
            output[outi][outj] = input[ini * in_num_cols + inj];
        }
    }
    output
}

fn pdq_image_domain_quality_metric<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    buffer64x64: &[[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) -> f32 {
    let mut gradient_sum = 0.0;

    for i in 0..(OUT_NUM_ROWS - 1) {
        for j in 0..OUT_NUM_COLS {
            let u = buffer64x64[i][j];
            let v = buffer64x64[i + 1][j];
            gradient_sum += ((u - v) / 255.0).abs();
        }
    }
    for i in 0..OUT_NUM_ROWS {
        for j in 0..(OUT_NUM_COLS - 1) {
            let u = buffer64x64[i][j];
            let v = buffer64x64[i][j + 1];
            gradient_sum += ((u - v) / 255.0).abs();
        }
    }

    let quality = gradient_sum / 90.0;
    if quality > 1.0 { 1.0 } else { quality }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jarosz_filter_logic() {
        // Create a 10x10 empty image
        let width = 10;
        let height = 10;
        let mut buffer = vec![0.0; width * height];

        // Set a single "hot" pixel in the center (5, 5)
        buffer[5 * width + 5] = 100.0;

        // Apply filter: 2 passes, window size 3
        // Window 3 implies neighbors should pick up some value
        jarosz_filter_float(
            &mut buffer,
            height,
            width,
            3, // Window row
            3, // Window col
            1, // 1 pass
        );

        // Check center pixel (should be dampened)
        let center = buffer[5 * width + 5];
        assert!(center < 100.0, "Center pixel should be averaged down");
        assert!(center > 0.0, "Center pixel should not be zero");

        // Check horizontal neighbor (5, 6) - Should get value from Row pass
        let right = buffer[5 * width + 6];
        assert!(right > 0.0, "Horizontal neighbor should receive blur");

        // Check vertical neighbor (6, 5) - Should get value from Col (Transposed) pass
        let down = buffer[6 * width + 5];
        assert!(down > 0.0, "Vertical neighbor should receive blur");

        // Check exact symmetry if window sizes are same
        assert!((right - down).abs() < 0.001, "Filter should be symmetric for symmetric windows");
    }
}

