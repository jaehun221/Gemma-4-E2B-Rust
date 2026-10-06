use image::{ImageReader, RgbImage};
use ndarray::{Array2, Array3, s};

use crate::config::ProcessorConfig;

pub struct PatchInput {
    pub pixels: Array2<f32>,
    pub positions: Vec<(usize, usize)>,
    pub grid: (usize, usize),
}

pub struct VisionWeights {}

// image 연산을 위한 전처리
pub fn preprocess(path: &str, prc_cfg: &ProcessorConfig) -> PatchInput {
    let img_cfg = &prc_cfg.image_processor;

    let img: RgbImage = ImageReader::open(path)
        .expect("failed to open image")
        .decode()
        .expect("failed to decode image")
        .to_rgb8();

    let (w, h) = img.dimensions();

    let cell_size = img_cfg.patch_size * img_cfg.pooling_kernel_size;
    let target_area = img_cfg.max_soft_tokens as f64 * (cell_size * cell_size) as f64;

    let scale = (target_area / (h * w) as f64).sqrt(); // iamge size가 클 경우 f32범위를 벗어날 수 있어 f64 사용
    let h_target = (h as f64 * scale / cell_size as f64).floor() as usize * cell_size;
    let w_target = (w as f64 * scale / cell_size as f64).floor() as usize * cell_size;

    if h_target != h as usize && w_target != w as usize {
        // TODO: resize는 정해둔 이미지 사이즈로 테스트 후 정상 작동하면 이후에 추가
        panic!("TODO: resize");
    }

    let img_data: Vec<f32> = img
        .as_raw()
        .iter()
        .map(|&x| x as f32 * img_cfg.rescale_factor)
        .collect();

    // resize logic 구현 후에는 h_target, w_target이 아닌 resize_img 사용
    let img_arr =
        Array3::from_shape_vec((h_target, w_target, 3), img_data).expect("shape not match");

    let p = img_cfg.patch_size;
    let ph = h_target / p;
    let pw = w_target / p;
    let patch_len = p * p * 3;

    let mut pixel_buf: Vec<f32> = Vec::with_capacity(ph * pw * patch_len);
    let mut positions: Vec<(usize, usize)> = Vec::with_capacity(ph * pw);

    for r in 0..ph {
        for c in 0..pw {
            let patch = img_arr.slice(s![r * p..r * p + p, c * p..c * p + p, ..]);
            pixel_buf.extend(patch.iter());
            positions.push((c, r));
        }
    }

    let pixels = Array2::from_shape_vec((ph * pw, patch_len), pixel_buf)
        .expect("patch buffer size mismatch");

    PatchInput {
        pixels,
        positions,
        grid: (pw, ph),
    }
}
