use image::{ImageReader, RgbImage};
use ndarray::Array3;

use crate::config::ProcessorConfig;

// image 연산을 위한 전처리 작업
pub fn preprocess(path: &str, prc_cfg: &ProcessorConfig) {
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

    let img_arr = Array3::from_shape_vec((h_target, w_target, 3), img_data).expect("shape not matched");
}
