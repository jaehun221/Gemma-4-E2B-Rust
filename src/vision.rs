use image::{ImageReader, RgbImage};
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, Ix0, Ix2, Ix3, s};
use safetensors::SafeTensors;

use crate::config::{ProcessorConfig, VisionConfig};
use crate::loader::get_tensor;
use crate::weights::Norms;

pub struct PatchInput {
    pub pixels: Array2<f32>,
    pub positions: Vec<(usize, usize)>,
    pub grid: (usize, usize),
}

pub struct ClippedLinear {
    pub weight: Array2<f32>,
    pub in_min: f32,
    pub in_max: f32,
    pub out_min: f32,
    pub out_max: f32,
}

pub struct VisionLayer {
    // attention
    pub q: ClippedLinear,
    pub k: ClippedLinear,
    pub v: ClippedLinear,
    pub o: ClippedLinear,
    pub q_norm: Array1<f32>, // [64]
    pub k_norm: Array1<f32>, // [64]
    // mlp
    pub gate: ClippedLinear,
    pub up: ClippedLinear,
    pub down: ClippedLinear,
    // sandwich norm
    pub norm: Norms, // 텍스트의 Norms 재사용
}

pub struct VisionWeights {
    pub patch_proj: Array2<f32>,
    pub pos_table_x: Array2<f32>,
    pub pos_table_y: Array2<f32>,
    pub layers: Vec<VisionLayer>,
    pub embed_proj: Array2<f32>,
}

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

impl VisionWeights {
    pub fn load(tensors: &SafeTensors, cfg: &VisionConfig) -> Self {
        let clipped = |p: &str| ClippedLinear {
            weight: get_tensor::<Ix2>(tensors, &format!("{p}.linear.weight")),
            in_min: get_tensor::<Ix0>(tensors, &format!("{p}.input_min")).into_scalar(),
            in_max: get_tensor::<Ix0>(tensors, &format!("{p}.input_max")).into_scalar(),
            out_min: get_tensor::<Ix0>(tensors, &format!("{p}.output_min")).into_scalar(),
            out_max: get_tensor::<Ix0>(tensors, &format!("{p}.output_max")).into_scalar(),
        };

        let layers: Vec<VisionLayer> = (0..cfg.num_hidden_layers)
            .map(|i| {
                let p = format!("model.vision_tower.encoder.layers.{i}");
                VisionLayer {
                    // attention
                    q: clipped(&format!("{p}.self_attn.q_proj")),
                    k: clipped(&format!("{p}.self_attn.k_proj")),
                    v: clipped(&format!("{p}.self_attn.v_proj")),
                    o: clipped(&format!("{p}.self_attn.o_proj")),
                    q_norm: get_tensor(tensors, &format!("{p}.self_attn.q_norm.weight")),
                    k_norm: get_tensor(tensors, &format!("{p}.self_attn.k_norm.weight")),
                    // mlp
                    gate: clipped(&format!("{p}.mlp.gate_proj")),
                    up: clipped(&format!("{p}.mlp.up_proj")),
                    down: clipped(&format!("{p}.mlp.down_proj")),
                    // sandwich norm
                    norm: Norms {
                        input_norm: get_tensor(tensors, &format!("{p}.input_layernorm.weight")),
                        post_attn_norm: get_tensor(
                            tensors,
                            &format!("{p}.post_attention_layernorm.weight"),
                        ),
                        pre_ffn_norm: get_tensor(
                            tensors,
                            &format!("{p}.pre_feedforward_layernorm.weight"),
                        ),
                        post_ffn_norm: get_tensor(
                            tensors,
                            &format!("{p}.post_feedforward_layernorm.weight"),
                        ),
                    },
                }
            })
            .collect();

        let pos = get_tensor::<Ix3>(
            tensors,
            "model.vision_tower.patch_embedder.position_embedding_table",
        );

        Self {
            patch_proj: get_tensor(
                tensors,
                "model.vision_tower.patch_embedder.input_proj.weight",
            ),
            pos_table_x: pos.index_axis(Axis(0), 0).to_owned(),
            pos_table_y: pos.index_axis(Axis(0), 1).to_owned(),
            layers,
            embed_proj: get_tensor(tensors, "model.embed_vision.embedding_projection.weight"),
        }
    }

    pub fn patch_embed(&self, input: &PatchInput) -> Array2<f32> {
        let x = input.pixels.mapv(|v| 2.0 * (v - 0.5));

        let mut h = x.dot(&self.patch_proj.t());

        for (n, &(px, py)) in input.positions.iter().enumerate() {
            let mut row = h.row_mut(n);
            row += &self.pos_table_x.row(px);
            row += &self.pos_table_y.row(py);
        }

        h
    }

    
}

impl ClippedLinear {

    pub fn forward(&self, x: ArrayView2<f32>) -> Array2<f32> {
        let x = x.mapv(|v| v.clamp(self.in_min, self.in_max));
        let y = x.dot(&self.weight.t());
        y.mapv(|v| v.clamp(self.out_min, self.out_max))
    }
}

