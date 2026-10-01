use serde::Deserialize;

// json이 계층 구조일때 객체를 struct로 만들고 객체의 key를 filed로 두면 됨

#[derive(Deserialize, Debug)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
}

#[derive(Deserialize, Debug)]
pub struct TextConfig {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub head_dim: usize,        // 256 (local)
    pub global_head_dim: usize, // 512 (global)
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,

    pub rms_norm_eps: f32,

    // local/global
    pub layer_types: Vec<String>, // sliding_attention, full_attention
    pub sliding_window: usize,
    pub num_kv_shared_layers: usize,
    pub rope_parameters: RopeParameters,

    pub hidden_size_per_layer_input: usize,

    pub final_logit_softcapping: f32,
    pub tie_word_embeddings: bool,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32, // imgae placeholder에 사용
}

#[derive(Deserialize, Debug)]
pub struct RopeParameters {
    pub full_attention: RopeConfig,
    pub sliding_attention: RopeConfig,
}

#[derive(Deserialize, Debug)]
pub struct RopeConfig {
    pub rope_theta: f32,

    // full_attention에만 있으며 없을경우 None
    pub partial_rotary_factor: Option<f32>,
}

// TextConfi와 중복되는 filed가 있으나 내부 값이 다르고 연산 함수 구현시 편리함
#[derive(Deserialize, Debug)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeConfig,
    pub standardize: bool,
    pub use_clipped_linears: bool,
    pub hidden_activation: String, // text generation에서 사용하던 gelu 재사용
}

#[derive(Deserialize, Debug)]
pub struct ProcessorConfig {
    pub image_processor: ImageProcessorConfig,
}

#[derive(Deserialize, Debug)]
pub struct ImageProcessorConfig {
    pub max_soft_tokens: usize,
    pub patch_size: usize,
    pub pooling_kernel_size: usize,
    pub rescale_factor: f32,
}

impl Config {
    pub fn load_config(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("config.json read failed");
        serde_json::from_str(&text).expect("config.json parse failed")
    }
}

impl ProcessorConfig {
    pub fn load_processor_config(path: &str) -> Self {
        let text = std::fs::read_to_string(path).expect("processor_config.json read failed");
        serde_json::from_str(&text).expect("processor_config.json parse failed")
    }
}
