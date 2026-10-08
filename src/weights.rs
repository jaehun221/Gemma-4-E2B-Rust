use std::assert_eq;

use memmap2::Mmap;
use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use safetensors::SafeTensors;
use tokenizers::Tokenizer;

use crate::config::{Config, TextConfig};
use crate::loader::{get_tensor, to_f32};
use crate::operation::{argmax, decoder_block, rms_norm, rope_tables};
use crate::vision::VisionWeights;

// 현재 f32로 처리하고 있으나, MultiModal 구현 후 양자화 적용을 고려
pub struct Weights {
    embd: Array2<f32>,
    layer: Vec<Block>,
    norm_f: Array1<f32>,
    mmap: Mmap,
    ple_table_offset: (usize, usize, usize),
    ple_model_proj: Array2<f32>,
    ple_proj_norm: Array1<f32>,

    pub vision: VisionWeights,
}

pub struct Block {
    pub attn: Attn,
    pub mlp: Mlp,
    pub norm: Norms,
    pub ple: PleLayer,
}

pub struct PleLayer {
    pub projection: Array2<f32>,
    pub input_gate: Array2<f32>,
    pub post_norm: Array1<f32>,
    pub scalar: f32,
}

pub struct Attn {
    pub attn_q: Array2<f32>,
    pub attn_k: Array2<f32>,
    pub attn_v: Array2<f32>,
    pub attn_o: Array2<f32>,

    pub q_norm: Array1<f32>,
    pub k_norm: Array1<f32>,
}

pub struct Norms {
    pub input_norm: Array1<f32>,
    pub post_attn_norm: Array1<f32>,
    pub pre_ffn_norm: Array1<f32>,
    pub post_ffn_norm: Array1<f32>,
}

pub struct Mlp {
    pub up_proj: Array2<f32>,
    pub gate_proj: Array2<f32>,
    pub down_proj: Array2<f32>,
}

impl Weights {
    pub fn generate(
        &self,
        prompt: &str,
        image_features: Option<ArrayView2<f32>>,
        tokenizer: &Tokenizer,
        cfg: &Config,
        max: usize,
    ) -> String {
        let tc = &cfg.text_config;

        let mut tokens = vec![tc.bos_token_id];
        tokens.extend(tokenizer.encode(prompt, false).unwrap().get_ids());

        for _ in 0..max {
            let logits = self.forward(&tokens, image_features, &cfg);
            let last = logits.row(logits.dim().0 - 1);

            let next = argmax(last) as u32;

            if next == tc.eos_token_id {
                break;
            }
            tokens.push(next);
        }

        tokenizer.decode(&tokens[1..], false).unwrap()
    }

    // .npy로 미리 생성해둔 token_id로 테스트 하기 위해 생성
    pub fn generate_input_token_id(
        &self,
        token_ids: &[u32],
        image_features: Option<ArrayView2<f32>>,
        tokenizer: &Tokenizer,
        cfg: &Config,
        max: usize,
    ) -> String {
        let tc = &cfg.text_config;

        let mut token: Vec<u32> = token_ids.to_vec();
        for _ in 0..max {
            let logits = self.forward(&token, image_features, &cfg);
            let last = logits.row(logits.dim().0 - 1);

            let next = argmax(last) as u32;

            if next == tc.eos_token_id {
                break;
            }
            token.push(next);
        }

        tokenizer.decode(&token[token_ids.len()..], false).unwrap()
    }

    pub fn forward(
        &self,
        token_ids: &[u32],
        image_features: Option<ArrayView2<f32>>,
        cfg: &Config,
    ) -> Array2<f32> {
        let tc = &cfg.text_config;

        let (mut hidden, ple) = self.prepare_inputs(token_ids, image_features, cfg);

        // cos_sliding, sin_sliding
        let (cos_s, sin_s) = rope_tables(
            token_ids.len(),
            tc.head_dim,
            tc.rope_parameters.sliding_attention.rope_theta,
            1.0,
        );

        // cos_global, sin_global
        let (cos_g, sin_g) = rope_tables(
            token_ids.len(),
            tc.global_head_dim,
            tc.rope_parameters.full_attention.rope_theta,
            0.25,
        );

        let mut kv_sliding: Option<(Array2<f32>, Array2<f32>)> = None;
        let mut kv_full: Option<(Array2<f32>, Array2<f32>)> = None;

        for (i, block) in self.layer.iter().enumerate() {
            let is_sliding = tc.layer_types[i] == "sliding_attention";
            let is_shared = i > 14;

            let kv_share = if is_shared {
                if is_sliding {
                    kv_sliding.as_ref().map(|(k, v)| (k, v))
                } else {
                    kv_full.as_ref().map(|(k, v)| (k, v))
                }
            } else {
                None
            };

            let per_layer_input = ple.slice(s![.., i, ..]);
            let (cos, sin) = if is_sliding {
                (&cos_s, &sin_s)
            } else {
                (&cos_g, &sin_g)
            };

            let (out, k, v) = decoder_block(
                hidden.view(),
                kv_share,
                block,
                per_layer_input,
                cos,
                sin,
                tc,
                is_sliding,
            );

            // KV Share를 사용하지 않는 레이어중 마지막 레이어를 타입별로 저장하여 공유한다.
            if i == 13 {
                kv_sliding = Some((k, v))
            } else if i == 14 {
                kv_full = Some((k, v))
            }

            hidden = out;
        }

        let hidden = rms_norm(hidden.view(), self.norm_f.view(), tc.rms_norm_eps);
        let logits = hidden.dot(&self.embd.t());
        let cap = tc.final_logit_softcapping;
        let logits = logits.mapv(|x| cap * (x / cap).tanh());

        logits
    }

    pub fn prepare_inputs(
        &self,
        token_ids: &[u32],
        image_features: Option<ArrayView2<f32>>,
        cfg: &Config,
    ) -> (Array2<f32>, Array3<f32>) {
        let tc = &cfg.text_config;
        let mut image_idx: Vec<usize> = Vec::new();

        for (i, &token_id) in token_ids.iter().enumerate() {
            if token_id == cfg.image_token_id {
                image_idx.push(i);
            }
        }

        let mut ids: Vec<u32> = token_ids.to_vec();

        for &i in &image_idx {
            ids[i] = tc.pad_token_id;
        }

        let mut hidden = self.embed(&ids);
        match image_features {
            Some(f) => {
                assert_eq!(
                    f.nrows(),
                    image_idx.len(),
                    "image feature rows != image token count"
                );
                assert_eq!(
                    f.ncols(),
                    tc.hidden_size,
                    "image feature dim != hidden_size"
                );

                for (i, &pos) in image_idx.iter().enumerate() {
                    hidden.row_mut(pos).assign(&f.row(i));
                }
            }
            None => {
                assert!(image_idx.is_empty(), "image tokens is not empty")
            }
        }

        let ple = self.prepare_ple(&ids, hidden.view(), tc);

        (hidden, ple)
    }

    pub fn weights_load(path: &str, cfg: &Config) -> Self {
        let file = std::fs::File::open(path).expect("file open failed");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let tensors = SafeTensors::deserialize(&mmap).expect("failed to parse safetensors");

        let mut layer: Vec<Block> = Vec::with_capacity(cfg.text_config.num_hidden_layers);

        for i in 0..cfg.text_config.num_hidden_layers {
            let p = format!("model.language_model.layers.{i}");
            layer.push(Block {
                attn: Attn {
                    attn_q: get_tensor(&tensors, &format!("{p}.self_attn.q_proj.weight")),
                    attn_k: get_tensor(&tensors, &format!("{p}.self_attn.k_proj.weight")),
                    attn_v: get_tensor(&tensors, &format!("{p}.self_attn.v_proj.weight")),
                    attn_o: get_tensor(&tensors, &format!("{p}.self_attn.o_proj.weight")),
                    q_norm: get_tensor(&tensors, &format!("{p}.self_attn.q_norm.weight")),
                    k_norm: get_tensor(&tensors, &format!("{p}.self_attn.k_norm.weight")),
                },
                mlp: Mlp {
                    up_proj: get_tensor(&tensors, &format!("{p}.mlp.up_proj.weight")),
                    gate_proj: get_tensor(&tensors, &format!("{p}.mlp.gate_proj.weight")),
                    down_proj: get_tensor(&tensors, &format!("{p}.mlp.down_proj.weight")),
                },
                norm: Norms {
                    input_norm: get_tensor(&tensors, &format!("{p}.input_layernorm.weight")),
                    post_attn_norm: get_tensor(
                        &tensors,
                        &format!("{p}.post_attention_layernorm.weight"),
                    ),
                    pre_ffn_norm: get_tensor(
                        &tensors,
                        &format!("{p}.pre_feedforward_layernorm.weight"),
                    ),
                    post_ffn_norm: get_tensor(
                        &tensors,
                        &format!("{p}.post_feedforward_layernorm.weight"),
                    ),
                },
                ple: PleLayer {
                    projection: get_tensor(&tensors, &format!("{p}.per_layer_projection.weight")),
                    input_gate: get_tensor(&tensors, &format!("{p}.per_layer_input_gate.weight")),
                    post_norm: get_tensor(
                        &tensors,
                        &format!("{p}.post_per_layer_input_norm.weight"),
                    ),
                    scalar: Self::get_scalar(&tensors, &format!("{p}.layer_scalar")),
                },
            });
        }

        // PLE packed 테이블: 상주 안 하고 mmap 좌표만 기록
        let ple_table_offset = {
            let t = tensors
                .tensor("model.language_model.embed_tokens_per_layer.weight")
                .expect("ple table not found");
            let offset = t.data().as_ptr() as usize - mmap.as_ptr() as usize;
            (offset, t.shape()[0], t.shape()[1])
        };

        let vision = VisionWeights::load(&tensors, &cfg.vision_config);

        Weights {
            embd: get_tensor(&tensors, "model.language_model.embed_tokens.weight"),
            norm_f: get_tensor(&tensors, "model.language_model.norm.weight"),
            ple_model_proj: get_tensor(
                &tensors,
                "model.language_model.per_layer_model_projection.weight",
            ),
            ple_proj_norm: get_tensor(
                &tensors,
                "model.language_model.per_layer_projection_norm.weight",
            ),
            ple_table_offset,
            layer,
            mmap,
            vision,
        }
    }

    fn embed(&self, token_ids: &[u32]) -> Array2<f32> {
        let hidden_size = self.embd.dim().1;
        let scale = (hidden_size as f32).sqrt();

        let mut out = Array2::zeros((token_ids.len(), hidden_size));

        for (i, &token_id) in token_ids.iter().enumerate() {
            let row = self.embd.row(token_id as usize);

            for (o, &e) in out.row_mut(i).iter_mut().zip(row.iter()) {
                *o = e * scale;
            }
        }

        out
    }

    // get_tensor 하나로 처리하기 위함
    // SafeTensors를 포함하는 Struct를 생성하여 get_tensor 함수의 위치를 변경하는 것을 고려
    // fn get_tensor<const N: usize>(tensors: &SafeTensors, name: &str) -> Array<f32, Dim<[Ix; N]>> {
    //     let t = tensors
    //         .tensor(name)
    //         .unwrap_or_else(|_| panic!("get_tensor1 failed: {name}"));
    //     let s = t.shape();
    //     let d = Array::from_shape_vec(s,Self::to_f32(t.data())).unwrap();
    //     d.into_dimensionality::<Dim<[Ix; N]>>().unwrap()
    // }

    fn get_scalar(tensors: &SafeTensors, name: &str) -> f32 {
        let t = tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("get_scalar failed: {name}"));
        to_f32(t.data())[0]
    }

    fn ple_token_identity(&self, token_ids: &[u32], cfg: &TextConfig) -> Array3<f32> {
        let num_layers = cfg.num_hidden_layers;
        let ple_dim = cfg.hidden_size_per_layer_input;
        let scale = (ple_dim as f32).sqrt(); // 256.sqrt()

        let mut out: Array3<f32> = Array3::zeros((token_ids.len(), num_layers, ple_dim));

        for (i, &token_id) in token_ids.iter().enumerate() {
            let row_byte = num_layers * ple_dim * 2; // bf16이기 때문에 *2
            let start = self.ple_table_offset.0 + row_byte * (token_id as usize);
            let end = start + row_byte;

            let byte = &self.mmap[start..end];
            let value = to_f32(byte);

            for layer in 0..num_layers {
                for dim in 0..ple_dim {
                    let flat_idx = ple_dim * layer + dim;
                    out[[i, layer, dim]] = value[flat_idx] * scale;
                }
            }
        }

        out
    }

    fn prepare_ple(
        &self,
        token_ids: &[u32],
        embed: ArrayView2<f32>,
        cfg: &TextConfig,
    ) -> Array3<f32> {
        let identity = self.ple_token_identity(token_ids, cfg);

        let token_len = token_ids.len();
        let num_layers = cfg.num_hidden_layers;
        let ple_dim = cfg.hidden_size_per_layer_input;

        let mut out = Array3::zeros((token_len, num_layers, ple_dim));
        let proj = embed.dot(&self.ple_model_proj.t());
        let scaled = proj * (1.0 / (cfg.hidden_size as f32).sqrt());

        let ple_scale = 1.0 / 2.0_f32.sqrt();

        for i in 0..token_len {
            for layer in 0..num_layers {
                let start = layer * ple_dim;
                let l = scaled.slice(s![i, start..start + ple_dim]);
                let mean_sq = l.iter().map(|&x| x * x).sum::<f32>() / ple_dim as f32;
                let rms = (mean_sq + cfg.rms_norm_eps).sqrt();

                for dim in 0..ple_dim {
                    let raw = scaled[(i, layer * ple_dim + dim)];
                    let normed = raw / rms * self.ple_proj_norm[dim];
                    out[[i, layer, dim]] = (normed + identity[[i, layer, dim]]) * ple_scale;
                }
            }
        }

        out
    }
}
