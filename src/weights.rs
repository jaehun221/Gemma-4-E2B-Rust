use memmap2::Mmap;
use ndarray::{Array1, Array2, Array3, ArrayView2, s};
use safetensors::SafeTensors;

use crate::{
    config::TextConfig,
    operation::{decoder_block, rms_norm, rope_tables},
};

pub struct Weights {
    embd: Array2<f32>,
    layer: Vec<Block>,
    norm_f: Array1<f32>,
    mmap: Mmap,
    ple_table_offset: (usize, usize, usize),
    ple_model_proj: Array2<f32>,
    ple_proj_norm: Array1<f32>,
    rms_eps: f32,
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
    pub q_norm: Array1<f32>,
    pub attn_o: Array2<f32>,
    pub attn_v: Array2<f32>,
    pub attn_k: Array2<f32>,
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
    pub fn debug_layer_15(&self, cfg: &TextConfig) {
        let token_ids = [2u32, 100, 500];
        let embeds = self.embed(&token_ids);
        let ple = self.prepare_ple(&token_ids, embeds.view(), cfg);
        let (cos_s, sin_s) = rope_tables(token_ids.len(), 256, 10000.0);
        let (cos_g, sin_g) = rope_tables(token_ids.len(), 512, 1000000.0);

        let mut hidden = embeds;
        for i in 0..=15 {   // 0~15까지
            let block = &self.layer[i];
            let is_sliding = cfg.layer_types[i] == "sliding_attention";
            let ple_i = ple.slice(s![.., i, ..]);
            let (cos, sin) = if is_sliding { (&cos_s, &sin_s) } else { (&cos_g, &sin_g) };
            hidden = decoder_block(hidden.view(), block, ple_i, cos, sin, cfg, is_sliding);

            if i == 14 {   // 레이어 14 출력
                println!("layer14 [0,:8]:");
                for d in 0..8 { print!("{:.5} ", hidden[[0, d]]); }
                println!();
            }
        }
        // 레이어 15 출력
        println!("layer15 [0,:8]:");
        for d in 0..8 { print!("{:.5} ", hidden[[0, d]]); }
        println!();
    }

    pub fn debug_layer4(&self, cfg: &TextConfig) {
        let token_ids = [2u32, 100, 500];

        let embeds = self.embed(&token_ids);
        let ple = self.prepare_ple(&token_ids, embeds.view(), cfg);

        let (cos_s, sin_s) = rope_tables(token_ids.len(), 256, 10000.0);
        let (cos_g, sin_g) = rope_tables(token_ids.len(), 512, 1000000.0);

        let mut hidden = embeds;

        for i in 0..=4 {   // 레이어 0~4까지만
            let block = &self.layer[i];
            let is_sliding = cfg.layer_types[i] == "sliding_attention";
            let ple_i = ple.slice(s![.., i, ..]);
            let (cos, sin) = if is_sliding {
                (&cos_s, &sin_s)
            } else {
                (&cos_g, &sin_g)
            };
            hidden = decoder_block(hidden.view(), block, ple_i, cos, sin, cfg, is_sliding);
        }

        // 레이어 4 출력
        println!("layer4 out [0, :8]:");
        for d in 0..8 { print!("{:.5} ", hidden[[0, d]]); }
        println!();
    }

    pub fn forward(&self, token_ids: &[u32], cfg: &TextConfig) -> Array2<f32> {
        let mut hidden = self.embed(token_ids);
        let ple = self.prepare_ple(token_ids, hidden.view(), cfg);

        // cos_sliding, sin_sliding
        let (cos_s, sin_s) = rope_tables(token_ids.len(), cfg.head_dim, 10000.0);

        // cos_global, sin_global
        let (cos_g, sin_g) = rope_tables(token_ids.len(), cfg.head_dim*2, 1000000.0);

        for (i, block) in self.layer.iter().enumerate() {
            let is_sliding = cfg.layer_types[i] == "sliding_attention";

            let per_layer_input = ple.slice(s![.., i, ..]);
            let (cos, sin) = if is_sliding {
                (&cos_s, &sin_s)
            } else {
                (&cos_g, &sin_g)
            };
            hidden = decoder_block(
                hidden.view(),
                block,
                per_layer_input,
                cos,
                sin,
                cfg,
                is_sliding,
            );
        }

        let hidden = rms_norm(hidden.view(), self.norm_f.view(), cfg.rms_norm_eps);

        let logits = hidden.dot(&self.embd.t());

        let logits = logits.mapv(|x| 30.0 * (x / 30.0).tanh());

        logits
    }

    const N_LAYERS: usize = 35;
    const RMS_EPS: f32 = 1e-6; // config.json에서 확인 후 확정

    pub fn debug_shapes(&self) {
        println!("embd: {:?}", self.embd.dim());
        println!("layers: {}", self.layer.len());
        println!("layer0 q_proj: {:?}", self.layer[0].attn.attn_q.dim());
        println!("layer4 q_proj: {:?}", self.layer[4].attn.attn_q.dim());
        println!("ple table: {:?}", self.ple_table_offset);
    }

    pub fn weights_load(path: &str) -> Self {
        let file = std::fs::File::open(path).expect("file open failed");
        let mmap = unsafe { Mmap::map(&file).expect("mmap failed") };
        let tensors = SafeTensors::deserialize(&mmap).expect("failed to parse safetensors");

        let mut layer: Vec<Block> = Vec::with_capacity(Self::N_LAYERS);

        for i in 0..Self::N_LAYERS {
            let p = format!("model.language_model.layers.{i}");
            layer.push(Block {
                attn: Attn {
                    attn_q: Self::get_tensor2(&tensors, &format!("{p}.self_attn.q_proj.weight")),
                    attn_k: Self::get_tensor2(&tensors, &format!("{p}.self_attn.k_proj.weight")),
                    attn_v: Self::get_tensor2(&tensors, &format!("{p}.self_attn.v_proj.weight")),
                    attn_o: Self::get_tensor2(&tensors, &format!("{p}.self_attn.o_proj.weight")),
                    q_norm: Self::get_tensor1(&tensors, &format!("{p}.self_attn.q_norm.weight")),
                    k_norm: Self::get_tensor1(&tensors, &format!("{p}.self_attn.k_norm.weight")),
                },
                mlp: Mlp {
                    up_proj: Self::get_tensor2(&tensors, &format!("{p}.mlp.up_proj.weight")),
                    gate_proj: Self::get_tensor2(&tensors, &format!("{p}.mlp.gate_proj.weight")),
                    down_proj: Self::get_tensor2(&tensors, &format!("{p}.mlp.down_proj.weight")),
                },
                norm: Norms {
                    input_norm: Self::get_tensor1(&tensors, &format!("{p}.input_layernorm.weight")),
                    post_attn_norm: Self::get_tensor1(
                        &tensors,
                        &format!("{p}.post_attention_layernorm.weight"),
                    ),
                    pre_ffn_norm: Self::get_tensor1(
                        &tensors,
                        &format!("{p}.pre_feedforward_layernorm.weight"),
                    ),
                    post_ffn_norm: Self::get_tensor1(
                        &tensors,
                        &format!("{p}.post_feedforward_layernorm.weight"),
                    ),
                },
                ple: PleLayer {
                    projection: Self::get_tensor2(
                        &tensors,
                        &format!("{p}.per_layer_projection.weight"),
                    ),
                    input_gate: Self::get_tensor2(
                        &tensors,
                        &format!("{p}.per_layer_input_gate.weight"),
                    ),
                    post_norm: Self::get_tensor1(
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

        Weights {
            embd: Self::get_tensor2(&tensors, "model.language_model.embed_tokens.weight"),
            norm_f: Self::get_tensor1(&tensors, "model.language_model.norm.weight"),
            ple_model_proj: Self::get_tensor2(
                &tensors,
                "model.language_model.per_layer_model_projection.weight",
            ),
            ple_proj_norm: Self::get_tensor1(
                &tensors,
                "model.language_model.per_layer_projection_norm.weight",
            ),
            ple_table_offset,
            layer,
            mmap,
            rms_eps: Self::RMS_EPS,
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

    fn get_tensor1(tensors: &SafeTensors, name: &str) -> Array1<f32> {
        let t = tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("get_tensor1 failed: {name}"));
        Array1::from_vec(Self::to_f32(t.data()))
    }

    fn get_tensor2(tensors: &SafeTensors, name: &str) -> Array2<f32> {
        let t = tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("get_tensor2 failed: {name}"));
        let s = t.shape();
        Array2::from_shape_vec((s[0], s[1]), Self::to_f32(t.data())).expect("Vec -> Array2 failed")
    }

    fn get_scalar(tensors: &SafeTensors, name: &str) -> f32 {
        let t = tensors
            .tensor(name)
            .unwrap_or_else(|_| panic!("get_scalar failed: {name}"));
        Self::to_f32(t.data())[0]
    }

    fn to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2)
            .map(|b| {
                let bits = u16::from_le_bytes([b[0], b[1]]);
                f32::from_bits((bits as u32) << 16)
            })
            .collect()
    }

    // 검증용
    pub fn debug_gain(&self) {
        let gain = &self.layer[0].norm.input_norm;
        for v in gain.iter().take(8) {
            print!("{} ", v);
        }
        println!();
        println!("평균: {}", gain.mean().unwrap());
    }

    // 검증용
    pub fn debug_mlp(&self) {
        let mut x = Array2::<f32>::zeros((2, 1536));

        x[[0, 0]] = 1.0;
        x[[0, 1]] = 0.5;
        x[[0, 2]] = -0.3;
        x[[1, 0]] = -1.0;
        x[[1, 1]] = 2.0;

        let mlp_w = &self.layer[0].mlp;

        let out = crate::operation::mlp(
            x.view(),
            mlp_w.gate_proj.view(),
            mlp_w.up_proj.view(),
            mlp_w.down_proj.view(),
        );

        println!("MLP 출력 첫 행 앞 8개:");
        for v in out.row(0).iter().take(8) {
            print!("{:.5} ", v);
        }
        println!();

        println!("MLP 출력 둘째 행 앞 8개:");
        for v in out.row(1).iter().take(8) {
            print!("{:.5} ", v);
        }
        println!();
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
            let value = Self::to_f32(byte);

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

    pub fn debug_ple(&self, cfg: &TextConfig) {
        let token_ids = [2u32, 100, 500]; // 파이썬과 동일

        let mut embeds = Array2::<f32>::zeros((3, 1536));
        embeds[[0, 0]] = 1.0;
        embeds[[0, 1]] = 0.5;
        embeds[[1, 0]] = -1.0;
        embeds[[2, 0]] = 2.0;

        // 1. identity만
        let identity = self.ple_token_identity(&token_ids, cfg);
        println!("identity [0,0,:8]:");
        for d in 0..8 {
            print!("{:.5} ", identity[[0, 0, d]]);
        }
        println!();

        // 3. 최종
        let ple = self.prepare_ple(&token_ids, embeds.view(), cfg);
        println!("최종 ple [0,0,:8]:");
        for d in 0..8 {
            print!("{:.5} ", ple[[0, 0, d]]);
        }
        println!();
    }
}
