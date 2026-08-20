use ndarray::{Array2, Array1};
use memmap2::Mmap;
use safetensors::SafeTensors;

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
    attn: Attn,
    mlp: Mlp,
    norm: Norms,
    ple: PleLayer,
}

struct PleLayer {
    projection: Array2<f32>,
    input_gate: Array2<f32>,
    post_norm: Array1<f32>,
    scalar: f32,
}

struct Attn {
    attn_q: Array2<f32>,
    q_norm: Array1<f32>,
    attn_o: Array2<f32>,
    attn_v: Array2<f32>,
    attn_k: Array2<f32>,
    k_norm: Array1<f32>,
    
}

struct Norms {
    input_norm: Array1<f32>,
    post_attn_norm: Array1<f32>,
    pre_ffn_norm: Array1<f32>,
    post_ffn_norm: Array1<f32>,
}

struct Mlp {
    up_proj: Array2<f32>,
    gate_proj: Array2<f32>,
    down_proj: Array2<f32>,
}

impl Weights {
    const N_LAYERS: usize = 35;
    const RMS_EPS: f32 = 1e-6;  // config.json에서 확인 후 확정

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
                    up_proj:   Self::get_tensor2(&tensors, &format!("{p}.mlp.up_proj.weight")),
                    gate_proj: Self::get_tensor2(&tensors, &format!("{p}.mlp.gate_proj.weight")),
                    down_proj: Self::get_tensor2(&tensors, &format!("{p}.mlp.down_proj.weight")),
                },
                norm: Norms {
                    input_norm:     Self::get_tensor1(&tensors, &format!("{p}.input_layernorm.weight")),
                    post_attn_norm: Self::get_tensor1(&tensors, &format!("{p}.post_attention_layernorm.weight")),
                    pre_ffn_norm:   Self::get_tensor1(&tensors, &format!("{p}.pre_feedforward_layernorm.weight")),
                    post_ffn_norm:  Self::get_tensor1(&tensors, &format!("{p}.post_feedforward_layernorm.weight")),
                },
                ple: PleLayer {
                    projection: Self::get_tensor2(&tensors, &format!("{p}.per_layer_projection.weight")),
                    input_gate: Self::get_tensor2(&tensors, &format!("{p}.per_layer_input_gate.weight")),
                    post_norm:  Self::get_tensor1(&tensors, &format!("{p}.post_per_layer_input_norm.weight")),
                    scalar:     Self::get_scalar(&tensors, &format!("{p}.layer_scalar")),
                },
            });
        }

        // PLE packed 테이블: 상주 안 하고 mmap 좌표만 기록
        let ple_table_offset = {
            let t = tensors.tensor("model.language_model.embed_tokens_per_layer.weight")
                .expect("ple table not found");
            let offset = t.data().as_ptr() as usize - mmap.as_ptr() as usize;
            (offset, t.shape()[0], t.shape()[1])
        };

        Weights {
            embd: Self::get_tensor2(&tensors, "model.language_model.embed_tokens.weight"),
            norm_f: Self::get_tensor1(&tensors, "model.language_model.norm.weight"),
            ple_model_proj: Self::get_tensor2(&tensors, "model.language_model.per_layer_model_projection.weight"),
            ple_proj_norm: Self::get_tensor1(&tensors, "model.language_model.per_layer_projection_norm.weight"),
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
        let t = tensors.tensor(name).unwrap_or_else(|_| panic!("get_tensor1 failed: {name}"));
        Array1::from_vec(Self::to_f32(t.data()))
    }

    fn get_tensor2(tensors: &SafeTensors, name: &str) -> Array2<f32> {
        let t = tensors.tensor(name).unwrap_or_else(|_| panic!("get_tensor2 failed: {name}"));
        let s = t.shape();
        Array2::from_shape_vec((s[0], s[1]), Self::to_f32(t.data()))
            .expect("Vec -> Array2 failed")
    }

    fn get_scalar(tensors: &SafeTensors, name: &str) -> f32 {
        let t = tensors.tensor(name).unwrap_or_else(|_| panic!("get_scalar failed: {name}"));
        Self::to_f32(t.data())[0]
    }

    fn to_f32(data: &[u8]) -> Vec<f32> {
        data.chunks_exact(2).map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        }).collect()
    }

    pub fn debug_gain(&self) {
        let gain = &self.layer[0].norm.input_norm;
        for v in gain.iter().take(8) {
            print!("{} ", v);
        }
        println!();
        println!("평균: {}", gain.mean().unwrap());
    }
}