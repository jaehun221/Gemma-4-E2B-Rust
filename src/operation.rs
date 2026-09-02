use ndarray::{ Array2, Axis, ArrayView1, ArrayView2, s, concatenate};
use crate::{config::TextConfig, weights::{ Attn, Block }};

pub fn rms_norm(x: ArrayView2<f32>, w: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let mut out = Array2::zeros(x.dim()); // x.dim(): [token수, 가중치 수]

    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let n = row.len() as f32;

        let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / n;

        let rms = (mean_sq + eps).sqrt();
         
        for ((o, &xi), &wi) in out.row_mut(i).iter_mut().zip(row.iter()).zip(w.iter()) {
            *o = (xi/rms) * wi
        }
    }

    out
}


fn rope_tables(seq_len: usize, head_dim: usize, base: f32) -> (Array2<f32>, Array2<f32>) {
    let half = head_dim / 2;

    let mut cos_table = Array2::zeros((seq_len, half));
    let mut sin_table = Array2::zeros((seq_len, half));

    for pos in 0..seq_len {
        for i in 0..half {

            let theta = base.powf(-2.0 * i as f32 / head_dim as f32);
            let angle = pos as f32 * theta;
            let (s, c) = angle.sin_cos();

            cos_table[[pos, i]] = c;
            sin_table[[pos, i]] = s;
        }
    }
    (cos_table, sin_table)
}


fn apply_rope(q: &mut Array2<f32>, cos_table: &Array2<f32>, sin_table: &Array2<f32>) {
    let half = q.dim().1 / 2;

    for pos in 0..q.dim().0 {
        for i in 0..half {
            let x = q[[pos, i]];
            let y = q[[pos, i + half]];

            let cos = cos_table[[pos, i]];
            let sin = sin_table[[pos, i]];

            q[[pos, i]]        = x * cos - y * sin;
            q[[pos, i + half]] = x * sin + y * cos;
        }
    }
}

 
pub fn mlp(x: ArrayView2<f32>, gate_proj: ArrayView2<f32>, up_proj: ArrayView2<f32>, down_proj: ArrayView2<f32>) -> Array2<f32> {
    let gate = x.dot(&gate_proj.t());
    let up = x.dot(&up_proj.t());

    let mut hidden = gate.mapv(gelu);
    hidden = hidden * up;

    hidden.dot(&down_proj.t())
}


fn gelu(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())
    
}


fn attention(x: ArrayView2<f32>, attn: &Attn, cos_table: &Array2<f32>, sin_table: &Array2<f32>, cfg: &TextConfig, is_sliding: bool, ) -> Array2<f32> {
    let q = x.dot(&attn.attn_q.t());
    let mut k = x.dot(&attn.attn_k.t());
    let v = x.dot(&attn.attn_v.t());

    k = rms_norm(k.view(), attn.k_norm.view(), cfg.rms_norm_eps);
    apply_rope(&mut k, cos_table, sin_table);

    let mut head_out: Vec<Array2<f32>> = Vec::new();
    let head_dim = cfg.head_dim;

    for head_idx in 0..cfg.num_attention_heads {
        let q_head = q.slice(s![.., head_idx*head_dim..head_idx*head_dim+head_dim]).to_owned();
        let mut q_head_norm = rms_norm(q_head.view(), attn.q_norm.view(), cfg.rms_norm_eps);
        apply_rope(&mut q_head_norm, cos_table, sin_table);

        // score [T, T]
        let mut score = q_head_norm.dot(&k.t()) / (head_dim as f32).sqrt();
        
        masking(&mut score, is_sliding, cfg.sliding_window);
        softmax(&mut score);

        let head_out_v = score.dot(&v);
        head_out.push(head_out_v);
    }


    let head_out: Vec<_> = head_out.iter().map(|a| a.view()).collect();
    let concat = concatenate(Axis(1), &head_out).unwrap();

    let out = concat.dot(&attn.attn_o.t());

    out
}

fn masking(score: &mut Array2<f32>, is_sliding: bool, sliding_window: usize) {
    let n = score.dim().0;

    for i in 0..n {
        for j in 0..n {
            
            if j > i {
                score[[i, j]] = f32::NEG_INFINITY;
            }

            if is_sliding && i - j > sliding_window {
                score[[i, j]] = f32::NEG_INFINITY;
            }
        }
    }
}

fn softmax(x: &mut Array2<f32>) {
    for mut row in x.axis_iter_mut(Axis(0)) {
        let max_vlaue = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        let mut sum = 0.0;

        for i in row.iter_mut() {
            *i = (*i - max_vlaue).exp();
            sum += *i;
        }

        for i in row.iter_mut() {
            *i /= sum;
        }

    }
}

// 반복되는 Layer 내부 연산 구현
pub fn decoder_blcok(x: ArrayView2<f32>, block: &Block, per_layer_input: ArrayView2<f32>, cos_table: &Array2<f32>, sin_table: &Array2<f32>, cfg: &TextConfig, is_sliding: bool,) -> Array2<f32> {

    // Attention
    let residual = x.to_owned();
    let h = rms_norm(x, block.norm.input_norm.view(), cfg.rms_norm_eps);
    let h = attention(h.view(), &block.attn, cos_table, sin_table, cfg, is_sliding);
    let h = rms_norm(h.view(), block.norm.post_attn_norm.view(), cfg.rms_norm_eps);
    let h = residual + h;

    // MLP
    let residual = h.clone();
    let h = rms_norm(h.view(), block.norm.pre_ffn_norm.view(), cfg.rms_norm_eps);
    let h = mlp(h.view(), block.mlp.gate_proj.view(), block.mlp.up_proj.view(), block.mlp.down_proj.view());
    let h = rms_norm(h.view(), block.norm.post_ffn_norm.view(), cfg.rms_norm_eps);
    let h = residual + h;

    // Per Layer Embedding
    let residual = h.clone();
    let mut h = residual.dot(&block.ple.input_gate.t());
    h = h.mapv(gelu);
    h = h * per_layer_input;
    h = h.dot(&block.ple.projection.t());
    h = rms_norm(h.view(), block.ple.post_norm.view(), cfg.rms_norm_eps);
    h = residual + h;


    h * block.ple.scalar
}

// TODO: Attention, PLE, KVCache, tokenizer 구현