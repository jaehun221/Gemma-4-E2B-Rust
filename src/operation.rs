use ndarray::{ Array2, Axis, ArrayView1, ArrayView2 };


pub fn rms_norm(x: ArrayView2<f32>, w: ArrayView1<f32>, eps: f32) -> Array2<f32> {
    let mut out = Array2::zeros(x.dim()); // x.dim(): [token수, 가중치 수]

    for (i, row) in x.axis_iter(Axis(0)).enumerate() {
        let n = row.len() as f32;

        // 각 토큰의 제곱 평균
        let mean_sq = row.iter().map(|&v| v * v).sum::<f32>() / n;

        let rms = (mean_sq + eps).sqrt();
         
        // out : 최종 반환할 Array, row : 입력값, w : 가중치 
        for ((o, &xi), &wi) in out.row_mut(i).iter_mut().zip(row.iter()).zip(w.iter()) {
            *o = (xi/rms) * wi
        }
    }

    out
}


// RoPE에 필요한 cos, sin Array를 생성
fn rope_tables(seq_len: usize, head_dim: usize, base: f32) -> (Array2<f32>, Array2<f32>) {
    let half = head_dim / 2;

    // 결과 테이블: head_dim 전체 크기
    let mut cos_table = Array2::zeros((seq_len, head_dim));
    let mut sin_table = Array2::zeros((seq_len, head_dim));

    for pos in 0..seq_len {
        for i in 0..half {
            // 속도 θ_i = base^(-2i/head_dim)
            let theta = base.powf(-2.0 * i as f32 / head_dim as f32);
            // 각도 = 위치 × 속도
            let angle = pos as f32 * theta;

            let (s, c) = angle.sin_cos();

            // 앞 절반(i)과 뒤 절반(i+half)에 같은 값
            cos_table[[pos, i]] = c;
            cos_table[[pos, i + half]] = c;
            sin_table[[pos, i]] = s;
            sin_table[[pos, i + half]] = s;
        }
    }
    (cos_table, sin_table)
}

fn apply_rope(q: &mut Array2<f32>, cos_table: Array2<f32>, sin_table: Array2<f32>) {
    let head_dim = q.dim().1;
    let half= head_dim / 2;

    for pos in 0..q.dim().0 {
        let row: Vec<f32> = q.row(pos).to_vec();

        for j in 0..head_dim { 
            let cos = cos_table[[pos, j]];
            let sin = sin_table[[pos, j]];

            let rotated = if j < half {
                -row[j + half]
            } else {
                row[j - half]
            };

            q[[pos, j]] = row[j] * cos + rotated * sin;
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


// fn attention() -> Array2<f32> {

// }