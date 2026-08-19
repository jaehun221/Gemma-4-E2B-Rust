use ndarray::{ ArrayView1, Axis, ArrayView1, ArrayView2 };


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


fn attention() {}

// fn rope(q: &mut Array2<f32>, cos_talbe: Array2<f32>, sin_talbe: Array2<f32>) -> Array2<f32> {
//     let head_dim = q.dim().1;
//     let half= head_dim / 2;

//     for p in 0..q.dim().0 {
//         let row: Vec<f32> = q.row(pos).to_vec();

//         for j in 0..head_dim {
//             let cos = cos_table[[pos, j]];
//             let sin = sin_table[[pos, j]];

//             let rotated = 
//         }
//     }

//     // 
// }

fn mlp() {}