use ndarray::{Array, Dimension};
use safetensors::SafeTensors;

pub fn to_f32(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(2)
        .map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            f32::from_bits((bits as u32) << 16)
        })
        .collect()
}

pub fn get_tensor<D: Dimension>(tensors: &SafeTensors, name: &str) -> Array<f32, D> {
    let t = tensors
        .tensor(name)
        .unwrap_or_else(|_| panic!("tensor not found: {name}"));
    let s = t.shape();
    let d = Array::from_shape_vec(s, to_f32(t.data())).unwrap();
    
    d.into_dimensionality::<D>()
        .unwrap_or_else(|_| panic!("dimension mismatch {name}, shape: {s:?}"))
}
