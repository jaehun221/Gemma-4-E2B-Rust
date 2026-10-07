mod config;
mod loader;
mod operation;
mod vision;
mod weights;

use std::iter::zip;

use config::Config;
use ndarray::{Array2, Array3, Array4, ArrayView, Axis, Dimension};
use ndarray_npy::read_npy;
use tokenizers::Tokenizer;
use weights::Weights;

fn main() {
    // TODO image 전처리 및 VIT

    // image 처리를 아직 구현하지 않았으므로 python으로 구해둔 .npy값을 임시로 사용
    let cfg = Config::load_config("gemma-4-e2b/config.json");
    let w = Weights::weights_load("gemma-4-e2b/model.safetensors", &cfg);

    let tokenizer = Tokenizer::from_file("gemma-4-e2b/tokenizer.json").unwrap();

    let i_npy: Array2<i64> = read_npy("ref/input_ids.npy").expect("read failed");
    let i_npy_u32: Vec<u32> = i_npy.iter().map(|&x| x as u32).collect();

    let image_features: Array2<f32> = read_npy("ref/image_features.npy").unwrap();
    let inputs_embeds_npy: Array2<f32> = read_npy::<_, Array3<f32>>("ref/inputs_embeds.npy")
        .unwrap()
        .remove_axis(Axis(0));
    let per_layer_inputs_npy: Array3<f32> = read_npy::<_, Array4<f32>>("ref/per_layer_inputs.npy")
        .unwrap()
        .remove_axis(Axis(0));
    let logits_npy: Array2<f32> = read_npy::<_, Array3<f32>>("ref/logits.npy")
        .unwrap()
        .remove_axis(Axis(0));

    // python에서 image를 포함한 logits이 Rust Decoder에서 정상적으로 연산되는지 검증
    let (hidden, ple) = w.prepare_inputs(&i_npy_u32, Some(image_features.view()), &cfg);
    let logits = w.forward(&i_npy_u32, Some(image_features.view()), &cfg);

    // python 라이브러리로 구한 값과 Rust로 직접 구현한 값이 일치하는지 검증
    // println!(
    //     "hidden: {:e}",
    //     max_diff(hidden.view(), inputs_embeds_npy.view())
    // );
    // println!(
    //     "ple: {:e}",
    //     max_diff(ple.view(), per_layer_inputs_npy.view())
    // );
    // println!(
    //     "logits: {:e}",
    //     max_diff(
    //         logits.row(logits.nrows() - 1),
    //         logits_npy.row(logits_npy.nrows() - 1)
    //     )
    // );


    let output = w.generate("The capital of France is", None, &tokenizer, &cfg, 20);
    println!("{}", output);

    let output = w.generate_input_token_id(
        &i_npy_u32,
        Some(image_features.view()),
        &tokenizer,
        &cfg,
        20,
    );
    println!("{}", output);
}

// 두 Array 요소별 차를 절댓값으로 변환해 가장 큰 값을 반환한다. 두 Array가 일치하는지 비교
fn max_diff<D: Dimension>(arr1: ArrayView<f32, D>, arr2: ArrayView<f32, D>) -> f32 {
    assert_eq!(arr1.shape(), arr2.shape(), "shape does not match");

    let mut max = 0.0;
    for (a1, a2) in zip(arr1, arr2) {
        if a1.is_nan() || a2.is_nan() {
            panic!("NaN found");
        }
        let a = (a1 - a2).abs();
        if a > max {
            max = a;
        }
    }

    max
}
