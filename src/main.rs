mod config;
mod operation;
mod weights;

use std::println;

use config::Config;
use tokenizers::Tokenizer;
use weights::Weights;

fn main() {
    let w = Weights::weights_load("gemma-4-e2b/model.safetensors");
    let cfg = Config::load("gemma-4-e2b/config.json");
    let tokenizer = Tokenizer::from_file("gemma-4-e2b/tokenizer.json").unwrap();

    let output = w.generate("What is your", &tokenizer, &cfg.text_config, 100);
    println!("{}", output);
}
