mod weights;
mod config;
use std::println;

use weights::Weights;
use config::Config;

fn main() {
    // let w = Weights::weights_load("gemma-4-e2b/model.safetensors");
    // w.debug_shapes();

    let path = "gemma-4-e2b/config.json";
    println!("{:?}", Config::load(&path));
}