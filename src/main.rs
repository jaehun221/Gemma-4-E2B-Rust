mod weights;
mod config;
mod operation;   

use weights::Weights;
use config::Config;
use operation::decoder_blcok;

fn main() {
    let w = Weights::weights_load("gemma-4-e2b/model.safetensors");
    let cfg = Config::load("gemma-4-e2b/config.json");

    w.debug_ple(&cfg.text_config);
}