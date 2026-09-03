mod config;
mod operation;
mod weights;

use config::Config;
use weights::Weights;

fn main() {
    let w = Weights::weights_load("gemma-4-e2b/model.safetensors");
    let cfg = Config::load("gemma-4-e2b/config.json");

    w.debug_layer_15(&cfg.text_config);
}
