mod weights;
mod config;
mod operation;   

use weights::Weights;
use config::Config;


fn main() {
    let w = Weights::weights_load("gemma-4-e2b/model.safetensors");

}