use std::path::Path;

use safetensors::tensor::TensorView;

pub const D_MODEL: usize = 64;
pub const D_STATE: usize = 16;
pub const D_CONV: usize = 4;
pub const EXPAND: usize = 2;
pub const N_LAYERS: usize = 2;
pub const VOCAB_SIZE: usize = 256;

fn d_inner() -> usize {
    D_MODEL * EXPAND
}

fn dt_rank() -> usize {
    D_MODEL.div_ceil(16)
}

fn xdbl_dim() -> usize {
    dt_rank() + 2 * D_STATE
}

fn simple_rng(seed: u64) -> impl FnMut() -> f32 {
    let mut state = seed.max(1);
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32 * 0.1
    }
}

fn rand_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = simple_rng(seed);
    (0..n).map(|_| rng()).collect()
}

fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn write_synthetic_checkpoint(dir: &Path, include_lm_head: bool) {
    let di = d_inner();
    let dr = dt_rank();
    let xd = xdbl_dim();

    let mut tensors: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::new();

    let embed = rand_vec(VOCAB_SIZE * D_MODEL, 100);
    tensors.push((
        "backbone.embeddings.weight".into(),
        f32_to_bytes(&embed),
        vec![VOCAB_SIZE, D_MODEL],
    ));

    if include_lm_head {
        let lm = rand_vec(VOCAB_SIZE * D_MODEL, 200);
        tensors.push((
            "lm_head.weight".into(),
            f32_to_bytes(&lm),
            vec![VOCAB_SIZE, D_MODEL],
        ));
    }

    tensors.push((
        "backbone.norm_f.weight".into(),
        f32_to_bytes(&vec![1.0f32; D_MODEL]),
        vec![D_MODEL],
    ));

    for i in 0..N_LAYERS {
        let seed_base = (i as u64 + 1) * 1000;
        let prefix = format!("backbone.layers.{i}");
        tensors.push((
            format!("{prefix}.norm.weight"),
            f32_to_bytes(&vec![1.0f32; D_MODEL]),
            vec![D_MODEL],
        ));
        tensors.push((
            format!("{prefix}.mixer.in_proj.weight"),
            f32_to_bytes(&rand_vec(2 * di * D_MODEL, seed_base + 1)),
            vec![2 * di, D_MODEL],
        ));
        tensors.push((
            format!("{prefix}.mixer.conv1d.weight"),
            f32_to_bytes(&rand_vec(di * D_CONV, seed_base + 2)),
            vec![di, 1, D_CONV],
        ));
        tensors.push((
            format!("{prefix}.mixer.conv1d.bias"),
            f32_to_bytes(&vec![0.0f32; di]),
            vec![di],
        ));
        tensors.push((
            format!("{prefix}.mixer.x_proj.weight"),
            f32_to_bytes(&rand_vec(xd * di, seed_base + 3)),
            vec![xd, di],
        ));
        tensors.push((
            format!("{prefix}.mixer.dt_proj.weight"),
            f32_to_bytes(&rand_vec(di * dr, seed_base + 4)),
            vec![di, dr],
        ));
        tensors.push((
            format!("{prefix}.mixer.dt_proj.bias"),
            f32_to_bytes(&rand_vec(di, seed_base + 5)),
            vec![di],
        ));
        let a_log: Vec<f32> = (0..di * D_STATE)
            .map(|j| -((j as f32 + 1.0).ln()))
            .collect();
        tensors.push((
            format!("{prefix}.mixer.A_log"),
            f32_to_bytes(&a_log),
            vec![di, D_STATE],
        ));
        tensors.push((
            format!("{prefix}.mixer.D"),
            f32_to_bytes(&vec![1.0f32; di]),
            vec![di],
        ));
        tensors.push((
            format!("{prefix}.mixer.out_proj.weight"),
            f32_to_bytes(&rand_vec(D_MODEL * di, seed_base + 6)),
            vec![D_MODEL, di],
        ));
    }

    let config_json = format!(
        r#"{{
            "model_type": "mamba",
            "hidden_size": {D_MODEL},
            "num_hidden_layers": {N_LAYERS},
            "state_size": {D_STATE},
            "conv_kernel": {D_CONV},
            "expand": {EXPAND},
            "vocab_size": {VOCAB_SIZE},
            "time_step_rank": {dr}
        }}"#,
        dr = dt_rank()
    );
    std::fs::write(dir.join("config.json"), config_json).unwrap();

    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .map(|(name, bytes, shape)| {
            let tv = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes).unwrap();
            (name.clone(), tv)
        })
        .collect();
    let serialized = safetensors::tensor::serialize(views, None).unwrap();
    std::fs::write(dir.join("model.safetensors"), serialized).unwrap();
}
