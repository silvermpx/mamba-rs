use mamba_rs::{MambaBackbone, MambaConfig};

fn main() {
    let cfg = MambaConfig::default();
    let input_dim = cfg.d_model;

    // Initialize backbone with paper-default weights
    let backbone = MambaBackbone::init(cfg, input_dim, 42);
    println!(
        "Mamba: {} layers, d_model={}, d_inner={}, {} params",
        backbone.n_layers(),
        backbone.config().d_model,
        backbone.config().d_inner(),
        backbone.param_count(),
    );

    // Allocate recurrent state + scratch (once, reuse across steps)
    let mut state = backbone.alloc_state();
    let mut scratch = backbone.alloc_scratch();
    let mut output = vec![0.0f32; backbone.config().d_model];

    // Run 10 inference steps
    for step in 0..10 {
        let input = vec![0.1 * step as f32; input_dim];
        backbone.forward_step(&input, &mut output, &mut state, &mut scratch);

        let norm: f32 = output.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("step {step}: output L2 norm = {norm:.6}");
    }

    // Reset state for new sequence
    state.reset();
    println!("state reset");
}
