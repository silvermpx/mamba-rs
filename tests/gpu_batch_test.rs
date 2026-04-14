#![cfg(all(feature = "hf", feature = "cuda"))]

fn find_model_dir(name: &str) -> Option<std::path::PathBuf> {
    let cache = std::path::Path::new("/root/.cache/huggingface/hub");
    for entry in std::fs::read_dir(cache).ok()? {
        let entry = entry.ok()?;
        let fname = entry.file_name().into_string().ok()?;
        if fname.contains(name) {
            let snaps = entry.path().join("snapshots");
            if snaps.exists() {
                let snap = std::fs::read_dir(&snaps).ok()?.next()?.ok()?;
                return Some(snap.path());
            }
        }
    }
    None
}

#[test]
#[ignore]
fn test_gpu_batch_generation() {
    let dir = find_model_dir("mamba-130m-hf").expect("model not in cache");
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    use mamba_rs::module::gpu_lm::GpuMambaLM;
    use mamba_rs::module::sample::SampleParams;

    let mut lm = GpuMambaLM::from_hf_with_dtype_batch(&dir, 0, WeightDtype::F32, 2).unwrap();
    assert_eq!(lm.batch, 2);

    let p1 = SampleParams {
        temperature: 0.0,
        max_tokens: 10,
        seed: 42,
        ..Default::default()
    };
    let p2 = SampleParams {
        temperature: 0.0,
        max_tokens: 10,
        seed: 42,
        ..Default::default()
    };
    let prompt1: &[u32] = &[1, 2, 3, 4, 5];
    let prompt2: &[u32] = &[1, 2, 3, 4, 5];

    let outs = lm
        .generate_batch(&[prompt1, prompt2], &[p1.clone(), p2.clone()])
        .unwrap();
    assert_eq!(outs.len(), 2);

    eprintln!("slot 0: {:?}", outs[0]);
    eprintln!("slot 1: {:?}", outs[1]);

    // Identical prompts + identical params → identical outputs.
    assert_eq!(outs[0], outs[1], "same prompt+params must give same tokens");

    // Verify matches batch=1 baseline.
    let mut lm1 = GpuMambaLM::from_hf_with_dtype(&dir, 0, WeightDtype::F32).unwrap();
    let out_single = lm1.generate(prompt1, &p1).unwrap();
    eprintln!("batch=1: {:?}", out_single);
    assert_eq!(
        outs[0], out_single,
        "batch=2 slot 0 must match batch=1 reference"
    );
}
