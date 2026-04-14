use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;

use mamba_rs::module::sample::SampleParams;

#[derive(Parser)]
#[command(name = "mamba-generate")]
#[command(about = "Generate text with pretrained Mamba language models")]
struct Args {
    /// HuggingFace model ID (e.g. state-spaces/mamba-130m-hf)
    #[arg(short, long)]
    model_id: String,

    /// HuggingFace model revision
    #[arg(long, default_value = "main")]
    revision: String,

    /// Local model directory (skip download)
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Prompt text
    prompt: String,

    /// Use GPU inference (requires CUDA)
    #[arg(long)]
    gpu: bool,

    /// GPU device index
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,

    /// Weight storage dtype on GPU: f32 (default), bf16 (half VRAM), f16
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Sampling temperature (0 = greedy)
    #[arg(short, long, default_value_t = 0.7)]
    temperature: f32,

    /// Top-k filtering (0 = disabled)
    #[arg(long, default_value_t = 40)]
    top_k: usize,

    /// Top-p nucleus sampling (1.0 = disabled)
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,

    /// Min-p filtering (0 = disabled)
    #[arg(long, default_value_t = 0.05)]
    min_p: f32,

    /// Repetition penalty (1.0 = disabled)
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,

    /// Maximum tokens to generate
    #[arg(short = 'n', long, default_value_t = 256)]
    max_tokens: usize,

    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let model_dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => download_model(&args.model_id, &args.revision),
    };

    let tokenizer = load_tokenizer(&model_dir, &args);

    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .unwrap_or_else(|e| {
            eprintln!("error: tokenization failed: {e}");
            std::process::exit(1);
        });
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    let n_prompt = prompt_ids.len();

    let params = SampleParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        repetition_penalty: args.repetition_penalty,
        max_tokens: args.max_tokens,
        eos_token_ids: find_eos_tokens(&tokenizer),
        seed: args.seed,
    };

    #[cfg(feature = "cuda")]
    if args.gpu {
        run_gpu(&model_dir, &args, &prompt_ids, n_prompt, &params, &tokenizer);
        return;
    }

    #[cfg(not(feature = "cuda"))]
    if args.gpu {
        eprintln!("error: --gpu requires the 'cuda' feature (rebuild with --features cli,cuda)");
        std::process::exit(1);
    }

    run_cpu(&model_dir, &args, &prompt_ids, n_prompt, &params, &tokenizer);
}

fn run_cpu(
    model_dir: &Path,
    args: &Args,
    prompt_ids: &[u32],
    n_prompt: usize,
    params: &SampleParams,
    tokenizer: &Tokenizer,
) {
    let t_load = Instant::now();
    let mut lm = mamba_rs::module::lm::MambaLM::from_hf(model_dir).unwrap_or_else(|e| {
        eprintln!("error: failed to load model: {e}");
        std::process::exit(1);
    });
    let load_ms = t_load.elapsed().as_millis();

    eprintln!(
        "model: {} (CPU) | d_model={} vocab={} | loaded in {load_ms}ms",
        args.model_id, lm.d_model, lm.vocab_size
    );
    eprintln!("prompt: {n_prompt} tokens");

    let mut stdout = std::io::stdout().lock();
    let mut n_generated = 0u64;
    let mut byte_buf = Vec::new();
    let t_start = Instant::now();
    let mut ttft: Option<u128> = None;

    lm.generate_streaming(prompt_ids, params, |token_id, _| {
        if ttft.is_none() {
            ttft = Some(t_start.elapsed().as_millis());
        }
        n_generated += 1;
        if let Some(piece) = decode_token(tokenizer, token_id, &mut byte_buf) {
            let _ = stdout.write_all(piece.as_bytes());
            let _ = stdout.flush();
        }
    });

    print_stats(&mut stdout, n_prompt, n_generated, ttft, t_start);
}

#[cfg(feature = "cuda")]
fn parse_dtype(s: &str) -> mamba_rs::mamba_ssm::gpu::dtype::WeightDtype {
    use mamba_rs::mamba_ssm::gpu::dtype::WeightDtype;
    match s.to_lowercase().as_str() {
        "f32" | "fp32" | "float32" => WeightDtype::F32,
        "bf16" | "bfloat16" => WeightDtype::Bf16,
        "f16" | "fp16" | "float16" | "half" => WeightDtype::F16,
        other => {
            eprintln!("error: unknown dtype '{other}', expected f32/bf16/f16");
            std::process::exit(1);
        }
    }
}

#[cfg(feature = "cuda")]
fn run_gpu(
    model_dir: &Path,
    args: &Args,
    prompt_ids: &[u32],
    n_prompt: usize,
    params: &SampleParams,
    tokenizer: &Tokenizer,
) {
    let dtype = parse_dtype(&args.dtype);
    let t_load = Instant::now();
    let mut lm =
        mamba_rs::module::gpu_lm::GpuMambaLM::from_hf_with_dtype(model_dir, args.gpu_device, dtype)
            .unwrap_or_else(|e| {
                eprintln!("error: failed to load GPU model: {e}");
                std::process::exit(1);
            });
    lm.capture_graph().unwrap_or_else(|e| {
        eprintln!("warning: CUDA Graph capture failed ({e}), running without graph");
    });
    let load_ms = t_load.elapsed().as_millis();

    eprintln!(
        "model: {} (GPU:{} {}) | d_model={} vocab={} | loaded in {load_ms}ms",
        args.model_id,
        args.gpu_device,
        dtype.as_str(),
        lm.d_model,
        lm.vocab_size
    );
    eprintln!("prompt: {n_prompt} tokens");

    let mut stdout = std::io::stdout().lock();
    let mut n_generated = 0u64;
    let mut byte_buf = Vec::new();
    let t_start = Instant::now();
    let mut ttft: Option<u128> = None;

    lm.generate_streaming(prompt_ids, params, |token_id, _| {
        if ttft.is_none() {
            ttft = Some(t_start.elapsed().as_millis());
        }
        n_generated += 1;
        if let Some(piece) = decode_token(tokenizer, token_id, &mut byte_buf) {
            let _ = stdout.write_all(piece.as_bytes());
            let _ = stdout.flush();
        }
    })
    .unwrap_or_else(|e| {
        eprintln!("\nerror: generation failed: {e}");
        std::process::exit(1);
    });

    print_stats(&mut stdout, n_prompt, n_generated, ttft, t_start);
}

fn print_stats(
    stdout: &mut std::io::StdoutLock,
    n_prompt: usize,
    n_generated: u64,
    ttft: Option<u128>,
    t_start: Instant,
) {
    let elapsed = t_start.elapsed();
    let _ = writeln!(stdout);
    let tok_s = if elapsed.as_secs_f64() > 0.0 {
        n_generated as f64 / elapsed.as_secs_f64()
    } else {
        0.0
    };
    eprintln!(
        "{n_prompt} prompt, prefill {}ms | {n_generated} tokens at {tok_s:.1} tok/s ({:.1}s total)",
        ttft.unwrap_or(0),
        elapsed.as_secs_f64()
    );
}

fn download_model(model_id: &str, revision: &str) -> PathBuf {
    eprintln!("downloading {model_id} (rev: {revision})...");
    let api = Api::new().unwrap_or_else(|e| {
        eprintln!("error: HF Hub API init failed: {e}");
        eprintln!("hint: set HF_TOKEN env var for gated models");
        std::process::exit(1);
    });
    let repo = api.repo(hf_hub::Repo::with_revision(
        model_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let config = repo.get("config.json").unwrap_or_else(|e| {
        eprintln!("error: cannot download config.json: {e}");
        std::process::exit(1);
    });
    let model_dir = config.parent().unwrap().to_path_buf();

    let st_path = model_dir.join("model.safetensors");
    let idx_path = model_dir.join("model.safetensors.index.json");

    if !st_path.exists() && !idx_path.exists() {
        if let Ok(idx) = repo.get("model.safetensors.index.json") {
            let idx_bytes = std::fs::read(&idx).unwrap();
            let index: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
            if let Some(wm) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut shards: Vec<&str> = wm.values().filter_map(|v| v.as_str()).collect();
                shards.sort();
                shards.dedup();
                for shard in shards {
                    eprintln!("  downloading {shard}...");
                    repo.get(shard).unwrap_or_else(|e| {
                        eprintln!("error: cannot download {shard}: {e}");
                        std::process::exit(1);
                    });
                }
            }
        } else {
            repo.get("model.safetensors").unwrap_or_else(|e| {
                eprintln!("error: cannot download model weights: {e}");
                std::process::exit(1);
            });
        }
    }

    for name in ["tokenizer.json", "tokenizer_config.json"] {
        let _ = repo.get(name);
    }

    model_dir
}

fn load_tokenizer(model_dir: &Path, args: &Args) -> Tokenizer {
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists() {
        Tokenizer::from_file(&tok_path).unwrap_or_else(|e| {
            eprintln!("error: cannot load tokenizer.json: {e}");
            std::process::exit(1);
        })
    } else {
        eprintln!("warning: tokenizer.json not found, trying HF download...");
        let api = Api::new().unwrap();
        let repo = api.repo(hf_hub::Repo::model(args.model_id.clone()));
        let path = repo.get("tokenizer.json").unwrap_or_else(|e| {
            eprintln!("error: cannot download tokenizer: {e}");
            std::process::exit(1);
        });
        Tokenizer::from_file(&path).unwrap_or_else(|e| {
            eprintln!("error: cannot parse tokenizer: {e}");
            std::process::exit(1);
        })
    }
}

fn find_eos_tokens(tokenizer: &Tokenizer) -> Vec<u32> {
    let mut eos = Vec::new();
    if let Some(id) = tokenizer.token_to_id("<|endoftext|>") {
        eos.push(id);
    }
    if let Some(id) = tokenizer.token_to_id("</s>") {
        eos.push(id);
    }
    if let Some(id) = tokenizer.token_to_id("<|end_of_text|>") {
        eos.push(id);
    }
    eos
}

fn decode_token(tokenizer: &Tokenizer, token_id: u32, buf: &mut Vec<u32>) -> Option<String> {
    buf.push(token_id);
    match tokenizer.decode(buf.as_slice(), false) {
        Ok(text) => {
            if text.ends_with('\u{FFFD}') {
                None
            } else {
                buf.clear();
                Some(text)
            }
        }
        Err(_) => None,
    }
}
