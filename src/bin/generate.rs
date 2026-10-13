use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;
use hf_hub::{HFClientSync, split_id};
use tokenizers::Tokenizer;

use mamba_rs::ModelFamily;
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

    /// Tokenizer file (tokenizer.json). The official Mamba-3 repositories
    /// ship none; they use the Llama-3.1 tokenizer.
    #[arg(long)]
    tokenizer: Option<PathBuf>,

    /// Prompt text
    prompt: String,

    /// Use GPU inference (requires CUDA)
    #[arg(long)]
    gpu: bool,

    /// GPU device index
    #[arg(long, default_value_t = 0)]
    gpu_device: usize,

    /// Weight storage precision on GPU: f32 (default), tf32 (f32 storage,
    /// deterministic TF32 products), bf16 (half VRAM), f16. The CPU runs
    /// f32 only.
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// GEMM mode on GPU: deterministic, cublas-fast or cublas-pedantic.
    /// Without it, MAMBA_RS_GEMM_MODE decides (deterministic by default).
    #[arg(long)]
    gemm_mode: Option<String>,

    /// Scan of the recurrence over the prompt on GPU: auto (default),
    /// sequential or parallel
    #[arg(long)]
    scan_mode: Option<String>,

    /// Continuations of the prompt to generate in one batched run on GPU,
    /// sequence i sampling with seed + i
    #[arg(long, default_value_t = 1)]
    batch: usize,

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

    let family = mamba_rs::hf::read_family(&model_dir).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        std::process::exit(1);
    });
    let tokenizer = load_tokenizer(&model_dir, &args, family);
    let mamba3 = matches!(family, ModelFamily::Mamba3 | ModelFamily::Mamba3Mimo);

    let encoding = tokenizer
        .encode(args.prompt.as_str(), false)
        .unwrap_or_else(|e| {
            eprintln!("error: tokenization failed: {e}");
            std::process::exit(1);
        });
    let mut prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    // The official Mamba-3 checkpoints were trained on Llama-3.1 token
    // streams that open with <|begin_of_text|>.
    if mamba3 && let Some(bos) = tokenizer.token_to_id("<|begin_of_text|>") {
        prompt_ids.insert(0, bos);
    }
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
        run_gpu(
            &model_dir,
            &args,
            &prompt_ids,
            n_prompt,
            &params,
            &tokenizer,
        );
        return;
    }

    #[cfg(not(feature = "cuda"))]
    if args.gpu {
        eprintln!("error: --gpu requires the 'cuda' feature (rebuild with --features cli,cuda)");
        std::process::exit(1);
    }

    // The CPU path takes none of the GPU settings: refuse them rather than
    // run something other than what was asked.
    let gpu_only = [
        (args.dtype.to_lowercase() != "f32", "--dtype other than f32"),
        (args.gemm_mode.is_some(), "--gemm-mode"),
        (args.scan_mode.is_some(), "--scan-mode"),
        (args.batch != 1, "--batch"),
    ];
    if let Some((_, flag)) = gpu_only.iter().find(|(set, _)| *set) {
        eprintln!("error: {flag} needs --gpu; the CPU runs f32, one sequence");
        std::process::exit(1);
    }

    run_cpu(
        &model_dir,
        &args,
        &prompt_ids,
        n_prompt,
        &params,
        &tokenizer,
    );
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
    let mut lm = mamba_rs::MambaLM::from_hf(model_dir).unwrap_or_else(|e| {
        eprintln!("error: failed to load model: {e}");
        std::process::exit(1);
    });
    let load_ms = t_load.elapsed().as_millis();

    eprintln!(
        "model: {} (CPU, {:?}) | d_model={} vocab={} | loaded in {load_ms}ms",
        args.model_id,
        lm.family(),
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
    });

    print_stats(&mut stdout, n_prompt, n_generated, ttft, t_start);
}

#[cfg(feature = "cuda")]
fn parse_dtype(s: &str) -> mamba_rs::WeightDtype {
    use mamba_rs::WeightDtype;
    match s.to_lowercase().as_str() {
        "f32" | "fp32" | "float32" => WeightDtype::F32,
        "tf32" => WeightDtype::Tf32,
        "bf16" | "bfloat16" => WeightDtype::Bf16,
        "f16" | "fp16" | "float16" | "half" => WeightDtype::F16,
        other => {
            eprintln!("error: unknown dtype '{other}', expected f32/tf32/bf16/f16");
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
    let mut opts = mamba_rs::GpuOptions::default()
        .device(args.gpu_device)
        .dtype(dtype);
    if let Some(mode) = &args.gemm_mode {
        opts = opts.mode(
            mamba_rs::GemmMode::parse_env_value(mode).unwrap_or_else(|e| {
                eprintln!("error: --gemm-mode: {e}");
                std::process::exit(1);
            }),
        );
    }
    if let Some(mode) = &args.scan_mode {
        opts = opts.scan_mode(mamba_rs::config::ScanMode::parse(mode).unwrap_or_else(|e| {
            eprintln!("error: --scan-mode: {e}");
            std::process::exit(1);
        }));
    }
    if args.batch == 0 {
        eprintln!("error: --batch must be at least 1");
        std::process::exit(1);
    }
    let t_load = Instant::now();
    let mut lm = mamba_rs::MambaGpuLM::from_hf(model_dir, args.batch, opts).unwrap_or_else(|e| {
        eprintln!("error: failed to load GPU model: {e}");
        std::process::exit(1);
    });
    lm.capture_graph().unwrap_or_else(|e| {
        eprintln!("warning: CUDA Graph capture failed ({e}), running without graph");
    });
    let load_ms = t_load.elapsed().as_millis();

    eprintln!(
        "model: {} (GPU:{} {}, {:?}) | d_model={} vocab={} | loaded in {load_ms}ms",
        args.model_id,
        args.gpu_device,
        dtype.as_str(),
        lm.family(),
        lm.d_model,
        lm.vocab_size
    );
    eprintln!("prompt: {n_prompt} tokens");

    let mut stdout = std::io::stdout().lock();
    let mut n_generated = 0u64;
    let mut byte_buf = Vec::new();
    let t_start = Instant::now();
    let mut ttft: Option<u128> = None;

    if args.batch > 1 {
        let prompts = vec![prompt_ids; args.batch];
        let seeds: Vec<SampleParams> = (0..args.batch as u64)
            .map(|i| SampleParams {
                seed: params.seed.wrapping_add(i),
                ..params.clone()
            })
            .collect();
        let outs = lm.generate_batch(&prompts, &seeds).unwrap_or_else(|e| {
            eprintln!("\nerror: generation failed: {e}");
            std::process::exit(1);
        });
        ttft = Some(t_start.elapsed().as_millis());
        for (i, tokens) in outs.iter().enumerate() {
            let _ = writeln!(stdout, "--- sequence {i} (seed {}) ---", seeds[i].seed);
            byte_buf.clear();
            for &token_id in tokens {
                n_generated += 1;
                if let Some(piece) = decode_token(tokenizer, token_id, &mut byte_buf) {
                    let _ = stdout.write_all(piece.as_bytes());
                }
            }
            let _ = writeln!(stdout);
        }
        print_stats(&mut stdout, n_prompt, n_generated, ttft, t_start);
        return;
    }

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
    let client = HFClientSync::new().unwrap_or_else(|e| {
        eprintln!("error: HF Hub client init failed: {e}");
        eprintln!("hint: set HF_TOKEN env var for gated models");
        std::process::exit(1);
    });
    let (owner, name) = split_id(model_id);
    let repo = client.model(owner, name);
    let get = |file: &str| {
        repo.download_file()
            .filename(file)
            .revision(revision.to_string())
            .send()
    };

    let config = get("config.json").unwrap_or_else(|e| {
        eprintln!("error: cannot download config.json: {e}");
        std::process::exit(1);
    });
    let model_dir = config.parent().unwrap().to_path_buf();

    let st_path = model_dir.join("model.safetensors");
    let idx_path = model_dir.join("model.safetensors.index.json");

    if !st_path.exists() && !idx_path.exists() {
        if let Ok(idx) = get("model.safetensors.index.json") {
            let idx_bytes = std::fs::read(&idx).unwrap();
            let index: serde_json::Value = serde_json::from_slice(&idx_bytes).unwrap();
            if let Some(wm) = index.get("weight_map").and_then(|v| v.as_object()) {
                let mut shards: Vec<&str> = wm.values().filter_map(|v| v.as_str()).collect();
                shards.sort();
                shards.dedup();
                for shard in shards {
                    eprintln!("  downloading {shard}...");
                    get(shard).unwrap_or_else(|e| {
                        eprintln!("error: cannot download {shard}: {e}");
                        std::process::exit(1);
                    });
                }
            }
        } else if get("model.safetensors").is_err() {
            // The original mamba_ssm repositories (the official Mamba-3
            // checkpoints among them) ship a torch archive instead.
            get("pytorch_model.bin").unwrap_or_else(|e| {
                eprintln!("error: cannot download model weights: {e}");
                std::process::exit(1);
            });
        }
    }

    for name in ["tokenizer.json", "tokenizer_config.json"] {
        let _ = get(name);
    }

    model_dir
}

fn load_tokenizer(model_dir: &Path, args: &Args, family: ModelFamily) -> Tokenizer {
    if let Some(path) = &args.tokenizer {
        return Tokenizer::from_file(path).unwrap_or_else(|e| {
            eprintln!("error: cannot load {}: {e}", path.display());
            std::process::exit(1);
        });
    }
    let tok_path = model_dir.join("tokenizer.json");
    if tok_path.exists() {
        Tokenizer::from_file(&tok_path).unwrap_or_else(|e| {
            eprintln!("error: cannot load tokenizer.json: {e}");
            std::process::exit(1);
        })
    } else {
        if matches!(family, ModelFamily::Mamba3 | ModelFamily::Mamba3Mimo) {
            eprintln!(
                "error: the official Mamba-3 repositories ship no tokenizer; pass \
                 --tokenizer with the Llama-3.1 tokenizer.json"
            );
            std::process::exit(1);
        }
        if family == ModelFamily::Mamba2 {
            eprintln!(
                "error: the official Mamba-2 repositories ship no tokenizer; pass \
                 --tokenizer with the GPT-NeoX (EleutherAI/gpt-neox-20b) tokenizer.json"
            );
            std::process::exit(1);
        }
        eprintln!("warning: tokenizer.json not found, trying HF download...");
        let client = HFClientSync::new().unwrap();
        let (owner, name) = split_id(&args.model_id);
        let path = client
            .model(owner, name)
            .download_file()
            .filename("tokenizer.json")
            .send()
            .unwrap_or_else(|e| {
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
