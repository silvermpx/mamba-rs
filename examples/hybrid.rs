//! A hybrid model trained end to end on the GPU: a Mamba-3 segment, a
//! layer of your own on the device, and a second Mamba-3 segment, with no
//! row crossing the host between them.
//!
//! ```bash
//! cargo run --release --example hybrid --features cuda
//! ```
//!
//! The first segment is built with `final_norm: false`, so its rows are its
//! residual stream; the second has an identity input (no input
//! projection), so it carries that stream on. Between them sits the
//! caller's layer, here a learned scale per channel where a real hybrid
//! would put attention. Each trainer hands out its own device buffers
//! (`input_ptr`, `temporal_ptr`, `d_temporal_ptr`, `d_input_ptr`); the
//! layer reads and writes them on the second segment's stream, and each
//! stream waits for the other's work through an event, not through the
//! host. Only the window's input and the loss touch the host.
//!
//! Before training, the gradient the device backward gives the layer's
//! scales is checked against central differences of the loss through both
//! segments.

#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("This example requires the `cuda` feature:");
    eprintln!("  cargo run --release --example hybrid --features cuda");
}

#[cfg(feature = "cuda")]
mod cuda_example {
    use std::sync::Arc;

    use cudarc::driver::sys::CUdeviceptr;
    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use mamba_rs::gpu::train::{BackwardOpts, TrainSessionCfg};
    use mamba_rs::{GpuOptions, Mamba3Config, Mamba3GpuTrainer, Mamba3Weights};

    const D_MODEL: usize = 64;
    const INPUT_DIM: usize = 16;
    const BATCH: usize = 4;
    const SEQ_LEN: usize = 128;
    /// The target is the input this many tokens back.
    const DELAY: usize = 4;
    const STEPS: usize = 300;
    /// The step of the plain gradient descent on the layer's scales.
    const LAYER_LR: f32 = 0.05;
    /// The channels whose gradient the check measures.
    const PROBES: [usize; 3] = [0, 17, 42];

    /// The caller's layer: `y = x * g` with one scale per channel, its
    /// backward, and gradient descent on the scales.
    const LAYER: &str = r#"
extern "C" __global__ void scale_fwd(float* y, const float* x, const float* g, int n, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * g[i % d];
}

// One thread per channel sums its rows in order, so the gradient of a
// scale is the same on every run.
extern "C" __global__ void scale_bwd(float* dx, float* dg, const float* dy, const float* x,
                                     const float* g, int rows, int d) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= d) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r) {
        int i = r * d + c;
        dx[i] = dy[i] * g[c];
        acc += dy[i] * x[i];
    }
    dg[c] = acc;
}

extern "C" __global__ void sgd(float* g, const float* dg, float lr, int d) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < d) g[c] -= lr * dg[c];
}
"#;

    fn cuda<T>(r: Result<T, impl std::fmt::Debug>) -> Result<T, String> {
        r.map_err(|e| format!("{e:?}"))
    }

    /// Host rows into a buffer a trainer handed out, on its stream.
    fn upload(stream: &CudaStream, dst: CUdeviceptr, rows: &[f32]) -> Result<(), String> {
        // SAFETY: `dst` holds at least `rows.len()` floats, and the stream
        // finishes the copy before `rows` can go.
        cuda(unsafe { cudarc::driver::result::memcpy_htod_async(dst, rows, stream.cu_stream()) })?;
        cuda(stream.synchronize())
    }

    /// A buffer a trainer handed out to the host, after its stream.
    fn download(stream: &CudaStream, src: CUdeviceptr, rows: &mut [f32]) -> Result<(), String> {
        // SAFETY: `src` holds at least `rows.len()` floats, and the stream
        // finishes the copy before `rows` can go.
        cuda(unsafe { cudarc::driver::result::memcpy_dtoh_async(rows, src, stream.cu_stream()) })?;
        cuda(stream.synchronize())
    }

    struct Layer {
        fwd: CudaFunction,
        bwd: CudaFunction,
        sgd: CudaFunction,
        g: CudaSlice<f32>,
        dg: CudaSlice<f32>,
    }

    struct Hybrid {
        first: Mamba3GpuTrainer,
        second: Mamba3GpuTrainer,
        layer: Layer,
    }

    impl Hybrid {
        fn new() -> Result<Self, String> {
            let opts = GpuOptions::default();
            let session = |input_dim| TrainSessionCfg {
                input_dim,
                batch: BATCH,
                seq_len: SEQ_LEN,
                lr: 3e-3,
                weight_decay: 1e-2,
            };
            let cfg = |final_norm| Mamba3Config {
                d_model: D_MODEL,
                n_layers: 2,
                final_norm,
                ..Mamba3Config::default()
            };
            // The first segment hands out its residual stream.
            let mut w = Mamba3Weights::init(&cfg(false), INPUT_DIM, 1);
            w.norm_f_weight.clear();
            w.norm_f_bias.clear();
            let first = Mamba3GpuTrainer::new(&w, cfg(false), session(INPUT_DIM), opts)?;
            // The second carries it on through an identity input.
            let mut w = Mamba3Weights::init(&cfg(true), D_MODEL, 2);
            w.input_proj_w.clear();
            w.input_proj_b.clear();
            let second = Mamba3GpuTrainer::new(&w, cfg(true), session(D_MODEL), opts)?;

            let stream = second.ctx().stream.clone();
            let ptx = cuda(cudarc::nvrtc::compile_ptx(LAYER))?;
            let module = cuda(stream.context().load_module(ptx))?;
            let layer = Layer {
                fwd: cuda(module.load_function("scale_fwd"))?,
                bwd: cuda(module.load_function("scale_bwd"))?,
                sgd: cuda(module.load_function("sgd"))?,
                g: cuda(stream.clone_htod(&[1.0f32; D_MODEL]))?,
                dg: cuda(stream.alloc_zeros::<f32>(D_MODEL))?,
            };
            Ok(Self {
                first,
                second,
                layer,
            })
        }

        fn streams(&self) -> (Arc<CudaStream>, Arc<CudaStream>) {
            (
                self.first.ctx().stream.clone(),
                self.second.ctx().stream.clone(),
            )
        }

        /// The forward of the window `x`: the first segment, the layer
        /// from its rows into the second segment's input, the second. The
        /// windows are sequences of their own, so each starts both
        /// segments from the empty state: a trainer otherwise carries its
        /// state into the next window, and the gradient check below would
        /// compare losses taken from different states.
        fn forward(&mut self, x: &[f32]) -> Result<(), String> {
            self.first.reset_state()?;
            self.second.reset_state()?;
            let (first, second) = self.streams();
            upload(&first, self.first.input_ptr(), x)?;
            self.first.forward_on_device()?;
            cuda(second.join(&first))?;
            let (n, d) = ((BATCH * SEQ_LEN * D_MODEL) as i32, D_MODEL as i32);
            let (y, rows) = (self.second.input_ptr(), self.first.temporal_ptr());
            let mut launch = second.launch_builder(&self.layer.fwd);
            launch.arg(&y).arg(&rows).arg(&self.layer.g).arg(&n).arg(&d);
            // SAFETY: both row buffers hold `n` floats and the scales `d`.
            cuda(unsafe { launch.launch(LaunchConfig::for_num_elems(n as u32)) })?;
            self.second.forward_on_device()
        }

        /// The mean squared error of the last forward's rows against
        /// `target`, its gradient left in the second segment's output
        /// gradient: the one place the rows meet the host.
        fn loss(&self, target: &[f32]) -> Result<f64, String> {
            let (_, second) = self.streams();
            let mut rows = vec![0.0f32; target.len()];
            download(&second, self.second.temporal_ptr(), &mut rows)?;
            let n = rows.len() as f64;
            let mut loss = 0.0f64;
            for (y, &t) in rows.iter_mut().zip(target) {
                let e = f64::from(*y - t);
                loss += 0.5 * e * e;
                *y = (e / n) as f32;
            }
            upload(&second, self.second.d_temporal_ptr(), &rows)?;
            Ok(loss / n)
        }

        /// The backward of the last forward: the second segment hands the
        /// gradient of its input to the layer, the layer the gradient of
        /// the first segment's rows to it, and each takes its step.
        fn backward(&mut self) -> Result<(), String> {
            let opts = BackwardOpts::default();
            let (first, second) = self.streams();
            self.second.backward_step_input_grad_on_device(opts)?;
            let (rows, d) = ((BATCH * SEQ_LEN) as i32, D_MODEL as i32);
            let (dx, dy, x) = (
                self.first.d_temporal_ptr(),
                self.second.d_input_ptr(),
                self.first.temporal_ptr(),
            );
            let lr = LAYER_LR;
            let mut launch = second.launch_builder(&self.layer.bwd);
            launch
                .arg(&dx)
                .arg(&mut self.layer.dg)
                .arg(&dy)
                .arg(&x)
                .arg(&self.layer.g)
                .arg(&rows)
                .arg(&d);
            // SAFETY: the row buffers hold `rows * d` floats, the scales and
            // their gradient `d`.
            cuda(unsafe { launch.launch(LaunchConfig::for_num_elems(d as u32)) })?;
            let mut launch = second.launch_builder(&self.layer.sgd);
            launch
                .arg(&mut self.layer.g)
                .arg(&self.layer.dg)
                .arg(&lr)
                .arg(&d);
            // SAFETY: the scales and their gradient hold `d` floats.
            cuda(unsafe { launch.launch(LaunchConfig::for_num_elems(d as u32)) })?;
            cuda(first.join(&second))?;
            self.first.backward_step_on_device(opts).map(|_| ())
        }

        fn set_scales(&mut self, g: &[f32]) -> Result<(), String> {
            let (_, second) = self.streams();
            cuda(second.memcpy_htod(g, &mut self.layer.g))?;
            cuda(second.synchronize())
        }
    }

    /// A window of inputs in [-1, 1) and its target: each channel of a
    /// row is an input channel `DELAY` tokens back, zero before.
    fn window(seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut s = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        let x: Vec<f32> = (0..BATCH * SEQ_LEN * INPUT_DIM)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                (s >> 40) as f32 / (1u64 << 23) as f32 - 1.0
            })
            .collect();
        let mut target = vec![0.0f32; BATCH * SEQ_LEN * D_MODEL];
        for b in 0..BATCH {
            for t in DELAY..SEQ_LEN {
                let src = &x[(b * SEQ_LEN + t - DELAY) * INPUT_DIM..][..INPUT_DIM];
                let dst = &mut target[(b * SEQ_LEN + t) * D_MODEL..][..D_MODEL];
                for (c, v) in dst.iter_mut().enumerate() {
                    *v = src[c % INPUT_DIM];
                }
            }
        }
        (x, target)
    }

    pub fn run() -> Result<(), String> {
        let mut h = Hybrid::new()?;

        // The gradient check, before any step moves the weights.
        let (x, target) = window(0);
        let ones = [1.0f32; D_MODEL];
        let eps = 1e-2f32;
        let mut differences = Vec::with_capacity(PROBES.len());
        for c in PROBES {
            let mut loss_at = |delta: f32| -> Result<f64, String> {
                let mut g = ones;
                g[c] += delta;
                h.set_scales(&g)?;
                h.forward(&x)?;
                h.loss(&target)
            };
            let (up, down) = (loss_at(eps)?, loss_at(-eps)?);
            differences.push((up - down) / (2.0 * f64::from(eps)));
        }
        h.set_scales(&ones)?;
        h.forward(&x)?;
        let first_loss = h.loss(&target)?;
        h.backward()?;
        let (_, second) = h.streams();
        let dg = cuda(second.clone_dtoh(&h.layer.dg))?;
        println!("the layer's scale gradients, device backward against central differences:");
        for (&c, &fd) in PROBES.iter().zip(&differences) {
            let bw = f64::from(dg[c]);
            println!("  channel {c:2}: backward {bw:+.6e}  differences {fd:+.6e}");
            let scale = bw.abs().max(fd.abs()).max(1e-8);
            if (bw - fd).abs() > 0.05 * scale {
                return Err(format!(
                    "channel {c}: the backward's {bw:e} is not the differences' {fd:e}"
                ));
            }
        }

        println!("step    0  loss {first_loss:.6}");
        let mut last = first_loss;
        for step in 1..STEPS {
            let (x, target) = window(step as u64);
            h.forward(&x)?;
            last = h.loss(&target)?;
            h.backward()?;
            if step % 50 == 0 || step == STEPS - 1 {
                println!("step {step:4}  loss {last:.6}");
            }
        }
        if last >= first_loss {
            return Err(format!(
                "the loss did not fall: {first_loss:.6} -> {last:.6}"
            ));
        }
        Ok(())
    }
}

#[cfg(feature = "cuda")]
fn main() {
    if let Err(e) = cuda_example::run() {
        eprintln!("hybrid: {e}");
        std::process::exit(1);
    }
}
