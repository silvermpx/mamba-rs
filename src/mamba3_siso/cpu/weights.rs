//! Training weight storage for Mamba-3 SISO.
//!
//! Mirrors the inference `Mamba3LayerWeights` but with gradient accumulation helpers.

use super::dims::Mamba3Dims;

/// Training weights for a single Mamba-3 SISO layer.
/// Same fields as inference weights — used for gradient accumulation.
#[derive(Clone)]
pub struct TrainMamba3LayerWeights {
    pub norm_weight: Vec<f32>,      // [d_model]
    pub in_proj_w: Vec<f32>,        // [d_model * in_proj_dim]
    pub dt_bias: Vec<f32>,          // [nheads]
    pub b_norm_weight: Vec<f32>,    // [d_state]
    pub c_norm_weight: Vec<f32>,    // [d_state]
    pub b_bias: Vec<f32>,           // [nheads * d_state]
    pub c_bias: Vec<f32>,           // [nheads * d_state]
    pub d_param: Vec<f32>,          // [nheads]
    pub norm_gate_weight: Vec<f32>, // [d_inner]
    pub out_proj_w: Vec<f32>,       // [d_inner * d_model]
}

impl TrainMamba3LayerWeights {
    pub fn zeros(dims: &Mamba3Dims) -> Self {
        let dm = dims.d_model;
        let di = dims.d_inner;
        let ds = dims.d_state;
        let nh = dims.nheads;
        let ip = dims.in_proj_dim;
        Self {
            norm_weight: vec![0.0; dm],
            in_proj_w: vec![0.0; dm * ip],
            dt_bias: vec![0.0; nh],
            b_norm_weight: vec![0.0; ds],
            c_norm_weight: vec![0.0; ds],
            b_bias: vec![0.0; nh * ds],
            c_bias: vec![0.0; nh * ds],
            d_param: vec![0.0; nh],
            norm_gate_weight: vec![0.0; di],
            out_proj_w: vec![0.0; di * dm],
        }
    }

    /// Zero all gradient accumulators.
    pub fn zero(&mut self) {
        self.norm_weight.fill(0.0);
        self.in_proj_w.fill(0.0);
        self.dt_bias.fill(0.0);
        self.b_norm_weight.fill(0.0);
        self.c_norm_weight.fill(0.0);
        self.b_bias.fill(0.0);
        self.c_bias.fill(0.0);
        self.d_param.fill(0.0);
        self.norm_gate_weight.fill(0.0);
        self.out_proj_w.fill(0.0);
    }

    /// Accumulate gradients from another set.
    pub fn add_inplace(&mut self, other: &Self) {
        for (a, &b) in self.norm_weight.iter_mut().zip(&other.norm_weight) {
            *a += b;
        }
        for (a, &b) in self.in_proj_w.iter_mut().zip(&other.in_proj_w) {
            *a += b;
        }
        for (a, &b) in self.dt_bias.iter_mut().zip(&other.dt_bias) {
            *a += b;
        }
        for (a, &b) in self.b_norm_weight.iter_mut().zip(&other.b_norm_weight) {
            *a += b;
        }
        for (a, &b) in self.c_norm_weight.iter_mut().zip(&other.c_norm_weight) {
            *a += b;
        }
        for (a, &b) in self.b_bias.iter_mut().zip(&other.b_bias) {
            *a += b;
        }
        for (a, &b) in self.c_bias.iter_mut().zip(&other.c_bias) {
            *a += b;
        }
        for (a, &b) in self.d_param.iter_mut().zip(&other.d_param) {
            *a += b;
        }
        for (a, &b) in self
            .norm_gate_weight
            .iter_mut()
            .zip(&other.norm_gate_weight)
        {
            *a += b;
        }
        for (a, &b) in self.out_proj_w.iter_mut().zip(&other.out_proj_w) {
            *a += b;
        }
    }

    /// Sum of squares of all parameters (for gradient norm).
    pub fn sum_sq(&self) -> f64 {
        let s = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        s(&self.norm_weight)
            + s(&self.in_proj_w)
            + s(&self.dt_bias)
            + s(&self.b_norm_weight)
            + s(&self.c_norm_weight)
            + s(&self.b_bias)
            + s(&self.c_bias)
            + s(&self.d_param)
            + s(&self.norm_gate_weight)
            + s(&self.out_proj_w)
    }
}

/// Training weights for the complete Mamba-3 backbone.
#[derive(Clone)]
pub struct TrainMamba3Weights {
    pub input_proj_w: Vec<f32>, // [input_dim * d_model]
    pub input_proj_b: Vec<f32>, // [d_model]
    pub layers: Vec<TrainMamba3LayerWeights>,
    pub norm_f_weight: Vec<f32>, // [d_model]
}

impl TrainMamba3Weights {
    pub fn zeros(dims: &Mamba3Dims, input_dim: usize) -> Self {
        Self {
            input_proj_w: vec![0.0; input_dim * dims.d_model],
            input_proj_b: vec![0.0; dims.d_model],
            layers: (0..dims.n_layers)
                .map(|_| TrainMamba3LayerWeights::zeros(dims))
                .collect(),
            norm_f_weight: vec![0.0; dims.d_model],
        }
    }

    pub fn zero(&mut self) {
        self.input_proj_w.fill(0.0);
        self.input_proj_b.fill(0.0);
        for l in &mut self.layers {
            l.zero();
        }
        self.norm_f_weight.fill(0.0);
    }

    pub fn add_inplace(&mut self, other: &Self) {
        for (a, &b) in self.input_proj_w.iter_mut().zip(&other.input_proj_w) {
            *a += b;
        }
        for (a, &b) in self.input_proj_b.iter_mut().zip(&other.input_proj_b) {
            *a += b;
        }
        for (l, o) in self.layers.iter_mut().zip(&other.layers) {
            l.add_inplace(o);
        }
        for (a, &b) in self.norm_f_weight.iter_mut().zip(&other.norm_f_weight) {
            *a += b;
        }
    }
}
