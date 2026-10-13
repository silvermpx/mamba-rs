//! `GpuOptions::scan_mode` reaches every GPU backbone and trainer: the
//! configuration each one runs carries the option's mode over the
//! configuration's own, and without the option the configuration's stays.
#![cfg(feature = "cuda")]

use mamba_rs::config::ScanMode;
use mamba_rs::{
    GpuOptions, Mamba2Config, Mamba2GpuBackbone, Mamba2GpuTrainer, Mamba2Weights, Mamba3Config,
    Mamba3GpuBackbone, Mamba3GpuTrainer, Mamba3MimoConfig, Mamba3MimoGpuBackbone,
    Mamba3MimoGpuTrainer, Mamba3MimoWeights, Mamba3Weights, MambaConfig, MambaGpuBackbone,
    MambaGpuTrainer, MambaWeights, TrainSessionCfg,
};

const DM: usize = 32;

#[test]
fn the_scan_mode_option_reaches_every_gpu_backbone_and_trainer() {
    let session = TrainSessionCfg::new(DM, 1, 16);
    let m1 = MambaConfig {
        d_model: DM,
        n_layers: 1,
        ..MambaConfig::default()
    };
    let m2 = Mamba2Config {
        d_model: DM,
        d_state: 16,
        headdim: 16,
        n_layers: 1,
        ..Mamba2Config::default()
    };
    let siso = Mamba3Config {
        d_model: DM,
        d_state: 16,
        headdim: 16,
        n_layers: 1,
        ..Mamba3Config::default()
    };
    let mimo = Mamba3MimoConfig {
        d_model: DM,
        d_state: 16,
        headdim: 16,
        n_layers: 1,
        ..Mamba3MimoConfig::default()
    };
    let (w1, w2) = (
        MambaWeights::init(&m1, DM, 1),
        Mamba2Weights::init(&m2, DM, 2),
    );
    let (w3, w4) = (
        Mamba3Weights::init(&siso, DM, 3),
        Mamba3MimoWeights::init(&mimo, DM, 4),
    );
    for (option, want) in [
        (None, ScanMode::Auto),
        (Some(ScanMode::Sequential), ScanMode::Sequential),
        (Some(ScanMode::Parallel), ScanMode::Parallel),
    ] {
        let opts = option.map_or(GpuOptions::default(), |m| {
            GpuOptions::default().scan_mode(m)
        });
        let got = [
            (
                "Mamba-1 backbone",
                MambaGpuBackbone::new(&w1, m1, DM, 1, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "Mamba-1 trainer",
                MambaGpuTrainer::new(&w1, m1, session, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "Mamba-2 backbone",
                Mamba2GpuBackbone::new(&w2, m2, DM, 1, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "Mamba-2 trainer",
                Mamba2GpuTrainer::new(&w2, m2, session, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "SISO backbone",
                Mamba3GpuBackbone::new(&w3, siso, DM, 1, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "SISO trainer",
                Mamba3GpuTrainer::new(&w3, siso, session, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "MIMO backbone",
                Mamba3MimoGpuBackbone::new(&w4, mimo, DM, 1, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
            (
                "MIMO trainer",
                Mamba3MimoGpuTrainer::new(&w4, mimo, session, opts)
                    .unwrap()
                    .config()
                    .scan_mode,
            ),
        ];
        for (what, mode) in got {
            assert_eq!(mode, want, "{what} with {option:?}");
        }
    }
    assert_eq!(ScanMode::parse(" sequential "), Ok(ScanMode::Sequential));
    assert_eq!(ScanMode::parse("parallel"), Ok(ScanMode::Parallel));
    assert!(ScanMode::parse("Sequential").is_err());
}
