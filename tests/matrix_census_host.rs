//! The matrix census: every family's `admit` over a grid of configurations
//! `validate` accepts and of requests (path, batch and window, scan mode,
//! precision, GEMM mode), on the host.
//!
//! A request the 32-bit kernel arguments cannot carry is a legitimate
//! refusal. A request no kernel serves, or one that runs another scan than
//! the one it names, is a defect: `qual/matrix_ledger.toml` lists each
//! with the cells the census finds it in, and the kernel work removes it.
//! The census fails when a gap appears that has no row, when a row's count
//! differs from the census (grown: a regression; shrunk: update the
//! ledger), and when a row's gap is gone.

use std::collections::BTreeMap;

use mamba_rs::config::ScanMode;
use mamba_rs::matrix::{INT_ARG_MAX, Path, Plan, Refusal, Request, Scan};
use mamba_rs::{GemmMode, Mamba2Config, Mamba3Config, Mamba3MimoConfig, MambaConfig, WeightDtype};

const LEDGER: &str = "qual/matrix_ledger.toml";

/// The windows every config is asked for: one token, the automatic scan
/// threshold and one past it, a long training window, a wide batch, and
/// the edges of the 32-bit kernel arguments.
const WINDOWS: [(usize, usize); 9] = [
    (1, 1),
    (3, 1),
    (1, 64),
    (1, 65),
    (2, 4621),
    (64, 2048),
    (1, INT_ARG_MAX),
    (INT_ARG_MAX, 1),
    (1 << 16, 1 << 15),
];

const SCANS: [ScanMode; 3] = [ScanMode::Auto, ScanMode::Sequential, ScanMode::Parallel];
const PRECISIONS: [WeightDtype; 4] = [
    WeightDtype::F32,
    WeightDtype::Tf32,
    WeightDtype::Bf16,
    WeightDtype::F16,
];
const GEMMS: [GemmMode; 3] = [
    GemmMode::Deterministic,
    GemmMode::CublasFast,
    GemmMode::CublasPedantic,
];

/// Every request of the census: each path over each window it takes (a
/// decode step is one token), in every scan mode, precision and GEMM mode.
fn requests() -> Vec<Request> {
    let mut out = Vec::new();
    for path in Path::ALL {
        for (batch, seq_len) in WINDOWS {
            if path == Path::Decode && seq_len != 1 {
                continue;
            }
            for scan in SCANS {
                for precision in PRECISIONS {
                    for gemm in GEMMS {
                        out.push(Request {
                            path,
                            batch,
                            seq_len,
                            scan,
                            precision,
                            gemm,
                        });
                    }
                }
            }
        }
    }
    out
}

/// The census of one family: the cells of every gap, and how many cells
/// were admitted and refused for capacity.
#[derive(Default)]
struct Tally {
    gaps: BTreeMap<String, u64>,
    admitted: u64,
    capacity: u64,
    configs: u64,
}

impl Tally {
    fn record(&mut self, family: &str, req: &Request, verdict: Result<Plan, Refusal>) {
        match verdict {
            Ok(plan) => {
                self.admitted += 1;
                let named = Scan::named(req.scan);
                if req.path != Path::Decode
                    && req.seq_len > 1
                    && named.is_some_and(|s| s != plan.scan)
                {
                    let gap = format!(
                        "{family} | reroute | {} {:?} runs {:?}",
                        req.path.as_str(),
                        req.scan,
                        plan.scan
                    );
                    *self.gaps.entry(gap).or_default() += 1;
                }
            }
            Err(Refusal::Capacity { .. }) => self.capacity += 1,
            Err(Refusal::Kernel { predicate, .. }) => {
                let gap = format!("{family} | kernel | {predicate}");
                *self.gaps.entry(gap).or_default() += 1;
            }
            Err(Refusal::Request { detail }) => {
                panic!("{family}: the census asked a malformed request {req:?}: {detail}")
            }
            Err(other) => panic!("{family}: {req:?} refused as no census row knows: {other}"),
        }
    }
}

/// Runs `admit` over every config of `configs` that `validate` accepts and
/// every request.
fn census<C: Copy + std::fmt::Debug>(
    family: &str,
    configs: impl IntoIterator<Item = C>,
    validate: impl Fn(&C) -> bool,
    admit: impl Fn(&C, &Request) -> Result<Plan, Refusal>,
) -> Tally {
    let requests = requests();
    let mut tally = Tally::default();
    for cfg in configs.into_iter().filter(|c| validate(c)) {
        tally.configs += 1;
        for req in &requests {
            tally.record(family, req, admit(&cfg, req));
        }
    }
    assert!(
        tally.configs > 0,
        "{family}: no config of the grid validates"
    );
    tally
}

const D_MODELS: [usize; 5] = [1, 32, 33, 384, 1 << 20];

fn mamba1() -> Tally {
    let mut configs = Vec::new();
    for d_model in D_MODELS {
        for d_state in [1, 16, 64, 65, 256] {
            for d_conv in [1, 4, 9, 64] {
                for expand in [1, 2, 3, 1024] {
                    configs.push(MambaConfig {
                        d_model,
                        d_state,
                        d_conv,
                        expand,
                        n_layers: 1,
                        ..MambaConfig::default()
                    });
                }
            }
        }
    }
    census(
        "mamba1",
        configs,
        |c| c.validate().is_ok(),
        |c, r| mamba_rs::mamba1::route::admit(c, c.d_model, r),
    )
}

fn mamba2() -> Tally {
    let mut configs = Vec::new();
    for d_model in D_MODELS {
        for expand in [1, 2, 3] {
            for headdim in [1, 7, 32, 64, 128, 4096] {
                for d_state in [1, 16, 64, 128, 1024] {
                    for ngroups in [1, 2, 8] {
                        for (d_conv, d_intermediate) in [(4, 0), (64, 1000)] {
                            configs.push(Mamba2Config {
                                d_model,
                                expand,
                                headdim,
                                d_state,
                                ngroups,
                                d_conv,
                                d_intermediate,
                                n_layers: 1,
                                ..Mamba2Config::default()
                            });
                        }
                    }
                }
            }
        }
    }
    census(
        "mamba2",
        configs,
        |c| c.validate().is_ok(),
        |c, r| mamba_rs::mamba2::route::admit(c, c.d_model, r),
    )
}

fn mamba3_siso() -> Tally {
    let mut configs = Vec::new();
    for d_model in D_MODELS {
        for expand in [1, 2] {
            for headdim in [1, 16, 32, 33, 64, 512, 1024] {
                for d_state in [1, 16, 32, 64, 128, 256] {
                    for ngroups in [1, 2] {
                        for (rope_fraction, d_intermediate) in [(0.5, 0), (1.0, 1000)] {
                            configs.push(Mamba3Config {
                                d_model,
                                expand,
                                headdim,
                                d_state,
                                ngroups,
                                rope_fraction,
                                d_intermediate,
                                n_layers: 1,
                                ..Mamba3Config::default()
                            });
                        }
                    }
                }
            }
        }
    }
    census(
        "mamba3_siso",
        configs,
        |c| c.validate().is_ok(),
        |c, r| mamba_rs::mamba3_siso::route::admit(c, c.d_model, r),
    )
}

fn mamba3_mimo() -> Tally {
    let mut configs = Vec::new();
    for d_model in D_MODELS {
        for expand in [1, 2] {
            for headdim in [1, 16, 64, 128, 1024] {
                for d_state in [2, 16, 64, 128, 256] {
                    for ngroups in [1, 2] {
                        for mimo_rank in [1, 2, 4, 16] {
                            for rope_fraction in [0.5, 1.0] {
                                configs.push(Mamba3MimoConfig {
                                    d_model,
                                    expand,
                                    headdim,
                                    d_state,
                                    ngroups,
                                    mimo_rank,
                                    rope_fraction,
                                    n_layers: 1,
                                    ..Mamba3MimoConfig::default()
                                });
                            }
                        }
                    }
                }
            }
        }
    }
    census(
        "mamba3_mimo",
        configs,
        |c| c.validate().is_ok(),
        |c, r| mamba_rs::mamba3_mimo::route::admit(c, c.d_model, r),
    )
}

/// The ledger's rows: `"<family> | <kind> | <predicate>" = <cells>` under
/// `[gaps]`, read with a narrow line reader as the lane table is.
fn parse_ledger(text: &str) -> BTreeMap<String, u64> {
    let mut rows = BTreeMap::new();
    let mut in_table = false;
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            in_table = line == "[gaps]";
            continue;
        }
        if !in_table {
            continue;
        }
        let (key, cells) = line
            .rsplit_once('=')
            .unwrap_or_else(|| panic!("{LEDGER}: a row without `=`: {line}"));
        let key = key.trim().trim_matches('"').to_string();
        let cells: u64 = cells
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("{LEDGER}: {key}: {e}"));
        assert!(
            rows.insert(key.clone(), cells).is_none(),
            "{LEDGER}: {key} twice"
        );
    }
    rows
}

/// The census matches the ledger row for row.
#[test]
fn the_census_matches_the_ledger() {
    let tallies = [mamba1(), mamba2(), mamba3_siso(), mamba3_mimo()];
    let mut census: BTreeMap<String, u64> = BTreeMap::new();
    for (family, t) in ["mamba1", "mamba2", "mamba3_siso", "mamba3_mimo"]
        .iter()
        .zip(&tallies)
    {
        eprintln!(
            "{family}: {} configs, {} cells admitted, {} past the 32-bit arguments",
            t.configs, t.admitted, t.capacity
        );
        census.extend(t.gaps.clone());
    }
    let text = std::fs::read_to_string(LEDGER).unwrap_or_else(|e| panic!("{LEDGER}: {e}"));
    let ledger = parse_ledger(&text);
    let mut problems = Vec::new();
    for (gap, &cells) in &census {
        match ledger.get(gap) {
            None => problems.push(format!("new gap, {cells} cells: {gap}")),
            Some(&rows) if cells > rows => {
                problems.push(format!("grew from {rows} to {cells} cells: {gap}"))
            }
            Some(&rows) if cells < rows => problems.push(format!(
                "shrank from {rows} to {cells} cells (update the ledger): {gap}"
            )),
            Some(_) => {}
        }
    }
    for gap in ledger.keys().filter(|g| !census.contains_key(*g)) {
        problems.push(format!("gone (remove the row): {gap}"));
    }
    if !problems.is_empty() {
        let rows: Vec<String> = census
            .iter()
            .map(|(gap, cells)| format!("\"{gap}\" = {cells}"))
            .collect();
        panic!(
            "{LEDGER} disagrees with the census:\n  {}\nThe census's rows:\n[gaps]\n{}",
            problems.join("\n  "),
            rows.join("\n")
        );
    }
}

/// Every shape the published models take is admitted on every path, scan,
/// precision and GEMM mode at the windows they run.
#[test]
fn the_published_shapes_take_every_request() {
    let windows = [(1, 1), (8, 1), (1, 2048), (2, 4621)];
    let siso = Mamba3Config {
        d_model: 384,
        headdim: 32,
        d_state: 16,
        rope_fraction: 1.0,
        n_layers: 24,
        ..Mamba3Config::default()
    };
    let m2 = Mamba2Config {
        d_model: 768,
        d_state: 128,
        headdim: 64,
        n_layers: 24,
        ..Mamba2Config::default()
    };
    let m1 = MambaConfig {
        d_model: 768,
        n_layers: 24,
        ..MambaConfig::default()
    };
    let mimo = Mamba3MimoConfig {
        d_model: 768,
        headdim: 64,
        d_state: 128,
        n_layers: 24,
        ..Mamba3MimoConfig::default()
    };
    for req in requests() {
        if !windows.contains(&(req.batch, req.seq_len)) {
            continue;
        }
        let siso_plan = mamba_rs::mamba3_siso::route::admit(&siso, 384, &req);
        let reroute = req.path == Path::Prefill && req.scan == ScanMode::Sequential;
        assert!(siso_plan.is_ok(), "siso {req:?}: {siso_plan:?}");
        if !reroute
            && req.seq_len > 1
            && let Some(named) = Scan::named(req.scan)
        {
            assert_eq!(siso_plan.unwrap().scan, named, "siso {req:?}");
        }
        assert!(
            mamba_rs::mamba2::route::admit(&m2, 768, &req).is_ok(),
            "m2 {req:?}"
        );
        assert!(
            mamba_rs::mamba1::route::admit(&m1, 768, &req).is_ok(),
            "m1 {req:?}"
        );
        assert!(
            mamba_rs::mamba3_mimo::route::admit(&mimo, 768, &req).is_ok(),
            "mimo {req:?}"
        );
    }
}
