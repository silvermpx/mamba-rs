//! Checkpoint weight files. Every layout `hf::source::Source` reads gives
//! the same tensors, a shard index is held to its files and a corrupt
//! header is refused; the Mamba-1 loader reads the `pytorch_model.bin` the
//! original repositories ship as it reads the `-hf` safetensors twin; and
//! the Mamba-2, SISO and MIMO loaders give an untied head as the language
//! models take it.
#![cfg(feature = "hf")]

use std::path::{Path, PathBuf};

use mamba_rs::hf::load_mamba;
use mamba_rs::hf::source::Source;
use mamba_rs::hf::{load_mamba2, parse_mamba2_config};
use mamba_rs::hf::{load_mamba3, load_mamba3_mimo, parse_mamba3_config};
use safetensors::Dtype;
use safetensors::tensor::TensorView;

/// A tensor to write: its name, shape and values, stored in `dtype`.
struct Tensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
    dtype: Dtype,
}

impl Tensor {
    fn new(name: &str, shape: &[usize], dtype: Dtype, seed: u64) -> Self {
        let n = shape.iter().product();
        Self {
            name: name.to_string(),
            shape: shape.to_vec(),
            values: values(n, seed),
            dtype,
        }
    }

    /// The values as stored, little endian.
    fn bytes(&self) -> Vec<u8> {
        self.values
            .iter()
            .flat_map(|&v| match self.dtype {
                Dtype::F32 => v.to_le_bytes().to_vec(),
                Dtype::BF16 => half::bf16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                Dtype::F16 => half::f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                other => panic!("no {other:?} in these tests"),
            })
            .collect()
    }

    /// The values a reader gives back: the stored ones, widened.
    fn read_back(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&v| match self.dtype {
                Dtype::BF16 => half::bf16::from_f32(v).to_f32(),
                Dtype::F16 => half::f16::from_f32(v).to_f32(),
                _ => v,
            })
            .collect()
    }
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).max(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

/// A fresh directory for one test.
fn scratch(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("mamba-rs-hf-{tag}-{}", std::process::id()));
    if dir.exists() {
        std::fs::remove_dir_all(&dir).unwrap();
    }
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_safetensors(path: &Path, tensors: &[&Tensor]) {
    let bytes: Vec<Vec<u8>> = tensors.iter().map(|t| t.bytes()).collect();
    let views: Vec<(String, TensorView<'_>)> = tensors
        .iter()
        .zip(&bytes)
        .map(|(t, b)| {
            let view = TensorView::new(t.dtype, t.shape.clone(), b).unwrap();
            (t.name.clone(), view)
        })
        .collect();
    std::fs::write(path, safetensors::serialize(views, None).unwrap()).unwrap();
}

/// The pickle `torch.save` writes for a state dict of these tensors, each
/// in its own storage `"<index>"`.
fn state_dict_pickle(tensors: &[&Tensor]) -> Vec<u8> {
    fn text(p: &mut Vec<u8>, s: &str) {
        p.push(b'X');
        p.extend_from_slice(&(s.len() as u32).to_le_bytes());
        p.extend_from_slice(s.as_bytes());
    }
    fn int(p: &mut Vec<u8>, v: usize) {
        p.push(b'J');
        p.extend_from_slice(&(v as i32).to_le_bytes());
    }
    fn tuple(p: &mut Vec<u8>, items: &[usize]) {
        p.push(b'(');
        for &v in items {
            int(p, v);
        }
        p.push(b't');
    }
    let mut p = vec![0x80, 2];
    p.extend_from_slice(b"ccollections\nOrderedDict\nq\x00)R(");
    for (i, t) in tensors.iter().enumerate() {
        let storage = match t.dtype {
            Dtype::F32 => "FloatStorage",
            Dtype::BF16 => "BFloat16Storage",
            Dtype::F16 => "HalfStorage",
            other => panic!("no {other:?} in these tests"),
        };
        text(&mut p, &t.name);
        p.extend_from_slice(b"ctorch._utils\n_rebuild_tensor_v2\n((");
        text(&mut p, "storage");
        p.extend_from_slice(format!("ctorch\n{storage}\n").as_bytes());
        text(&mut p, &i.to_string());
        text(&mut p, "cpu");
        int(&mut p, t.values.len());
        p.extend_from_slice(b"tQ");
        int(&mut p, 0);
        tuple(&mut p, &t.shape);
        let mut strides = vec![1usize; t.shape.len()];
        for d in (0..t.shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * t.shape[d + 1];
        }
        tuple(&mut p, &strides);
        p.extend_from_slice(b"\x89h\x00)RtR");
    }
    p.extend_from_slice(b"u.");
    p
}

/// A `pytorch_model.bin`: a zip of stored entries holding the pickle and
/// one storage per tensor.
fn write_torch(path: &Path, tensors: &[&Tensor]) {
    let mut entries: Vec<(String, Vec<u8>)> =
        vec![("archive/data.pkl".into(), state_dict_pickle(tensors))];
    for (i, t) in tensors.iter().enumerate() {
        entries.push((format!("archive/data/{i}"), t.bytes()));
    }
    let (mut zip, mut central) = (Vec::new(), Vec::new());
    for (name, data) in &entries {
        let offset = zip.len() as u32;
        let header = |sig: u32, central: bool| {
            let mut h = sig.to_le_bytes().to_vec();
            if central {
                h.extend_from_slice(&20u16.to_le_bytes());
            }
            for v in [20u16, 0, 0, 0, 0] {
                h.extend_from_slice(&v.to_le_bytes());
            }
            for v in [0u32, data.len() as u32, data.len() as u32] {
                h.extend_from_slice(&v.to_le_bytes());
            }
            h.extend_from_slice(&(name.len() as u16).to_le_bytes());
            h.extend_from_slice(&0u16.to_le_bytes());
            if central {
                for v in [0u16, 0, 0] {
                    h.extend_from_slice(&v.to_le_bytes());
                }
                h.extend_from_slice(&0u32.to_le_bytes());
                h.extend_from_slice(&offset.to_le_bytes());
            }
            h.extend_from_slice(name.as_bytes());
            h
        };
        zip.extend_from_slice(&header(0x0403_4b50, false));
        zip.extend_from_slice(data);
        central.extend_from_slice(&header(0x0201_4b50, true));
    }
    let cd_offset = zip.len() as u32;
    zip.extend_from_slice(&central);
    zip.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    for v in [0u16, 0, entries.len() as u16, entries.len() as u16] {
        zip.extend_from_slice(&v.to_le_bytes());
    }
    zip.extend_from_slice(&(central.len() as u32).to_le_bytes());
    zip.extend_from_slice(&cd_offset.to_le_bytes());
    zip.extend_from_slice(&0u16.to_le_bytes());
    std::fs::write(path, zip).unwrap();
}

/// A shard index mapping each tensor to the file named beside it.
fn write_index(path: &Path, map: &[(&str, &str)]) {
    let weight_map: serde_json::Map<String, serde_json::Value> = map
        .iter()
        .map(|&(t, f)| (t.to_string(), serde_json::Value::from(f)))
        .collect();
    let index = serde_json::json!({ "metadata": {}, "weight_map": weight_map });
    std::fs::write(path, serde_json::to_vec(&index).unwrap()).unwrap();
}

fn refs(ts: &[Tensor]) -> Vec<&Tensor> {
    ts.iter().collect()
}

fn sample() -> Vec<Tensor> {
    vec![
        Tensor::new("a.weight", &[3, 5], Dtype::F32, 1),
        Tensor::new("b", &[7], Dtype::BF16, 2),
        Tensor::new("c.bias", &[2, 2, 3], Dtype::F16, 3),
        Tensor::new("d", &[1], Dtype::F32, 4),
    ]
}

fn assert_reads(dir: &Path, tensors: &[Tensor]) {
    let mut src = Source::open(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
    let mut names = src.names().to_vec();
    names.sort();
    let mut want: Vec<String> = tensors.iter().map(|t| t.name.clone()).collect();
    want.sort();
    assert_eq!(names, want, "{}", dir.display());
    for t in tensors {
        assert_eq!(src.shape(&t.name), Some(t.shape.as_slice()), "{}", t.name);
        let got = src.read(&t.name).unwrap();
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&got),
            bits(&t.read_back()),
            "{}: {}",
            dir.display(),
            t.name
        );
    }
    assert!(src.read("absent").is_err());
}

#[test]
fn every_layout_gives_the_same_tensors() {
    let ts = sample();
    let (first, second) = ts.split_at(2);
    let map: Vec<(&str, &str)> = ts
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), if i < 2 { "one" } else { "two" }))
        .collect();
    let named = |ext: &str| -> Vec<(String, String)> {
        map.iter()
            .map(|&(t, f)| (t.to_string(), format!("{f}.{ext}")))
            .collect()
    };

    let dir = scratch("st-single");
    write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
    assert_reads(&dir, &ts);

    let dir = scratch("st-shards");
    write_safetensors(&dir.join("one.safetensors"), &refs(first));
    write_safetensors(&dir.join("two.safetensors"), &refs(second));
    let m = named("safetensors");
    let m: Vec<(&str, &str)> = m.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    write_index(&dir.join("model.safetensors.index.json"), &m);
    assert_reads(&dir, &ts);

    let dir = scratch("bin-single");
    write_torch(&dir.join("pytorch_model.bin"), &refs(&ts));
    assert_reads(&dir, &ts);

    let dir = scratch("bin-shards");
    write_torch(&dir.join("one.bin"), &refs(first));
    write_torch(&dir.join("two.bin"), &refs(second));
    let m = named("bin");
    let m: Vec<(&str, &str)> = m.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    write_index(&dir.join("pytorch_model.bin.index.json"), &m);
    assert_reads(&dir, &ts);
}

#[test]
fn a_shard_index_is_held_to_its_files() {
    let ts = sample();
    let (first, second) = ts.split_at(2);
    let dir = scratch("index-held");
    write_safetensors(&dir.join("one.safetensors"), &refs(first));
    write_safetensors(&dir.join("two.safetensors"), &refs(second));
    let index = dir.join("model.safetensors.index.json");
    let refused = |map: &[(&str, &str)], what: &str| {
        write_index(&index, map);
        let err = Source::open(&dir).err();
        assert!(err.is_some(), "{what}: opened");
    };
    // A tensor mapped to the file that does not hold it.
    refused(
        &[
            ("a.weight", "two.safetensors"),
            ("b", "one.safetensors"),
            ("c.bias", "two.safetensors"),
            ("d", "two.safetensors"),
        ],
        "tensor in the other shard",
    );
    // A shard holding a tensor the index does not name.
    refused(
        &[
            ("a.weight", "one.safetensors"),
            ("c.bias", "two.safetensors"),
            ("d", "two.safetensors"),
        ],
        "unindexed tensor",
    );
    // A shard path outside the directory, and a shard that is not there.
    refused(
        &[
            ("a.weight", "../one.safetensors"),
            ("b", "../one.safetensors"),
            ("c.bias", "two.safetensors"),
            ("d", "two.safetensors"),
        ],
        "path outside",
    );
    refused(
        &[
            ("a.weight", "one.safetensors"),
            ("b", "one.safetensors"),
            ("c.bias", "three.safetensors"),
            ("d", "three.safetensors"),
        ],
        "missing shard",
    );
    // The same tensor stored in two shards.
    let dup = Tensor::new("a.weight", &[3, 5], Dtype::F32, 9);
    write_safetensors(&dir.join("two.safetensors"), &[&ts[2], &ts[3], &dup]);
    refused(
        &[
            ("a.weight", "one.safetensors"),
            ("b", "one.safetensors"),
            ("c.bias", "two.safetensors"),
            ("d", "two.safetensors"),
        ],
        "tensor stored twice",
    );
}

/// A safetensors file of one header and a data region, as given.
fn raw_safetensors(header: &str, data_len: usize) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + data_len, 0);
    bytes
}

#[test]
fn a_corrupt_safetensors_header_is_refused() {
    let dir = scratch("corrupt");
    let path = dir.join("model.safetensors");
    let open = |bytes: &[u8]| {
        std::fs::write(&path, bytes).unwrap();
        Source::open(&dir)
    };
    let x = |offsets: &str| {
        format!(r#"{{"x":{{"dtype":"F32","shape":[2],"data_offsets":{offsets}}}}}"#)
    };
    assert!(open(&raw_safetensors(&x("[0,8]"), 8)).is_ok());
    // The data region shorter than the tensor.
    assert!(open(&raw_safetensors(&x("[0,8]"), 7)).is_err());
    // Offsets past the data region, and offsets that do not span the shape.
    assert!(open(&raw_safetensors(&x("[8,16]"), 8)).is_err());
    assert!(open(&raw_safetensors(&x("[0,12]"), 12)).is_err());
    assert!(open(&raw_safetensors(&x("[8,0]"), 8)).is_err());
    // A header that claims more bytes than the file has.
    let mut long = raw_safetensors(&x("[0,8]"), 8);
    let claimed = long.len() as u64;
    long[..8].copy_from_slice(&claimed.to_le_bytes());
    assert!(open(&long).is_err());
    // An empty file, a file without a header, a header that is not JSON.
    assert!(open(&[]).is_err());
    assert!(open(&[1, 0, 0]).is_err());
    assert!(open(&raw_safetensors("{not json", 0)).is_err());
    // A dtype that is not a float weight lists, and refuses to read.
    let ints = r#"{"i":{"dtype":"I32","shape":[2],"data_offsets":[0,8]}}"#;
    let mut src = open(&raw_safetensors(ints, 8)).unwrap();
    assert_eq!(src.shape("i"), Some([2usize].as_slice()));
    assert!(src.read("i").is_err());
}

/// A small Mamba-1 checkpoint's tensors under the original repositories'
/// names: two layers of d_model 16, d_state 4, d_conv 4, expand 2.
fn mamba1_tensors(embedding_name: &str, dtype: Dtype) -> Vec<Tensor> {
    mamba1_tensors_with(embedding_name, dtype, (1, false, true, 0))
}

/// [`mamba1_tensors`] at dt rank `dr`, with the in_proj and out_proj
/// biases when `proj_bias`, the conv bias only when `conv_bias`, and a
/// gated MLP `hidden` wide after every mixer when `hidden` is not 0.
fn mamba1_tensors_with(
    embedding_name: &str,
    dtype: Dtype,
    (dr, proj_bias, conv_bias, hidden): (usize, bool, bool, usize),
) -> Vec<Tensor> {
    let (d, di, ds, w, vocab) = (16, 32, 4, 4, 32);
    let mut ts = vec![
        Tensor::new(embedding_name, &[vocab, d], dtype, 100),
        Tensor::new("backbone.norm_f.weight", &[d], dtype, 101),
    ];
    for i in 0..2u64 {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        let s = 200 + 20 * i;
        for (name, shape, k) in [
            (p("mixer.in_proj.weight"), vec![2 * di, d], 0),
            (p("mixer.conv1d.weight"), vec![di, 1, w], 1),
            (p("mixer.conv1d.bias"), vec![di], 2),
            (p("mixer.x_proj.weight"), vec![dr + 2 * ds, di], 3),
            (p("mixer.dt_proj.weight"), vec![di, dr], 4),
            (p("mixer.dt_proj.bias"), vec![di], 5),
            (p("mixer.A_log"), vec![di, ds], 6),
            (p("mixer.D"), vec![di], 7),
            (p("mixer.out_proj.weight"), vec![d, di], 8),
            (p("norm.weight"), vec![d], 9),
            (p("mixer.in_proj.bias"), vec![2 * di], 10),
            (p("mixer.out_proj.bias"), vec![d], 11),
        ] {
            let absent = match k {
                2 => !conv_bias,
                10 | 11 => !proj_bias,
                _ => false,
            };
            if !absent {
                ts.push(Tensor::new(&name, &shape, dtype, s + k));
            }
        }
        if hidden > 0 {
            for (name, shape, k) in [
                (format!("backbone.layers.{i}.norm2.weight"), vec![d], 12),
                (
                    format!("backbone.layers.{i}.mlp.fc1.weight"),
                    vec![2 * hidden, d],
                    13,
                ),
                (
                    format!("backbone.layers.{i}.mlp.fc2.weight"),
                    vec![d, hidden],
                    14,
                ),
            ] {
                ts.push(Tensor::new(&name, &shape, dtype, s + k));
            }
        }
    }
    ts
}

/// An original-format checkpoint with `d_intermediate` loads its gated
/// MLPs: the norm as stored, both projections in the `[n_in, n_out]`
/// layout; one whose MLP tensors are missing or of another width is
/// refused.
#[test]
fn mamba1_gated_mlp_loads() {
    let config = br#"{"d_model": 16, "n_layer": 2, "vocab_size": 32, "d_intermediate": 100,
        "ssm_cfg": {"d_state": 4}}"#;
    let ts = mamba1_tensors_with(
        "backbone.embedding.weight",
        Dtype::F32,
        (1, false, true, 128),
    );
    let dir = scratch("m1-mlp");
    write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
    std::fs::write(dir.join("config.json"), config).unwrap();
    let m = load_mamba(&dir).unwrap();
    let cfg = &m.cfg;
    assert_eq!((cfg.d_intermediate, cfg.mlp_hidden()), (100, 128));
    let tensor = |name: &str| {
        ts.iter()
            .find(|t| t.name == name)
            .map(|t| t.read_back())
            .unwrap()
    };
    let w = &m.weights;
    assert_eq!(w.mlp.len(), 2);
    for (i, mw) in w.mlp.iter().enumerate() {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        assert_eq!(mw.norm_weight, tensor(&p("norm2.weight")));
        // fc1 is stored [2 hidden, d] and comes out [d, 2 hidden].
        let fc1 = tensor(&p("mlp.fc1.weight"));
        assert_eq!(mw.fc1_w[3 * 256 + 5], fc1[5 * 16 + 3]);
        let fc2 = tensor(&p("mlp.fc2.weight"));
        assert_eq!(mw.fc2_w[7 * 16 + 2], fc2[2 * 128 + 7]);
    }

    // Without the MLP's tensors, or with another width, the load is refused.
    let refused = |tag: &str, tensors: &[Tensor]| {
        let dir = scratch(tag);
        write_safetensors(&dir.join("model.safetensors"), &refs(tensors));
        std::fs::write(dir.join("config.json"), config).unwrap();
        assert!(load_mamba(&dir).is_err(), "{tag}");
    };
    let name = "backbone.embedding.weight";
    refused(
        "m1-mlp-missing",
        &mamba1_tensors_with(name, Dtype::F32, (1, false, true, 0)),
    );
    refused(
        "m1-mlp-narrow",
        &mamba1_tensors_with(name, Dtype::F32, (1, false, true, 256)),
    );
    // A width or a depth past the bounds is refused before anything is
    // sized from it.
    for (tag, config) in [
        (
            "m1-mlp-huge",
            r#"{"d_model": 16, "n_layer": 2, "vocab_size": 32,
                "d_intermediate": 18446744073709551615, "ssm_cfg": {"d_state": 4}}"#,
        ),
        (
            "m1-deep",
            r#"{"d_model": 16, "n_layer": 18446744073709551615, "vocab_size": 32,
                "ssm_cfg": {"d_state": 4}}"#,
        ),
    ] {
        let dir = scratch(tag);
        write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
        std::fs::write(dir.join("config.json"), config).unwrap();
        let err = load_mamba(&dir).err().unwrap_or_default();
        assert!(err.contains("out of supported range"), "{tag}: {err}");
    }
}

/// An original-format Mamba-1 checkpoint built with `rms_norm = false`
/// loads its LayerNorm block norms: every norm's bias as stored, the gated
/// MLP's norm2 among them. One missing a bias is refused, and so is an
/// RMSNorm config whose checkpoint carries one.
#[test]
fn mamba1_layer_norm_checkpoints_load() {
    let config = br#"{"d_model": 16, "n_layer": 2, "vocab_size": 32, "d_intermediate": 100,
        "ssm_cfg": {"d_state": 4}, "rms_norm": false}"#;
    let mut ts = mamba1_tensors_with(
        "backbone.embedding.weight",
        Dtype::F32,
        (1, false, true, 128),
    );
    ts.push(Tensor::new("backbone.norm_f.bias", &[16], Dtype::F32, 300));
    for i in 0..2u64 {
        ts.push(Tensor::new(
            &format!("backbone.layers.{i}.norm.bias"),
            &[16],
            Dtype::F32,
            301 + i,
        ));
        ts.push(Tensor::new(
            &format!("backbone.layers.{i}.norm2.bias"),
            &[16],
            Dtype::F32,
            311 + i,
        ));
    }
    let tensor = |name: &str| {
        ts.iter()
            .find(|t| t.name == name)
            .map(|t| t.read_back())
            .unwrap()
    };
    let dir = scratch("m1-layer-norm");
    write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
    std::fs::write(dir.join("config.json"), config).unwrap();
    let m = load_mamba(&dir).unwrap();
    assert_eq!(m.cfg.block_norm, mamba_rs::BlockNorm::Layer);
    let w = &m.weights;
    assert_eq!(w.norm_f_bias, tensor("backbone.norm_f.bias"));
    for i in 0..2 {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        assert_eq!(w.layers[i].norm_bias, tensor(&p("norm.bias")));
        assert_eq!(w.mlp[i].norm_bias, tensor(&p("norm2.bias")));
    }

    let refused = |tag: &str, tensors: &[&Tensor], config: &[u8]| {
        let dir = scratch(tag);
        write_safetensors(&dir.join("model.safetensors"), tensors);
        std::fs::write(dir.join("config.json"), config).unwrap();
        assert!(load_mamba(&dir).is_err(), "{tag}");
    };
    for missing in [
        "backbone.norm_f.bias",
        "backbone.layers.1.norm.bias",
        "backbone.layers.0.norm2.bias",
    ] {
        let kept: Vec<&Tensor> = ts.iter().filter(|t| t.name != missing).collect();
        refused(&format!("m1-layer-norm-no-{missing}"), &kept, config);
    }
    let rms = String::from_utf8_lossy(config).replace("\"rms_norm\": false", "\"rms_norm\": true");
    refused("m1-layer-norm-as-rms", &refs(&ts), rms.as_bytes());
}

/// The projection biases, a missing conv bias and a dt rank of its own load
/// from either format's flags, the biases as they are stored; a checkpoint
/// whose tensors disagree with its flags is refused.
#[test]
fn mamba1_bias_and_rank_options_load() {
    let opts = (3, true, false, 0);
    let ts = mamba1_tensors_with("backbone.embeddings.weight", Dtype::F32, opts);
    let hf = scratch("m1-options-hf");
    write_safetensors(&hf.join("model.safetensors"), &refs(&ts));
    let hf_config = r#"{"model_type": "mamba", "hidden_size": 16, "num_hidden_layers": 2,
        "state_size": 4, "conv_kernel": 4, "expand": 2, "time_step_rank": 3,
        "vocab_size": 32, "use_bias": true, "use_conv_bias": false}"#;
    std::fs::write(hf.join("config.json"), hf_config).unwrap();
    let original = scratch("m1-options-original");
    write_safetensors(&original.join("model.safetensors"), &refs(&ts));
    std::fs::write(
        original.join("config.json"),
        br#"{"d_model": 16, "n_layer": 2, "vocab_size": 32, "pad_vocab_size_multiple": 8,
            "ssm_cfg": {"d_state": 4, "dt_rank": 3, "bias": true, "conv_bias": false}}"#,
    )
    .unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let tensor = |name: &str| {
        ts.iter()
            .find(|t| t.name == name)
            .map(|t| t.read_back())
            .unwrap()
    };
    let (a, b) = (load_mamba(&hf).unwrap(), load_mamba(&original).unwrap());
    for m in [&a, &b] {
        let cfg = &m.cfg;
        assert_eq!(
            (cfg.dt_rank(), cfg.proj_bias, cfg.conv_bias),
            (3, true, false)
        );
        for (i, lw) in m.weights.layers.iter().enumerate() {
            let p = |n: &str| format!("backbone.layers.{i}.mixer.{n}");
            assert_eq!(bits(&lw.in_proj_b), bits(&tensor(&p("in_proj.bias"))));
            assert_eq!(bits(&lw.out_proj_b), bits(&tensor(&p("out_proj.bias"))));
            assert!(lw.conv1d_bias.is_empty());
            assert_eq!(lw.dt_proj_w.len(), 3 * 32);
        }
    }
    for (la, lb) in a.weights.layers.iter().zip(&b.weights.layers) {
        assert_eq!(bits(&la.dt_proj_w), bits(&lb.dt_proj_w));
        assert_eq!(bits(&la.x_proj_w), bits(&lb.x_proj_w));
    }

    // Flags the tensors contradict are refused: a conv bias the file
    // lacks, a conv bias the flags say the model has not, biases the file
    // lacks.
    let refused = |tag: &str, tensors: &[Tensor], config: &str, needle: &str| {
        let dir = scratch(tag);
        write_safetensors(&dir.join("model.safetensors"), &refs(tensors));
        std::fs::write(dir.join("config.json"), config).unwrap();
        let err = load_mamba(&dir).err().unwrap();
        assert!(err.contains(needle), "{tag}: {err}");
    };
    let conv_on = hf_config.replace("\"use_conv_bias\": false", "\"use_conv_bias\": true");
    refused("m1-options-no-conv-tensor", &ts, &conv_on, "conv1d_bias");
    let name = "backbone.embeddings.weight";
    let extra_conv = mamba1_tensors_with(name, Dtype::F32, (3, true, true, 0));
    refused(
        "m1-options-extra-conv",
        &extra_conv,
        hf_config,
        "conv1d_bias",
    );
    let unbiased = mamba1_tensors_with(name, Dtype::F32, (3, false, false, 0));
    refused("m1-options-no-biases", &unbiased, hf_config, "in_proj_b");
}

/// The original `state-spaces/mamba-*` layout (config in the mamba_ssm
/// format, `pytorch_model.bin` with the tied head saved beside the
/// embedding) loads the same model as the `-hf` twin.
#[test]
fn mamba1_reads_the_original_bin_as_its_hf_twin() {
    for dtype in [Dtype::F32, Dtype::BF16] {
        let original = scratch(&format!("m1-original-{dtype:?}"));
        let mut ts = mamba1_tensors("backbone.embedding.weight", dtype);
        let mut head = Tensor::new("lm_head.weight", &[32, 16], dtype, 100);
        head.values = ts[0].values.clone();
        ts.push(head);
        write_torch(
            &original.join("pytorch_model.bin"),
            &ts.iter().collect::<Vec<_>>(),
        );
        std::fs::write(
            original.join("config.json"),
            br#"{"d_model": 16, "n_layer": 2, "vocab_size": 30, "ssm_cfg": {"d_state": 4},
                "rms_norm": true, "residual_in_fp32": true, "fused_add_norm": true,
                "pad_vocab_size_multiple": 8, "tie_embeddings": true}"#,
        )
        .unwrap();

        let twin = scratch(&format!("m1-hf-{dtype:?}"));
        let hf = mamba1_tensors("backbone.embeddings.weight", dtype);
        write_safetensors(
            &twin.join("model.safetensors"),
            &hf.iter().collect::<Vec<_>>(),
        );
        std::fs::write(
            twin.join("config.json"),
            br#"{"model_type": "mamba", "hidden_size": 16, "num_hidden_layers": 2,
                "state_size": 4, "conv_kernel": 4, "expand": 2, "intermediate_size": 32,
                "time_step_rank": 1, "vocab_size": 32, "use_bias": false,
                "use_conv_bias": true, "layer_norm_epsilon": 1e-5,
                "tie_word_embeddings": true}"#,
        )
        .unwrap();

        let a = load_mamba(&original).unwrap();
        let b = load_mamba(&twin).unwrap();
        assert!(a.lm_head.is_none() && b.lm_head.is_none());
        assert_eq!(
            (a.embed.len() / a.cfg.d_model),
            (b.embed.len() / b.cfg.d_model)
        );
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&a.embed), bits(&b.embed), "{dtype:?} embedding");
        let (wa, wb) = (&a.weights, &b.weights);
        assert_eq!(bits(&wa.norm_f_weight), bits(&wb.norm_f_weight));
        for (la, lb) in wa.layers.iter().zip(&wb.layers) {
            for (x, y) in [
                (&la.norm_weight, &lb.norm_weight),
                (&la.in_proj_w, &lb.in_proj_w),
                (&la.conv1d_weight, &lb.conv1d_weight),
                (&la.conv1d_bias, &lb.conv1d_bias),
                (&la.x_proj_w, &lb.x_proj_w),
                (&la.dt_proj_w, &lb.dt_proj_w),
                (&la.dt_proj_b, &lb.dt_proj_b),
                (&la.a_log, &lb.a_log),
                (&la.a_neg, &lb.a_neg),
                (&la.d_param, &lb.d_param),
                (&la.out_proj_w, &lb.out_proj_w),
            ] {
                assert_eq!(bits(x), bits(y), "{dtype:?} layer weights");
            }
        }
        // in_proj comes out as [d_model, 2 d_inner].
        let src = &ts[2].read_back();
        assert_eq!(wa.layers[0].in_proj_w[1], src[16]);
    }
}

/// A Mamba-3 config with its own block norm epsilon parses: the block
/// norms take `norm_epsilon`, the mixer's norms the reference's fixed 1e-5,
/// in SISO and in MIMO.
#[test]
fn mamba3_block_norm_epsilon_is_its_own() {
    for mimo in [false, true] {
        let json = format!(
            r#"{{"d_model": 32, "n_layer": 1, "vocab_size": 30, "norm_epsilon": 1e-6,
            "ssm_cfg": {{"layer": "Mamba3", "d_state": 16, "headdim": 16, "is_mimo": {mimo},
            "mimo_rank": 2}}}}"#
        );
        let hf = parse_mamba3_config(json.as_bytes()).unwrap();
        assert_eq!((hf.cfg.rms_norm_eps, hf.cfg.ssm_norm_eps), (1e-6, 1e-5));
        if let Some(m) = hf.mimo_config().unwrap() {
            assert_eq!((m.rms_norm_eps, m.ssm_norm_eps), (1e-6, 1e-5));
        }
    }
}

/// A head `[vocab_rows, d]` checkpoint tensor as the language models take
/// an untied head: `[d, vocab]`.
fn head_layout(rows: &[f32], vocab: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; d * vocab];
    for v in 0..vocab {
        for j in 0..d {
            out[j * vocab + v] = rows[v * d + j];
        }
    }
    out
}

#[test]
fn untied_heads_load_in_mamba2_siso_and_mimo() {
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    // Mamba-2: d_model 32, d_state 8, headdim 16, vocab 30 padded to 32.
    let json = br#"{"d_model": 32, "n_layer": 1, "vocab_size": 30, "pad_vocab_size_multiple": 8,
        "ssm_cfg": {"layer": "Mamba2", "d_state": 8, "headdim": 16}, "rms_norm": true,
        "tie_embeddings": false}"#;
    let cfg = parse_mamba2_config(json).unwrap().cfg;
    let (d, vp) = (cfg.d_model, 32);
    let p = |n: &str| format!("backbone.layers.0.{n}");
    let mut ts = vec![
        Tensor::new("backbone.embedding.weight", &[vp, d], Dtype::F32, 1),
        Tensor::new("backbone.norm_f.weight", &[d], Dtype::F32, 2),
        Tensor::new("lm_head.weight", &[vp, d], Dtype::F32, 3),
        Tensor::new(&p("norm.weight"), &[d], Dtype::F32, 4),
    ];
    for (name, shape, seed) in [
        ("mixer.in_proj.weight", vec![cfg.in_proj_out_dim(), d], 5),
        (
            "mixer.conv1d.weight",
            vec![cfg.conv_dim(), 1, cfg.d_conv],
            6,
        ),
        ("mixer.conv1d.bias", vec![cfg.conv_dim()], 7),
        ("mixer.dt_bias", vec![cfg.nheads()], 8),
        ("mixer.A_log", vec![cfg.nheads()], 9),
        ("mixer.D", vec![cfg.nheads()], 10),
        ("mixer.norm.weight", vec![cfg.ssm_dim()], 11),
        ("mixer.out_proj.weight", vec![d, cfg.d_inner()], 12),
    ] {
        ts.push(Tensor::new(&p(name), &shape, Dtype::F32, seed));
    }
    let dir = scratch("m2-untied");
    std::fs::write(dir.join("config.json"), json).unwrap();
    write_safetensors(
        &dir.join("model.safetensors"),
        &ts.iter().collect::<Vec<_>>(),
    );
    let m = load_mamba2(&dir).unwrap();
    let head = m.lm_head.expect("an untied head");
    assert_eq!(bits(&head), bits(&head_layout(&ts[2].values, 30, d)));
    // The same checkpoint read as tied leaves the saved head unused.
    std::fs::write(
        dir.join("config.json"),
        String::from_utf8_lossy(json)
            .replace("\"tie_embeddings\": false", "\"tie_embeddings\": true"),
    )
    .unwrap();
    assert!(load_mamba2(&dir).unwrap().lm_head.is_none());

    // SISO and MIMO: d_model 32, d_state 16, headdim 16.
    for mimo in [false, true] {
        let json = format!(
            r#"{{"d_model": 32, "n_layer": 1, "vocab_size": 30, "pad_vocab_size_multiple": 8,
            "ssm_cfg": {{"layer": "Mamba3", "d_state": 16, "headdim": 16, "is_mimo": {mimo},
            "mimo_rank": 2}}, "rms_norm": true, "tie_embeddings": false}}"#
        );
        let hf = parse_mamba3_config(json.as_bytes()).unwrap();
        let (cfg, rank) = (hf.cfg, hf.mimo_rank.unwrap_or(1));
        let nh = cfg.nheads();
        let ip = match hf.mimo_config().unwrap() {
            Some(m) => m.in_proj_out_dim(),
            None => cfg.in_proj_out_dim(),
        };
        let (d, ds) = (cfg.d_model, cfg.d_state);
        let mut ts = vec![
            Tensor::new("backbone.embedding.weight", &[vp, d], Dtype::BF16, 21),
            Tensor::new("backbone.norm_f.weight", &[d], Dtype::BF16, 22),
            Tensor::new("lm_head.weight", &[vp, d], Dtype::BF16, 23),
            Tensor::new(&p("norm.weight"), &[d], Dtype::BF16, 24),
        ];
        let mut mixer = vec![
            ("mixer.in_proj.weight", vec![ip, d]),
            ("mixer.dt_bias", vec![nh]),
            ("mixer.B_bias", vec![nh, rank, ds]),
            ("mixer.C_bias", vec![nh, rank, ds]),
            ("mixer.D", vec![nh]),
            ("mixer.B_norm.weight", vec![ds]),
            ("mixer.C_norm.weight", vec![ds]),
            ("mixer.out_proj.weight", vec![d, cfg.d_inner()]),
        ];
        if mimo {
            for name in ["mixer.mimo_x", "mixer.mimo_z", "mixer.mimo_o"] {
                mixer.push((name, vec![nh, rank, cfg.headdim]));
            }
        }
        for (k, (name, shape)) in mixer.into_iter().enumerate() {
            ts.push(Tensor::new(&p(name), &shape, Dtype::BF16, 30 + k as u64));
        }
        let dir = scratch(&format!("m3-untied-{mimo}"));
        std::fs::write(dir.join("config.json"), &json).unwrap();
        write_torch(
            &dir.join("pytorch_model.bin"),
            &ts.iter().collect::<Vec<_>>(),
        );
        let want = head_layout(&ts[2].read_back(), 30, d);
        let head = if mimo {
            load_mamba3_mimo(&dir).unwrap().lm_head
        } else {
            load_mamba3(&dir).unwrap().lm_head
        };
        assert_eq!(
            bits(&head.expect("an untied head")),
            bits(&want),
            "mimo {mimo}"
        );
    }
}

/// The transformers layout of a Mamba-2 checkpoint (Codestral's: its own
/// field names, `Infinity` in the dt limit, a `norm_before_gate` field its
/// weights do not run with, groups, an untied head, sharded safetensors)
/// loads the model its original-format twin loads.
#[test]
fn mamba2_transformers_layout_loads_as_its_original_twin() {
    let (d, layers, vocab) = (32usize, 2u64, 32usize);
    let original =
        br#"{"d_model": 32, "n_layer": 2, "vocab_size": 32, "pad_vocab_size_multiple": 8,
        "ssm_cfg": {"layer": "Mamba2", "d_state": 8, "headdim": 16, "ngroups": 2},
        "rms_norm": true, "tie_embeddings": false}"#;
    let transformers = br#"{"model_type": "mamba2", "architectures": ["Mamba2ForCausalLM"],
        "hidden_size": 32, "num_hidden_layers": 2, "state_size": 8, "head_dim": 16,
        "num_heads": 4, "n_groups": 2, "expand": 2, "intermediate_size": 64, "conv_kernel": 4,
        "vocab_size": 32, "tie_word_embeddings": false, "time_step_limit": [0.0, Infinity],
        "norm_before_gate": true, "rms_norm": true, "layer_norm_epsilon": 1e-05,
        "use_bias": false, "use_conv_bias": true, "chunk_size": 256}"#;
    let a_cfg = parse_mamba2_config(original).unwrap();
    let b_cfg = parse_mamba2_config(transformers).unwrap();
    assert_eq!(a_cfg.cfg, b_cfg.cfg);
    assert!(!b_cfg.cfg.norm_before_gate && b_cfg.cfg.dt_limit.1.is_infinite());
    let cfg = a_cfg.cfg;

    let layer = |i: u64| {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        let s = 50 + 20 * i;
        vec![
            Tensor::new(&p("norm.weight"), &[d], Dtype::BF16, s),
            Tensor::new(
                &p("mixer.in_proj.weight"),
                &[cfg.in_proj_out_dim(), d],
                Dtype::BF16,
                s + 1,
            ),
            Tensor::new(
                &p("mixer.conv1d.weight"),
                &[cfg.conv_dim(), 1, cfg.d_conv],
                Dtype::BF16,
                s + 2,
            ),
            Tensor::new(
                &p("mixer.conv1d.bias"),
                &[cfg.conv_dim()],
                Dtype::BF16,
                s + 3,
            ),
            Tensor::new(&p("mixer.dt_bias"), &[cfg.nheads()], Dtype::BF16, s + 4),
            Tensor::new(&p("mixer.A_log"), &[cfg.nheads()], Dtype::BF16, s + 5),
            Tensor::new(&p("mixer.D"), &[cfg.nheads()], Dtype::BF16, s + 6),
            Tensor::new(
                &p("mixer.norm.weight"),
                &[cfg.ssm_dim()],
                Dtype::BF16,
                s + 7,
            ),
            Tensor::new(
                &p("mixer.out_proj.weight"),
                &[d, cfg.d_inner()],
                Dtype::BF16,
                s + 8,
            ),
        ]
    };
    let tensors = |embedding: &str| {
        let mut ts = vec![
            Tensor::new(embedding, &[vocab, d], Dtype::BF16, 1),
            Tensor::new("backbone.norm_f.weight", &[d], Dtype::BF16, 2),
            Tensor::new("lm_head.weight", &[vocab, d], Dtype::BF16, 3),
        ];
        for i in 0..layers {
            ts.extend(layer(i));
        }
        ts
    };

    let a_dir = scratch("m2-original-twin");
    std::fs::write(a_dir.join("config.json"), original).unwrap();
    let a_ts = tensors("backbone.embedding.weight");
    write_torch(&a_dir.join("pytorch_model.bin"), &refs(&a_ts));

    let b_dir = scratch("m2-transformers");
    std::fs::write(b_dir.join("config.json"), transformers).unwrap();
    let b_ts = tensors("backbone.embeddings.weight");
    let (first, second) = b_ts.split_at(7);
    write_safetensors(
        &b_dir.join("model-00001-of-00002.safetensors"),
        &refs(first),
    );
    write_safetensors(
        &b_dir.join("model-00002-of-00002.safetensors"),
        &refs(second),
    );
    let map: Vec<(&str, &str)> = b_ts
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let file = if i < 7 {
                "model-00001-of-00002.safetensors"
            } else {
                "model-00002-of-00002.safetensors"
            };
            (t.name.as_str(), file)
        })
        .collect();
    write_index(&b_dir.join("model.safetensors.index.json"), &map);

    let a = load_mamba2(&a_dir).unwrap();
    let b = load_mamba2(&b_dir).unwrap();
    assert_eq!(a.cfg, b.cfg);
    assert_eq!(a.vocab_size, b.vocab_size);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.embed), bits(&b.embed));
    assert_eq!(
        bits(a.lm_head.as_deref().unwrap()),
        bits(b.lm_head.as_deref().unwrap())
    );
    assert_eq!(
        bits(&a.weights.norm_f_weight),
        bits(&b.weights.norm_f_weight)
    );
    for (la, lb) in a.weights.layers.iter().zip(&b.weights.layers) {
        for (x, y) in [
            (&la.norm_weight, &lb.norm_weight),
            (&la.in_proj_w, &lb.in_proj_w),
            (&la.in_proj_b, &lb.in_proj_b),
            (&la.conv_w, &lb.conv_w),
            (&la.conv_b, &lb.conv_b),
            (&la.dt_bias, &lb.dt_bias),
            (&la.a_log, &lb.a_log),
            (&la.d_param, &lb.d_param),
            (&la.gate_norm_weight, &lb.gate_norm_weight),
            (&la.out_proj_w, &lb.out_proj_w),
            (&la.out_proj_b, &lb.out_proj_b),
        ] {
            assert_eq!(bits(x), bits(y));
        }
    }
    // A tied transformers checkpoint without the head tensor loads tied.
    let tied = String::from_utf8_lossy(transformers).replace(
        "\"tie_word_embeddings\": false",
        "\"tie_word_embeddings\": true",
    );
    let c_dir = scratch("m2-transformers-tied");
    std::fs::write(c_dir.join("config.json"), tied).unwrap();
    let c_ts: Vec<&Tensor> = b_ts.iter().filter(|t| t.name != "lm_head.weight").collect();
    write_safetensors(&c_dir.join("model.safetensors"), &c_ts);
    assert!(load_mamba2(&c_dir).unwrap().lm_head.is_none());
}

/// A FalconMamba checkpoint (transformers layout, `model_type`
/// `falcon_mamba`, an untied head) loads as Mamba-1 with its mixer norms.
#[test]
fn falcon_mamba_loads_with_its_mixer_norms() {
    let dir = scratch("falcon");
    let mut ts = mamba1_tensors("backbone.embeddings.weight", Dtype::BF16);
    ts.push(Tensor::new("lm_head.weight", &[32, 16], Dtype::BF16, 7));
    write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
    std::fs::write(
        dir.join("config.json"),
        br#"{"model_type": "falcon_mamba", "hidden_size": 16, "num_hidden_layers": 2,
            "state_size": 4, "conv_kernel": 4, "expand": 16, "intermediate_size": 32,
            "time_step_rank": 1, "vocab_size": 32, "use_bias": false, "use_conv_bias": true,
            "layer_norm_epsilon": 1e-5, "mixer_rms_eps": 1e-6, "tie_word_embeddings": false}"#,
    )
    .unwrap();
    let m = load_mamba(&dir).unwrap();
    let cfg = &m.cfg;
    assert_eq!(cfg.bcdt_rms_eps, Some(1e-6));
    assert_eq!((cfg.expand, cfg.rms_norm_eps), (2, 1e-5));
    assert!(m.lm_head.is_some());
}

/// A Mamba-2 checkpoint built with `rms_norm = false` loads its LayerNorm
/// block norms: every norm's bias as stored, the gated MLP's norm2 among
/// them. One missing a bias is refused.
#[test]
fn mamba2_layer_norm_checkpoints_load() {
    let json = br#"{"d_model": 32, "n_layer": 2, "vocab_size": 32, "d_intermediate": 64,
        "ssm_cfg": {"layer": "Mamba2", "d_state": 8, "headdim": 16}, "rms_norm": false}"#;
    let cfg = parse_mamba2_config(json).unwrap().cfg;
    assert_eq!(cfg.block_norm, mamba_rs::BlockNorm::Layer);
    let (d, h) = (cfg.d_model, cfg.mlp_hidden());
    let mut ts = vec![
        Tensor::new("backbone.embedding.weight", &[32, d], Dtype::F32, 1),
        Tensor::new("backbone.norm_f.weight", &[d], Dtype::F32, 2),
        Tensor::new("backbone.norm_f.bias", &[d], Dtype::F32, 3),
    ];
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        let seed = 10 * (i as u64 + 1);
        for (k, (name, shape)) in [
            ("norm.weight", vec![d]),
            ("norm.bias", vec![d]),
            ("norm2.weight", vec![d]),
            ("norm2.bias", vec![d]),
            ("mlp.fc1.weight", vec![2 * h, d]),
            ("mlp.fc2.weight", vec![d, h]),
            ("mixer.in_proj.weight", vec![cfg.in_proj_out_dim(), d]),
            ("mixer.conv1d.weight", vec![cfg.conv_dim(), 1, cfg.d_conv]),
            ("mixer.conv1d.bias", vec![cfg.conv_dim()]),
            ("mixer.dt_bias", vec![cfg.nheads()]),
            ("mixer.A_log", vec![cfg.nheads()]),
            ("mixer.D", vec![cfg.nheads()]),
            ("mixer.norm.weight", vec![cfg.ssm_dim()]),
            ("mixer.out_proj.weight", vec![d, cfg.d_inner()]),
        ]
        .into_iter()
        .enumerate()
        {
            ts.push(Tensor::new(&p(name), &shape, Dtype::F32, seed + k as u64));
        }
    }
    let tensor = |name: &str| {
        ts.iter()
            .find(|t| t.name == name)
            .map(|t| t.read_back())
            .unwrap()
    };
    let dir = scratch("m2-layer-norm");
    std::fs::write(dir.join("config.json"), json).unwrap();
    write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
    let m = load_mamba2(&dir).unwrap();
    assert_eq!(m.weights.norm_f_bias, tensor("backbone.norm_f.bias"));
    for i in 0..cfg.n_layers {
        let p = |n: &str| format!("backbone.layers.{i}.{n}");
        assert_eq!(m.weights.layers[i].norm_bias, tensor(&p("norm.bias")));
        assert_eq!(m.weights.mlp[i].norm_bias, tensor(&p("norm2.bias")));
    }

    let missing: Vec<Tensor> = ts
        .into_iter()
        .filter(|t| t.name != "backbone.layers.1.norm2.bias")
        .collect();
    let dir = scratch("m2-layer-norm-missing");
    std::fs::write(dir.join("config.json"), json).unwrap();
    write_safetensors(&dir.join("model.safetensors"), &refs(&missing));
    assert!(load_mamba2(&dir).is_err());
}

/// Mamba-3 checkpoints built with `rms_norm = false`, SISO and MIMO, load
/// their LayerNorm block norms: every norm's bias as stored, the gated
/// MLP's norm2 among them. One missing a bias is refused.
#[test]
fn mamba3_layer_norm_checkpoints_load() {
    for mimo in [false, true] {
        let json = format!(
            r#"{{"d_model": 32, "n_layer": 2, "vocab_size": 32, "d_intermediate": 64,
            "ssm_cfg": {{"layer": "Mamba3", "d_state": 16, "headdim": 16, "is_mimo": {mimo},
            "mimo_rank": 2}}, "rms_norm": false, "tie_embeddings": true}}"#
        );
        let hf = parse_mamba3_config(json.as_bytes()).unwrap();
        let (cfg, rank) = (hf.cfg, hf.mimo_rank.unwrap_or(1));
        assert_eq!(cfg.block_norm, mamba_rs::BlockNorm::Layer);
        let ip = match hf.mimo_config().unwrap() {
            Some(m) => m.in_proj_out_dim(),
            None => cfg.in_proj_out_dim(),
        };
        let (d, ds, nh, h) = (cfg.d_model, cfg.d_state, cfg.nheads(), cfg.mlp_hidden());
        let mut ts = vec![
            Tensor::new("backbone.embedding.weight", &[32, d], Dtype::F32, 1),
            Tensor::new("backbone.norm_f.weight", &[d], Dtype::F32, 2),
            Tensor::new("backbone.norm_f.bias", &[d], Dtype::F32, 3),
        ];
        for i in 0..cfg.n_layers {
            let p = |n: &str| format!("backbone.layers.{i}.{n}");
            let seed = 100 * (i as u64 + 1);
            let mut layer = vec![
                ("norm.weight", vec![d]),
                ("norm.bias", vec![d]),
                ("norm2.weight", vec![d]),
                ("norm2.bias", vec![d]),
                ("mlp.fc1.weight", vec![2 * h, d]),
                ("mlp.fc2.weight", vec![d, h]),
                ("mixer.in_proj.weight", vec![ip, d]),
                ("mixer.dt_bias", vec![nh]),
                ("mixer.B_bias", vec![nh, rank, ds]),
                ("mixer.C_bias", vec![nh, rank, ds]),
                ("mixer.D", vec![nh]),
                ("mixer.B_norm.weight", vec![ds]),
                ("mixer.C_norm.weight", vec![ds]),
                ("mixer.out_proj.weight", vec![d, cfg.d_inner()]),
            ];
            if mimo {
                for name in ["mixer.mimo_x", "mixer.mimo_z", "mixer.mimo_o"] {
                    layer.push((name, vec![nh, rank, cfg.headdim]));
                }
            }
            for (k, (name, shape)) in layer.into_iter().enumerate() {
                ts.push(Tensor::new(&p(name), &shape, Dtype::F32, seed + k as u64));
            }
        }
        let tensor = |name: &str| {
            ts.iter()
                .find(|t| t.name == name)
                .map(|t| t.read_back())
                .unwrap()
        };
        let dir = scratch(&format!("m3-layer-norm-{mimo}"));
        std::fs::write(dir.join("config.json"), &json).unwrap();
        write_safetensors(&dir.join("model.safetensors"), &refs(&ts));
        // Every bias in the order the model holds them: the final norm's,
        // then each layer's block norm and norm2.
        let mut want = vec![tensor("backbone.norm_f.bias")];
        for i in 0..cfg.n_layers {
            want.push(tensor(&format!("backbone.layers.{i}.norm.bias")));
            want.push(tensor(&format!("backbone.layers.{i}.norm2.bias")));
        }
        let got: Vec<Vec<f32>> = if mimo {
            let w = load_mamba3_mimo(&dir).unwrap().weights;
            std::iter::once(w.norm_f_bias.clone())
                .chain(
                    w.layers
                        .iter()
                        .zip(&w.mlp)
                        .flat_map(|(l, m)| [l.norm_bias.clone(), m.norm_bias.clone()]),
                )
                .collect()
        } else {
            let w = load_mamba3(&dir).unwrap().weights;
            std::iter::once(w.norm_f_bias.clone())
                .chain(
                    w.layers
                        .iter()
                        .zip(&w.mlp)
                        .flat_map(|(l, m)| [l.norm_bias.clone(), m.norm_bias.clone()]),
                )
                .collect()
        };
        assert_eq!(got, want, "mimo {mimo}");

        let missing: Vec<Tensor> = ts
            .into_iter()
            .filter(|t| t.name != "backbone.layers.1.norm.bias")
            .collect();
        let dir = scratch(&format!("m3-layer-norm-missing-{mimo}"));
        std::fs::write(dir.join("config.json"), &json).unwrap();
        write_safetensors(&dir.join("model.safetensors"), &refs(&missing));
        let refused = if mimo {
            load_mamba3_mimo(&dir).is_err()
        } else {
            load_mamba3(&dir).is_err()
        };
        assert!(refused, "mimo {mimo}: a missing block-norm bias loaded");
    }
}
