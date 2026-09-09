#[path = "../src/mamba_ssm/gpu/gemm_bi_triad/sm89_tf32_joint_source.rs"]
mod joint;

#[path = "support/triad_tf32_nn_n96_direct_epilogue_source.rs"]
mod nn_direct;
#[path = "support/triad_nn_n96_source.rs"]
mod nn_parent;
#[path = "support/triad_tf32_tn_transpose_rna_m64n64_source.rs"]
mod tn_m64n64;
#[path = "support/triad_tf32_tn_transpose_rna_n96_source.rs"]
mod tn_n96;
#[path = "support/triad_tn_transpose_n96_source.rs"]
mod tn_raw;

const FIXED_N96: &str = include_str!("../kernels/gemm_bi_fixed/tf32_rna_n96.cu");
const FIXED_TF32: &str = include_str!("../kernels/gemm_bi_fixed/tf32.cu");

const TN_N96_SECTION: &str = "TN_N96";
const TN_M64N64_SECTION: &str = "TN_M64N64";
const NN_DIRECT_N96_SECTION: &str = "NN_DIRECT_N96";

fn retained_tn_n96() -> String {
    let raw = tn_raw::candidate_source(FIXED_N96).expect("compose raw TN N96 parent");
    tn_n96::compose_candidate_source(&raw).expect("compose retained pre-RNA TN N96")
}

fn retained_tn_m64n64() -> String {
    let raw = tn_raw::candidate_source(FIXED_N96).expect("compose raw TN N96 parent");
    tn_m64n64::candidate_source(&raw).expect("compose retained pre-RNA TN M64N64")
}

fn retained_nn_direct_n96() -> String {
    let add_half =
        nn_parent::compose_triad_nn_n96_source(FIXED_N96).expect("compose add-half NN N96");
    nn_direct::compose_candidate_source(&add_half).expect("compose retained direct NN N96")
}

fn extract_section(source: &str, label: &str) -> String {
    let begin = format!("// BEGIN RETAINED {label}\n");
    let end = format!("// END RETAINED {label}\n");
    let body = source
        .split_once(&begin)
        .unwrap_or_else(|| panic!("missing {label} begin marker"))
        .1
        .split_once(&end)
        .unwrap_or_else(|| panic!("missing {label} end marker"))
        .0;
    let namespace = format!(
        "namespace sm89_tf32_joint_{} {{\n",
        label.to_ascii_lowercase()
    );
    let body = body
        .strip_prefix(&namespace)
        .unwrap_or_else(|| panic!("missing {label} namespace wrapper"));
    body.strip_suffix("}\n")
        .unwrap_or_else(|| panic!("missing {label} namespace terminator"))
        .to_owned()
}

fn remove_single_export(source: &str, symbol: &str) -> String {
    let name = format!("void {symbol}(");
    let name_at = source
        .find(&name)
        .unwrap_or_else(|| panic!("missing removable export {symbol}"));
    assert_eq!(
        source.matches(&name).count(),
        1,
        "ambiguous export {symbol}"
    );
    let begin = source[..name_at]
        .rfind("extern \"C\"")
        .expect("missing extern-C export prefix");
    let mut depth = 0_i32;
    let mut opened = false;
    let mut end = None;
    for (offset, byte) in source[name_at..].bytes().enumerate() {
        match byte {
            b'{' => {
                opened = true;
                depth += 1;
            }
            b'}' if opened => {
                depth -= 1;
                if depth == 0 {
                    end = Some(name_at + offset + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let mut end = end.expect("unterminated removable export");
    while source.as_bytes().get(end) == Some(&b'\n') {
        end += 1;
    }
    format!("{}{}", &source[..begin], &source[end..])
}

fn extract_device_function<'a>(source: &'a str, name: &str) -> &'a str {
    let signature = format!("void {name}(");
    let name_at = source
        .find(&signature)
        .unwrap_or_else(|| panic!("missing device function {name}"));
    assert_eq!(
        source.matches(&signature).count(),
        1,
        "ambiguous device function {name}"
    );
    let start = source[..name_at]
        .rfind("__device__")
        .unwrap_or_else(|| panic!("missing __device__ prefix for {name}"));
    let mut depth = 0_i32;
    let mut opened = false;
    for (offset, byte) in source[name_at..].bytes().enumerate() {
        match byte {
            b'{' => {
                opened = true;
                depth += 1;
            }
            b'}' if opened => {
                depth -= 1;
                if depth == 0 {
                    return &source[start..=name_at + offset];
                }
            }
            _ => {}
        }
    }
    panic!("unterminated device function {name}")
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let padded_len = (bytes.len() + 9).div_ceil(64) * 64;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded[bytes.len()] = 0x80;
    padded[padded_len - 8..].copy_from_slice(&bit_len.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ (!e & g);
            let temp1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(K[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut digest = [0_u8; 32];
    for (index, word) in state.into_iter().enumerate() {
        digest[index * 4..index * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

fn digest_hex(digest: [u8; 32]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

#[test]
fn test_sha256_implementation_matches_the_standard_abc_vector() {
    assert_eq!(
        digest_hex(sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn joint_source_has_frozen_exact_inventory_and_one_transpose_export() {
    joint::validate_source().expect("sealed joint source contract");
    assert_eq!(
        digest_hex(sha256(joint::SOURCE.as_bytes())),
        joint::SOURCE_SHA256
    );
    assert_eq!(
        joint::export_inventory(joint::SOURCE).unwrap(),
        joint::SM89_TF32_JOINT_SYMBOLS
    );
    assert_eq!(
        joint::SOURCE
            .matches(joint::TN_PRE_RNA_TRANSPOSE_SYMBOL)
            .count(),
        1,
        "the common transpose must have one physical definition"
    );
}

#[test]
fn materialized_sections_are_normalized_retained_bodies_only() {
    let expected_tn_n96 = retained_tn_n96()
        .replace(tn_n96::CANDIDATE_GEMM_SYMBOL, joint::TN_PRE_RNA_N96_SYMBOL)
        .replace(
            tn_n96::CANDIDATE_TRANSPOSE_SYMBOL,
            joint::TN_PRE_RNA_TRANSPOSE_SYMBOL,
        );
    assert_eq!(
        extract_section(joint::SOURCE, TN_N96_SECTION),
        expected_tn_n96
    );

    let expected_tn_m64n64 =
        retained_tn_m64n64().replace(tn_m64n64::GEMM_SYMBOL, joint::TN_PRE_RNA_M64N64_SYMBOL);
    let expected_tn_m64n64 = remove_single_export(&expected_tn_m64n64, tn_m64n64::TRANSPOSE_SYMBOL);
    assert_eq!(
        extract_section(joint::SOURCE, TN_M64N64_SECTION),
        expected_tn_m64n64
    );

    let expected_nn =
        retained_nn_direct_n96().replace(nn_direct::SYMBOL, joint::NN_ADD_HALF_DIRECT_N96_SYMBOL);
    assert_eq!(
        extract_section(joint::SOURCE, NN_DIRECT_N96_SECTION),
        expected_nn
    );
}

#[test]
fn retained_transforms_are_reversible_and_joint_source_has_no_discovery_markers() {
    let retained_tn = retained_tn_n96();
    let raw = tn_raw::candidate_source(FIXED_N96).unwrap();
    assert_eq!(tn_n96::restore_retained_source(&retained_tn).unwrap(), raw);

    let retained_nn = retained_nn_direct_n96();
    let add_half = nn_parent::compose_triad_nn_n96_source(FIXED_N96).unwrap();
    assert_eq!(
        nn_direct::restore_retained_source(&retained_nn).unwrap(),
        add_half
    );
    assert!(!joint::SOURCE.contains("_test_"));
    assert!(!joint::SOURCE.contains("_exp_"));
}

#[test]
fn sealed_validator_rejects_extra_exports_and_discovery_markers() {
    let extra = format!(
        "{}\nextern \"C\" __global__ void gemm_bi_unqualified_extra() {{}}\n",
        joint::SOURCE
    );
    assert!(joint::validate_source_text(&extra).is_err());

    for marker in ["_test_", "_exp_"] {
        let contaminated = format!("{}\n// {marker}\n", joint::SOURCE);
        assert!(joint::validate_source_text(&contaminated).is_err());
    }
}

#[test]
fn composed_source_contains_only_the_four_sealed_exports() {
    let composed = joint::compose_source().expect("compose standalone joint source");
    assert_eq!(
        joint::export_inventory(&composed).unwrap(),
        joint::SM89_TF32_JOINT_SYMBOLS
    );
    assert_eq!(composed.matches("extern \"C\"").count(), 4);
    assert!(composed.starts_with(joint::PRIMITIVES));
    joint::validate_source_text(&composed).expect("composed inventory remains sealed");
}

#[test]
fn primitive_owner_is_frozen_and_byte_equal_to_fixed_tf32_definitions() {
    assert_eq!(
        digest_hex(sha256(joint::PRIMITIVES.as_bytes())),
        joint::PRIMITIVES_SHA256
    );
    joint::validate_primitives_text(joint::PRIMITIVES).expect("sealed primitive owner");
    for name in [joint::COPY_CG_PRIMITIVE, joint::MMA_M16N8K8_PRIMITIVE] {
        assert_eq!(
            extract_device_function(joint::PRIMITIVES, name),
            extract_device_function(FIXED_TF32, name),
            "primitive {name} drifted from the proven Fixed TF32 definition"
        );
    }
    assert_eq!(
        joint::export_inventory(joint::PRIMITIVES).unwrap(),
        Vec::<&str>::new()
    );
}

#[test]
fn primitive_validator_rejects_missing_foreign_and_exported_code() {
    let missing = joint::PRIMITIVES.replacen("gbf_tf32_copy_cg", "gbf_tf32_copy_removed", 1);
    assert!(joint::validate_primitives_text(&missing).is_err());

    let foreign = format!(
        "{}\n__device__ __forceinline__ void gbf_tf32_copy_ca(unsigned, const void*, int) {{}}\n",
        joint::PRIMITIVES
    );
    assert!(joint::validate_primitives_text(&foreign).is_err());

    let foreign_nonvoid = format!(
        "{}\n__device__ __forceinline__ unsigned gbf_tf32_rna(float value) {{ return (unsigned)value; }}\n",
        joint::PRIMITIVES
    );
    assert!(joint::validate_primitives_text(&foreign_nonvoid).is_err());

    let exported = format!(
        "{}\nextern \"C\" __global__ void gemm_bi_foreign() {{}}\n",
        joint::PRIMITIVES
    );
    assert!(joint::validate_primitives_text(&exported).is_err());
}

#[test]
fn typed_params_match_the_frozen_driver_abi() {
    assert_eq!(
        std::mem::size_of::<joint::Sm89Tf32JointTransposeParams>(),
        12
    );
    assert_eq!(
        std::mem::align_of::<joint::Sm89Tf32JointTransposeParams>(),
        4
    );
    assert_eq!(std::mem::size_of::<joint::Sm89Tf32JointGemmParams>(), 32);
    assert_eq!(std::mem::align_of::<joint::Sm89Tf32JointGemmParams>(), 4);
    assert_eq!(
        joint::TRANSPOSE_DRIVER_ABI,
        [
            joint::Sm89Tf32JointAbiParameter { offset: 0, size: 8 },
            joint::Sm89Tf32JointAbiParameter { offset: 8, size: 8 },
            joint::Sm89Tf32JointAbiParameter {
                offset: 16,
                size: 12
            },
        ]
    );
    assert_eq!(joint::TRANSPOSE_TERMINAL_ARGUMENT, 3);
    assert_eq!(
        joint::GEMM_DRIVER_ABI,
        [
            joint::Sm89Tf32JointAbiParameter { offset: 0, size: 8 },
            joint::Sm89Tf32JointAbiParameter { offset: 8, size: 8 },
            joint::Sm89Tf32JointAbiParameter {
                offset: 16,
                size: 8
            },
            joint::Sm89Tf32JointAbiParameter {
                offset: 24,
                size: 8
            },
            joint::Sm89Tf32JointAbiParameter {
                offset: 32,
                size: 32
            },
        ]
    );
    assert_eq!(joint::GEMM_TERMINAL_ARGUMENT, 5);
    assert!(joint::SOURCE.contains(
        "void gemm_bi_tn_sm89_tf32_pre_rna_transpose_32x32_v1(\n    const unsigned* input, unsigned* output, GbfTf32TnTransposeParams params)"
    ));
}

#[test]
fn four_typed_specs_bind_symbols_abi_and_retained_resources() {
    use joint::Sm89Tf32JointKernelKind as Kind;

    assert_eq!(joint::SM89_TF32_JOINT_KERNEL_SPECS.len(), 4);
    let expected = [
        (
            joint::NN_ADD_HALF_DIRECT_N96_SYMBOL,
            Kind::NnAddHalfDirectM128N96Bk32S3,
            (256, 1, 1),
            86_016,
            0,
            124,
            Some(1),
            5,
            64,
        ),
        (
            joint::TN_PRE_RNA_N96_SYMBOL,
            Kind::TnPreRnaM128N96Bk32S3,
            (256, 1, 1),
            86_016,
            0,
            127,
            Some(1),
            5,
            64,
        ),
        (
            joint::TN_PRE_RNA_M64N64_SYMBOL,
            Kind::TnPreRnaM64N64Bk32S3,
            (256, 1, 1),
            49_152,
            0,
            83,
            Some(2),
            5,
            64,
        ),
        (
            joint::TN_PRE_RNA_TRANSPOSE_SYMBOL,
            Kind::TnPreRnaTranspose32x32,
            (32, 8, 1),
            0,
            4_224,
            26,
            None,
            3,
            28,
        ),
    ];
    for (symbol, kind, block, dynamic, static_bytes, regs, occupancy, argc, abi_bytes) in expected {
        let spec = joint::kernel_spec(symbol).expect("typed joint kernel spec");
        assert_eq!(spec.kind, kind);
        assert_eq!(spec.block, block);
        assert_eq!(spec.dynamic_shared_bytes, dynamic);
        assert_eq!(spec.static_shared_bytes, static_bytes);
        assert_eq!(spec.register_cap, regs);
        assert_eq!(spec.local_bytes, 0);
        assert_eq!(spec.minimum_max_threads, 256);
        assert_eq!(spec.minimum_active_blocks, occupancy);
        assert_eq!(spec.abi_parameters.len(), argc);
        assert_eq!(spec.abi_parameter_bytes, abi_bytes);
    }
    assert_eq!(
        joint::SM89_TF32_JOINT_KERNEL_SPECS.map(|spec| spec.symbol),
        joint::SM89_TF32_JOINT_SYMBOLS
    );
    assert!(joint::kernel_spec("gemm_bi_unknown").is_none());
}
