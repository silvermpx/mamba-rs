require 'open3'
require 'tmpdir'
path = File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
source = "include!(#{path.dump});\n" + <<'RS'
#[test]
fn root_dense_copy_exports_and_exact_change_boundary() {
    let source = CandidateVariant::PaddedDenseCopy.source().unwrap();
    let old_symbol = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3";
    let symbol = CandidateVariant::PaddedDenseCopy.symbol();
    assert_eq!(source.matches(old_symbol).count(), 0);
    assert_eq!(source.matches(symbol).count(), 2);
    let exports: std::collections::BTreeSet<_> = source.lines()
        .filter_map(|line| line.strip_prefix("GEMM_BI_TF32_DEFINE_KERNEL("))
        .map(|line| line.split(',').next().unwrap()).collect();
    let assertions: std::collections::BTreeSet<_> = source.lines()
        .filter_map(|line| line.strip_prefix("TF32_ASSERT_KERNEL_SIGNATURE("))
        .map(|line| line.strip_suffix(");").unwrap()).collect();
    assert_eq!(exports.len(), 18);
    assert_eq!(exports, assertions);
    let added_header = format!("\n\n{PADDED_DENSE_COPY_CUDA}");
    assert_eq!(source.matches(&added_header).count(), 1);
    let mut restored = source.replace(&added_header, "").replace(symbol, old_symbol);
    let start = "    if (wide_a && wide_b) {";
    let end = "    } else if (gemm_bi_tf32_can_stage_async_4(a, b, params)) {";
    let a = restored.find(start).unwrap();
    let b = a + restored[a..].find(end).unwrap();
    let pa = PRODUCTION_CUDA.find(start).unwrap();
    let pb = pa + PRODUCTION_CUDA[pa..].find(end).unwrap();
    let branch = &restored[a..b];
    assert!(branch.contains("if constexpr (Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 3)"));
    assert_eq!(branch.matches("if (gemm_bi_tf32_nt_test_padded_dense_target(params))").count(), 1);
    assert_eq!(branch.matches("gemm_bi_tf32_nt_test_padded_dense_mainloop<MAtoms, NAtoms>").count(), 1);
    assert_eq!(branch.matches("Op, BM, BN, Stages, MAtoms, NAtoms, false, false>").count(), 2);
    restored.replace_range(a..b, &PRODUCTION_CUDA[pa..pb]);
    assert_eq!(restored, PRODUCTION_CUDA, "unexpected change outside helper insertion, guarded wide branch, export/assert names");
}
RS
Dir.mktmpdir('triad-dense-source-host-') do |dir|
  exe = File.join(dir, 'host-tests')
  output, status = Open3.capture2e(
    '/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc',
    '--edition', '2024', '--test', '-', '-o', exe, stdin_data: source)
  abort "compile failure #{output}" unless status.success?
  output, status = Open3.capture2e(exe)
  puts output
  abort "host tests exit #{status.exitstatus}" unless status.success?
end
