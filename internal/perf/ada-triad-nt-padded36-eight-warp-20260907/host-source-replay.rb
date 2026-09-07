require 'open3'
require 'tmpdir'
path = File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
source = "include!(#{path.dump});\n" + <<'RS'
#[test]
fn root_eight_warp_exports_and_exact_change_boundary() {
    let source = CandidateVariant::PaddedEightWarp.source().unwrap();
    let old_symbol = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3";
    let symbol = CandidateVariant::PaddedEightWarp.symbol();
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
    let changes = [
        (
            concat!(
                "    constexpr bool eight_compute_warps =\n",
                "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 3;\n",
                "    constexpr int MAtoms = eight_compute_warps ? 2\n",
                "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
            ),
            "    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);",
            1
        ),
        (
            "    bool compute = eight_compute_warps || BM != 128 || warp < 4;",
            "    bool compute = BM != 128 || warp < 4;",
            1
        ),
        (
            concat!(
                "    int warp_m = eight_compute_warps ? (warp >> 1) * 32\n",
                "        : (BM == 128 ? (warp >> 1) * 64\n",
                "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
            ),
            concat!(
                "    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
                "        : (BM == 64 ? (warp >> 1) * 32 : 0);"
            ),
            1
        ),
        (symbol, old_symbol, 2)
    ];
    let mut restored = source;
    for (new, old, count) in changes {
        assert_eq!(restored.matches(new).count(), count);
        restored = restored.replace(new, old);
    }
    assert_eq!(restored, PRODUCTION_CUDA, "unexpected change outside three ownership declarations and export/assert names");
}
RS
Dir.mktmpdir('triad-eight-warp-host-') do |dir|
  exe = File.join(dir, 'host-tests')
  output, status = Open3.capture2e(
    '/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc',
    '--edition', '2024', '--test', '-', '-o', exe, stdin_data: source)
  abort "compile failure #{output}" unless status.success?
  output, status = Open3.capture2e(exe)
  puts output
  abort "host tests exit #{status.exitstatus}" unless status.success?
end
