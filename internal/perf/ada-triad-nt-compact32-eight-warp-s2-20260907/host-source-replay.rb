require 'open3'
require 'tmpdir'
path = File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
source = "include!(#{path.dump});\n" + <<'RS'
#[test]
fn root_compact8s2_exports_and_change_boundary() {
    let source = CandidateVariant::CompactEightWarpS2.source().unwrap();
    let symbol = CandidateVariant::CompactEightWarpS2.symbol();
    let old = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2";
    assert_eq!(source.matches(old).count(), 0);
    assert_eq!(source.matches(symbol).count(), 2);
    let exports: std::collections::BTreeSet<_> = source.lines()
        .filter_map(|line| line.strip_prefix("GEMM_BI_TF32_DEFINE_KERNEL("))
        .map(|line| line.split(',').next().unwrap()).collect();
    let assertions: std::collections::BTreeSet<_> = source.lines()
        .filter_map(|line| line.strip_prefix("TF32_ASSERT_KERNEL_SIGNATURE("))
        .map(|line| line.strip_suffix(");").unwrap()).collect();
    assert_eq!(exports.len(), 18);
    assert_eq!(exports, assertions);
    let prefix = format!("{CANDIDATE_CUDA}\n");
    let mut restored = source.strip_prefix(&prefix).unwrap().replace(symbol,old);
    // The separate actual-generated C++ replay checks this full storage/slot
    // region, including all nine NT specializations and other extent asserts.
    let begin = "enum SgbTf32Op {";
    let end = "struct SgbTf32Problem {";
    let a=restored.find(begin).unwrap(); let b=restored.find(end).unwrap();
    let pa=PRODUCTION_CUDA.find(begin).unwrap(); let pb=PRODUCTION_CUDA.find(end).unwrap();
    restored.replace_range(a..b,&PRODUCTION_CUDA[pa..pb]);
    let changes = [
      (concat!(
        "    constexpr bool compact_eight_warp_s2 =\n",
        "        Op == SgbTf32Nt && BM == 128 && BN == 64 && Stages == 2;\n",
        "    constexpr int MAtoms = compact_eight_warp_s2 ? 2\n",
        "        : (BM == 128 ? 4 : (BM == 64 ? 2 : 1));"
       ),"    constexpr int MAtoms = BM == 128 ? 4 : (BM == 64 ? 2 : 1);"),
      ("    bool compute = compact_eight_warp_s2 || BM != 128 || warp < 4;",
       "    bool compute = BM != 128 || warp < 4;"),
      (concat!(
        "    int warp_m = compact_eight_warp_s2 ? (warp >> 1) * 32\n",
        "        : (BM == 128 ? (warp >> 1) * 64\n",
        "        : (BM == 64 ? (warp >> 1) * 32 : 0));"
       ),concat!("    int warp_m = BM == 128 ? (warp >> 1) * 64\n",
                 "        : (BM == 64 ? (warp >> 1) * 32 : 0);"))
    ];
    for(new,old) in changes {
      assert_eq!(restored.matches(new).count(),1);
      restored=restored.replace(new,old);
    }
    assert_eq!(restored,PRODUCTION_CUDA,"unexpected change outside separately verified storage/slots, ownership declarations and symbol pair");
}
RS
Dir.mktmpdir('triad-compact8s2-source-host-') do |dir|
  exe=File.join(dir,'host-tests')
  output,status=Open3.capture2e('/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc','--edition','2024','--test','-','-o',exe,stdin_data:source)
  abort "compile failure #{output}" unless status.success?
  output,status=Open3.capture2e(exe)
  puts output
  abort "host tests exit #{status.exitstatus}" unless status.success?
end
