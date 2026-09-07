require 'open3'
require 'tmpdir'
path=File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
source = "include!(#{path.dump});\n" + <<'RS'
#[test]
fn root_generated_exports_resolve_all_signature_assertions() {
    let old = "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3";
    for variant in [CandidateVariant::PaddedCopyPlan, CandidateVariant::PaddedLdmatrix] {
        let source = variant.source().unwrap();
        assert_eq!(source.matches(old).count(), 0);
        assert_eq!(source.matches(variant.symbol()).count(), 2);
        let exports: std::collections::BTreeSet<_> = source.lines()
            .filter_map(|line| line.strip_prefix("GEMM_BI_TF32_DEFINE_KERNEL("))
            .map(|line| line.split(',').next().unwrap()).collect();
        let assertions: std::collections::BTreeSet<_> = source.lines()
            .filter_map(|line| line.strip_prefix("TF32_ASSERT_KERNEL_SIGNATURE("))
            .map(|line| line.strip_suffix(");").unwrap()).collect();
        assert_eq!(exports.len(), 18);
        assert_eq!(exports, assertions, "{} export/assert resolution", variant.name());
    }
}
RS
Dir.mktmpdir('triad-signature-root-') do |dir|
  exe=File.join(dir,'host-tests')
  out,err,status=Open3.capture3('/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc','--edition','2024','--test','-','-o',exe,stdin_data:source)
  puts out,err
  abort "compile #{status.exitstatus}" unless status.success?
  out,err,status=Open3.capture3(exe)
  puts out,err
  abort "test #{status.exitstatus}" unless status.success?
end
