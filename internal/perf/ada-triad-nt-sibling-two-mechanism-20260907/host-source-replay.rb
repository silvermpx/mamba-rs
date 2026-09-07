require 'open3'
require 'tmpdir'
require 'digest'
path = File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
source = "include!(#{path.dump});\n" + <<'RS'
#[test]
fn root_four_sibling_exports_and_exact_boundaries() {
    for variant in [CandidateVariant::PaddedDenseD768Out,
                    CandidateVariant::PaddedDensePrism,
                    CandidateVariant::CompactEightWarpS2D768Out,
                    CandidateVariant::CompactEightWarpS2Prism] {
        let source = variant.source().unwrap();
        let exports: std::collections::BTreeSet<_> = source.lines()
            .filter_map(|l| l.strip_prefix("GEMM_BI_TF32_DEFINE_KERNEL("))
            .map(|l| l.split(',').next().unwrap()).collect();
        let assertions: std::collections::BTreeSet<_> = source.lines()
            .filter_map(|l| l.strip_prefix("TF32_ASSERT_KERNEL_SIGNATURE("))
            .map(|l| l.strip_suffix(");").unwrap()).collect();
        assert_eq!(exports.len(),18);
        assert_eq!(exports,assertions);
        assert_eq!(source.matches(variant.symbol()).count(),2);
        let old = if variant.required_occupancy()==2 {
            "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s2"
        } else { "gemm_bi_nt_sm80_mma_tf32_v1_m128n64_bk32_s3" };
        assert_eq!(source.matches(old).count(),0);
    }
    for variant in [CandidateVariant::CompactEightWarpS2D768Out,
                    CandidateVariant::CompactEightWarpS2Prism] {
        assert_eq!(variant.source().unwrap(),CandidateVariant::CompactEightWarpS2.source().unwrap());
        assert_eq!(variant.symbol(),CandidateVariant::CompactEightWarpS2.symbol());
    }
    let base=CandidateVariant::PaddedDenseCopy.source().unwrap();
    let mut out=CandidateVariant::PaddedDenseD768Out.source().unwrap();
    let guard=format!("\n\n{PADDED_DENSE_D768_OUT_GUARD_CUDA}");
    assert_eq!(out.matches(&guard).count(),1);
    out=out.replace(&guard,"")
        .replace("gemm_bi_tf32_nt_test_padded_dense_d768_out_target(params)",
                 "gemm_bi_tf32_nt_test_padded_dense_target(params)")
        .replace(PADDED_DENSE_D768_OUT_SYMBOL,PADDED_DENSE_COPY_SYMBOL);
    assert_eq!(out,base,"out changes outside added guard, caller and symbol pair");
    let mut prism=CandidateVariant::PaddedDensePrism.source().unwrap();
    let header=format!("\n\n{PADDED_DENSE_PRISM_CUDA}");
    assert_eq!(prism.matches(&header).count(),1);
    prism=prism.replace(&header,"")
        .replace("gemm_bi_tf32_nt_test_padded_dense_prism_target(params)",
                 "gemm_bi_tf32_nt_test_padded_dense_target(params)")
        .replace("gemm_bi_tf32_nt_test_padded_dense_prism_mainloop<MAtoms, NAtoms>(",
                 "gemm_bi_tf32_nt_test_padded_dense_mainloop<MAtoms, NAtoms>(")
        .replace(PADDED_DENSE_PRISM_SYMBOL,PADDED_DENSE_COPY_SYMBOL);
    assert_eq!(prism,base,"prism changes outside added header, caller and symbol pair");
}
RS
Dir.mktmpdir('triad-sibling-source-host-') do |dir|
  exe=File.join(dir,'tests')
  rustc='/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc'
  output,status=Open3.capture2e(rustc,'--edition','2024','--test','-','-o',exe,stdin_data:source)
  abort output unless status.success?
  output,status=Open3.capture2e(exe); puts output
  abort "native exit #{status.exitstatus}" unless status.success?
  # Compile the actual runtime composition function on the host, not just
  # the transformed sm80 body: the recorded digest includes five preambles.
  runtime=File.read(path)
  start=runtime.index('    fn compose_source(variant: CandidateVariant)') or raise 'composition missing'
  stop=runtime.index('    fn compile_candidate(',start) or raise 'composition end missing'
  compose=runtime[start...stop].gsub(/include_str!\("([^"]+)"\)/) do
    "include_str!(#{File.expand_path($1,File.dirname(path)).dump})"
  end
  emitter="include!(#{path.dump});\n#{compose}\n" + <<'RS'
fn main() {
  let variant=match std::env::args().nth(1).unwrap().as_str() {
    "compact"=>CandidateVariant::CompactEightWarpS2,
    "out"=>CandidateVariant::PaddedDenseD768Out,
    "prism"=>CandidateVariant::PaddedDensePrism,
    _=>panic!("unknown variant")
  };
  print!("{}",compose_source(variant).unwrap());
}
RS
  exe=File.join(dir,'emit')
  output,status=Open3.capture2e(rustc,'--edition','2024','-A','warnings','-','-o',exe,stdin_data:emitter)
  abort output unless status.success?
  %w[compact out prism].each do |variant|
    generated,status=Open3.capture2e(exe,variant); abort generated unless status.success?
    hash=Digest::SHA256.hexdigest(generated)
    if variant=='compact'
      raise "compact CUDA changed #{hash}" unless hash=='fd5bd92f7cc56276a1b95b24b59319142b247f7ddedb14d6b06cac91843205b4'
    end
    puts "PASS actual composed CUDA: #{variant} #{hash}"
  end
end
