require 'open3'
require 'tmpdir'
require 'digest'
require 'json'

# Native source equivalence only. No CUDA compilation, GPU emulation or
# performance claim. The original GPU discovery composition is the reference.
root=Dir.pwd
manifest_path=File.join(root,'internal/perf/ada-triad-nt-sibling-two-mechanism-20260907/build-cuda132/source-manifest.json')
raise 'reference manifest changed' unless Digest::SHA256.file(manifest_path).hexdigest=='b4a7d40dde86a31fc82cc808aa955b11dc8dea8d437b3e3805b89fb90fd37910'
manifest=JSON.parse(File.read(manifest_path))
%w[tests/gemm_bi_tf32_nt_compact_xor.rs tests/gemm_bi_tf32_nt_compact_xor.cu kernels/_typed_prelude.cuh kernels/gemm_bi_triad/contract.cuh kernels/gemm_bi_triad/common.cuh kernels/gemm_bi_triad/epilogue.cuh kernels/gemm_bi_triad/mma16.cuh kernels/gemm_bi_triad/sm80.cu].each do |path|
  raise "measured reference changed: #{path}" unless Digest::SHA256.file(path).hexdigest==manifest.fetch('sources').fetch(path)
end
discovery=File.join(root,'tests/gemm_bi_tf32_nt_compact_xor.rs')
production=File.join(root,'src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs')
old_helper=File.join(root,'tests/gemm_bi_tf32_nt_compact_xor.cu')
new_helper=File.join(root,'kernels/gemm_bi_triad/sm89_nt_compact.cuh')
text=File.read(discovery)
start=text.index('    fn compose_source(variant: CandidateVariant)') or raise 'reference composition missing'
stop=text.index('    fn compile_candidate(',start) or raise 'reference composition end missing'
compose=text[start...stop].gsub(/include_str!\("([^"]+)"\)/) do
  "include_str!(#{File.expand_path($1,File.dirname(discovery)).dump})"
end
source="mod measured {\ninclude!(#{discovery.dump});\n#{compose}\n"+
  "pub fn source()->String { compose_source(CandidateVariant::CompactEightWarpS2).unwrap() }\n}\n"+
  "mod production { include!(#{production.dump}); }\n"+
  "const OLD_HELPER:&str=include_str!(#{old_helper.dump});\n"+
  "const NEW_HELPER:&str=include_str!(#{new_helper.dump});\n"+<<'RS'
fn main() {
    let old_name="gemm_bi_nt_test_compact_xor_k";
    let new_name="gemm_bi_nt_compact8_xor_k";
    let strip_comments=|s:&str|s.lines().filter(|l|!l.trim().starts_with("//") && !l.trim().is_empty()).collect::<Vec<_>>().join("\n");
    assert_eq!(strip_comments(OLD_HELPER).replace(old_name,new_name),strip_comments(NEW_HELPER),"helper code changed beyond its name");
    let measured=measured::source();
    assert_eq!(measured.matches(OLD_HELPER).count(),1,"exact original helper seam");
    let old_symbol="gemm_bi_nt_test_compact_eight_warp_sm80_mma_tf32_v1_m128n64_bk32_s2";
    assert_eq!(measured.matches(old_symbol).count(),2,"original DEFINE/ASSERT pair");
    let expected=measured.replace(OLD_HELPER,NEW_HELPER)
        .replace(old_name,new_name)
        .replace(old_symbol,production::SM89_FINALIST_SYMBOL);
    let actual=production::compose_sm89_finalist_source().unwrap();
    assert_eq!(actual,expected,"production source differs from the measured compact candidate beyond helper comments/names and export name");
    print!("{actual}");
}
RS
Dir.mktmpdir('triad-finalist-source-compare-') do |dir|
  exe=File.join(dir,'compare')
  out,status=Open3.capture2e('/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc','--edition','2024','-A','warnings','-','-o',exe,stdin_data:source)
  abort out unless status.success?
  out,err,status=Open3.capture3(exe)
  abort err unless status.success?
  puts "PASS:8 frozen reference-source hashes; actual complete production source equals measured candidate after allowed naming/comment normalization."
  puts "production_composed_sha256=#{Digest::SHA256.hexdigest(out)} bytes=#{out.bytesize}"
end
