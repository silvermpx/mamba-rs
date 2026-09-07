require 'open3'
require 'tmpdir'
path=File.expand_path('src/mamba_ssm/gpu/gemm_bi_triad/sm89_finalist_source.rs')
source="mod sm89_finalist_source { include!(#{path.dump}); }\n"
rustc='/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc'
Dir.mktmpdir('triad-finalist-native-') do |dir|
  {'native'=>[], 'cuda_cfg_only'=>['--cfg','feature="cuda"']}.each do |label,options|
    exe=File.join(dir,label)
    out,status=Open3.capture2e(rustc,'--edition','2024','--test',*options,'-','-o',exe,stdin_data:source)
    puts "#{label} compile_exit=#{status.exitstatus}",out
    abort 'native compile failed' unless status.success?
    out,status=Open3.capture2e(exe,'sm89_finalist_source::tests::','--nocapture')
    puts "#{label} test_exit=#{status.exitstatus}",out
    abort 'native tests failed' unless status.success?
  end
end
# cuda_cfg_only checks Rust conditional-compilation boundaries. It is still
# native macOS rustc: no CUDA headers, NVRTC, GPU execution or ABI proof.
