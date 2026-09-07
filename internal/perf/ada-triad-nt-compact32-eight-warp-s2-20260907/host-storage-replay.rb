require 'open3'
require 'tmpdir'

# Compile only the actual generated storage/slot address functions on the host.
# This is not CUDA compilation or an async/MMA emulator.
prefix = <<'CPP'
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#define __device__
#define __forceinline__ inline
#define __align__(n) alignas(n)
CPP
suffix = <<'CPP'
void require(bool b,const char* m) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",m); std::exit(1); } }
template<int BM,int BN,int Stages> void verify_nt() {
  constexpr bool compact=BM==128 && BN==64 && Stages==2;
  constexpr int stride=compact?32:36;
  SgbTf32Storage<SgbTf32Nt,BM,BN,Stages> storage;
  require(sizeof(storage)==Stages*(BM+BN)*stride*4,"actual storage extent");
  for(int stage=0;stage<Stages;++stage) {
    for(int row=0;row<BM;++row) for(int k=0;k<32;++k) {
      const int physical=compact? k^((row&7)<<2):k;
      require(&gemm_bi_tf32_a_slot<SgbTf32Nt>(&storage,stage,row,k)==&storage.a[stage][row][physical],"A slot mapping");
    }
    for(int col=0;col<BN;++col) for(int k=0;k<32;++k) {
      const int physical=compact? k^((col&7)<<2):k;
      require(&gemm_bi_tf32_b_slot<SgbTf32Nt>(&storage,stage,k,col)==&storage.b[stage][col][physical],"B slot mapping");
    }
  }
}
int main() {
  verify_nt<128,64,2>(); verify_nt<128,64,3>();
  verify_nt<64,64,2>(); verify_nt<64,64,3>();
  verify_nt<16,32,4>(); verify_nt<16,32,3>();
  verify_nt<32,32,3>(); verify_nt<32,32,4>(); verify_nt<16,16,4>();
  std::puts("PASS actual generated compactS2 storage/slots: all9 NT instantiations; every active A/B word; all original NN/TN extent assertions");
}
CPP
Dir.mktmpdir('triad-compact8s2-storage-') do |dir|
  rust_bin=File.join(dir,'emit-source')
  path=File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
  rust="include!(#{path.dump});\nfn main() { print!(\"{}\", CandidateVariant::CompactEightWarpS2.source().unwrap()); }\n"
  out,status=Open3.capture2e('/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc','--edition','2024','-A','warnings','-','-o',rust_bin,stdin_data:rust)
  abort "Rust source emitter compile failed: #{out}" unless status.success?
  generated,status=Open3.capture2e(rust_bin)
  abort "Rust source emitter failed: #{generated}" unless status.success?
  first=generated.index('enum SgbTf32Op {') or raise 'storage start missing'
  last=generated.index('struct SgbTf32Problem {',first) or raise 'slot end missing'
  helper=File.read('tests/gemm_bi_tf32_nt_compact_xor.cu')
  raise 'actual helper binding' unless generated.start_with?(helper+"\n")
  actual=helper+"\n"+generated[first...last]
  variants={
    'actual'=>actual,
    'mutate_xor_rows'=>actual.sub('(row_or_column & 7)','(row_or_column & 3)'),
    'mutate_b_slot'=>actual.sub('gemm_bi_nt_test_compact_xor_k(column, reduction)','gemm_bi_nt_test_compact_xor_k(column + 1, reduction)')
  }
  variants.each do |name,body|
    raise "mutation missing #{name}" if name!='actual' && body==actual
    exe=File.join(dir,name)
    out,err,status=Open3.capture3('xcrun','clang++','-std=c++17','-O2','-Werror','-x','c++','-','-o',exe,stdin_data:prefix+body+suffix)
    puts "#{name} compile_exit=#{status.exitstatus}"
    abort out+err unless status.success?
    out,err,status=Open3.capture3(exe)
    puts "#{name} run_exit=#{status.exitstatus}",out,err
    raise 'unexpected behavior' unless name=='actual' ? status.success? : !status.success?
  end
end
