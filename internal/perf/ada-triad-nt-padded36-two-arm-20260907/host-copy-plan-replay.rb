require 'open3'
require 'tmpdir'
header = File.read('tests/gemm_bi_tf32_nt_padded_copy_plan.cuh').split('template <int MAtoms, int NAtoms>', 2).first
commit = 'asm volatile("cp.async.commit_group;\n" ::);'
raise 'commit seam count' unless header.scan(commit).size == 1
header = header.sub(commit, '/* host address-only: opaque commit removed */')
prefix = <<'CPP'
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __device__
#define __forceinline__ inline
struct { int x; } threadIdx;
enum SgbTf32Op { SgbTf32Nt };
template<SgbTf32Op Op, int BM, int BN, int Stages>
struct SgbTf32Storage { float a[Stages][BM][36]; float b[Stages][BN][36]; };
struct Params { int m,k,n,lda,ldb; };
struct SgbTf32Problem { const float* a; const float* b; Params params; int tile_row,tile_column; };
template<SgbTf32Op Op,int BM,int BN,int Stages>
float& gemm_bi_tf32_a_slot(SgbTf32Storage<Op,BM,BN,Stages>* p,int s,int r,int k) { return p->a[s][r][k]; }
template<SgbTf32Op Op,int BM,int BN,int Stages>
float& gemm_bi_tf32_b_slot(SgbTf32Storage<Op,BM,BN,Stages>* p,int s,int k,int c) { return p->b[s][c][k]; }
std::uintptr_t storage_base;
unsigned __cvta_generic_to_shared(const float* p) { return (unsigned)((std::uintptr_t)p-storage_base); }
const float* gemm_bi_cp_async_source(const float* p,long long off,int) { return (const float*)((std::uintptr_t)p+off*4); }
struct Copy { unsigned dst; std::uintptr_t src; int bytes; };
std::vector<Copy> observed;
template<bool Narrow,int BM>
void gemm_bi_tf32_cp_async_zfill(unsigned d,const float* s,int b) { observed.push_back({d,(std::uintptr_t)s,b}); }
CPP
suffix = <<'CPP'
void require(bool b,const char* m) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",m); std::exit(1); } }
int main() {
  SgbTf32Storage<SgbTf32Nt,128,64,3> storage;
  storage_base=(std::uintptr_t)&storage;
  const Params fixtures[]={{2048,768,3072,3072,3072},{129,65,36,36,36},{129,65,35,40,44}};
  long checks=0;
  for(const Params p:fixtures) {
    for(int tr=0;tr<p.m;tr+=128) for(int tc=0;tc<p.k;tc+=64) {
      SgbTf32Problem problem{(const float*)0x10000000ULL,(const float*)0x30000000ULL,p,tr,tc};
      const int kbases[]={0,32,((p.n+31)/32-1)*32};
      for(int tid=0;tid<256;++tid) {
        threadIdx.x=tid;
        SgbTf32NtTestPaddedCopyPlan plan{};
        gemm_bi_tf32_nt_test_make_padded_copy_plan(&storage,problem,plan);
        for(int stage=0;stage<3;++stage) for(int kb:kbases) {
          observed.clear();
          gemm_bi_tf32_nt_test_padded_copy_stage(plan,problem,stage,kb);
          require(observed.size()==6,"copy count");
          int index=0;
          for(int operand=0;operand<2;++operand) {
            const int rows=operand?64:128;
            for(int linear=tid;linear<rows*8;linear+=256) {
              int row=linear/8;
              int reduction=(linear%8)*4;
              int global=(operand?tc:tr)+row;
              int extent=operand?p.k:p.m;
              int bytes=global<extent?4*std::max(0,std::min(4,p.n-kb-reduction)):0;
              long long off=bytes? (long long)global*(operand?p.ldb:p.lda)+kb+reduction :0;
              unsigned dst=operand?
                (unsigned)((std::uintptr_t)&storage.b[stage][row][reduction]-storage_base):
                (unsigned)((std::uintptr_t)&storage.a[stage][row][reduction]-storage_base);
              std::uintptr_t src=(operand?0x30000000ULL:0x10000000ULL)+4*off;
              const Copy got=observed[index++];
              require(got.dst==dst,operand?"B stage destination":"A stage destination");
              require(got.src==src,operand?"B global safe offset":"A global safe offset");
              require(got.bytes==bytes,operand?"B zero-fill bytes":"A zero-fill bytes");
              ++checks;
            }
          }
        }
      }
    }
  }
  std::printf("PASS actual copy-plan/stage addresses: %ld copy tuples, target/tail/K35-padded, all threads and stages\n",checks);
}
CPP
variants = {
 'actual' => header,
 'mutate_a_stage' => header.sub('stage * AStageBytes','stage * BStageBytes'),
 'mutate_b_stage' => header.sub('stage * BStageBytes','stage * AStageBytes'),
 'mutate_tail_bytes' => header.sub('int full_bytes = remaining * 4;', 'int full_bytes = remaining == 3 ? 16 : remaining * 4;')
}
Dir.mktmpdir('triad-copy-plan-actual-') do |dir|
  variants.each do |name,body|
    raise "mutation not applied #{name}" if name!='actual' && body==header
    exe=File.join(dir,name)
    out,err,status=Open3.capture3('xcrun','clang++','-std=c++17','-O2','-Werror','-Wno-unknown-pragmas','-x','c++','-','-o',exe,stdin_data:prefix+body+suffix)
    puts "#{name} compile_exit=#{status.exitstatus}"
    puts out,err unless status.success?
    raise 'compile failure' unless status.success?
    out,err,status=Open3.capture3(exe)
    puts "#{name} run_exit=#{status.exitstatus}",out,err
    raise 'unexpected run outcome' unless name=='actual' ? status.success? : !status.success?
  end
end
