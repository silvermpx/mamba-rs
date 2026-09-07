require 'open3'
require 'tmpdir'

# Address-only execution of the actual header. No CUDA or async emulation.
header = File.read('tests/gemm_bi_tf32_nt_padded_dense_copy.cuh').split('template <int MAtoms, int NAtoms>', 2).first
commit = 'asm volatile("cp.async.commit_group;\n" ::);'
raise 'commit seam count' unless header.scan(commit).size == 1
header = header.sub(commit, '/* opaque async commit omitted on host */')
prefix = <<'CPP'
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define __device__
#define __forceinline__ inline
struct { int x; } threadIdx;
enum SgbTf32Op { SgbTf32Nt };
template<SgbTf32Op Op,int BM,int BN,int Stages>
struct SgbTf32Storage { float a[Stages][BM][36]; float b[Stages][BN][36]; };
struct Sm80Tf32KernelParams { float alpha,beta; int m,k,n,lda,ldb,ldc; };
struct SgbTf32Problem { const float* a; const float* b; Sm80Tf32KernelParams params; int tile_row,tile_column; };
template<SgbTf32Op Op,int BM,int BN,int Stages>
float& gemm_bi_tf32_a_slot(SgbTf32Storage<Op,BM,BN,Stages>* p,int s,int r,int k) { return p->a[s][r][k]; }
template<SgbTf32Op Op,int BM,int BN,int Stages>
float& gemm_bi_tf32_b_slot(SgbTf32Storage<Op,BM,BN,Stages>* p,int s,int k,int c) { return p->b[s][c][k]; }
std::uintptr_t storage_base;
unsigned __cvta_generic_to_shared(const float* p) { return (unsigned)((std::uintptr_t)p-storage_base); }
struct Copy { unsigned dst; std::uintptr_t src; int bytes; };
std::vector<Copy> observed;
template<int BM>
void gemm_bi_tf32_cp_async_16_zfill(unsigned d,const float* s,int b) { observed.push_back({d,(std::uintptr_t)s,b}); }
CPP
suffix = <<'CPP'
void require(bool b,const char* m) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",m); std::exit(1); } }
int main() {
  const Sm80Tf32KernelParams p{1,0,2048,768,3072,3072,3072,768};
  require(gemm_bi_tf32_nt_test_padded_dense_target(p),"target guard");
  const Sm80Tf32KernelParams bad[]={
    {1,0,2049,768,3072,3072,3072,768}, {1,0,2048,769,3072,3072,3072,768},
    {1,0,2048,768,3073,3072,3072,768}, {1,0,2048,768,3072,3076,3072,768},
    {1,0,2048,768,3072,3072,3076,768}, {1,0,2048,768,3072,3072,3072,772},
    {1,0,129,65,36,36,36,65}
  };
  for(const auto q:bad) require(!gemm_bi_tf32_nt_test_padded_dense_target(q),"negative guard");
  alignas(16) SgbTf32Storage<SgbTf32Nt,128,64,3> storage;
  storage_base=(std::uintptr_t)&storage;
  std::vector<float> a((long long)p.m*p.lda+16),b((long long)p.k*p.ldb+16);
  const float* ap=(const float*)(((std::uintptr_t)a.data()+15)&~std::uintptr_t(15));
  const float* bp=(const float*)(((std::uintptr_t)b.data()+15)&~std::uintptr_t(15));
  observed.reserve(12);
  long checks=0;
  for(int tr=0;tr<p.m;tr+=128) for(int tc=0;tc<p.k;tc+=64) {
    SgbTf32Problem problem{ap,bp,p,tr,tc};
    for(int tile=0;tile<96;++tile) {
      const int stage=tile%3,kb=tile*32;
      std::vector<unsigned char> ownership(sizeof(storage)/4,0);
      for(int tid=0;tid<256;++tid) {
        threadIdx.x=tid;
        observed.clear();
        gemm_bi_tf32_nt_test_padded_dense_stage(&storage,stage,problem,kb);
        require(observed.size()==6,"six copies per256-thread stage");
        int index=0;
        for(int operand=0;operand<2;++operand) {
          const int rows=operand?64:128;
          const float* base=operand?bp:ap;
          const int stride=operand?p.ldb:p.lda;
          const long extent=(long)(operand?p.k:p.m)*stride;
          for(int linear=tid;linear<rows*8;linear+=256) {
            const int row=linear/8,red=(linear%8)*4;
            const long off=(long)((operand?tc:tr)+row)*stride+kb+red;
            const unsigned dst=operand?
              (unsigned)((std::uintptr_t)&storage.b[stage][row][red]-storage_base):
              (unsigned)((std::uintptr_t)&storage.a[stage][row][red]-storage_base);
            const Copy got=observed[index++];
            require(got.dst==dst,operand?"B shared address":"A shared address");
            require(got.src==(std::uintptr_t)(base+off),operand?"B global address":"A global address");
            require(got.bytes==16 && got.dst%16==0 && got.src%16==0,"literal aligned16B");
            require(off>=0 && off+4<=extent && got.dst+16<=sizeof(storage),"copy ranges");
            for(int element=0;element<4;++element)
              require(++ownership[got.dst/4+element]==1,"unique shared word ownership");
            ++checks;
          }
        }
      }
      int covered=0;
      for(auto count:ownership) covered+=count;
      require(covered==6144,"full active tile coverage");
    }
  }
  require(checks==28311552,"complete target copy count");
  std::printf("PASS actual dense header: %ld copy tuples,192CTAs x96tiles x256threads x6; guard negatives7; unique/range/alignment\n",checks);
}
CPP
variants = {
  'actual' => header,
  'mutate_threads' => header.sub('Threads = 256', 'Threads = 128'),
  'mutate_b_row' => header.sub('problem.tile_column + column', 'problem.tile_column + column + 1'),
  'mutate_guard' => header.sub('params.m == 2048', 'params.m >= 2048')
}
Dir.mktmpdir('triad-dense-copy-actual-') do |dir|
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
