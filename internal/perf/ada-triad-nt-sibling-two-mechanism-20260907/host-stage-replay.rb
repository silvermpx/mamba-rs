require 'open3'
require 'tmpdir'

# Actual generated guards + actual dense stage and prism dispatcher. Generic
# staging is an opaque call marker; no CUDA/async/MMA emulation is attempted.
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
int generic_calls=0;
template<int BM> void gemm_bi_tf32_cp_async_16_zfill(unsigned d,const float* s,int b) { observed.push_back({d,(std::uintptr_t)s,b}); }
template<SgbTf32Op Op,int BM,int BN,int Stages,bool NA,bool NB>
void gemm_bi_tf32_stage_async(SgbTf32Storage<Op,BM,BN,Stages>*,int,const SgbTf32Problem&,int) { ++generic_calls; }
CPP
suffix = <<'CPP'
void require(bool b,const char* m) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",m); std::exit(1); } }
int main() {
  const Sm80Tf32KernelParams fixtures[]={{1,0,2048,1536,768,768,768,1536},{1,0,4621,384,1928,1928,1928,384},{1,0,129,65,36,36,36,65}};
  require(out_guard::target(fixtures[0]),"out exact guard");
  require(prism_guard::target(fixtures[1]),"prism exact guard");
  for(int f=0;f<2;++f) for(int field=0;field<6;++field) {
    auto bad=fixtures[f];
    if(field==0)++bad.m; if(field==1)++bad.k; if(field==2)++bad.n;
    if(field==3)++bad.lda; if(field==4)++bad.ldb; if(field==5)++bad.ldc;
    require(!(f?prism_guard::target(bad):out_guard::target(bad)),"negative exact guard");
  }
  long copies=0;
  for(int fixture=0;fixture<3;++fixture) {
    const auto p=fixtures[fixture];
    alignas(16) SgbTf32Storage<SgbTf32Nt,128,64,3> storage;
    storage_base=(std::uintptr_t)&storage;
    // Allocation padding also keeps deliberately bad mutation pointer
    // arithmetic defined; range checks still use the logical extents.
    std::vector<float> a((long)((p.m+127)/128*128)*p.lda+64),b((long)((p.k+63)/64*64)*p.ldb+64);
    const float* ap=(const float*)(((std::uintptr_t)a.data()+15)&~std::uintptr_t(15));
    const float* bp=(const float*)(((std::uintptr_t)b.data()+15)&~std::uintptr_t(15));
    long dense_positions=0,generic_positions=0;
    for(int tr=0;tr<p.m;tr+=128) for(int tc=0;tc<p.k;tc+=64) for(int kb=0;kb<p.n;kb+=32) {
      const bool expected_dense=fixture==0 || (fixture==1 && tr+128<=p.m && tc+64<=p.k && kb+32<=p.n);
      expected_dense?++dense_positions:++generic_positions;
      const int stage=(kb/32)%3;
      SgbTf32Problem problem{ap,bp,p,tr,tc};
      std::vector<unsigned char> ownership(sizeof(storage)/4,0);
      for(int tid=0;tid<256;++tid) {
        threadIdx.x=tid; observed.clear(); generic_calls=0;
        if(out_guard::target(p))
          gemm_bi_tf32_nt_test_padded_dense_stage(&storage,stage,problem,kb);
        else if(prism_guard::target(p))
          gemm_bi_tf32_nt_test_padded_dense_prism_stage(&storage,stage,problem,kb);
        else gemm_bi_tf32_stage_async<SgbTf32Nt,128,64,3,false,false>(&storage,stage,problem,kb);
        require(generic_calls==(expected_dense?0:1),"actual generic/dense routing");
        require(observed.size()==(expected_dense?6:0),"actual copy count");
        if(!expected_dense) continue;
        int index=0;
        for(int operand=0;operand<2;++operand) for(int linear=tid;linear<(operand?64:128)*8;linear+=256) {
          const int axis=linear/8,red=(linear%8)*4;
          const long off=(long)((operand?tc:tr)+axis)*(operand?p.ldb:p.lda)+kb+red;
          const float* base=operand?bp:ap;
          const unsigned dst=operand?(unsigned)((std::uintptr_t)&storage.b[stage][axis][red]-storage_base):(unsigned)((std::uintptr_t)&storage.a[stage][axis][red]-storage_base);
          const Copy got=observed[index++];
          require(got.dst==dst && got.src==(std::uintptr_t)(base+off),"actual copy address");
          require(got.bytes==16 && got.dst%16==0 && got.src%16==0,"aligned literal16B");
          require(off>=0 && off+4<=(long)(operand?p.k:p.m)*p.n && got.dst+16<=sizeof(storage),"logical copy bounds");
          for(int element=0;element<4;++element) require(++ownership[got.dst/4+element]==1,"unique shared ownership");
          ++copies;
        }
      }
      if(expected_dense) {int count=0;for(auto seen:ownership)count+=seen;require(count==6144,"full dense coverage");}
    }
    const long expected_d[]={9216,12960,0},expected_g[]={0,582,8};
    require(dense_positions==expected_d[fixture] && generic_positions==expected_g[fixture],"complete position counts");
    std::printf("fixture%d dense=%ld generic=%ld\n",fixture,dense_positions,generic_positions);
  }
  require(copies==34062336,"all dense copy tuples");
  std::printf("PASS actual sibling guards/stage routing and%ld dense copy tuples; forced-tail generic;12 negative guards\n",copies);
}
CPP
Dir.mktmpdir('triad-sibling-stage-host-') do |dir|
  guards=''
  %w[PaddedDenseD768Out PaddedDensePrism].each_with_index do |variant,index|
    exe=File.join(dir,"emit#{index}")
    path=File.expand_path('tests/gemm_bi_tf32_nt_compact_xor.rs')
    rust="include!(#{path.dump});\nfn main(){print!(\"{}\",CandidateVariant::#{variant}.source().unwrap());}\n"
    out,status=Open3.capture2e('/Users/silvermpx/.rustup/toolchains/1.98.1-aarch64-apple-darwin/bin/rustc','--edition','2024','-A','warnings','-','-o',exe,stdin_data:rust)
    abort out unless status.success?
    generated,status=Open3.capture2e(exe); abort generated unless status.success?
    name="gemm_bi_tf32_nt_test_padded_dense_#{index==0?'d768_out':'prism'}_target"
    start=generated.index("__device__ __forceinline__ bool #{name}(") or raise 'guard missing'
    stop=generated.index("\n}",start) or raise 'guard end missing'
    guard=generated[start...stop+2].sub(name,'target')
    guards+="namespace #{index==0?'out_guard':'prism_guard'} {\n#{guard}\n}\n"
  end
  dense=File.read('tests/gemm_bi_tf32_nt_padded_dense_copy.cuh').split('template <int MAtoms, int NAtoms>',2).first
  dense=dense[dense.index('__device__ __forceinline__ void gemm_bi_tf32_nt_test_padded_dense_stage(')..-1]
  commit='asm volatile("cp.async.commit_group;\n" ::);'
  raise 'commit seam count' unless dense.scan(commit).size==1
  dense=dense.sub(commit,'/* opaque async commit omitted on host */')
  prism=File.read('tests/gemm_bi_tf32_nt_padded_dense_prism.cuh').split('template <int MAtoms, int NAtoms>',2).first
  variants={'actual'=>prism,'mutate_m_tail'=>prism.sub('problem.tile_row + 128 <=','problem.tile_row <='),'mutate_k_tail'=>prism.sub('reduction_base + 32 <=','reduction_base <=')}
  variants.each do |name,body|
    raise "mutation missing #{name}" if name!='actual' && body==prism
    exe=File.join(dir,name)
    out,err,status=Open3.capture3('xcrun','clang++','-std=c++17','-O2','-Werror','-Wno-unknown-pragmas','-x','c++','-','-o',exe,stdin_data:prefix+guards+dense+body+suffix)
    puts "#{name} compile_exit=#{status.exitstatus}"; abort out+err unless status.success?
    out,err,status=Open3.capture3(exe); puts "#{name} run_exit=#{status.exitstatus}",out,err
    raise 'unexpected behavior' unless name=='actual' ? status.success? : !status.success?
  end
end
