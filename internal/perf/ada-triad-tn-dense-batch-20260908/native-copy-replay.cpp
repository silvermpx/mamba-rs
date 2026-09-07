#include <cassert>
#include <cstdint>
#include <cstdio>
#include <vector>
#define __device__
#define __forceinline__ inline
enum SgbTf32Op { SgbTf32Nn, SgbTf32Tn, SgbTf32Nt };
struct Params { int m,k,n,lda,ldb,ldc; };
struct SgbTf32Problem { float* a; float* b; Params params; int tile_row,tile_column; };
template<SgbTf32Op,int,int,int> struct alignas(16) SgbTf32Storage {
    float a[3][32][136]; float b[3][32][72];
};
using Storage=SgbTf32Storage<SgbTf32Tn,128,64,3>;
struct { unsigned x; } threadIdx;
static Storage storage;
static SgbTf32Problem current;
static int current_stage, current_reduction, calls;
static std::vector<bool> seen;
inline bool gemm_bi_is_aligned_16(const void* p) { return (uintptr_t(p)&15)==0; }
inline uintptr_t __cvta_generic_to_shared(const void* p) { return uintptr_t(p)-uintptr_t(&storage); }
template<SgbTf32Op> float& gemm_bi_tf32_a_slot(Storage* s,int stage,int row,int reduction) { return s->a[stage][reduction][row]; }
template<SgbTf32Op> float& gemm_bi_tf32_b_slot(Storage* s,int stage,int reduction,int col) { return s->b[stage][reduction][col]; }
template<int> void gemm_bi_tf32_cp_async_16_zfill(unsigned dst,const float* src,int bytes) {
    assert(bytes==16 && (dst&15)==0 && gemm_bi_is_aligned_16(src));
    bool a=dst<sizeof(storage.a);
    unsigned offset=(dst-(a?0:sizeof(storage.a)))/4, stride=a?136:72;
    unsigned stage=offset/(32*stride), reduction=(offset/stride)%32, axis=offset%stride;
    assert(stage==unsigned(current_stage) && axis%4==0 && axis+3<unsigned(a?128:64));
    const float* expected=(a?current.a:current.b)+(int64_t(current_reduction)+reduction)*(a?current.params.lda:current.params.ldb)+(a?current.tile_row:current.tile_column)+axis;
    assert(src==expected);
    for(unsigned e=0;e<4;++e) { assert(!seen[dst/4+e]); seen[dst/4+e]=true; }
    ++calls;
}
#include "tests/gemm_bi_tf32_tn_dense_copy.cuh"
int main() {
    unsigned long long full=0,tail=0,copy_calls=0;
    for(auto dims: {Params{2048,768,3072,768,3072,3072}, Params{2048,1536,768,1536,768,768}, Params{4621,384,1928,384,1928,1928}}) {
        std::vector<float> a(size_t(dims.m)*dims.lda), b(size_t(dims.m)*dims.ldb);
        current={a.data(),b.data(),dims,0,0};
        assert(gemm_bi_is_aligned_16(current.a)&&gemm_bi_is_aligned_16(current.b));
        for(int row=0;row<dims.k;row+=128) for(int col=0;col<dims.n;col+=64) for(int red=0;red<dims.m;red+=32) {
            current.tile_row=row; current.tile_column=col;
            bool want=row+128<=dims.k && col+64<=dims.n && red+32<=dims.m;
            assert(gemm_bi_tf32_tn_test_dense_full_stage(current,red)==want);
            if(!want) { ++tail; continue; }
            ++full; current_stage=(red/32)%3; current_reduction=red;
            seen.assign(sizeof(storage)/4,false); calls=0;
            for(unsigned t=0;t<256;++t) { threadIdx.x=t; gemm_bi_tf32_tn_test_dense_stage(&storage,current_stage,current,red); }
            assert(calls==1536); copy_calls+=calls;
        }
        current.tile_row=0; current.tile_column=0;
        assert(!gemm_bi_tf32_tn_test_dense_full_stage(current,-1));
        ++current.a; assert(!gemm_bi_tf32_tn_test_dense_full_stage(current,0)); --current.a;
        ++current.params.ldb; assert(!gemm_bi_tf32_tn_test_dense_full_stage(current,0));
    }
    std::printf("actual-header host replay: full=%llu tail=%llu 16B-copies=%llu; guards and unique destinations PASS\n",full,tail,copy_calls);
}
