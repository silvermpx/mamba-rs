#pragma once
#include <vector>
namespace ada_half_cutlass_s3 {
enum class Kind { CopySlice, Commit, Wait, Barrier, Load, Mma, Rotate, AliasEpilogue };
struct Event { Kind kind; int tile, slot, issue, slice; };
inline std::vector<Event> candidate_schedule(int tiles, int mutant = 0) {
    std::vector<Event> e;
    auto emit = [&](Kind k, int t, int s, int i=-1, int sl=-1) { e.push_back({k,t,s,i,sl}); };
    for (int t=0; t<tiles && t<2; ++t) {
        for(int s=0;s<4;++s) emit(Kind::CopySlice,t,t,-1,s);
        emit(Kind::Commit,t,t);
    }
    if(tiles>0) {
        emit(Kind::Wait,0,0,mutant == 1 ? 1 : (tiles>1 ? 1 : 0));
        emit(Kind::Barrier,0,0); emit(Kind::Load,0,0,0);
    }
    int read=0, write=2;
    for(int t=0;t<tiles;++t) {
        bool refill=t+2<tiles, next=t+1<tiles;
        for(int i=0;i<3;++i) {
            if(refill) {
                emit(Kind::CopySlice,t+2,write,-1,i);
                if(i==2) emit(Kind::CopySlice,t+2,write,-1,3);
            }
            emit(Kind::Load,t,read,i+1); emit(Kind::Mma,t,read,i);
        }
        int old=read;
        if(refill) emit(Kind::Commit,t+2,write);
        if(next) {
            int nr=read==2?0:read+1;
            emit(Kind::Wait,t+1,nr,mutant ? 1 : (refill ? 1 : 0));
            emit(Kind::Barrier,t+1,nr); emit(Kind::Rotate,t,nr);
            read=nr; write=write==2?0:write+1;
            emit(Kind::Load,t+1,read,0);
        }
        emit(Kind::Mma,t,old,3);
    }
    emit(Kind::AliasEpilogue,-1,-1);
    return e;
}
}
