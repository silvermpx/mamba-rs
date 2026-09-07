#include "schedule_model.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace s = ada_half_cutlass_s3;

[[noreturn]] static void fail(int tiles, size_t index, const char* message) {
    std::fprintf(stderr, "FAIL tiles=%d event=%zu: %s\n", tiles, index, message);
    std::exit(1);
}

static void require(bool condition, int tiles, size_t index, const char* message) {
    if (!condition) fail(tiles, index, message);
}

static void validate(const std::vector<s::Event>& events, int tiles) {
    std::vector<int> copies(tiles, 0), commits(tiles, 0);
    std::vector<bool> waited(tiles, false), barred(tiles, false);
    std::vector<std::array<bool, 4>> loaded(tiles);
    std::array<int, 3> resident{{-1, -1, -1}};
    std::vector<int> pending;
    std::vector<bool> complete(tiles, false);
    std::vector<std::pair<int, int>> mma_sequence;
    std::vector<size_t> issue3_position(tiles, size_t(-1));
    std::vector<size_t> next_issue0_position(tiles, size_t(-1));
    bool aliased = false;

    for (size_t index = 0; index < events.size(); ++index) {
        const auto& event = events[index];
        require(!aliased, tiles, index, "event occurs after epilogue alias");
        switch (event.kind) {
            case s::Kind::CopySlice: {
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "copy tile out of range");
                require(event.slot == (event.tile % 3), tiles, index,
                        "copy uses wrong S3 slot");
                require(event.slice >= 0 && event.slice < 4, tiles, index,
                        "copy slice out of range");
                require(copies[event.tile] == event.slice, tiles, index,
                        "copy slices are not exact ascending 0..3");
                int previous = resident[event.slot];
                if (copies[event.tile] == 0) {
                    if (previous >= 0) {
                        for (int issue = 0; issue < 4; ++issue)
                            require(loaded[previous][issue], tiles, index,
                                    "stage slot overwritten before all four ldmatrix issues");
                    }
                    resident[event.slot] = event.tile;
                }
                require(resident[event.slot] == event.tile, tiles, index,
                        "copy changes resident tile mid-group");
                ++copies[event.tile];
                break;
            }
            case s::Kind::Commit:
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "empty or out-of-range commit group");
                require(copies[event.tile] == 4, tiles, index,
                        "commit before four exact copy slices");
                require(commits[event.tile]++ == 0, tiles, index,
                        "copy group committed more than once");
                pending.push_back(event.tile);
                break;
            case s::Kind::Wait:
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "wait for nonexistent tile");
                require(commits[event.tile] == 1, tiles, index,
                        "wait before committed copy group");
                require(!waited[event.tile], tiles, index, "duplicate wait");
                require(event.issue == 0 || event.issue == 1, tiles, index, "invalid wait depth");
                while (pending.size() > size_t(event.issue)) {
                    complete[pending.front()] = true;
                    pending.erase(pending.begin());
                }
                require(complete[event.tile], tiles, index, "wait1 left the required sole pending group unfinished");
                waited[event.tile] = true;
                break;
            case s::Kind::Barrier:
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "barrier for nonexistent tile");
                require(waited[event.tile], tiles, index, "barrier before wait");
                require(!barred[event.tile], tiles, index, "duplicate tile-ready barrier");
                barred[event.tile] = true;
                break;
            case s::Kind::Load:
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "load tile out of range");
                require(event.slot == (event.tile % 3), tiles, index,
                        "load uses wrong S3 slot");
                require(barred[event.tile], tiles, index,
                        "ldmatrix before corresponding wait/barrier");
                require(resident[event.slot] == event.tile, tiles, index,
                        "ldmatrix reads overwritten resident slab");
                require(event.issue >= 0 && event.issue < 4, tiles, index,
                        "ldmatrix issue out of range");
                require(!loaded[event.tile][event.issue], tiles, index,
                        "duplicate ldmatrix issue");
                if (event.issue > 0)
                    require(loaded[event.tile][event.issue - 1], tiles, index,
                            "ldmatrix issues are not ascending");
                loaded[event.tile][event.issue] = true;
                if (event.issue == 0 && event.tile > 0)
                    next_issue0_position[event.tile - 1] = index;
                break;
            case s::Kind::Mma:
                require(event.tile >= 0 && event.tile < tiles, tiles, index,
                        "MMA tile out of range");
                require(event.issue >= 0 && event.issue < 4, tiles, index,
                        "MMA issue out of range");
                require(loaded[event.tile][event.issue], tiles, index,
                        "MMA consumes an unloaded fragment");
                mma_sequence.push_back({event.tile, event.issue});
                if (event.issue == 3) issue3_position[event.tile] = index;
                break;
            case s::Kind::Rotate:
                require(event.tile >= 0 && event.tile + 1 < tiles, tiles, index,
                        "final drain rotates to nonexistent stage");
                require(barred[event.tile + 1], tiles, index,
                        "S3 ring rotates before next stage is ready");
                break;
            case s::Kind::AliasEpilogue:
                for (int tile = 0; tile < tiles; ++tile)
                    for (int issue = 0; issue < 4; ++issue)
                        require(loaded[tile][issue], tiles, index,
                                "epilogue aliases shared before fragment reads retire");
                require(pending.empty(), tiles, index, "epilogue aliases outstanding cp.async writes");
                aliased = true;
                break;
        }
    }

    require(aliased, tiles, events.size(), "missing epilogue alias event");
    require(mma_sequence.size() == size_t(tiles * 4), tiles, events.size(),
            "missing or duplicate MMA");
    for (int tile = 0; tile < tiles; ++tile) {
        require(copies[tile] == 4, tiles, events.size(),
                "tile copy group does not contain four slices");
        require(commits[tile] == 1, tiles, events.size(),
                "tile copy group does not commit exactly once");
        require(waited[tile] && barred[tile], tiles, events.size(),
                "tile is consumed without one wait/barrier");
        for (int issue = 0; issue < 4; ++issue) {
            const auto want = std::pair<int, int>{tile, issue};
            require(mma_sequence[size_t(tile * 4 + issue)] == want, tiles,
                    events.size(), "MMA sequence is not tile-major issue0..3");
        }
        if (tile + 1 < tiles)
            require(next_issue0_position[tile] < issue3_position[tile], tiles,
                    events.size(), "next-stage issue0 load does not precede current issue3 MMA");
    }
}

int main(int argc, char** argv) {
    int old = argc == 2 && std::string(argv[1]) == "--mutant-wait1" ? 1 :
              (argc == 2 && std::string(argv[1]) == "--mutant-drain" ? 2 : 0);
    if (argc > 2 || (argc == 2 && !old)) {
        std::fprintf(stderr, "usage: %s [--mutant-wait1]\n", argv[0]);
        return 2;
    }
    for (int tiles = 0; tiles <= 97; ++tiles) {
        auto events = s::candidate_schedule(tiles, old);
        validate(events, tiles);
    }
    std::puts("PASS CUTLASS-style S3 schedule for BK64 tiles 0..97");
}
