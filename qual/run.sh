#!/usr/bin/env bash
# Lane runner: executes or lists one lane of qual/lanes.toml on a GPU box.
#
#   qual/run.sh gate           every-run correctness (plain battery)
#   qual/run.sh toolkit        compile gates and identity contracts: the
#                              toolkit, no device; once per toolkit build
#   qual/run.sh toolkits       the toolkit lane under every CUDA toolkit
#                              installed as /usr/local/cuda-X.Y
#   qual/run.sh board          the gate lane on a board: every gate target,
#                              the toolkit lane left out
#   qual/run.sh contract       pre-release / post-edit / post-merge bit gates
#   qual/run.sh record         lists the manual instruments under tests/
#   qual/run.sh bench          lists the bench entry points
#   qual/run.sh qualification  lists the qualification tools
#
# Set MAMBA_RS_ACCEPTANCE_TSV to also capture the contract lane's cells
# for examples/acceptance_diff.
set -u
cd "$(dirname "$0")/.."
lane="${1:-gate}"

suites() {
    grep -E "^\"[a-z0-9_]+\" = \"$1\"" qual/lanes.toml | cut -d'"' -f2
}

case "$lane" in
gate)
    exec cargo test --release --features cuda,cublas
    ;;
toolkit | board)
    # Each target on its own; the lane names each red at its end. The
    # toolkit lane needs the toolkit on PATH and no device, so it runs on
    # the build box once per toolkit build; a rented board runs `board`,
    # which is the gate lane without that work.
    red=""
    for t in $(suites "$( [ "$lane" = board ] && echo gate || echo toolkit )"); do
        echo "== $lane: $t"
        cargo test --release --features cuda,cublas,hf --test "$t" || red="$red $t"
    done
    if [ -n "$red" ]; then
        echo "$(echo "$lane" | tr a-z A-Z) RED:$red"
        exit 1
    fi
    echo "$(echo "$lane" | tr a-z A-Z) GREEN"
    ;;
toolkits)
    # The toolkit lane once under each installed toolkit, each build in a
    # target directory of its own (two toolkits sharing one would leave the
    # second build fresh on the first one's bindings). NVRTC versions refuse
    # different constructs, so a kernel that compiles under one toolkit can
    # fail under another: every toolkit a release supports runs this before
    # it ships. cudarc names bindings up to CUDA 13.3; a newer toolkit
    # builds with those and loads its own libraries at run time.
    base="${CARGO_TARGET_DIR:-target}"
    red=""
    found=""
    for cuda in /usr/local/cuda-[0-9]*.[0-9]*; do
        [ -x "$cuda/bin/nvcc" ] || continue
        version="${cuda##*/cuda-}"
        found="$found $version"
        binding=""
        major="${version%%.*}"
        minor="${version#*.}"
        if [ "$major" -gt 13 ] || { [ "$major" -eq 13 ] && [ "$minor" -gt 3 ]; }; then
            binding=13030
        fi
        echo "=== toolkits: CUDA $version"
        env ${binding:+CUDARC_CUDA_VERSION=$binding} PATH="$cuda/bin:$PATH" \
            LD_LIBRARY_PATH="$cuda/lib64" CUDA_PATH="$cuda" CUDA_HOME="$cuda" CUDA_ROOT="$cuda" \
            CARGO_TARGET_DIR="$base/toolkit-$version" "$0" toolkit || red="$red $version"
    done
    if [ -z "$found" ]; then
        echo "TOOLKITS RED: no toolkit under /usr/local/cuda-X.Y"
        exit 1
    fi
    if [ -n "$red" ]; then
        echo "TOOLKITS RED:$red (of$found)"
        exit 1
    fi
    echo "TOOLKITS GREEN:$found"
    ;;
contract)
    # diag_-prefixed tests are manual position printers (record-grade
    # instruments living inside contract files); the lane skips them.
    # Every target runs; the lane names each red at the end instead of
    # stopping at the first, since a target written for another board is
    # red here by design and would hide the rest. The hf feature is on
    # because one contract target loads a checkpoint.
    red=""
    for t in $(suites contract); do
        echo "== contract: $t"
        cargo test --release --features cuda,cublas,hf --test "$t" -- --ignored --skip diag_ --nocapture || red="$red $t"
    done
    if [ -n "$red" ]; then
        echo "CONTRACT RED:$red"
        exit 1
    fi
    echo "CONTRACT GREEN"
    ;;
record)
    echo "record-lane instruments (run by hand, one at a time):"
    suites record | sed 's/^/  cargo test --release --features "cuda cublas hf" --test /;s/$/ -- --ignored --nocapture/'
    ;;
bench)
    echo "bench entry points (each runs every instrument, or the names given after --):"
    suites bench | sed 's/^/  cargo bench --features cuda,cublas --bench /'
    ;;
qualification)
    echo "qualification tools (built only with the qualification feature; run one on its board):"
    suites qualification | sed 's/^/  cargo test --release --features "cuda cublas hf qualification" --test /;s/$/ -- --ignored --nocapture/'
    ;;
*)
    echo "usage: qual/run.sh [gate|board|toolkit|toolkits|contract|record|bench|qualification]" >&2
    exit 2
    ;;
esac
