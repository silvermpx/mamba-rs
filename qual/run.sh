#!/usr/bin/env bash
# Lane runner: executes or lists one lane of qual/lanes.toml on a GPU box.
#
#   qual/run.sh gate           every-run correctness (plain battery)
#   qual/run.sh toolkit        compile gates and identity contracts: the
#                              toolkit, no device; once per toolkit build
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
    exec cargo test --release --features cuda
    ;;
toolkit | board)
    # Each target on its own; the lane names each red at its end. The
    # toolkit lane needs the toolkit on PATH and no device, so it runs on
    # the build box once per toolkit build; a rented board runs `board`,
    # which is the gate lane without that work.
    red=""
    for t in $(suites "$( [ "$lane" = board ] && echo gate || echo toolkit )"); do
        echo "== $lane: $t"
        cargo test --release --features cuda,hf --test "$t" || red="$red $t"
    done
    if [ -n "$red" ]; then
        echo "$(echo "$lane" | tr a-z A-Z) RED:$red"
        exit 1
    fi
    echo "$(echo "$lane" | tr a-z A-Z) GREEN"
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
        cargo test --release --features cuda,hf --test "$t" -- --ignored --skip diag_ --nocapture || red="$red $t"
    done
    if [ -n "$red" ]; then
        echo "CONTRACT RED:$red"
        exit 1
    fi
    echo "CONTRACT GREEN"
    ;;
record)
    echo "record-lane instruments (run by hand, one at a time):"
    suites record | sed 's/^/  cargo test --release --features "cuda hf" --test /;s/$/ -- --ignored --nocapture/'
    ;;
bench)
    echo "bench entry points (each runs every instrument, or the names given after --):"
    suites bench | sed 's/^/  cargo bench --features cuda --bench /'
    ;;
qualification)
    echo "qualification tools (built only with the qualification feature; run one on its board):"
    suites qualification | sed 's/^/  cargo test --release --features "cuda hf qualification" --test /;s/$/ -- --ignored --nocapture/'
    ;;
*)
    echo "usage: qual/run.sh [gate|board|toolkit|contract|record|bench|qualification]" >&2
    exit 2
    ;;
esac
