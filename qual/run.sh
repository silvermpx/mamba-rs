#!/usr/bin/env bash
# Qualification runner: executes one lane of qual/lanes.toml on a GPU box.
#
#   qual/run.sh gate       every-run correctness (plain battery)
#   qual/run.sh contract   pre-release / post-edit / post-merge bit gates
#   qual/run.sh record     lists the manual instruments (never auto-runs)
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
contract)
    # diag_-prefixed tests are manual position printers (record-grade
    # instruments living inside contract files); the lane skips them.
    rc=0
    for t in $(suites contract); do
        echo "== contract: $t"
        cargo test --release --features cuda --test "$t" -- --ignored --skip diag_ --nocapture || rc=1
        if [ "$rc" -ne 0 ]; then
            echo "CONTRACT RED: $t"
            exit 1
        fi
    done
    echo "CONTRACT GREEN"
    ;;
record)
    echo "record-lane instruments (run by hand, one at a time):"
    suites record | sed 's/^/  cargo test --release --features cuda --test /;s/$/ -- --ignored --nocapture/'
    ;;
*)
    echo "usage: qual/run.sh [gate|contract|record]" >&2
    exit 2
    ;;
esac
