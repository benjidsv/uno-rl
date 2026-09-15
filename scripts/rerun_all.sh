#!/usr/bin/env bash
# Re-run every experiment of docs/results/, in dependency order, one log per step.
#
#   caffeinate -is bash scripts/rerun_all.sh
#
# Logs: logs/rerun-<date>-<time>/NN_<step>.log, and summary.txt with the start, exit code and
# duration of every step. The test suite runs first and stops everything if it fails; any other
# failing step is recorded and the run goes on.
#
# Order: `cfr TOY`, `symmetry`, `toy2 <ruleset>` save the Nash strategies that `transfer`,
# `toy2_columns` and `validate` load, so they come first. Trees are compiled on first use and
# cached in .cache/ (a file from before CACHE_VERSION 1 is stale and recompiled). The TOY2_FULL
# steps come last: each holds 2-5 GB of tree in memory.
set -u -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

out="logs/rerun-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
summary="$out/summary.txt"
n=0

run() {  # run <step> <command...>
    local step=$1
    shift
    n=$((n + 1))
    local log
    log=$(printf "%s/%02d_%s.log" "$out" "$n" "$step")
    local t0=$SECONDS
    echo "[$(date '+%H:%M:%S')] start $step: $*" | tee -a "$summary"
    "$@" > "$log" 2>&1
    local rc=$?
    echo "[$(date '+%H:%M:%S')] end   $step: exit $rc after $(( (SECONDS - t0) / 60 )) min $(( (SECONDS - t0) % 60 )) s" | tee -a "$summary"
    return $rc
}

echo "rerun in $out, $(git rev-parse --short HEAD)$(git diff --quiet || echo ' + uncommitted changes')" | tee -a "$summary"
df -h . | tail -1 | tee -a "$summary"

run pytest uv run pytest -q || { echo "tests failed: stopping (see $out)" | tee -a "$summary"; exit 1; }

# steps 3-7: pure-Python reference walkers on TOY
run census uv run uno census
run ladder uv run uno ladder TOY
run mirror uv run uno mirror TOY

# steps 8-8.6: compiled trees, Nash of TOY and TOY_FULL (saved), transfer
run cfr_TOY uv run uno cfr TOY 1000
run symmetry uv run uno symmetry 1000
run transfer uv run uno transfer

# step 8.7: sizes, loop cost, the TOY2 family (Nash saved), CFR variants
run sizing uv run uno sizing
run loop_cost uv run uno loop_cost
run toy2_TOY2 uv run uno toy2 TOY2 1000
run toy2_TOY2_SKIP uv run uno toy2 TOY2_SKIP 1000
run cfr_variants uv run uno cfr_variants
run toy2_TOY2_FULL uv run uno toy2 TOY2_FULL 150 light dcfr
run toy2_columns uv run uno toy2_columns TOY2_FULL

# steps 8.7-8.8: validation from the cached trees and Nash strategies
run validate_TOY uv run uno validate TOY
run validate_TOY2 uv run uno validate TOY2
run validate_TOY2_SKIP uv run uno validate TOY2_SKIP
run validate_TOY2_FULL uv run uno validate TOY2_FULL 4000 light

echo "done: $(grep -c 'exit 0 after' "$summary") of $n steps exited 0; failed checks per validate run:" | tee -a "$summary"
grep -h "failed checks" "$out"/*validate*.log | tee -a "$summary"
