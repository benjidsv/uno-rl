#!/usr/bin/env bash
# Re-run every experiment of docs/foundations/results/, in dependency order, with a live status line and a final report.
#
#   bash scripts/rerun_all.sh
#
# In logs/rerun-<date>-<time>/, per step NN_<step>:
#   .log       the experiment's own output
#   .progress  its tqdm bars (uno/progress.py)
#   .timings   one line per loop of 2 s or more: "<what>: took <s> s for <count> <unit> (<rate>)"
# and for the whole run:
#   summary.txt  start, exit code and duration of every step, as they happen
#   report.txt   at the end (or on Ctrl-C): every step with its exit code, duration and expected
#                duration, its timed loops indented below it, and the failed checks of each
#                validate, robustness and pairing step
#
# On a terminal a status line shows the step, its elapsed and expected time, the time left for the
# run and the latest progress bar. A step's expected time is its last successful duration in
# logs/durations.tsv, or the rough guess below on a first run; the time left is rescaled by how the
# finished steps compared with their expectation.
#
# The test suite runs first and stops everything if it fails; any other failing step is recorded
# and the run goes on. Order: `cfr TOY`, `symmetry` and `toy2 <ruleset>` save the Nash strategies
# that `transfer`, `toy2_columns`, `validate`, `deal_model` (TOY_FULL / TOY2_FULL) and `memory`
# load. The TOY2_FULL steps come last: each holds 2-5 GB of tree in memory. Trees are compiled on
# first use into .cache/ (files from before CACHE_VERSION 1 are stale and recompiled). The report's
# failed-check summary greps every validate, robustness, pairing, edge, memory, deal_model, aivat, pimc_value, report and lbr log.
# The lbr steps need the cached FULL_UNO value networks, like colfree, so they sit beside it.
set -u -o pipefail
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1

# step, expected minutes on a first run (guesses), command
STEPS="
pytest               4  uv run pytest -q
census              10  uv run uno census
ladder              25  uv run uno ladder TOY
mirror              25  uv run uno mirror TOY
cfr_TOY             10  uv run uno cfr TOY 1000
symmetry            15  uv run uno symmetry 1000
transfer            15  uv run uno transfer
sizing              25  uv run uno sizing
loop_cost            5  uv run uno loop_cost
toy2_TOY2            8  uv run uno toy2 TOY2 1000
toy2_TOY2_SKIP      12  uv run uno toy2 TOY2_SKIP 1000
cfr_variants        10  uv run uno cfr_variants
toy2_TOY2_FULL      25  uv run uno toy2 TOY2_FULL 150 light dcfr
toy2_columns        20  uv run uno toy2_columns TOY2_FULL
validate_TOY         8  uv run uno validate TOY
validate_TOY2       12  uv run uno validate TOY2
validate_TOY2_SKIP  15  uv run uno validate TOY2_SKIP
validate_TOY2_FULL  20  uv run uno validate TOY2_FULL 4000 light
robustness_TOY       1  uv run uno robustness TOY
robustness_TOY_FULL  9  uv run uno robustness TOY_FULL
robustness_TOY2      5  uv run uno robustness TOY2
robustness_TOY2_SKIP 16  uv run uno robustness TOY2_SKIP
robustness_TOY2_FULL 71  uv run uno robustness TOY2_FULL 150 light
pairing             14  uv run uno pairing
edge_TOY             2  uv run uno edge TOY 1000
edge_TOY_FULL         2  uv run uno edge TOY_FULL 1000
edge_TOY2             2  uv run uno edge TOY2 1000
edge_TOY2_SKIP        2  uv run uno edge TOY2_SKIP 1000
edge_TOY2_FULL       15  uv run uno edge TOY2_FULL 150 light
report_TOY_heuristic  1  uv run uno report TOY heuristic
report_TOY_random     1  uv run uno report TOY random
report_TOY2_heuristic 2  uv run uno report TOY2 heuristic
report_TOY2_SKIP      3  uv run uno report TOY2_SKIP heuristic
target_TOY            2  uv run uno target TOY
target_TOY_FULL       2  uv run uno target TOY_FULL 1000 1000 0
shaping_TOY           1  uv run uno shaping TOY
shaping_TOY_FULL      1  uv run uno shaping TOY_FULL
shaping_TOY2          2  uv run uno shaping TOY2
shaping_TOY2_SKIP     3  uv run uno shaping TOY2_SKIP
rungs_TOY             4  uv run uno rungs TOY
colfree              45  uv run uno colfree
lbr_exact_TOY         4  uv run uno lbr exact TOY
lbr_games_TOY         1  uv run uno lbr games TOY 200 4
lbr_pilot             5  uv run uno lbr pilot 200 32 2 "" 100
lbr_eval             10  uv run uno lbr eval 4000
memory_TOY            1  uv run uno memory TOY 20
memory_TOY_FULL       1  uv run uno memory TOY_FULL 20
memory_TOY2           1  uv run uno memory TOY2 20
memory_TOY2_SKIP      2  uv run uno memory TOY2_SKIP 20
memory_TOY2_FULL     15  uv run uno memory TOY2_FULL 20
deal_model_TOY        2  uv run uno deal_model TOY 100000
deal_model_TOY_FULL   1  uv run uno deal_model TOY_FULL 100000
deal_model_TOY2       2  uv run uno deal_model TOY2 100000
deal_model_TOY2_SKIP  2  uv run uno deal_model TOY2_SKIP 100000
deal_model_TOY2_FULL  2  uv run uno deal_model TOY2_FULL 100000
aivat_TOY             2  uv run uno aivat TOY
aivat_TOY_FULL        2  uv run uno aivat TOY_FULL
aivat_TOY2            2  uv run uno aivat TOY2
aivat_TOY2_SKIP       2  uv run uno aivat TOY2_SKIP
aivat_DUPS            4  uv run uno aivat DUPS
aivat_uno_sweep       2  uv run uno aivat_uno sweep 1000
aivat_uno            60  uv run uno aivat_uno 4000 10000
pimc_value_check      5  uv run uno pimc_value check 400 400
pimc_value_floor     10  uv run uno pimc_value floor 2000 32
pimc_value_pilot     30  uv run uno pimc_value pilot 5000
pimc_value_aivat     85  uv run uno pimc_value aivat 4000 20000 2
pimc_value_mc        15  uv run uno pimc_value mc 200 16
aivat_TOY2_FULL       8  uv run uno aivat TOY2_FULL 0 150 light
"

names=()
guesses=()
cmds=()
while read -r name guess cmd; do
    [ -z "$name" ] && continue
    names+=("$name")
    guesses+=("$guess")
    cmds+=("$cmd")
done <<< "${RERUN_STEPS:-$STEPS}"
total=${#names[@]}

out="${RERUN_LOGS:-logs}/rerun-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"
summary="$out/summary.txt"
report_file="$out/report.txt"
durations="${RERUN_LOGS:-logs}/durations.tsv"
commit="$(git rev-parse --short HEAD)$(git diff --quiet || echo ' + uncommitted changes')"

hms() { printf "%d:%02d:%02d" $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }
say() { echo "$1" | tee -a "$summary"; }
tag() { printf "%02d_%s" $(($1 + 1)) "${names[$1]}"; }

expected=()
exp_total=0
for ((i = 0; i < total; i++)); do
    d=""
    [ -f "$durations" ] && d=$(awk -F'\t' -v s="${names[$i]}" '$1 == s {d = $2} END {print d}' "$durations")
    expected+=("${d:-$((guesses[i] * 60))}")
    exp_total=$((exp_total + expected[i]))
done

rcs=()
dts=()
failed=0
run_start=$SECONDS

report() {  # report <how the run ended>
    {
        echo "rerun $out at $commit: $1 after $(hms $((SECONDS - run_start))), ${#rcs[@]} of $total steps run, $failed failed"
        echo
        for ((j = 0; j < ${#rcs[@]}; j++)); do
            printf "%2d %-20s exit %-3s took %s (expected ~%s)\n" $((j + 1)) "${names[$j]}" "${rcs[$j]}" "$(hms "${dts[$j]}")" "$(hms "${expected[$j]}")"
            t="$out/$(tag "$j").timings"
            [ -s "$t" ] && sed 's/^\[[0-9:]*\] /      /' "$t"
        done
        checks=$(ls "$out"/*validate*.log "$out"/*robustness*.log "$out"/*pairing*.log "$out"/*edge*.log "$out"/*memory*.log "$out"/*deal_model*.log "$out"/*aivat*.log "$out"/*pimc_value*.log "$out"/*report*.log "$out"/*target*.log "$out"/*shaping*.log "$out"/*rungs*.log "$out"/*colfree*.log "$out"/*lbr*.log 2> /dev/null)
        if [ -n "$checks" ]; then
            echo
            grep -H "failed checks" $checks | sed "s|$out/||"
        fi
    } > "$report_file"
    echo
    cat "$report_file"
}

say "rerun in $out at $commit: $total steps, expected ~$(hms $exp_total)"
df -h . | tail -1 >> "$summary"

pid=""
trap '[ -n "$pid" ] && { pkill -TERM -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; }; printf "\n"; say "interrupted"; report interrupted; exit 130' INT TERM

done_exp=0
done_real=0
for ((i = 0; i < total; i++)); do
    name=${names[$i]}
    cmd=${cmds[$i]}
    exp=${expected[$i]}
    log="$out/$(tag "$i").log"
    prog="$out/$(tag "$i").progress"
    timings="$out/$(tag "$i").timings"
    : > "$prog"
    echo "[$(date '+%H:%M:%S')] start $name: $cmd" >> "$summary"
    t0=$SECONDS
    env UNO_PROGRESS_FILE="$prog" UNO_TIMINGS_FILE="$timings" $cmd > "$log" 2>&1 < /dev/null &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [ -t 1 ]; then
            el=$((SECONDS - t0))
            left=$(awk -v de="$done_exp" -v dr="$done_real" -v rest="$((exp_total - done_exp - exp))" -v e="$exp" -v el="$el" \
                'BEGIN { s = (de > 0) ? dr / de : 1; cur = e * s - el; if (cur < 0) cur = 0; printf "%d", rest * s + cur }')
            src=$log
            [ -s "$prog" ] && [ "$prog" -nt "$log" ] && src=$prog
            # tail -c can cut a multibyte character (bar blocks, ±): split bytes under C, then drop the pieces
            last=$(tail -c 400 "$src" | LC_ALL=C tr '\r' '\n' | LC_ALL=C awk 'NF' | tail -1 | iconv -c -f UTF-8 -t UTF-8 2>/dev/null)
            line="[$((i + 1))/$total] $name $(hms $el) (~$(hms $exp)) | run $(hms $((SECONDS - run_start))), ~$(hms $left) left | $last"
            cols=$(tput cols 2>/dev/null || echo 120)
            printf "\r\033[K%s" "${line:0:$((cols - 1))}"
        fi
        sleep 2
    done
    wait "$pid"
    rc=$?
    pid=""
    dt=$((SECONDS - t0))
    [ -t 1 ] && printf "\r\033[K"
    rcs+=("$rc")
    dts+=("$dt")
    say "[$(date '+%H:%M:%S')] $(printf '%2d/%d' $((i + 1)) "$total") $name: exit $rc, took $(hms $dt) (expected ~$(hms $exp))"
    done_exp=$((done_exp + exp))
    done_real=$((done_real + dt))
    if [ "$rc" -eq 0 ]; then
        printf "%s\t%s\t%s\n" "$name" "$dt" "$(date +%Y-%m-%d)" >> "$durations"
    else
        failed=$((failed + 1))
        if [ "$name" = pytest ]; then
            say "tests failed: stopping, see $log"
            report "stopped: tests failed"
            exit 1
        fi
    fi
done

report finished
