#!/usr/bin/env bash
# Build the PRISM case studies and convert them to IntervalMDP.jl .nc + .json problems
# (infinite-time reach-avoid) under problems/<name>/.
#
# Each model declares the labels "reach" and "avoid" used by the spec; PRISM adds "init".
# The PRISM explicit export is only an intermediate and is deleted afterwards.
#
# Usage: bash convert_prism.sh [name ...]   (no names = build every problem below)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/../.." && pwd)"
PROBLEMS="$BENCH/problems"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# name | model file | PRISM constants | strategy | satisfaction
#
# robot: Pmaxmin [ F goal1 ], avoiding the absorbing traps goal2 = {s2,s3}.
#        Closed-form value 0.5 - delta.
# coin:  Pmaxmin [ F finished & all_coins_equal_1 ], avoiding the absorbing
#        "finished with another decision" states: the scheduler is the maximizing
#        controller, as for every other benchmark problem. (PRISM's published C2 is
#        Pminmin, an adversarial scheduler, with analytic value (K-1)/(2K); that is not
#        the question the benchmark asks.) Cross-check with the Pmaxmin line in
#        consensus/coin.props.
#        Only process 1's coin is biased (bias1); the size ladder varies N and K.
PROBLEM_TABLE="
robot_delta0.1 | simple/robot.prism     | delta=0.1         | max | pessimistic
coin2_K2       | consensus/coin2.prism  | K=2,bias1=0.01    | max | pessimistic
coin2_K4       | consensus/coin2.prism  | K=4,bias1=0.01    | max | pessimistic
coin2_K16      | consensus/coin2.prism  | K=16,bias1=0.01   | max | pessimistic
coin2_K64      | consensus/coin2.prism  | K=64,bias1=0.01   | max | pessimistic
coin2_K256     | consensus/coin2.prism  | K=256,bias1=0.01  | max | pessimistic
coin2_K1024    | consensus/coin2.prism  | K=1024,bias1=0.01 | max | pessimistic
coin4_K2       | consensus/coin4.prism  | K=2,bias1=0.01    | max | pessimistic
coin4_K4       | consensus/coin4.prism  | K=4,bias1=0.01    | max | pessimistic
coin4_K8       | consensus/coin4.prism  | K=8,bias1=0.01    | max | pessimistic
coin4_K16      | consensus/coin4.prism  | K=16,bias1=0.01   | max | pessimistic
coin4_K32      | consensus/coin4.prism  | K=32,bias1=0.01   | max | pessimistic
"

selected() {
    [ "$#" -eq 1 ] && return 0   # no names given: build everything
    local n; for n in "${@:2}"; do [ "$n" = "$1" ] && return 0; done
    return 1
}

while IFS='|' read -r name model consts strategy satisfaction; do
    name="$(echo "$name" | xargs)"
    [ -z "$name" ] && continue
    selected "$name" "$@" || continue
    model="$(echo "$model" | xargs)"; consts="$(echo "$consts" | xargs)"
    strategy="$(echo "$strategy" | xargs)"; satisfaction="$(echo "$satisfaction" | xargs)"

    echo "==> $name  ($model, $consts)"
    base="$WORK/$name"
    prism "$HERE/$model" -const "$consts" \
        -exportstates "$base.sta" -exporttrans "$base.tra" -exportlabels "$base.lab" \
        > "$base.log" || { cat "$base.log"; exit 1; }

    out="$PROBLEMS/$name"
    rm -f "$out"/*.sta "$out"/*.tra "$out"/*.lab "$out"/*.pctl
    julia --project="$BENCH" "$HERE/prism_to_imdp.jl" "$base" \
        --out "$out" --name "$name" --strategy "$strategy" --satisfaction "$satisfaction"
    rm -f "$base".*
done <<< "$PROBLEM_TABLE"
