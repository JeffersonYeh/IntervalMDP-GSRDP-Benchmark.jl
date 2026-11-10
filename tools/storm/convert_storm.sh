#!/usr/bin/env bash
# Convert Storm's interval-MDP test models (DRN) to IntervalMDP.jl .nc + .json problems
# (infinite-time reach-avoid) under problems/<name>/.
#
# models/*.drn are copied verbatim from Storm's resources/examples/testfiles/imdp/
# (github.com/stormchecker/storm @ 927db56d). coin2.prism and robot.prism from the same
# directory are not converted: they are the tools/prism models (coin2_K*, robot_delta0.1).
#
# Usage: bash convert_storm.sh [name ...]   (no names = build every problem below)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH="$(cd "$HERE/../.." && pwd)"
PROBLEMS="$BENCH/problems"

# name | model file | reach label | avoid label | strategy | satisfaction
#
# avoid "@absorbing" = the absorbing non-target traps (value 0 either way; see drn_to_imdp.jl
# for why a reach-only model still needs a non-empty avoid set). "-" = none.
#
# All max/pessimistic, i.e. Storm's robust "Pmax=? [F reach]"; reference values from
# src/test/storm/modelchecker/prctl/mdp/RobustMdpPrctlModelCheckerTest.cpp:
#   tiny-01 0.4, tiny-03 0.4, tiny-04 1, tiny-05 0.3, crowds-quotient-3-5 0.1383409.
#   tiny-02 has no reference; tiny-bounded is only checked on bounded-time properties.
# brp:    "target" = the failure states of the bounded retransmission protocol; the three
#         "deadlock" states are its absorbing terminal states (made the avoid set).
# crowds: quotient IDTMC of Crowds (N=3, K=5); target "observe0Greater1".
# tiny04_noec: tiny-04 with state 1 made an avoid state. In storm_tiny04, state 1's self-loop
#         action is an end component on which GSRDP never terminates (the upper-greedy policy keeps
#         the self-loop, so the lower bound stalls at 3/7). Avoiding state 1 removes it but is a
#         different problem: V(s0) = 3/7 instead of 1. Avoid items are a comma-separated union of
#         labels, "state:<0-based id>" and "@absorbing" (which keeps state 4, itself a trap, out).
PROBLEM_TABLE="
storm_tiny01        | tiny-01.drn             | target           | @absorbing | max | pessimistic
storm_tiny02        | tiny-02.drn             | target           | @absorbing | max | pessimistic
storm_tiny03        | tiny-03.drn             | target           | @absorbing | max | pessimistic
storm_tiny04        | tiny-04.drn             | target           | @absorbing | max | pessimistic
storm_tiny04_noec   | tiny-04.drn             | target           | @absorbing,state:1 | max | pessimistic
storm_tiny05        | tiny-05.drn             | target           | @absorbing | max | pessimistic
storm_tiny_bounded  | tiny-bounded.drn        | target           | @absorbing | max | pessimistic
brp_16_2            | brp-16-2.drn            | target           | deadlock | max | pessimistic
crowds_quotient_3_5 | crowds-quotient-3-5.drn | observe0Greater1 | @absorbing | max | pessimistic
"

selected() {
    [ "$#" -eq 1 ] && return 0   # no names given: build everything
    local n; for n in "${@:2}"; do [ "$n" = "$1" ] && return 0; done
    return 1
}

while IFS='|' read -r name model reach avoid strategy satisfaction; do
    name="$(echo "$name" | xargs)"
    [ -z "$name" ] && continue
    selected "$name" "$@" || continue
    model="$(echo "$model" | xargs)"; reach="$(echo "$reach" | xargs)"
    avoid="$(echo "$avoid" | xargs)"
    strategy="$(echo "$strategy" | xargs)"; satisfaction="$(echo "$satisfaction" | xargs)"

    avoid_args=()
    [ "$avoid" != "-" ] && avoid_args=(--avoid-label "$avoid")

    echo "==> $name  ($model)"
    julia --project="$BENCH" "$HERE/drn_to_imdp.jl" "$HERE/models/$model" \
        --out "$PROBLEMS/$name" --name "$name" --reach-label "$reach" "${avoid_args[@]}" \
        --strategy "$strategy" --satisfaction "$satisfaction"
done <<< "$PROBLEM_TABLE"
