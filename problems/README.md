# Problems

The 38 interval MDPs of the benchmark. The problem files are not in the repository (about
1.3 GB); the scripts below build them into `problems/<name>/<name>.{nc,json}`, which is where
the configs in [configs/](../configs/) look for them.

Every problem is an interval MDP with the same kind of property, so the algorithms are
compared on the model, not on the spec:

| | |
|---|---|
| Property | infinite-horizon reach-avoid: reach `reach` without entering `avoid` |
| Satisfaction | pessimistic (adversary picks transition probabilities within the intervals) |
| Strategy | maximise |
| Convergence | `convergence_eps = 1e-5` on the initial states (`restrict_to_initial = true` in the configs). This overrides the `eps` in each `.json` |
| Initial states `S0` | embedded in each `.nc`; see the `\|S0\|` column |

## Generate

All commands run from the repository root, after `julia --project -e 'using Pkg; Pkg.instantiate()'`.

| family | problems | command | needs |
|---|---|---|---|
| Storm | `storm_tiny*`, `crowds_quotient_3_5`, `brp_16_2` | `bash tools/storm/convert_storm.sh` | nothing else (the `.drn` models are in `tools/storm/models/`) |
| PRISM | `robot_delta0.1`, `coin2_K{2,4,16,64}`, `coin4_K{2,4,8}` | `bash tools/prism/convert_prism.sh robot_delta0.1 coin2_K2 coin2_K4 coin2_K16 coin2_K64 coin4_K2 coin4_K4 coin4_K8` | [PRISM](https://www.prismmodelchecker.org/) on the `PATH` |
| Abstractions | `car_parking_*`, `linear_stochastically_switched_*`, `robot_2d_*`, `van_der_pol_*`, `almost_identity_*`, `package_delivery_*`, `patrol_robot_*`, `bas_4d_*`, `impact_robot_2d_ra_*`, `room_temp_3d_*` | `julia --project tools/IntervalMDPAbstractions/run.jl` | about 2.5 GB of memory for the largest; minutes to an hour in total |
| Reproducibility package | `multiObj_robotIMDP`, `pimdp_0`, `pimdp_1` | `REPRO_PACKAGE=<checkout> bash tools/repro_package/fetch_lfs.sh /tmp/repro multiObj_robotIMDP pimdp_0 pimdp_1`, then `julia --project tools/repro_package/import_repro.jl /tmp/repro multiObj_robotIMDP pimdp_0 pimdp_1` | read access to [IntervalMDP.jl_ReproducibilityPackage](https://github.com/Zinoex/IntervalMDP.jl_ReproducibilityPackage) (not public): a checkout for the LFS pointer files and an SSH key for the LFS download |

The problem facts every run writes (`problems.csv` of a results dir: states, actions,
non-terminal states, transitions, initial states) are a quick check against the inventory
below.

## Inventory

Cheapest first by measured full-sweep time (the order of the configs). Column meanings:

- `maybe` is the number of states in neither `reach` nor `avoid`, i.e. the states whose value
  is actually computed.
- `transitions` is the number of stored interval transitions, summed over all state-action
  pairs.
- `Known V(S0)` is an independent reference value, where one exists. External values
  (Storm's test suite, closed forms) are cited under Families. The IMPaCT values are from
  our own runs.

| # | Problem | Family | \|S\| | \|A\| | maybe | transitions | \|S0\| | Known V(S0) |
|--:|---|---|--:|--:|--:|--:|--:|---|
| 1 | storm_tiny01 | Storm | 3 | 1 | 1 | 4 | 1 | 0.4 |
| 2 | storm_tiny02 | Storm | 3 | 1 | 1 | 5 | 1 | |
| 3 | storm_tiny03 | Storm | 3 | 2 | 1 | 8 | 1 | 0.4 |
| 4 | storm_tiny05 | Storm | 3 | 1 | 1 | 4 | 1 | 0.3 |
| 5 | storm_tiny_bounded | Storm | 4 | 1 | 2 | 7 | 1 | |
| 6 | storm_tiny04_noec | Storm | 5 | 3 | | | 1 | 3/7 ≈ 0.4286 |
| 7 | robot_delta0.1 | PRISM | 6 | 2 | 3 | 19 | 1 | 0.4 (= 0.5 − δ) |
| 8 | crowds_quotient_3_5 | Storm | 33 | 1 | 31 | 53 | 1 | 0.1383409 |
| 9 | brp_16_2 | Storm | 613 | 1 | 578 | 803 | 1 | |
| 10 | linear_stochastically_switched_S10 | IMDPAbstractions | 101 | 1 | 83 | 5,972 | 1 | |
| 11 | car_parking_S10_I3 | IMDPAbstractions | 101 | 9 | 88 | 37,404 | 1 | |
| 12 | linear_stochastically_switched_S20 | IMDPAbstractions | 401 | 1 | 325 | 76,272 | 4 | |
| 13 | coin2_K2 | PRISM coin | 272 | 2 | 264 | 668 | 1 | |
| 14 | van_der_pol_S20_I5 | IMDPAbstractions | 401 | 5 | 398 | 222,026 | 1 | |
| 15 | package_delivery_S12_I3 | ARCH-COMP | 145 | 9 | 136 | 47,212 | 2 | |
| 16 | robot_2d_S12_I5 | IMDPAbstractions | 145 | 25 | 127 | 121,368 | 1 | |
| 17 | coin2_K4 | PRISM coin | 528 | 2 | 520 | 1,308 | 1 | |
| 18 | room_temp_3d_S8_I6 | IMPaCT | 513 | 36 | 440 | 3,130,504 | 1 | ≈ 0.52 (ours) |
| 19 | bas_4d_S5x5x7x7_I4 | ARCH-COMP | 1,226 | 4 | 1,176 | 4,373,997 | 1 | |
| 20 | linear_stochastically_switched_S40 | IMDPAbstractions | 1,601 | 1 | 1,300 | 1,054,492 | 16 | |
| 21 | car_parking_S20_I3 | IMDPAbstractions | 401 | 9 | 352 | 457,416 | 4 | |
| 22 | package_delivery_S24_I5 | ARCH-COMP | 577 | 25 | 544 | 1,537,468 | 8 | |
| 23 | patrol_robot_S20_I5 | ARCH-COMP | 401 | 25 | 380 | 714,020 | 1 | |
| 24 | van_der_pol_S40_I10 | IMDPAbstractions | 1,601 | 10 | 1,588 | 5,767,067 | 4 | |
| 25 | almost_identity_n3_S16 | IMDPAbstractions | 4,097 | 1 | 4,032 | 2,796,044 | 1 | |
| 26 | multiObj_robotIMDP | Repro package | 207 | 4 | 205 | 2,784 | 1 | |
| 27 | package_delivery_S50_I5 | ARCH-COMP | 2,501 | 25 | 2,355 | 23,508,800 | 32 | |
| 28 | almost_identity_n3_S8 | IMDPAbstractions | 513 | 1 | 504 | 74,490 | 1 | |
| 29 | patrol_robot_S40_I5 | ARCH-COMP | 1,601 | 25 | 1,520 | 9,096,564 | 4 | |
| 30 | pimdp_0 | Repro package | 21,174 | 3 | 21,172 | 33,908,939 | 1 | |
| 31 | pimdp_1 | Repro package | 32,336 | 3 | 32,334 | 36,019,325 | 1 | |
| 32 | impact_robot_2d_ra_S20_I11 | IMPaCT | 401 | 121 | 375 | 4,437,048 | 1 | ≈ 0.86 (ours) |
| 33 | robot_2d_S30_I15 | IMDPAbstractions | 901 | 225 | 860 | 30,045,016 | 4 | |
| 34 | coin2_K16 | PRISM coin | 2,064 | 2 | 2,056 | 5,148 | 1 | |
| 35 | coin4_K2 | PRISM coin | 22,656 | 4 | 22,592 | 112,576 | 1 | |
| 36 | coin4_K4 | PRISM coin | 43,136 | 4 | 43,072 | 214,976 | 1 | |
| 37 | coin2_K64 | PRISM coin | 8,208 | 2 | 8,200 | 20,508 | 1 | |
| 38 | coin4_K8 | PRISM coin | 84,096 | 4 | 84,032 | 419,776 | 1 | |

**Naming** for abstraction problems: `S<n>` is the number of state cells per dimension (plus
one sink state for leaving the domain) and `I<n>` is the number of input cells per dimension.
For example, `robot_2d_S30_I15` has 30² + 1 = 901 states and 15² = 225 actions.

## Families

### Storm test models: `storm_tiny*`, `brp_16_2`, `crowds_quotient_3_5`

- **Source.** Storm's IMDP test files (`resources/examples/testfiles/imdp/*.drn`,
  stormchecker/storm @ `927db56d`), converted by
  [tools/storm/convert_storm.sh](../tools/storm/convert_storm.sh).
- **Spec.** Storm's robust `Pmax=? [F reach]`.
- **Avoid set.** Absorbing non-target traps. The reader needs a non-empty avoid set, and making
  traps the avoid set does not change any value.
- **Reference values.** From Storm's `RobustMdpPrctlModelCheckerTest.cpp`.
- **Models.**
  - `brp_16_2`: the bounded retransmission protocol (N=16, MAX=2). The target is the failure
    states.
  - `crowds_quotient_3_5`: the quotient interval DTMC of Crowds (N=3, K=5). The target is
    `observe0Greater1`.
  - The tiny models are hand-made corner cases, 3–5 states each.
- **Role.** Correctness checks with known answers, plus a cost floor. Most have 1 action, so
  they are interval Markov chains and only state sampling matters.
- **storm_tiny04_noec.** This is storm_tiny04 with state 1 made an avoid state. That removes the
  end component GSRDP cannot leave (see Not included), but it is a different problem: V(S0) = 3/7 instead of 1.

### PRISM: `robot_delta0.1`, `coin2_K*`, `coin4_K*`

- **Source.** Converted by [tools/prism/convert_prism.sh](../tools/prism/convert_prism.sh).
- **robot_delta0.1.**
  - Model: the 6-state robot IMDP ([tools/prism/simple/robot.prism](../tools/prism/simple/robot.prism)),
    with δ = 0.1.
  - Property: `Pmaxmin [F goal1]`, with the absorbing traps `goal2` as the avoid set.
  - Value: closed form 0.5 − δ.
- **coin.**
  - Model: the shared-coin protocol of Aspnes & Herlihy's randomised consensus, as an IMDP.
    Only process 1's coin is biased (`bias1 = 0.01`).
  - Scaling: `coin2` has N = 2 processes and `coin4` has N = 4, and K is the random-walk
    bound, so the ladder scales K for each N.
  - Property: `Pmaxmin [F finished & all_coins_equal_1]`, avoiding the "finished with the other
    decision" states. PRISM's published C2 is `Pminmin` (an adversarial scheduler); the
    benchmark uses the maximising controller, as for every other problem.
  - Structure: 2 or 4 actions and very sparse (~1.25 transitions per state-action pair).
    These problems have the most states and the least work per state.

### IntervalMDPAbstractions.jl examples: `car_parking`, `linear_stochastically_switched`, `robot_2d`, `van_der_pol`, `almost_identity`

- **Source.** Abstractions of continuous stochastic systems from the upstream examples
  (`IntervalMDPAbstractions.jl/examples/systems/`), rebuilt in
  [tools/IntervalMDPAbstractions/](../tools/IntervalMDPAbstractions/) so that an initial region
  can be embedded.
- **Common structure.** Leaving the domain goes to a sink state, which is always in `avoid`.
  `S0` is the cells overlapping a small initial region placed away from both reach and avoid,
  so restricting convergence to `S0` is meaningful.

| Family | Dynamics | Reach | Avoid (besides sink) | Actions |
|---|---|---|---|---|
| car_parking | 2-D, controlled | [4,10]×[−4,0] | [4,10]×[0,4] | I² |
| linear_stochastically_switched | 2-D linear, stochastically switched, no input | [−1,0]×[−1,1] | [1,2]×[0,1] | 1 |
| robot_2d | 2-D robot | [5,7]² | [−2,2]² | I² |
| van_der_pol | 2-D Van der Pol oscillator | [−1.4,−0.7]×[−2.9,−2.0] | — | I |
| almost_identity (n=3) | x⁺ = (0.7 I − 0.1 shift) x + w, w ~ N(0, 0.01 I), on [−1,1]³, no input | [−0.25,0.25]³ | — | 1 |

almost_identity's property is **augmented**: upstream it is finite-horizon safety. It serves as
a dimension-scaling family where action sampling plays no role.

### ARCH-COMP: `patrol_robot`, `package_delivery`, `bas_4d`

Abstracted with IntervalMDPAbstractions from the models of ArchCompStochasticModels.jl, in
[tools/IntervalMDPAbstractions/](../tools/IntervalMDPAbstractions/).

- **patrol_robot.** The "reduced patrol robot" from ArchCompStochasticModels.
  - Dynamics: x⁺ = x + 10 u₁ [cos u₂, sin u₂] + w, with w ~ N(0, 0.75 I).
  - Regions: reach [5,7]², avoid [−2,2]². This is the only ARCH-COMP benchmark whose
    published property is already infinite-horizon reach-avoid.
  - Upstream bug: the published spec swaps the reach and avoid sets, so the regions are taken
    from `reduced_patrol_robot_regions()` instead.
- **package_delivery.** ArchCompStochasticModels' `package_delivery_synthesis`.
  - Regions: `p1` is the initial region, `p2` is avoid, `p3` is reach.
  - The number of initial states grows with resolution (2, 8 and 32 cells).
- **bas_4d.** 4-D building automation (ARCH-COMP BAS CS1): two zone temperatures, two radiator
  temperatures, one heater input.
  - Property (**augmented**; upstream is finite-horizon safety): reach the comfort band
    x₁, x₂ ∈ [19.75, 20.25] without leaving the region of interest.

### IMPaCT: `impact_robot_2d_ra`, `room_temp_3d`

- **Source.** IMPaCT's examples (github.com/Kiguli/IMPaCT), abstracted here with
  IntervalMDPAbstractions rather than with IMPaCT itself
  ([tools/IntervalMDPAbstractions/](../tools/IntervalMDPAbstractions/)).
- **impact_robot_2d_ra.**
  - Dynamics: x⁺ = x + 2 u₁ [cos u₂, sin u₂] + w, with w ~ N(0, I/1.3333).
  - Regions: reach [5,8]², avoid [−1.5,1.5]².
  - Not a duplicate of patrol_robot or robot_2d: its step size, input range and regions differ.
    With 121 actions it is the most action-heavy problem after robot_2d_S30.
- **room_temp_3d.** Three rooms, with heaters in rooms 1 and 3 (bilinear in u·x).
  - Property (**augmented**; upstream is safety): warm both heated rooms to ≥ 20.25 while no
    room leaves [19, 21].

### IntervalMDP.jl reproducibility package: `multiObj_robotIMDP`, `pimdp_0`, `pimdp_1`

- **Source.** Imported by
  [tools/repro_package/import_repro.jl](../tools/repro_package/import_repro.jl).
- **Spec.** The published specs are finite-horizon (H=200) reachability. Here they are
  infinite-horizon:
  - `reach`: the published reach state(s), unchanged.
  - `avoid`: every other absorbing state (value-neutral).
- **Initial states.** The files carry no initial state. `S0` is the non-terminal state farthest
  from the goal that can still reach it.
- **pimdp_0/1.** Large product IMDPs (21k and 32k states, ~34M transitions) with *possible*
  end components (adversary-dependent, 10k and 23k states reachable from S0). GSRDP still
  terminates on them.

## Not included

| Problem | Why |
|---|---|
| storm_tiny04 | Sure end component (a self-loop action with U = 1). GSRDP's upper-greedy policy keeps the self-loop, so L(S0) stalls at 3/7 while U stays at 1 forever. It needs MEC collapsing. `convert_storm.sh` still builds it as the end-component stress case; storm_tiny04_noec (#6) runs instead. |
| coin2_K256 (33k states) | Built by `convert_prism.sh`, but not part of the thesis dataset. |
| coin4_K16 (166k states) | Extrapolated full-sweep solve ~5 h, so warm-up + 5 samples exceeds 24 h even in a job of its own. |
| coin4_K32 (330k states) | Extrapolated ~35 h per solve. |
| coin2_K1024 (131k states) | Extrapolated ~60 h per solve. |

`convert_prism.sh` builds coin4_K16, coin4_K32 and coin2_K1024 too when run without names; pass
names to skip them. Run [tools/end_components.jl](../tools/end_components.jl) when adding
problems: of the problems above, only storm_tiny04 (sure) and pimdp_0/1 (possible only) have
end components.

