# Builds the 19 abstraction problems of the benchmark into problems/<name>/ (below), from the
# repository root:
#
#   julia --project tools/IntervalMDPAbstractions/run.jl
#
# or loads every problem-construction script so their `main` can be called by hand
# (julia --project -i, then include this file with the calls below commented out), e.g.:
#   CarParking.main(5, 3)
#   LinearStochasticallySwitched.main(5)
#   Robot2D.main(5, 3)
#   VanDerPol.main(5, 3)
#   PackageDelivery.main(5, 3)
#   PatrolRobot.main(20, 5)
#   Bas4D.main((5, 5, 7, 7), 4)
#   AlmostIdentity.main(3, 8)
#   ImpactRobot2D.main(20, 11)
#   RoomTemp.main(3, 8, 6)
#
# Each is wrapped in its own module because they share identical
# `const DIR/NAME/SPARSE/EPS` and `main` names that would otherwise clash.

module CarParking; include(joinpath(@__DIR__, "car_parking.jl")); end
module LinearStochasticallySwitched; include(joinpath(@__DIR__, "linear_stochastically_switched.jl")); end
module Robot2D; include(joinpath(@__DIR__, "robot_2d.jl")); end
module VanDerPol; include(joinpath(@__DIR__, "van_der_pol.jl")); end
module PackageDelivery; include(joinpath(@__DIR__, "package_delivery.jl")); end
module PatrolRobot; include(joinpath(@__DIR__, "patrol_robot.jl")); end
module Bas4D; include(joinpath(@__DIR__, "bas_4d.jl")); end
module AlmostIdentity; include(joinpath(@__DIR__, "almost_identity.jl")); end
# IMPaCT examples (github.com/Kiguli/IMPaCT, examples/), abstracted here rather than with IMPaCT
module ImpactRobot2D; include(joinpath(@__DIR__, "impact_robot_2d.jl")); end
module RoomTemp; include(joinpath(@__DIR__, "room_temp.jl")); end


# Every abstraction problem of configs/thesis/*.toml. All of these embed an initial-state set
# (each generator's `INITIAL_REGION`), without which `restrict_to_initial` is silently a no-op
# and the traces' `initial_gap_*` columns are NaN.

CarParking.main(10, 3); GC.gc()
CarParking.main(20, 3); GC.gc()
LinearStochasticallySwitched.main(10); GC.gc()
LinearStochasticallySwitched.main(20); GC.gc()
LinearStochasticallySwitched.main(40); GC.gc()
Robot2D.main(12, 5); GC.gc()
Robot2D.main(30, 15); GC.gc()
VanDerPol.main(20, 5); GC.gc()
VanDerPol.main(40, 10); GC.gc()
PackageDelivery.main(12, 3); GC.gc()
PackageDelivery.main(24, 5); GC.gc()
PackageDelivery.main(50, 5); GC.gc()

PatrolRobot.main(20, 5); GC.gc()
PatrolRobot.main(40, 5); GC.gc()
Bas4D.main((5, 5, 7, 7), 4); GC.gc()
AlmostIdentity.main(3, 8); GC.gc()
AlmostIdentity.main(3, 16); GC.gc()

ImpactRobot2D.main(20, 11); GC.gc()
RoomTemp.main(3, 8, 6); GC.gc()

# Cluster-only: each peaked (or is estimated to peak) above what a shared WSL box can spare.
#   PatrolRobot.main(40, 9)       -- ~29M transitions (S40_I5 is ~9M)
#   Bas4D.main((10, 10, 7, 7), 4) -- ~70M transitions (S5x5x7x7 is already ~4.4M, rows near-dense)
#   ImpactRobot2D.main(40, 11)    -- est. ~70M transitions (S20_I11: 4.4M, ~92 successors per choice,
#                                    1.7 GB peak). IMPaCT's own grid, (40, 21), is est. ~260M: impractical
#   RoomTemp.main(3, 16, 6)       -- est. ~200M transitions (S8: 3.1M, ~170 successors per choice of 513)

# VanDerPol.main(10, 3)  -- S10 cannot represent the reach region, see spec.jl
#
# Tried and dropped (every state outside reach has value 0 under the pessimistic abstraction,
# converging in 1-5 sweeps): the noise per step is far smaller than any tractable cell, so each
# cell's lower bounds into its neighbours are ~0 and the adversary can hold the state in place.
#   ArchComp fully-automated anaesthesia (sigma ~0.03 vs cells >= 0.5; Minimize *and* Maximize),
#   ArchComp integrator chain (sigma 0.1 vs cells of 1, and of 0.2 on a shrunk [-2,2]^n),
#   ArchComp reduced automated vehicle (heading noise var 2/3 on a non-wrapping heading: sink),
#   almost_identity n=4 at S8 (V(S0) ~ 1e-4, below eps),
#   IMPaCT 5-room temperature, RoomTemp.main(5, 5, 3) (sigma 0.1 vs cells of 0.4: V(S0) = 0.01 in
#     7 sweeps; heated >= 19.8/20.2, either heated, all rooms >= 19.4/19.8 are all <= 7 sweeps and
#     V(S0) <= 0.03 or exactly 1; S10 would be 100k states),
#   IMPaCT 7-D BAS, bas_7d.jl at (10,10,2,2,2,2,2) (x3..x7 have two 0.5-wide cells vs sigma ~0.2, so
#     the pessimistic exit probability swamps everything: max V <= 0.006 in 4-5 sweeps for reach
#     x1,x2 in [-0.1,0.1] up to [-0.3,0.3]; only the optimistic bound, ~0.5 at S0, is informative).
#     IMPaCT's own (20,20,2,...) grid refines only x1,x2 and would not change that.
