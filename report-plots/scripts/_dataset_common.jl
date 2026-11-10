# Shared setup of dataset-metrics.jl and dataset-cluster.jl (not run on its own: `_`-prefixed
# scripts are skipped by make_all.jl): the 38 problems (problems.csv of THESIS_DIR), their
# source and class groupings, the mixing-statistics folders and the short problem names.

# `CATEGORIES` (source) and `CLASSES` (problem class: what a problem models) are the
# single source of the two problem groupings for the dataset scripts. CATEGORIES deliberately
# differs from `family()` in tools/mixing_stats.jl (the problem-directory view):
# PRISM coin is folded into PRISM, the IntervalMDPAbstractions examples and the pimdp product
# IMDPs are both "IntervalMDP.jl", van_der_pol counts as ARCH-COMP, and multiObj_robotIMDP
# is attributed to the BMDP tool it originates from.

include(joinpath(@__DIR__, "..", "common.jl"))
using CSV
using LinearAlgebra: svd, norm
import Distributions, StatsBase

const DIR = THESIS_DIR

# Display order, which the figures also use for legends.
const CATEGORIES = [
    "Storm"         => ["storm_tiny01", "storm_tiny02", "storm_tiny03", "storm_tiny05",
                        "storm_tiny_bounded", "storm_tiny04_noec", "crowds_quotient_3_5",
                        "brp_16_2"],
    "PRISM"         => ["robot_delta0.1", "coin2_K2", "coin2_K4", "coin2_K16", "coin2_K64",
                        "coin4_K2", "coin4_K4", "coin4_K8"],
    "IntervalMDP.jl" => ["car_parking_S10_I3", "car_parking_S20_I3",
                        "linear_stochastically_switched_S10", "linear_stochastically_switched_S20",
                        "linear_stochastically_switched_S40", "robot_2d_S12_I5", "robot_2d_S30_I15",
                        "almost_identity_n3_S8", "almost_identity_n3_S16", "pimdp_0", "pimdp_1"],
    "ARCH-COMP"     => ["package_delivery_S12_I3", "package_delivery_S24_I5",
                        "package_delivery_S50_I5", "patrol_robot_S20_I5", "patrol_robot_S40_I5",
                        "bas_4d_S5x5x7x7_I4", "van_der_pol_S20_I5", "van_der_pol_S40_I10"],
    "IMPaCT"        => ["impact_robot_2d_ra_S20_I11", "room_temp_3d_S8_I6"],
    "BMDP-tool"     => ["multiObj_robotIMDP"],
]
const CATEGORY_NAMES = first.(CATEGORIES)
# Source is carried by marker shape alone (colour is taken by the transition count).
const CATEGORY_MARKER = Dict(zip(CATEGORY_NAMES,
                                 [:circle, :rect, :diamond, :utriangle, :dtriangle, :star5]))
const CATEGORY_OF = Dict(p => c for (c, ps) in CATEGORIES for p in ps)
@assert length(CATEGORY_OF) == sum(length ∘ last, CATEGORIES) "a problem is listed under two categories"

# Second grouping: what each problem models (problem class), independent of its source.
# Display order as in the thesis; a class may be empty (resource management / scheduling
# has no problem in this dataset yet) and is kept so tables and legends can show the gap.
const CLASSES = [
    "Communication / network protocols"  => ["brp_16_2"],
    "Consensus and coordination"         => ["coin2_K2", "coin2_K4", "coin2_K16", "coin2_K64",
                                             "coin4_K2", "coin4_K4", "coin4_K8"],
    "Security protocols"                 => ["crowds_quotient_3_5"],
    "Resource management, scheduling"    => String[],
    "Motion planning"                    => ["robot_delta0.1", "car_parking_S10_I3",
                                             "car_parking_S20_I3", "robot_2d_S12_I5",
                                             "robot_2d_S30_I15", "package_delivery_S12_I3",
                                             "package_delivery_S24_I5", "package_delivery_S50_I5",
                                             "patrol_robot_S20_I5", "patrol_robot_S40_I5",
                                             "impact_robot_2d_ra_S20_I11", "multiObj_robotIMDP"],
    "Stochastic Dynamical Systems"       => ["linear_stochastically_switched_S10",
                                             "linear_stochastically_switched_S20",
                                             "linear_stochastically_switched_S40",
                                             "van_der_pol_S20_I5", "van_der_pol_S40_I10",
                                             "almost_identity_n3_S8", "almost_identity_n3_S16",
                                             "bas_4d_S5x5x7x7_I4", "room_temp_3d_S8_I6"],
    "Synthetic models"                   => ["storm_tiny01", "storm_tiny02", "storm_tiny03",
                                             "storm_tiny05", "storm_tiny_bounded",
                                             "storm_tiny04_noec", "pimdp_0", "pimdp_1"],
]
const CLASS_NAMES = first.(CLASSES)
const CLASS_OF = Dict(p => c for (c, ps) in CLASSES for p in ps)
@assert length(CLASS_OF) == sum(length ∘ last, CLASSES) "a problem is listed under two classes"
@assert keys(CLASS_OF) == keys(CATEGORY_OF) "CLASSES and CATEGORIES must list the same problems"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# The fixed row order of every per-problem table. A combined results dir lists its problems in
# the order its sources were merged, and the clustering (Ward's tie-breaking, the dendrogram's
# leaf order, the label placement) depends on the row order, so the problems are sorted by this.
const DATASET_ORDER = [
    "storm_tiny01", "storm_tiny02", "storm_tiny03", "storm_tiny05", "storm_tiny_bounded",
    "storm_tiny04_noec", "robot_delta0.1", "crowds_quotient_3_5", "brp_16_2",
    "linear_stochastically_switched_S10", "car_parking_S10_I3",
    "linear_stochastically_switched_S20", "coin2_K2", "van_der_pol_S20_I5",
    "package_delivery_S12_I3", "robot_2d_S12_I5", "coin2_K4", "room_temp_3d_S8_I6",
    "bas_4d_S5x5x7x7_I4", "linear_stochastically_switched_S40", "car_parking_S20_I3",
    "package_delivery_S24_I5", "patrol_robot_S20_I5", "van_der_pol_S40_I10",
    "almost_identity_n3_S16", "multiObj_robotIMDP", "package_delivery_S50_I5",
    "almost_identity_n3_S8", "patrol_robot_S40_I5", "pimdp_0", "pimdp_1",
    "impact_robot_2d_ra_S20_I11", "robot_2d_S30_I15", "coin2_K16", "coin4_K2", "coin4_K4",
    "coin2_K64", "coin4_K8"]


problems = CSV.read(joinpath(DIR, "problems.csv"), DataFrame)
issetequal(problems.problem, DATASET_ORDER) ||
    error("_dataset_common.jl: $(basename(DIR))/problems.csv does not hold the 38 problems of DATASET_ORDER")
problems = problems[sortperm([findfirst(==(p), DATASET_ORDER) for p in problems.problem]), :]
unmapped = setdiff(problems.problem, keys(CATEGORY_OF))
isempty(unmapped) || error("_dataset_common.jl: no category for $(join(unmapped, ", ")); add them to CATEGORIES")
problems.category = [CATEGORY_OF[p] for p in problems.problem]
problems.class = [CLASS_OF[p] for p in problems.problem]
rename!(problems, :num_maybe_states => :num_nt)
absent = setdiff(keys(CATEGORY_OF), problems.problem)
isempty(absent) || error("_dataset_common.jl: $(join(absent, ", ")) not in $(basename(DIR)); remove from CATEGORIES")

const INK = "#0b0b0b"

# Per-problem structure statistics (tools/mixing_stats.jl and friends)
const MIXING_DIR = joinpath(BENCH_DIR, "results", "mixing")
const DISTS_DIR = joinpath(MIXING_DIR, "dists")
const STOCH_DIR = joinpath(MIXING_DIR, "dists_stoch")
const SPECTRAL_DIR = joinpath(MIXING_DIR, "dists_spectral")
const S0_CSV = joinpath(MIXING_DIR, "s0_horizon.csv")

# Short problem names for crowded axes.
const SHORT = ["linear_stochastically_switched_" => "lss ", "almost_identity_n3_" => "aid ",
               "package_delivery_" => "pkg ", "impact_robot_2d_ra_S20_I11" => "impact",
               "patrol_robot_" => "patrol ", "car_parking_" => "carpark ", "van_der_pol_" => "vdp ",
               "robot_2d_" => "robot2d ", "_quotient_3_5" => "", "multiObj_robotIMDP" => "multiObj",
               "bas_4d_S5x5x7x7_I4" => "bas4d", "room_temp_3d_S8_I6" => "roomtemp", "storm_tiny" => "tiny",
               "robot_delta0.1" => "robot_delta", "_16_2" => "", r"_I\d+$" => "", "coin2_" => "coin2 ",
               "coin4_" => "coin4 "]
short_name(p) = replace(p, SHORT...)
