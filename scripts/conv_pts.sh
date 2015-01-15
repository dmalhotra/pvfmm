#!/bin/bash

CORES=16;
export EXEC=examples/bin/fmm_pts

# List arrays and corresponding executable option prefix
declare -a opt_array=(nodes cores mpi_proc threads ker n_pts m_pts b_len dist m depth sin_pr max_time);
declare -a opt_names=(    -     -        -     omp ker     N     M     b dist m     d     sp        -);
for (( i=0; i<${#opt_names[@]}; i++ )) ; do # Declare arrays
  eval "declare -a ${opt_array[$i]}=()";
done


###################################################################################################
#                    Convergence Laplace kernel, 1M points, uniform distribution                  #
###################################################################################################
nodes+=(            1         1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              1         1         1         1         1         1         1         1         1 :)
n_pts+=(         1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6 :)
m_pts+=(          300       300       300       300       300       300       300       300       300 :)
b_len+=(       0.8125    0.8125      0.75      0.75     0.625       1.0       1.0     0.875      0.75 :)
dist+=(             0         0         0         0         0         0         0         0         0 :)
m+=(                2         4         6         6         8        10        12        14        16 :)
depth+=(            6         6         6         6         6         5         5         5         5 :)
sin_pr+=(           1         1         1         0         0         0         0         0         0 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)


###################################################################################################
#                  Convergence Laplace kernel, 1M points, non-uniform distribution (sphere)       #
###################################################################################################
nodes+=(            1         1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              1         1         1         1         1         1         1         1         1 :)
n_pts+=(         1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6 :)
m_pts+=(          180       190       200       180       200       700       750       750      1500 :)
b_len+=(            1         1         1         1         1         1         1         1         1 :)
dist+=(             1         1         1         1         1         1         1         1         1 :)
m+=(                2         4         6         6         8        10        12        14        16 :)
depth+=(           15        15        15        15        15        15        15        15        15 :)
sin_pr+=(           1         1         1         0         0         0         0         0         0 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)


###################################################################################################
#                  Convergence Laplace kernel, 1M points, non-uniform distribution (ellipse)      #
###################################################################################################
nodes+=(            1         1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              1         1         1         1         1         1         1         1         1 :)
n_pts+=(         1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6 :)
m_pts+=(          190       270       370       260       370       430       640      1000      1400 :)
b_len+=(            1         1         1         1         1         1         1         1         1 :)
dist+=(             2         2         2         2         2         2         2         2         2 :)
m+=(                2         4         6         6         8        10        12        14        16 :)
depth+=(           15        15        15        15        15        15        15        15        15 :)
sin_pr+=(           1         1         1         0         0         0         0         0         0 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)


###################################################################################################
#    Convergence Helmholtz kernel (wave-number=10), 1M points, uniform distribution (ellipse)     #
###################################################################################################
nodes+=(            1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              3         3         3         3         3         3         3         3 :)
n_pts+=(         1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6      1e+6 :)
m_pts+=(          370       430       640       370       430       640       640       640 :)
m_pts+=(          150       150       150       200       300       400       400       400 :)
b_len+=(            1         1         1         1         1         1         1         1 :)
dist+=(             0         0         0         0         0         0         0         0 :)
m+=(                8        10        10        12        14        16        18        20 :)
depth+=(           15        15        15        15        15        15        15        15 :)
sin_pr+=(           1         1         0         0         0         0         0         0 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)


RESULT_HEADER=" Script: $0          Convergence with multipole order for 1M points"

declare -a RESULT_FIELDS=()
RESULT_FIELDS+=("FMM Kernel name"                    "kernel" )
RESULT_FIELDS+=("Point distribution"                 "dist"   )
RESULT_FIELDS+=("Number of Leaf Nodes"               "Noct"   )
RESULT_FIELDS+=("Tree Depth"                         "d"      )
RESULT_FIELDS+=("Maximum points per octant"          "M"      )
RESULT_FIELDS+=("Order of multipole expansions"      "m"      )
RESULT_FIELDS+=("|"                                  "|"      )
RESULT_FIELDS+=("Maximum Relative Error \[Output\]"  "Linf(e)")

declare -a PROF_FIELDS=()
PROF_FIELDS+=("InitTree"    )
PROF_FIELDS+=("SetupFMM"    )
PROF_FIELDS+=("RunFMM"      )
#PROF_FIELDS+=("UpwardPass"  )
#PROF_FIELDS+=("ReduceBcast" )
#PROF_FIELDS+=("DownwardPass")

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

