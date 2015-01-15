#!/bin/bash

CORES=16;
export EXEC=examples/bin/fmm_cheb

# List arrays and corresponding executable option prefix
declare -a opt_array=(nodes cores mpi_proc threads testcase n_pts m_pts m q tol depth unif adap max_time);
declare -a opt_names=(    -     -        -     omp     test     N     M m q tol     d unif adap        -);
for (( i=0; i<${#opt_names[@]}; i++ )) ; do # Declare arrays
  eval "declare -a ${opt_array[$i]}=()";
done

# Set run parameters
nodes+=(            1         1         1         1 ) # Number of compute nodes
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES} ) # Number of CPU cores / node
mpi_proc+=(         1         1         1         1 ) # Number of MPI processes
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES} ) # Number of OpenMP threads / MPI process
testcase+=(         1         1         1         1 ) # test case: 1) Laplace (smooth) 2) Laplace (discontinuous) ...
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) ) # Total number of points for tree construction
m_pts+=(            1         1         1         1 ) # Maximum number of points per octant
m+=(               10        10        10        10 ) # Multipole order
q+=(               14        14        14        14 ) # Chebyshev order
tol+=(           1e-4      1e-5      1e-6      1e-7 ) # Refinement tolerance
depth+=(           15        15        15        15 ) # Octree maximum depth
unif+=(             0         0         0         0 ) # Uniform point distribution
adap+=(             1         1         1         1 ) # Adaptive refinement
max_time+=(   1000000   1000000   1000000   1000000 ) # Maximum run time

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

