#!/bin/bash

CORES=16;
export EXEC=examples/bin/fmm_cheb

# List arrays and corresponding executable option prefix
declare -a opt_array=(nodes cores mpi_proc threads testcase n_pts m_pts m q tol depth unif adap max_time);
declare -a opt_names=(    -     -        -     omp     test     N     M m q tol     d unif adap        -);
for (( i=0; i<${#opt_names[@]}; i++ )) ; do # Declare arrays
  eval "declare -a ${opt_array[$i]}=()";
done


###################################################################################################
#                 NON-UNIFORM OCTREE, LAPLACE KERNEL, STRONG SCALABILITY                          #
###################################################################################################

# m=10, q=14, octants=
nodes+=(            1         8        64       512      4096     32768 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         4        32       256      2048     16384     16384 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1         1         1         1 :)
n_pts+=(    $((8**7)) $((8**7)) $((8**7)) $((8**7)) $((8**7)) $((8**7)) :)
m_pts+=(            1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10 :)
q+=(               14        14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15        15 :)
unif+=(             0         0         0         0         0         0 :)
adap+=(             0         0         0         0         0         0 :)
max_time+=(       800       800       800       800       800       800 :)


###################################################################################################
#                 NON-UNIFORM OCTREE, STOKES KERNEL, STRONG SCALABILITY                           #
###################################################################################################

# m=10, q=14, octants=
nodes+=(            1         8        64       512      4096     32768 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         8        64       512      4096     32768 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         3         3         3         3         3         3 :)
n_pts+=(    $((8**7)) $((8**7)) $((8**7)) $((8**7)) $((8**7)) $((8**7)) :)
m_pts+=(            1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10 :)
q+=(               14        14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15        15 :)
unif+=(             0         0         0         0         0         0 :)
adap+=(             0         0         0         0         0         0 :)
max_time+=(      2400      2400      2400      2400      2400      2400 :)


###################################################################################################
#                  UNIFORM OCTREE, HELMHOLTZ KERNEL, STRONG SCALABILITY                           #
###################################################################################################

# m=10, q=14, octants=262144
nodes+=(            4         8        16        32        64        128        256        512       1024 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
mpi_proc+=(         4         8        16        32        64        128        256        512       1024 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
testcase+=(         5         5         5         5         5          5          5          5          5 :)
n_pts+=(    $((8**6)) $((8**6)) $((8**6)) $((8**6)) $((8**6))  $((8**6))  $((8**6))  $((8**6))  $((8**6)) :)
m_pts+=(            1         1         1         1         1          1          1          1          1 :)
m+=(               10        10        10        10        10         10         10         10         10 :)
q+=(               14        14        14        14        14         14         14         14         14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0       1e-0       1e-0       1e-0       1e-0 :)
depth+=(           15        15        15        15        15         15         15         15         15 :)
unif+=(             1         1         1         1         1          1          1          1          1 :)
adap+=(             0         0         0         0         0          0          0          0          0 :)
max_time+=(      2400      2400      2400      2400      2400       2400       2400       2400       2400 :)


###################################################################################################

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

