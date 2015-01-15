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
#                  SMOOTH INPUT, LAPLACE KERNEL, SINGLE NODE CONVERGENCE RESULTS                  #
###################################################################################################

# m=10, q=9, tol=1e-{0,1,2,3,4,5,6}
nodes+=(            1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1         1         1         1         1 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10        10 :)
q+=(                9         9         9         9         9         9         9 :)
tol+=(           1e-0      1e-1      1e-2      1e-3      1e-4      1e-5      1e-6 :)
depth+=(           15        15        15        15        15        15        15 :)
unif+=(             1         1         1         1         1         1         1 :)
adap+=(             1         1         1         1         1         1         1 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)

# m=10, q=14, tol=1e-{0,1,2,3,4,5,6,7}
nodes+=(            1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1         1         1         1         1         1 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10        10        10 :)
q+=(               14        14        14        14        14        14        14        14 :)
tol+=(           1e-0      1e-1      1e-2      1e-3      1e-4      1e-5      1e-6      1e-7 :)
depth+=(           15        15        15        15        15        15        15        15 :)
unif+=(             1         1         1         1         1         1         1         1 :)
adap+=(             1         1         1         1         1         1         1         1 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)

# m={4,6,8,10,12,14,16,18}, q=9, tol=1e-9
nodes+=(            1         1         1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1         1         1         1         1         1 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1         1         1         1 :)
m+=(                4         6         8        10        12        14        16        18 :)
q+=(                9         9         9         9         9         9         9         9 :)
tol+=(           1e-9      1e-9      1e-9      1e-9      1e-9      1e-9      1e-9      1e-9 :)
depth+=(           15        15        15        15        15        15        15        15 :)
unif+=(             1         1         1         1         1         1         1         1 :)
adap+=(             1         1         1         1         1         1         1         1 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 :)

# m={4,6,8,10,12,14,16,18}, q=14, tol=1e-9
nodes+=(            1         1         1         1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         1         1         1         1         1         1         1         1 : :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) : :)
m_pts+=(            1         1         1         1         1         1         1         1 : :)
m+=(                4         6         8        10        12        14        16        18 : :)
q+=(               14        14        14        14        14        14        14        14 : :)
tol+=(           1e-9      1e-9      1e-9      1e-9      1e-9      1e-9      1e-9      1e-9 : :)
depth+=(           15        15        15        15        15        15        15        15 : :)
unif+=(             1         1         1         1         1         1         1         1 : :)
adap+=(             1         1         1         1         1         1         1         1 : :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000   1000000   1000000 : :)


###################################################################################################
#                  DISCONTINUOUS SPHERE, LAPLACE KERNEL, SINGLE NODE CONVERGENCE RESULTS          #
###################################################################################################

# m=10, q=9, depth={4,5,6,7,8,9}
nodes+=(            1         1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         2         2         2         2         2         2 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10 :)
q+=(                9         9         9         9         9         9 :)
tol+=(           1e-6      1e-6      1e-6      1e-6      1e-6      1e-6 :)
depth+=(            4         5         6         7         8         9 :)
unif+=(             1         1         1         1         1         1 :)
adap+=(             1         1         1         1         1         1 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000 :)

# m=10, q=14, depth={4,5,6,7,8,9}
nodes+=(            1         1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         2         2         2         2         2         2 : :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) : :)
m_pts+=(            1         1         1         1         1         1 : :)
m+=(               10        10        10        10        10        10 : :)
q+=(               14        14        14        14        14        14 : :)
tol+=(           1e-6      1e-6      1e-6      1e-6      1e-6      1e-6 : :)
depth+=(            4         5         6         7         8         9 : :)
unif+=(             1         1         1         1         1         1 : :)
adap+=(             1         1         1         1         1         1 : :)
max_time+=(   1000000   1000000   1000000   1000000   1000000   1000000 : :)


###################################################################################################
#                  SMOOTH INPUT, STOKES KERNEL, SINGLE NODE CONVERGENCE RESULTS                   #
###################################################################################################

# m={4,6,8,10,12}, q=9, tol=1e-4
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(                4         6         8        10        12 :)
q+=(                9         9         9         9         9 :)
tol+=(           1e-4      1e-4      1e-4      1e-4      1e-4 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             1         1         1         1         1 :)
max_time+=(   1000000   1000000   1000000   1000000   1000000 :)

# m={4,6,8,10,12}, q=14, tol=1e-4
nodes+=(            1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         3         3         3         3         3 : :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(                4         6         8        10        12 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-4      1e-4      1e-4      1e-4      1e-4 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             1         1         1         1         1 : :)
max_time+=(   1000000   1000000   1000000   1000000   1000000 : :)


###################################################################################################
#                SMOOTH INPUT, HELMHOLTZ KERNEL, SINGLE NODE CONVERGENCE RESULTS                  #
###################################################################################################

# m={4,6,8,10,12}, q=9, tol=1e-5
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         5         5         5         5         5 :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(                4         6         8        10        12 :)
q+=(                9         9         9         9         9 :)
tol+=(           1e-5      1e-5      1e-5      1e-5      1e-5 :)
depth+=(           10        10        10        10        10 :)
unif+=(             1         1         1         1         1 :)
adap+=(             1         1         1         1         1 :)
max_time+=(  10000000  10000000  10000000  10000000  10000000 :)

# m={4,6,8,10,12}, q=14, tol=1e-5
nodes+=(            1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         5         5         5         5         5 : :)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)) $((8**1)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(                4         6         8        10        12 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-5      1e-5      1e-5      1e-5      1e-5 : :)
depth+=(           10        10        10        10        10 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             1         1         1         1         1 : :)
max_time+=(  10000000  10000000  10000000  10000000  10000000 : :)


###################################################################################################

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

