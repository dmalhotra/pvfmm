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
#                      UNIFORM OCTREE, LAPLACE KERNEL, WEAK SCALABILITY                           #
###################################################################################################

# m=10, q=14, octants=64k oct/node
nodes+=(            4        32       256      2048      16384 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
mpi_proc+=(         4        32       256      2048      16384 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
testcase+=(         1         1         1         1          1 :)
n_pts+=(    $((8**6)) $((8**7)) $((8**8)) $((8**9)) $((8**10)) :)
m_pts+=(            1         1         1         1          1 :)
m+=(               10        10        10        10         10 :)
q+=(               14        14        14        14         14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0       1e-0 :)
depth+=(           15        15        15        15         15 :)
unif+=(             1         1         1         1          1 :)
adap+=(             0         0         0         0          0 :)
max_time+=(       800       800       800       800        800 :)

# m=10, q=14, octants=32k oct/node
nodes+=(            1         8        64       512      4096      32768 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
mpi_proc+=(         1         8        64       512      4096      32768 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
testcase+=(         1         1         1         1         1          1 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) $((8**10)) :)
m_pts+=(            1         1         1         1         1          1 :)
m+=(               10        10        10        10        10         10 :)
q+=(               14        14        14        14        14         14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0       1e-0 :)
depth+=(           15        15        15        15        15         15 :)
unif+=(             1         1         1         1         1          1 :)
adap+=(             0         0         0         0         0          0 :)
max_time+=(       400       400       400       400       400        400 :)

# m=10, q=14, octants=16k oct/node
nodes+=(            2        16       128      1024      8192 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         2        16       128      1024      8192 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1         1         1 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(       200       200       200       200       200 :)

# m=10, q=14, octants=8k oct/node
nodes+=(            4        32       256      2048     16384 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         4        32       256      2048     16384 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         1         1         1         1         1 : :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(       100       100       100       100       100 : :)



###################################################################################################
#                    NON-UNIFORM OCTREE, LAPLACE KERNEL, WEAK SCALABILITY                         #
###################################################################################################

# m=10, q=13, octants=16k oct/node
nodes+=(             1          4         16         64        256       1024       4096 :)
cores+=(      ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
mpi_proc+=(          1          4         16         64        256       1024       4096 :)
threads+=(    ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
testcase+=(          1          1          1          1          1          1          1 :)
n_pts+=(    $((2**20)) $((2**22)) $((2**24)) $((2**26)) $((2**28)) $((2**30)) $((2**32)) :)
m_pts+=(           500        500        500        500        500        500        500 :)
m+=(                10         10         10         10         10         10         10 :)
q+=(                13         13         13         13         13         13         13 :)
tol+=(            1e-0       1e-0       1e-0       1e-0       1e-0       1e-0       1e-0 :)
depth+=(            30         30         30         30         30         30         30 :)
unif+=(              0          0          0          0          0          0          0 :)
adap+=(              0          0          0          0          0          0          0 :)
max_time+=(        500        500        500        500        500        500        500 :)

# m=10, q=13, octants=32k oct/node
nodes+=(             1          4         16         64        256       1024       4096 :)
cores+=(      ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
mpi_proc+=(          1          4         16         64        256       1024       4096 :)
threads+=(    ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES}   ${CORES} :)
testcase+=(          1          1          1          1          1          1          1 :)
n_pts+=(    $((2**21)) $((2**23)) $((2**25)) $((2**27)) $((2**29)) $((2**31)) $((2**33)) :)
m_pts+=(           500        500        500        500        500        500        500 :)
m+=(                10         10         10         10         10         10         10 :)
q+=(                13         13         13         13         13         13         13 :)
tol+=(            1e-0       1e-0       1e-0       1e-0       1e-0       1e-0       1e-0 :)
depth+=(            30         30         30         30         30         30         30 :)
unif+=(              0          0          0          0          0          0          0 :)
adap+=(              0          0          0          0          0          0          0 :)
max_time+=(        500        500        500        500        500        500        500 :)



###################################################################################################
#                      UNIFORM OCTREE, STOKES KERNEL, WEAK SCALABILITY                            #
###################################################################################################

# m=10, q=14, octants=32k oct/node
nodes+=(            1         8        64       512      4096      32768 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
mpi_proc+=(         1         8        64       512      4096      32768 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
testcase+=(         3         3         3         3         3          3 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) $((8**10)) :)
m_pts+=(            1         1         1         1         1          1 :)
m+=(               10        10        10        10        10         10 :)
q+=(               14        14        14        14        14         14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0       1e-0 :)
depth+=(           15        15        15        15        15         15 :)
unif+=(             1         1         1         1         1          1 :)
adap+=(             0         0         0         0         0          0 :)
max_time+=(      2400      2400      2400      2400      2400       2400 :)

# m=10, q=14, octants=16k oct/node
nodes+=(            2        16       128      1024      8192 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         2        16       128      1024      8192 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200 :)

# m=10, q=14, octants=8k oct/node
nodes+=(            4        32       256      2048     16384 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         4        32       256      2048     16384 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         3         3         3         3         3 : :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(       600       600       600       600       600 : :)



###################################################################################################
#                      UNIFORM OCTREE, HELMHOLTZ KERNEL, WEAK SCALABILITY                         #
###################################################################################################

# m=10, q=14, octants=32k oct/node
nodes+=(            1         8        64       512      4096      32768 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
mpi_proc+=(         1         8        64       512      4096      32768 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}   ${CORES} :)
testcase+=(         5         5         5         5         5          5 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) $((8**10)) :)
m_pts+=(            1         1         1         1         1          1 :)
m+=(               10        10        10        10        10         10 :)
q+=(               14        14        14        14        14         14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0       1e-0 :)
depth+=(           15        15        15        15        15         15 :)
unif+=(             1         1         1         1         1          1 :)
adap+=(             0         0         0         0         0          0 :)
max_time+=(      2400      2400      2400      2400      2400       2400 :)

# m=10, q=14, octants=16k oct/node
nodes+=(            2        16       128      1024      8192 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         2        16       128      1024      8192 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         5         5         5         5         5 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200 :)

# m=10, q=14, octants=8k oct/node
nodes+=(            4        32       256      2048     16384 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         4        32       256      2048     16384 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         5         5         5         5         5 :)
n_pts+=(    $((8**5)) $((8**6)) $((8**7)) $((8**8)) $((8**9)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200 :)


###################################################################################################

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

