#!/bin/bash

CORES=16;
export EXEC=examples/bin/fmm_cheb

# Set run parameters
declare -a    nodes=();
declare -a    cores=();
declare -a mpi_proc=();
declare -a  threads=();
declare -a testcase=();
declare -a    n_pts=();
declare -a        m=();
declare -a        q=();
declare -a      tol=();
declare -a    depth=();
declare -a     unif=();
declare -a     adap=();
declare -a max_time=();


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
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(       100       100       100       100       100 : :)



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
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(       600       600       600       600       600 : :)



###################################################################################################

# Export arrays
export    nodes_="$(declare -p    nodes)";
export    cores_="$(declare -p    cores)";
export mpi_proc_="$(declare -p mpi_proc)";
export  threads_="$(declare -p  threads)";
export testcase_="$(declare -p testcase)";
export    n_pts_="$(declare -p    n_pts)";
export        m_="$(declare -p        m)";
export        q_="$(declare -p        q)";
export      tol_="$(declare -p      tol)";
export    depth_="$(declare -p    depth)";
export     unif_="$(declare -p     unif)";
export     adap_="$(declare -p     adap)";
export max_time_="$(declare -p max_time)";

export RESULT_FNAME=$(basename ${0%.*}).out;
export WORK_DIR=$(dirname ${PWD}/$0)/..
cd ${WORK_DIR}

TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
./scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

