#!/bin/bash

CORES=12;
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

nodes+=(            1         1         1         1)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES})
mpi_proc+=(         1         1         1         1)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES})
testcase+=(         1         1         1         1)
n_pts+=(    $((8**1)) $((8**1)) $((8**1)) $((8**1)))
m+=(               10        10        10        10)
q+=(               14        14        14        14)
tol+=(           1e-4      1e-5      1e-6      1e-7)
depth+=(           15        15        15        15)
unif+=(             0         0         0         0)
adap+=(             1         1         1         1)
max_time+=(   1000000   1000000   1000000   1000000)

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

