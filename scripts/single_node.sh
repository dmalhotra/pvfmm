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
#                  UNIFORM OCTREE, LAPLACE KERNEL, OMP SCALABILITY RESULTS                        #
###################################################################################################

# m=10, q=14, octants=512, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         1         1         1         1         1 :)
n_pts+=(    $((8**3)) $((8**3)) $((8**3)) $((8**3)) $((8**3)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(       320       160        80        40        20 :)

# m=10, q=14, octants=4096, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         1         1         1         1         1 :)
n_pts+=(    $((8**4)) $((8**4)) $((8**4)) $((8**4)) $((8**4)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      2560      1280       640       320       160 :)

# m=10, q=14, octants=32768, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1 : :)
threads+=(          1         2         4         8        16 : :)
testcase+=(         1         1         1         1         1 : :)
n_pts+=(    $((8**5)) $((8**5)) $((8**5)) $((8**5)) $((8**5)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(      2560      1280       640       320       160 : :)



###################################################################################################
#                  NON-UNIFORM OCTREE, LAPLACE KERNEL, OMP SCALABILITY RESULTS                    #
###################################################################################################

# m=10, q=14, octants=939, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         1         1         1         1         1 :)
n_pts+=(           32        32        32        32        32 :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             0         0         0         0         0 :)
adap+=(             0         0         0         0         0 :)
max_time+=(       320       160        80        40        20 :)

# m=10, q=14, octants=5685, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         1         1         1         1         1 :)
n_pts+=(          256       256       256       256       256 :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             0         0         0         0         0 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      2560      1280       640       320       160 :)

# m=10, q=14, octants=37416, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1 : :)
threads+=(          1         2         4         8        16 : :)
testcase+=(         1         1         1         1         1 : :)
n_pts+=(         2048      2048      2048      2048      2048 : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             0         0         0         0         0 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(      2560      1280       640       320       160 : :)


###################################################################################################

###################################################################################################
#                   UNIFORM OCTREE, STOKES KERNEL, OMP SCALABILITY RESULTS                        #
###################################################################################################

# m=10, q=14, octants=512, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(    $((8**3)) $((8**3)) $((8**3)) $((8**3)) $((8**3)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      3080      1540       720       360       180 :)

# m=10, q=14, octants=4096, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(    $((8**4)) $((8**4)) $((8**4)) $((8**4)) $((8**4)) :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             1         1         1         1         1 :)
adap+=(             0         0         0         0         0 :)
max_time+=(     23040     11520      5760      2880      1440 :)

# m=10, q=14, octants=32768, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1         1         1 : :)
threads+=(          1         2         4         8        16 : :)
testcase+=(         3         3         3         3         3 : :)
n_pts+=(    $((8**5)) $((8**5)) $((8**5)) $((8**5)) $((8**5)) : :)
m_pts+=(            1         1         1         1         1 : :)
m+=(               10        10        10        10        10 : :)
q+=(               14        14        14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15        15        15 : :)
unif+=(             1         1         1         1         1 : :)
adap+=(             0         0         0         0         0 : :)
max_time+=(    184320     92160     46080     23040     11520 : :)



###################################################################################################
#                   NON-UNIFORM OCTREE, STOKES KERNEL, OMP SCALABILITY RESULTS                    #
###################################################################################################

# m=10, q=14, octants=939, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(           32        32        32        32        32 :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             0         0         0         0         0 :)
adap+=(             0         0         0         0         0 :)
max_time+=(      3080      1540       720       360       180 :)

# m=10, q=14, octants=5685, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1         1         1 :)
threads+=(          1         2         4         8        16 :)
testcase+=(         3         3         3         3         3 :)
n_pts+=(          256       256       256       256       256 :)
m_pts+=(            1         1         1         1         1 :)
m+=(               10        10        10        10        10 :)
q+=(               14        14        14        14        14 :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 :)
depth+=(           15        15        15        15        15 :)
unif+=(             0         0         0         0         0 :)
adap+=(             0         0         0         0         0 :)
max_time+=(     23040     11520      5760      2880      1440 :)

# m=10, q=14, octants=37416, threads={1,2,4,8,16}
nodes+=(            1         1         1         1         1 : : : :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} : : : :)
mpi_proc+=(         1         1         1         1         1 : : : :)
threads+=(          1         2         4         8        16 : : : :)
testcase+=(         3         3         3         3         3 : : : :)
n_pts+=(         2048      2048      2048      2048      2048 : : : :)
m_pts+=(            1         1         1         1         1 : : : :)
m+=(               10        10        10        10        10 : : : :)
q+=(               14        14        14        14        14 : : : :)
tol+=(           1e-0      1e-0      1e-0      1e-0      1e-0 : : : :)
depth+=(           15        15        15        15        15 : : : :)
unif+=(             0         0         0         0         0 : : : :)
adap+=(             0         0         0         0         0 : : : :)
max_time+=(    184320     92160     46080     23040     11520 : : : :)


###################################################################################################

###################################################################################################
#                  UNIFORM OCTREE, LAPLACE KERNEL, SINGLE NODE PERFORMANCE                        #
###################################################################################################

# m=6, q=9, octants={512,4096,3276}
nodes+=(            1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1 :)
n_pts+=(    $((8**3)) $((8**4)) $((8**5)) :)
m_pts+=(            1         1         1 :)
m+=(                6         6         6 :)
q+=(                9         9         9 :)
tol+=(           1e-0      1e-0      1e-0 :)
depth+=(           15        15        15 :)
unif+=(             1         1         1 :)
adap+=(             0         0         0 :)
max_time+=(        20       160      1280 :)

# m=10, q=14, octants={512,4096,3276}
nodes+=(            1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         1         1         1 : :)
n_pts+=(    $((8**3)) $((8**4)) $((8**5)) : :)
m_pts+=(            1         1         1 : :)
m+=(               10        10        10 : :)
q+=(               14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15 : :)
unif+=(             1         1         1 : :)
adap+=(             0         0         0 : :)
max_time+=(        20       160      1280 : :)



###################################################################################################
#                  NON-UNIFORM OCTREE, LAPLACE KERNEL, SINGLE NODE PERFORMANCE                    #
###################################################################################################

# m=6, q=9, octants={512,4096,3276}
nodes+=(            1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         1         1         1 :)
n_pts+=(           32       256      2048 :)
m_pts+=(            1         1         1 :)
m+=(                6         6         6 :)
q+=(                9         9         9 :)
tol+=(           1e-0      1e-0      1e-0 :)
depth+=(           15        15        15 :)
unif+=(             0         0         0 :)
adap+=(             0         0         0 :)
max_time+=(        20       160      1280 :)

# m=10, q=14, octants={512,4096,3276}
nodes+=(            1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         1         1         1 : :)
n_pts+=(           32       256      2048 : :)
m_pts+=(            1         1         1 : :)
m+=(               10        10        10 : :)
q+=(               14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15 : :)
unif+=(             0         0         0 : :)
adap+=(             0         0         0 : :)
max_time+=(        20       160      1280 : :)

###################################################################################################
#                   UNIFORM OCTREE, STOKES KERNEL, SINGLE NODE PERFORMANCE                        #
###################################################################################################

# m=6, q=9, octants={512,4096,3276}
nodes+=(            1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         3         3         3 :)
n_pts+=(    $((8**3)) $((8**4)) $((8**5)) :)
m_pts+=(            1         1         1 :)
m+=(                6         6         6 :)
q+=(                9         9         9 :)
tol+=(           1e-0      1e-0      1e-0 :)
depth+=(           15        15        15 :)
unif+=(             1         1         1 :)
adap+=(             0         0         0 :)
max_time+=(       180      1440     11520 :)

# m=10, q=14, octants={512,4096,3276}
nodes+=(            1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         3         3         3 : :)
n_pts+=(    $((8**3)) $((8**4)) $((8**5)) : :)
m_pts+=(            1         1         1 : :)
m+=(               10        10        10 : :)
q+=(               14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15 : :)
unif+=(             1         1         1 : :)
adap+=(             0         0         0 : :)
max_time+=(       180      1440     11520 : :)



###################################################################################################
#                   NON-UNIFORM OCTREE, STOKES KERNEL, SINGLE NODE PERFORMANCE                    #
###################################################################################################

# m=6, q=9, octants={512,4096,3276}
nodes+=(            1         1         1 :)
cores+=(     ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         1         1         1 :)
threads+=(   ${CORES}  ${CORES}  ${CORES} :)
testcase+=(         3         3         3 :)
n_pts+=(           32       256      2048 :)
m_pts+=(            1         1         1 :)
m+=(                6         6         6 :)
q+=(                9         9         9 :)
tol+=(           1e-0      1e-0      1e-0 :)
depth+=(           15        15        15 :)
unif+=(             0         0         0 :)
adap+=(             0         0         0 :)
max_time+=(       180      1440     11520 :)

# m=10, q=14, octants={512,4096,3276}
nodes+=(            1         1         1 : :)
cores+=(     ${CORES}  ${CORES}  ${CORES} : :)
mpi_proc+=(         1         1         1 : :)
threads+=(   ${CORES}  ${CORES}  ${CORES} : :)
testcase+=(         3         3         3 : :)
n_pts+=(           32       256      2048 : :)
m_pts+=(            1         1         1 : :)
m+=(               10        10        10 : :)
q+=(               14        14        14 : :)
tol+=(           1e-0      1e-0      1e-0 : :)
depth+=(           15        15        15 : :)
unif+=(             0         0         0 : :)
adap+=(             0         0         0 : :)
max_time+=(       180      1440     11520 : :)



###################################################################################################

# Export arrays
export    nodes_="$(declare -p    nodes)";
export    cores_="$(declare -p    cores)";
export mpi_proc_="$(declare -p mpi_proc)";
export  threads_="$(declare -p  threads)";
export testcase_="$(declare -p testcase)";
export    n_pts_="$(declare -p    n_pts)";
export    m_pts_="$(declare -p    m_pts)";
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

