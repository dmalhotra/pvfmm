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

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

