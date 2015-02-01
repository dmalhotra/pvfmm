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
#          Strong Scaling Laplace kernel, 100M points, uniform distribution                       #
###################################################################################################
nodes+=(            2         4         8        16        32        64       128       256       512      1024      1024 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         2         4         8        16        32        64       128       256       512      1024      1024 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              1         1         1         1         1         1         1         1         1         1         1 :)
n_pts+=(         1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8 :)
m_pts+=(          450       450       450       450       450       450       450       450       450       450       450 :)
b_len+=(         0.75      0.75      0.75      0.75      0.75      0.75      0.75      0.75      0.75      0.75         1 :)
dist+=(             0         0         0         0         0         0         0         0         0         0         0 :)
m+=(                6         6         6         6         6         6         6         6         6         6         6 :)
depth+=(           15        15        15        15        15        15        15        15        15        15        15 :)
sin_pr+=(           0         0         0         0         0         0         0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200      1200      1200      1200      1200      1200      1200 :)


###################################################################################################
#          Strong Scaling Laplace kernel, 100M points, non-uniform distribution (ellipse)         #
###################################################################################################
nodes+=(            2         4         8        16        32        64       128       256       512      1024 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         2         4         8        16        32        64       128       256       512      1024 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              1         1         1         1         1         1         1         1         1         1 :)
n_pts+=(         1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8 :)
m_pts+=(          450       450       450       450       450       450       450       450       450       450 :)
b_len+=(            1         1         1         1         1         1         1         1         1         1 :)
dist+=(             2         2         2         2         2         2         2         2         2         2 :)
m+=(                6         6         6         6         6         6         6         6         6         6 :)
depth+=(           28        28        28        28        28        28        28        28        28        28 :)
sin_pr+=(           0         0         0         0         0         0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200      1200      1200      1200      1200      1200 :)




###################################################################################################
# Strong Scaling Helmholtz kernel (wave-number=10), 100M points, uniform distribution             #
###################################################################################################
nodes+=(            4         8        16        32        64       128       256       512      1024 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         4         8        16        32        64       128       256       512      1024 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              4         4         4         4         4         4         4         4         4 :)
n_pts+=(         1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8 :)
m_pts+=(          400       400       400       400       400       400       400       400       400 :)
b_len+=(         0.75      0.75      0.75      0.75      0.75      0.75      0.75      0.75      0.75 :)
dist+=(             0         0         0         0         0         0         0         0         0 :)
m+=(               10        10        10        10        10        10        10        10        10 :)
depth+=(           15        15        15        15        15        15        15        15        15 :)
sin_pr+=(           0         0         0         0         0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200      1200      1200      1200      1200 :)


###################################################################################################
# Strong Scaling Helmholtz kernel (wave-number=10), 100M points, non-uniform distribution (sphere)#
###################################################################################################
nodes+=(            8        16        32        64       128       256       512      1024 :)
cores+=(     ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
mpi_proc+=(         8        16        32        64       128       256       512      1024 :)
threads+=(   ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES}  ${CORES} :)
ker+=(              4         4         4         4         4         4         4         4 :)
n_pts+=(         1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8      1e+8 :)
m_pts+=(          400       400       400       400       400       400       400       400 :)
b_len+=(            1         1         1         1         1         1         1         1 :)
dist+=(             1         1         1         1         1         1         1         1 :)
m+=(               10        10        10        10        10        10        10        10 :)
depth+=(           15        15        15        15        15        15        15        15 :)
sin_pr+=(           0         0         0         0         0         0         0         0 :)
max_time+=(      1200      1200      1200      1200      1200      1200      1200      1200 :)




RESULT_HEADER=" Script: $0      Strong scaling with 100M points"

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
#PROF_FIELDS+=("InitTree"    )
#PROF_FIELDS+=("SetupFMM"    )
#PROF_FIELDS+=("RunFMM"      )
##PROF_FIELDS+=("UpwardPass"  )
##PROF_FIELDS+=("ReduceBcast" )
##PROF_FIELDS+=("DownwardPass")

PROF_FIELDS+=("TotalTime"   )
PROF_FIELDS+=("InitTree"    )
PROF_FIELDS+=("InitFMM_Tree")
PROF_FIELDS+=("SetupFMM"    )
PROF_FIELDS+=("RunFMM"      )
PROF_FIELDS+=("Scatter"     )

WORK_DIR=$(dirname ${PWD}/$0)/..
TERM_WIDTH=$(stty size | cut -d ' ' -f 2)
source ${WORK_DIR}/scripts/.submit_jobs.sh | cut -b -${TERM_WIDTH}

