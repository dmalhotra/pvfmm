#!/bin/bash

cd ${WORK_DIR}
make ${EXEC} -j
if [ ! -f ${EXEC} ] ; then exit -1; fi;

export RESULT_DIR=${WORK_DIR}/result
mkdir -p ${RESULT_DIR}
#find ${RESULT_DIR} -type f -size 0 -exec rm {} \;

if command -v timeout >/dev/null; then
  export TIMEOUT="timeout";
else
  export TIMEOUT="scripts/.timeout3 -t ";
fi

declare -a     args=();
declare -a    fname=();
for (( k=0; k<${#nodes[@]}; k++ )) ; do
  if [ "${nodes[k]}" == ":" ]; then continue; fi;

  # Set output filename
  case $HOSTNAME in
    *titan*) fname[$k]="host_titan";;
    *stampede*) fname[$k]="host_stampede";;
    *ls4*) fname[$k]="host_lonestar";;
    *ronaldo*) fname[$k]="host_ronaldo";;
    *) fname[$k]="host_${HOSTNAME}";;
  esac
  fname[$k]="${fname[$k]}_nds${nodes[k]}_mpi${mpi_proc[k]}";

  # Set executable options
  for (( i=0; i<${#opt_names[@]}; i++ )) ; do
    if [ "${opt_names[i]}" != "-" ]; then
      eval "opt_val=\${${opt_array[i]}[$k]}";
      args[$k]="${args[k]} -${opt_names[i]} $opt_val";
      fname[$k]="${fname[k]}_${opt_names[i]}$opt_val";
    fi
  done
done

# export arrays
export    nodes_="$(declare -p    nodes)";
export    cores_="$(declare -p    cores)";
export mpi_proc_="$(declare -p mpi_proc)";
export  threads_="$(declare -p  threads)";
export max_time_="$(declare -p max_time)";
export     args_="$(declare -p     args)";
export    fname_="$(declare -p    fname)";

for (( k=0; k<${#nodes[@]}; k++ )) ; do
  if [ "${nodes[k]}" == ":" ] ||
     [ -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[k]}.out ]; then
    continue;
  fi;
  for (( j=0; j<$k; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      continue 2;
    fi
  done;
  TOTAL_TIME=0;
  for (( j=0; j<${#nodes[@]}; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      TOTAL_TIME=$(( ${TOTAL_TIME} + ${max_time[j]} ))
    fi
  done;

  export    NODES=${nodes[k]};    # Number of compute nodes.
  export MPI_PROC=${mpi_proc[k]}; # Number of MPI processes.
  FNAME=${RESULT_DIR}/$(basename ${EXEC})_nds${NODES}_mpi${MPI_PROC}

  #Submit Job
  case $HOSTNAME in
    *titan*) #titan.ccs.ornl.gov (Portable Batch System)
        qsub -l nodes=${NODES} \
             -o ${FNAME}.out -e ${FNAME}.err \
             -l walltime=${TOTAL_TIME} \
             ./scripts/.job.titan
      ;;
    *stampede*) #stampede.tacc.utexas.edu (Slurm Batch)
        if (( ${TOTAL_TIME} > 14400 )); then TOTAL_TIME="14400"; fi
        #if (( ${NODES} > 128 )) ; then continue; fi;
        sbatch -N${NODES} -n${MPI_PROC} \
               -o ${FNAME}.out -e ${FNAME}.err -D ${PWD} \
               --time=00:00:${TOTAL_TIME} \
               ./scripts/.job.stampede
      ;;
    *ls4*) #lonestar.tacc.utexas.edu (Sun Grid Engine)
        qsub -pe $((${MPI_PROC}/${NODES}))way $((${NODES}*${CORES})) \
             -o ${FNAME}.out -e ${FNAME}.err \
             -l h_rt=${TOTAL_TIME} \
             ./scripts/.job.lonestar
      ;;
    *ronaldo*) #ronaldo.ices.utexas.edu (Portable Batch System)
        qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) \
             -o ${FNAME}.out -e ${FNAME}.err \
             -l walltime=${TOTAL_TIME} \
             ./scripts/.job.ronaldo
      ;;
    *) # none of the known machines
      if command -v qsub >/dev/null; then # Portable Batch System
        qsub -l nodes=${NODES}:ppn=$((${MPI_PROC}/${NODES})) \
             -o ${FNAME}.out -e ${FNAME}.err \
             -l walltime=${TOTAL_TIME} \
             ./scripts/.job.qsub
      elif command -v sbatch >/dev/null; then # Slurm Batch
        sbatch -N${NODES} -n${MPI_PROC} \
               -o ${FNAME}.out -e ${FNAME}.err -D ${PWD} \
               --time=${TOTAL_TIME} \
               ./scripts/.job.sbatch
      else # Shell
        ./scripts/.job.sh
      fi
  esac

  #Exit on error.
  if (( $? != 0 )) ; then continue; fi;
  for (( j=0; j<${#nodes[@]}; j++ )) ; do
    if [ "${nodes[k]}" == "${nodes[j]}" ] &&
       [ "${mpi_proc[k]}" == "${mpi_proc[j]}" ] &&
       [ ! -f ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out ]; then
      touch ${RESULT_DIR}/$(basename ${EXEC})_${fname[j]}.out;
    fi
  done;
done;

# Display results
source ./scripts/.results.sh
