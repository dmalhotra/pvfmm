#!/bin/bash

if [ -z "$RESULT_FIELDS" ]; then
  declare -a RESULT_FIELDS=()
  RESULT_FIELDS+=("FMM Kernel name"                    "kernel" )
  RESULT_FIELDS+=("Maximum Tree Depth"                 "max_d"  )
  RESULT_FIELDS+=("Order of multipole expansions"      "m"      )
  RESULT_FIELDS+=("|"                                  "|"      )
  RESULT_FIELDS+=("Maximum Relative Error \[Output\]"  "Linf(e)")
fi

if [ -z "$PROF_FIELDS" ]; then
  declare -a PROF_FIELDS=()
  PROF_FIELDS+=("InitTree"    )
  PROF_FIELDS+=("SetupFMM"    )
  PROF_FIELDS+=("RunFMM"      )
  PROF_FIELDS+=("UpwardPass"  )
  PROF_FIELDS+=("ReduceBcast" )
  PROF_FIELDS+=("DownwardPass")
fi

declare -a DEFAULT_FIELDS=()
DEFAULT_FIELDS+=("nodes"   )
DEFAULT_FIELDS+=("mpi_proc")
DEFAULT_FIELDS+=("threads" )

DEFAULT_FIELDS+=("|" )
RESULT_FIELDS+=("|" "|")
PROF_FIELDS+=("|" )

RESULT_FNAME=${RESULT_DIR}/$(basename ${0%.*}).out;
rm -f ${RESULT_FNAME};

# Print Result Header
RESULT_HEADER="#    ${RESULT_HEADER-"Script: $0          Profiling Result Format: Time (FLOP/s)"}    #"
printf "%${#RESULT_HEADER}s\n" |tr " " "#" | tee -a ${RESULT_FNAME};
printf   "${RESULT_HEADER} \n"             | tee -a ${RESULT_FNAME};
printf "%${#RESULT_HEADER}s\n" |tr " " "#" | tee -a ${RESULT_FNAME};

# Default Field Columns
COLUMN_HEADER=""
for (( i = 0; i < ${#DEFAULT_FIELDS[@]}; i++ )) do
  COLUMN=${DEFAULT_FIELDS[$i]};
  if [ "$COLUMN" == "|" ]; then
    COLUMN_HEADER="${COLUMN_HEADER} |";
  else
    COLUMN_HEADER="${COLUMN_HEADER}$(printf " %10s" "$COLUMN")";
  fi
done

# Result Field Columns
for (( i = 1; i < ${#RESULT_FIELDS[@]}; i=$(($i+2)) )) do
  COLUMN=${RESULT_FIELDS[$i]};
  if [ "$COLUMN" == "|" ]; then
    COLUMN_HEADER="${COLUMN_HEADER} |";
  else
    COLUMN_HEADER="${COLUMN_HEADER}$(printf " %10s" "$COLUMN")";
  fi
done

# Profiling Field Columns
for (( i = 0; i < ${#PROF_FIELDS[@]}; i++ )) do
  COLUMN=${PROF_FIELDS[$i]};
  if [ "$COLUMN" == "|" ]; then
    COLUMN_HEADER="${COLUMN_HEADER} |";
  else
    COLUMN_HEADER="${COLUMN_HEADER}$(printf " %17s" "$COLUMN")";
  fi
done

# Print Column Headers
COLUMN_HEADER="|${COLUMN_HEADER}";
printf "%${#COLUMN_HEADER}s\n" |tr " " "=" | tee -a ${RESULT_FNAME}; #==========
printf   "${COLUMN_HEADER} \n"             | tee -a ${RESULT_FNAME};
printf "%${#COLUMN_HEADER}s\n" |tr " " "=" | tee -a ${RESULT_FNAME}; #==========

# Create output rows
declare -a RESULT_ROW=()
for (( i=0; i<${#nodes[@]}; i++ )) ; do
  RESULT_ROW[i]="";
done;

# Row Values (Default Fields)
for (( i = 0; i < ${#DEFAULT_FIELDS[@]}; i++ )) do
  COLUMN=${DEFAULT_FIELDS[$i]};
  for (( j=0; j<${#nodes[@]}; j++ )) ; do
    if [ "${nodes[j]}" == ":" ]; then continue; fi;
    if [ "$COLUMN" == "|" ]; then
      RESULT_ROW[j]="${RESULT_ROW[j]} |";
    else
      eval "colval=\${${COLUMN}[$j]}";
      RESULT_ROW[j]="${RESULT_ROW[j]}$(printf " %10s" "${colval}")";
    fi
  done
done

################### Loop over all runs ###################
for (( l=0; l<${#nodes[@]}; l++ )) ; do
  ( # Begin parallel subshell
  RESULT_FNAME=${RESULT_FNAME}_${l};

  # File name.
  FNAME_NOMIC=${RESULT_DIR}/$(basename ${EXEC})_${fname[l]}.out;
  FNAME_MIC=${RESULT_DIR}/$(basename ${EXEC})_mic_${fname[l]}.out;
  FNAME_ASYNC=${RESULT_DIR}/$(basename ${EXEC})_async_${fname[l]}.out;

  subrow_cnt=0;
  for (( k=0; k<3; k++ )) ; do
    case $k in
      0) FNAME=${FNAME_NOMIC};;
      1) FNAME=${FNAME_MIC};;
      2) FNAME=${FNAME_ASYNC};;
    esac
    if [ ! -f ${FNAME} ] ; then  continue; fi;
    if [ ! -s ${FNAME} ] ; then  continue; fi;
    subrow_cnt=$(( $subrow_cnt + 1 ))

    ######################### Parse Data #################################
    ROW_VALUE=${RESULT_ROW[l]};

    # Parse Data: Results
    for (( i = 0; i < ${#RESULT_FIELDS[@]}; i=$(($i+2)) )) do
      x="${RESULT_FIELDS[i]}"
      if [ "$x" == "|" ]; then
        ROW_VALUE="${ROW_VALUE} |";
        continue;
      fi
      PARAM[i]="$(grep -hir "$x" ${FNAME} | tail -n 1 | rev | cut -d ' ' -f 1 | rev)";
      ROW_VALUE="${ROW_VALUE}$(printf " %10s" "${PARAM[i]}")";
    done
    #---------------------------------------------------------------------
    # Parse Data: Time, Flop, Flop/s 
    PROC_NOMIC="$(grep -hir "Number of MPI processes:" ${FNAME_NOMIC} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"
    PROC="$(grep -hir "Number of MPI processes:" ${FNAME} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"
    for (( i = 0; i < ${#PROF_FIELDS[@]}; i++ )) do
      x="${PROF_FIELDS[i]}"
      if [ "$x" == "|" ]; then
        ROW_VALUE="${ROW_VALUE} |";
        continue;
      fi
      T_MAX[i]="$(grep -hir "$x  " ${FNAME} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 10 | rev)"
      if [ "${T_MAX[i]}" == "" ]; then continue; fi;
      FP_AVG[i]="$(grep -hir "$x  " ${FNAME_NOMIC} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 8 | rev)"
      FP_AVG[i]=$(echo "scale=10;${FP_AVG[i]}*${PROC_NOMIC}/${mpi_proc[l]}" | bc 2> /dev/null)
      FLOPS[i]=$(echo "scale=10;${FP_AVG[i]}/(${T_MAX[i]}+0.0001)" | bc 2> /dev/null)

      if [ "${FLOPS[i]}" != "" ] && [ -f ${FNAME_MIC} ] && [ -f ${FNAME_ASYNC} ] && [ -f ${FNAME_NOMIC} ] ; then 
        T_MAX_NOASYNC[i]="$(grep -hir "$x  " ${FNAME_MIC} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 10 | rev)"
        if [ "${T_MAX_NOASYNC[i]}" == "" ]; then continue; fi;
        compare_result1=$(echo "${T_MAX[i]}<0.5*${T_MAX_NOASYNC[i]}" | bc)
        compare_result2=$(echo "${T_MAX[i]}<0.01" | bc)
        if [ ${compare_result1} -eq 1 ] && [ ${compare_result2} -eq 1 ] ;  then
          FLOPS[i]=$(echo "scale=10;${FP_AVG[i]}/${T_MAX_NOASYNC[i]}" | bc 2> /dev/null)
        fi
      fi
      FLOPS[i]=$(echo "scale=10;${FLOPS[i]}*${mpi_proc[l]}/${nodes[l]}" | bc 2> /dev/null)
      TIMING_FORMAT=" %9.3f (%5.1f)"
      ROW_VALUE="${ROW_VALUE}$(printf "${TIMING_FORMAT}" "${T_MAX[i]}" "${FLOPS[i]}")";
    done
    #=====================================================================

    ######################### Print Data #################################
    printf "|${ROW_VALUE}\n" >> ${RESULT_FNAME};
    #=====================================================================
  done

  if [[  $l == $(( ${#nodes[@]}-1 )) ]] || [ "${nodes[l]}" == ":" ]; then
    printf "%${#COLUMN_HEADER}s\n" |tr " " "=" >> ${RESULT_FNAME}; #==========
  elif [[ $subrow_cnt > 1 ]]; then
    printf "%${#COLUMN_HEADER}s\n" |tr " " "-" >> ${RESULT_FNAME}; #----------
  fi
  )& # End parallel subshell

  # Combine results
  if (( ($l+1) % 10 == 0 )) || [[ $l == $(( ${#nodes[@]}-1 )) ]] ; then
    wait;
    for (( i=0; i<${#nodes[@]}; i++ )) ; do
      RESULT_FNAME_=${RESULT_FNAME}_${i};
      if [ -f ${RESULT_FNAME_} ] ; then
        cat ${RESULT_FNAME_} | tee -a ${RESULT_FNAME};
        rm ${RESULT_FNAME_};
      fi;
    done
  fi
done

