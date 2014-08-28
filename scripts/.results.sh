#!/bin/bash

eval    $nodes_;
eval    $cores_;
eval $mpi_proc_;
eval  $threads_;
eval $testcase_;
eval    $n_pts_;
eval        $m_;
eval        $q_;
eval      $tol_;
eval    $depth_;
eval     $unif_;
eval     $adap_;
eval $max_time_;
eval    $fname_;
eval     $args_;

export RESULT_FNAME=${RESULT_DIR}/${RESULT_FNAME}
rm -f ${RESULT_FNAME}
echo "#########################################################################################################" | tee -a ${RESULT_FNAME}
echo "#         CPU + MIC Results :   Time (FLOP/s)          [CPU Only, CPU+MIC, CPU+MIC (async)]             #" | tee -a ${RESULT_FNAME}
echo "#########################################################################################################" | tee -a ${RESULT_FNAME}


################### Input Parameter Fields ###################
export PARAMSTR="FMM Kernel name;Order of multipole expansions;Order of Chebyshev polynomials;Maximum Tree Depth;Chebyshev Tolerance"
IFS=';' read -ra PARAMARR <<< "$PARAMSTR"
Nparam=${#PARAMARR[@]};
export PARAMHEADERSTR="kernel;m;q;max_d;tol"
IFS=';' read -ra PARAMHEADERARR <<< "$PARAMHEADERSTR"

################### Time (FLOP/s) Fields ###################
export COLSTR="RunFMM;UpwardPass;ReduceBcast;DownwardPass;U-List;V-List;W-List;X-List;D2H_Wait:Trg;D2D;D2T"
IFS=';' read -ra COLARR <<< "$COLSTR"
N=${#COLARR[@]};

################### Error Fields ###################
export ERRSTR="Maximum Relative Error \[Input\];Relative L2 Error \[Input\];Relative L2 Error \[Output\];Maximum Relative Error \[Output\];Relative L2 Error \[OutputGrad\];Maximum Relative Error \[OutputGrad\]"
IFS=';' read -ra ERRARR <<< "$ERRSTR"
Nerr=${#ERRARR[@]};
export ERRHEADERSTR="Linf(f);L2(f);L2(u);Linf(u);L2(grad_u);Linf(grad_u)"
IFS=';' read -ra ERRHEADERARR <<< "$ERRHEADERSTR"


################### Print Column Headers ###################
printf "%$((16+$(($Nparam+4))*12+$N*18+$Nerr*14))s\n" |tr " " "=" | tee -a ${RESULT_FNAME} #=================================================
#-----------------------------------------------------------
for (( i=0; i<$Nparam; i++ )) ; do
  printf "%11s " "${PARAMHEADERARR[i]}" | tee -a ${RESULT_FNAME}
done;
printf "   |" | tee -a ${RESULT_FNAME}
#-----------------------------------------------------------
HEADER_FORMAT="%11s "
printf "${HEADER_FORMAT}" "MPI_PROC" | tee -a ${RESULT_FNAME}
printf "${HEADER_FORMAT}"  "THREADS" | tee -a ${RESULT_FNAME}
printf "${HEADER_FORMAT}"    "NODES" | tee -a ${RESULT_FNAME}
printf "${HEADER_FORMAT}" "OCT/NODE" | tee -a ${RESULT_FNAME}
printf "   |" | tee -a ${RESULT_FNAME}
#-----------------------------------------------------------
for (( i=0; i<$N; i++ )) ; do
  printf " %17s" "${COLARR[i]}" | tee -a ${RESULT_FNAME}
done;
printf "   |" | tee -a ${RESULT_FNAME}
#-----------------------------------------------------------
for (( i=0; i<$Nerr; i++ )) ; do
  printf " %13s" "${ERRHEADERARR[i]}" | tee -a ${RESULT_FNAME}
done;
printf "   |\n" | tee -a ${RESULT_FNAME}
#-----------------------------------------------------------
printf "%$((16+$(($Nparam+4))*12+$N*18+$Nerr*14))s\n" |tr " " "=" | tee -a ${RESULT_FNAME} #=================================================
#===========================================================


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
    if [ $k -eq 0 ] ; then
      FNAME=${FNAME_NOMIC};
    fi
    if [ $k -eq 1 ] ; then
      FNAME=${FNAME_MIC};
    fi
    if [ $k -eq 2 ] ; then
      FNAME=${FNAME_ASYNC};
    fi
    if [ ! -f ${FNAME} ] ; then 
      #echo >> ${RESULT_FNAME}      
      continue; 
    fi;
    subrow_cnt=$(( $subrow_cnt + 1 ))

    ######################### Parse Data #################################
    # Parse Data: Parameters
    for (( i=0; i<$Nparam; i++ )) ; do
      x="${PARAMARR[i]}"
      PARAM[i]="$(grep -hir "$x" ${FNAME} | tail -n 1 | rev | cut -d ' ' -f 1 | rev)";
    done
    #---------------------------------------------------------------------
    # Parse Data: Leaf Count
    NODE_LIST="$(grep -hir 'Leaf Nodes:' ${FNAME} | tail -n 1 | cut -d " " -f 3-)"
    NODES=0;
    for i in ${NODE_LIST} ; do 
      NODES=$(( $NODES + $i )); 
    done;
    PROC_NOMIC="$(grep -hir "Number of MPI processes:" ${FNAME_NOMIC} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"
    PROC="$(grep -hir "Number of MPI processes:" ${FNAME} | tail -n 1 | tr -s ' ' | rev | cut -d ' ' -f 1 | rev)"
    #---------------------------------------------------------------------
    # Parse Data: Time, Flop, Flop/s 
    for (( i=0; i<$N; i++ )) ; do
      x="${COLARR[i]}"

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
    done
    #---------------------------------------------------------------------
    # Parse Data: Error
    for (( i=0; i<$Nerr; i++ )) ; do
      x="${ERRARR[i]}"
      ERR[i]="$(grep -hir "$x" ${FNAME} | tail -n 1 | rev | cut -d ' ' -f 1 | rev)";
    done
    #=====================================================================


    ######################### Print Data #################################
    PARAM_FORMAT="%11s "
    for (( i=0; i<$Nparam; i++ )) ; do
      printf "${PARAM_FORMAT}" "${PARAM[i]}" >> ${RESULT_FNAME}      
    done;
    printf "   |" >> ${RESULT_FNAME}      
    #---------------------------------------------------------------------
    printf "${PARAM_FORMAT}"            "${PROC}" >> ${RESULT_FNAME}      
    printf "${PARAM_FORMAT}"             "${threads[l]}" >> ${RESULT_FNAME}      
    printf "${PARAM_FORMAT}"               "${nodes[l]}" >> ${RESULT_FNAME}      
    printf "${PARAM_FORMAT}" "$((${NODES}/${nodes[l]}))" >> ${RESULT_FNAME}      
    printf "   |" >> ${RESULT_FNAME}      
    #---------------------------------------------------------------------
    TIMING_FORMAT=" %9.3f (%5.1f)"
    for (( i=0; i<$N; i++ )) ; do
      printf "${TIMING_FORMAT}" "${T_MAX[i]}" "${FLOPS[i]}" >> ${RESULT_FNAME}      
    done;
    printf "   |" >> ${RESULT_FNAME}      
    #---------------------------------------------------------------------
    ERR_FORMAT="      %1.2e"
    for (( i=0; i<$Nerr; i++ )) ; do
      printf "${ERR_FORMAT}" "${ERR[i]}" >> ${RESULT_FNAME}      
    done;
    printf "   |\n" >> ${RESULT_FNAME}      
    #=====================================================================

  done
  if [[  $l == $(( ${#nodes[@]}-1 )) ]] || [ "${nodes[l]}" == ":" ]; then
    printf "%$((16+$(($Nparam+4))*12+$N*18+$Nerr*14))s\n" |tr " " "=" >> ${RESULT_FNAME}       #=================================================
  elif [[ $subrow_cnt > 1 ]]; then
    printf "%$((16+$(($Nparam+4))*12+$N*18+$Nerr*14))s\n" |tr " " "-" >> ${RESULT_FNAME}       #-------------------------------------------------
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

