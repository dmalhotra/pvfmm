
/**
  @file parUtils.txx
  @brief Definitions of the templated functions in the par module.
  @author Rahul S. Sampath, rahul.sampath@gmail.com
  @author Hari Sundar, hsundar@gmail.com
  @author Shravan Veerapaneni, shravan@seas.upenn.edu
  @author Santi Swaroop Adavani, santis@gmail.com
  */

#include "dtypes.h"
#include <cassert>
#include <iostream>
#include <algorithm>
#include <cstring>
#include "ompUtils.h"
#include <mpi.h>

namespace pvfmm{
namespace par{

  template <typename T>
    int Mpi_Alltoallv_sparse(T* sendbuf, int* sendcnts, int* sdispls,
        T* recvbuf, int* recvcnts, int* rdispls, const MPI_Comm &comm) {

#ifndef ALLTOALLV_FIX
      return Mpi_Alltoallv
        (sendbuf, sendcnts, sdispls,
         recvbuf, recvcnts, rdispls, comm);
#else

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);

      int commCnt = 0;

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      #pragma omp parallel for reduction(+:commCnt)
      for(int i = (rank+1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          commCnt++;
        }
        if(recvcnts[i] > 0) {
          commCnt++;
        }
      }

      MPI_Request* requests = new MPI_Request[commCnt];
      assert(requests);

      MPI_Status* statuses = new MPI_Status[commCnt];
      assert(statuses);

      commCnt = 0;

      //First place all recv requests. Do not recv from self.
      for(int i = 0; i < rank; i++) {
        if(recvcnts[i] > 0) {
          MPI_Irecv( &(recvbuf[rdispls[i]]) , recvcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(recvcnts[i] > 0) {
          MPI_Irecv( &(recvbuf[rdispls[i]]) , recvcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Next send the messages. Do not send to self.
      for(int i = 0; i < rank; i++) {
        if(sendcnts[i] > 0) {
          MPI_Issend( &(sendbuf[sdispls[i]]), sendcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      for(int i = (rank + 1); i < npes; i++) {
        if(sendcnts[i] > 0) {
          MPI_Issend( &(sendbuf[sdispls[i]]), sendcnts[i], par::Mpi_datatype<T>::value(), i, 1,
              comm, &(requests[commCnt]) );
          commCnt++;
        }
      }

      //Now copy local portion.
      #pragma omp parallel for
      for(int i = 0; i < sendcnts[rank]; i++) {
        recvbuf[rdispls[rank] + i] = sendbuf[sdispls[rank] + i];
      }

      MPI_Waitall(commCnt, requests, statuses);

      delete [] requests;
      delete [] statuses;
      return 0;
#endif
    }


  template <typename T>
    int Mpi_Alltoallv_dense(T* sbuff_, int* s_cnt_, int* sdisp_,
        T* rbuff_, int* r_cnt_, int* rdisp_, const MPI_Comm& comm){

#ifndef ALLTOALLV_FIX
      return Mpi_Alltoallv
        (sbuff_, s_cnt_, sdisp_,
         rbuff_, r_cnt_, rdisp_, c);
#else
      int np, pid;
      MPI_Comm_size(comm,&np);
      MPI_Comm_rank(comm,&pid);
      int range[2]={0,np-1};
      int split_id, partner;

      std::vector<int> s_cnt(np);
      #pragma omp parallel for
      for(int i=0;i<np;i++){
        s_cnt[i]=s_cnt_[i]*sizeof(T)+2*sizeof(int);
      }
      std::vector<int> sdisp(np); sdisp[0]=0;
      omp_par::scan(&s_cnt[0],&sdisp[0],np);

      char* sbuff=new char[sdisp[np-1]+s_cnt[np-1]];
      #pragma omp parallel for
      for(int i=0;i<np;i++){
        ((int*)&sbuff[sdisp[i]])[0]=s_cnt[i];
        ((int*)&sbuff[sdisp[i]])[1]=pid;
        memcpy(&sbuff[sdisp[i]]+2*sizeof(int),&sbuff_[sdisp_[i]],s_cnt[i]-2*sizeof(int));
      }

      while(range[0]<range[1]){
        split_id=(range[0]+range[1])/2;

        int new_range[2]={(pid<=split_id?range[0]:split_id+1),
          (pid<=split_id?split_id:range[1]  )};
        int cmp_range[2]={(pid> split_id?range[0]:split_id+1),
          (pid> split_id?split_id:range[1]  )};
        int new_np=new_range[1]-new_range[0]+1;
        int cmp_np=cmp_range[1]-cmp_range[0]+1;

        partner=pid+cmp_range[0]-new_range[0];
        if(partner>range[1]) partner=range[1];
        assert(partner>=range[0]);
        bool extra_partner=( (range[1]-range[0])%2==0  &&
            range[1]            ==pid  );

        //Communication.
        {
          int* s_lengths=&s_cnt[cmp_range[0]-range[0]];
          std::vector<int> s_len_ext(cmp_np,0);
          std::vector<int> r_cnt    (new_np,0);
          std::vector<int> r_cnt_ext(new_np,0);
          MPI_Status status;

          //Exchange send sizes.
          MPI_Sendrecv                  (&s_lengths[0],cmp_np,MPI_INT, partner,0,   &r_cnt    [0],new_np,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&s_len_ext[0],cmp_np,MPI_INT,split_id,0,   &r_cnt_ext[0],new_np,MPI_INT,split_id,   0,comm,&status);

          //Allocate receive buffer.
          std::vector<int> rdisp    (new_np,0);
          std::vector<int> rdisp_ext(new_np,0);
          omp_par::scan(&r_cnt    [0],&rdisp    [0],new_np);
          omp_par::scan(&r_cnt_ext[0],&rdisp_ext[0],new_np);
          int rbuff_size    =rdisp    [new_np-1]+r_cnt    [new_np-1];
          int rbuff_size_ext=rdisp_ext[new_np-1]+r_cnt_ext[new_np-1];
          char* rbuff   =                new char[rbuff_size    ];
          char* rbuffext=(extra_partner? new char[rbuff_size_ext]: NULL);

          //Sendrecv data.
          {
            int * s_cnt_tmp=&s_cnt[cmp_range[0]-range[0]] ;
            int * sdisp_tmp=&sdisp[cmp_range[0]-range[0]];
            char* sbuff_tmp=&sbuff[sdisp_tmp[0]];
            int  sbuff_size=sdisp_tmp[cmp_np-1]+s_cnt_tmp[cmp_np-1]-sdisp_tmp[0];
            MPI_Sendrecv                  (sbuff_tmp,sbuff_size,MPI_BYTE, partner,0,   &rbuff   [0],rbuff_size    ,MPI_BYTE, partner,   0,comm,&status);
            if(extra_partner) MPI_Sendrecv(     NULL,         0,MPI_BYTE,split_id,0,   &rbuffext[0],rbuff_size_ext,MPI_BYTE,split_id,   0,comm,&status);
          }

          //Rearrange received data.
          {
            //assert(!extra_partner);
            int * s_cnt_old=&s_cnt[new_range[0]-range[0]];
            int * sdisp_old=&sdisp[new_range[0]-range[0]];

            std::vector<int> s_cnt_new(&s_cnt_old[0],&s_cnt_old[new_np]);
            std::vector<int> sdisp_new(new_np       ,0                 );
            #pragma omp parallel for
            for(int i=0;i<new_np;i++){
              s_cnt_new[i]+=r_cnt[i]+r_cnt_ext[i];
            }
            omp_par::scan(&s_cnt_new[0],&sdisp_new[0],new_np);

            //Copy data to sbuff_new.
            char* sbuff_new=new char[sdisp_new[new_np-1]+s_cnt_new[new_np-1]];
            #pragma omp parallel for
            for(int i=0;i<new_np;i++){
              memcpy(&sbuff_new[sdisp_new[i]                      ],&sbuff   [sdisp_old[i]],s_cnt_old[i]);
              memcpy(&sbuff_new[sdisp_new[i]+s_cnt_old[i]         ],&rbuff   [rdisp    [i]],r_cnt    [i]);
              memcpy(&sbuff_new[sdisp_new[i]+s_cnt_old[i]+r_cnt[i]],&rbuffext[rdisp_ext[i]],r_cnt_ext[i]);
            }

            //Free memory.
            if(sbuff   !=NULL) delete[] sbuff   ;
            if(rbuff   !=NULL) delete[] rbuff   ;
            if(rbuffext!=NULL) delete[] rbuffext;

            //Substitute data for next iteration.
            s_cnt=s_cnt_new;
            sdisp=sdisp_new;
            sbuff=sbuff_new;
          }
        }

        range[0]=new_range[0];
        range[1]=new_range[1];
      }

      //Copy data to rbuff_.
      std::vector<char*> buff_ptr(np);
      char* tmp_ptr=sbuff;
      for(int i=0;i<np;i++){
        int& blk_size=((int*)tmp_ptr)[0];
        buff_ptr[i]=tmp_ptr;
        tmp_ptr+=blk_size;
      }
      #pragma omp parallel for
      for(int i=0;i<np;i++){
        int& blk_size=((int*)buff_ptr[i])[0];
        int& src_pid=((int*)buff_ptr[i])[1];
        assert(blk_size-2*sizeof(int)<=r_cnt_[src_pid]*sizeof(T));
        memcpy(&rbuff_[rdisp_[src_pid]],buff_ptr[i]+2*sizeof(int),blk_size-2*sizeof(int));
      }

      //Free memory.
      if(sbuff   !=NULL) delete[] sbuff;
      return 0;
#endif
    }


  template<typename T>
    int partitionW(Vector<T>& nodeList, long long* wts, const MPI_Comm& comm){

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      long long npesLong = npes;

      long long nlSize = nodeList.Dim();
      long long off1= 0, off2= 0, localWt= 0, totalWt = 0;

      // First construct arrays of wts.
      Vector<long long> wts_(nlSize);
      if(wts == NULL) {
        wts=&wts_[0];
        #pragma omp parallel for
        for (long long i = 0; i < nlSize; i++){
          wts[i] = 1;
        }
      }
      #pragma omp parallel for reduction(+:localWt)
      for (long long i = 0; i < nlSize; i++){
        localWt+=wts[i];
      }

      // compute the total weight of the problem ...
      MPI_Allreduce(&localWt, &totalWt, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
      MPI_Scan(&localWt, &off2, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm );
      off1=off2-localWt;

      // perform a local scan on the weights first ...
      Vector<long long> lscn(nlSize);
      if(nlSize) {
        lscn[0]=off1;
        omp_par::scan(&wts[0],&lscn[0],nlSize);
      }

      Vector<int> int_buff(npesLong*4);
      Vector<int> sendSz (npesLong,&int_buff[0]+npesLong*0,false);
      Vector<int> recvSz (npesLong,&int_buff[0]+npesLong*1,false);
      Vector<int> sendOff(npesLong,&int_buff[0]+npesLong*2,false);
      Vector<int> recvOff(npesLong,&int_buff[0]+npesLong*3,false);

      // compute the partition offsets and sizes so that All2Allv can be performed.
      // initialize ...

      #pragma omp parallel for
      for (size_t i = 0; i < npesLong; i++) {
        sendSz[i] = 0;
      }

      //The Heart of the algorithm....
      if(nlSize>0 && totalWt>0) {
        long long pid1=( off1   *npesLong)/totalWt;
        long long pid2=((off2+1)*npesLong)/totalWt+1;
        assert((totalWt*pid2)/npesLong>=off2);
        pid1=(pid1<       0?       0:pid1);
        pid2=(pid2>npesLong?npesLong:pid2);
        #pragma omp parallel for
        for(int i=pid1;i<pid2;i++){
          long long wt1=(totalWt*(i  ))/npesLong;
          long long wt2=(totalWt*(i+1))/npesLong;
          long long start = std::lower_bound(&lscn[0], &lscn[0]+nlSize, wt1, std::less<long long>())-&lscn[0];
          long long end   = std::lower_bound(&lscn[0], &lscn[0]+nlSize, wt2, std::less<long long>())-&lscn[0];
          if(i==         0) start=0     ;
          if(i==npesLong-1) end  =nlSize;
          sendSz[i]=end-start;
        }
      }else sendSz[0]=nlSize;

      // communicate with other procs how many you shall be sending and get how
      // many to recieve from whom.
      MPI_Alltoall(&sendSz[0], 1, par::Mpi_datatype<int>::value(),
          &recvSz[0], 1, par::Mpi_datatype<int>::value(), comm);

      // compute offsets ...
      sendOff[0] = 0; omp_par::scan(&sendSz[0],&sendOff[0],npesLong);
      recvOff[0] = 0; omp_par::scan(&recvSz[0],&recvOff[0],npesLong);

      // new value of nlSize, ie the local nodes.
      long long nn = recvSz[npesLong-1] + recvOff[npes-1];

      // allocate memory for the new arrays ...
      Vector<T> newNodes;
      {
        if(nodeList.Capacity()>nn+std::max(nn,nlSize)){
          newNodes.ReInit(nn,&nodeList[0]+std::max(nn,nlSize),false);
        //}else if(buff!=NULL && buff->Dim()>nn*sizeof(T)){
        //  newNodes.ReInit(nn,(T*)&(*buff)[0],false);
        }else newNodes.Resize(nn);
      }

      // perform All2All  ...
      par::Mpi_Alltoallv_sparse<T>(&nodeList[0], &sendSz[0], &sendOff[0],
          &newNodes[0], &recvSz[0], &recvOff[0], comm);

      // reset the pointer ...
      nodeList=newNodes;

      return 0;
    }//end function

  template<typename T>
    int partitionW(std::vector<T>& nodeList, long long* wts, const MPI_Comm& comm){
      Vector<T> nodeList_=nodeList;
      int ret = par::partitionW<T>(nodeList_, wts, comm);

      nodeList.assign(&nodeList_[0],&nodeList_[0]+nodeList_.Dim());
      return ret;
    }


  template<typename T>
    int HyperQuickSort(const Vector<T>& arr_, Vector<T>& SortedElem, const MPI_Comm& comm_){ // O( ((N/p)+log(p))*(log(N/p)+log(p)) )

      // Copy communicator.
      MPI_Comm comm=comm_;

      // Get comm size and rank.
      int npes, myrank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &myrank);
      int omp_p=omp_get_max_threads();
      srand(myrank);

      // Local and global sizes. O(log p)
      long long totSize, nelem = arr_.Dim();
      //assert(nelem); // TODO: Check if this is needed.
      MPI_Allreduce(&nelem, &totSize, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);

      // Local sort.
      Vector<T> arr=arr_;
      omp_par::merge_sort(&arr[0], &arr[0]+nelem);

      // Allocate memory.
      Vector<T> nbuff;
      Vector<T> nbuff_ext;
      Vector<T> rbuff    ;
      Vector<T> rbuff_ext;

      // Binary split and merge in each iteration.
      while(npes>1 && totSize>0){ // O(log p) iterations.

        //Determine splitters. O( log(N/p) + log(p) )
        T split_key;
        long long totSize_new;
        //while(true)
        {
          // Take random splitters. O( 1 ) -- Let p * splt_count = glb_splt_count = const = 100~1000
          int splt_count=(1000*nelem)/totSize;
          if(npes>1000) splt_count=(((float)rand()/(float)RAND_MAX)*totSize<(1000*nelem)?1:0);
          if(splt_count>nelem) splt_count=nelem;
          std::vector<T> splitters(splt_count);
          for(int i=0;i<splt_count;i++)
            splitters[i]=arr[rand()%nelem];

          // Gather all splitters. O( log(p) )
          int glb_splt_count;
          std::vector<int> glb_splt_cnts(npes);
          std::vector<int> glb_splt_disp(npes,0);
          MPI_Allgather(&splt_count      , 1, par::Mpi_datatype<int>::value(),
              &glb_splt_cnts[0], 1, par::Mpi_datatype<int>::value(), comm);

          omp_par::scan(&glb_splt_cnts[0],&glb_splt_disp[0],npes);
          glb_splt_count=glb_splt_cnts[npes-1]+glb_splt_disp[npes-1];
          std::vector<T> glb_splitters(glb_splt_count);
          MPI_Allgatherv(&    splitters[0], splt_count, par::Mpi_datatype<T>::value(),
              &glb_splitters[0], &glb_splt_cnts[0], &glb_splt_disp[0],
              par::Mpi_datatype<T>::value(), comm);

          // Determine split key. O( log(N/p) + log(p) )
          std::vector<long long> disp(glb_splt_count,0);
          if(nelem>0){
            #pragma omp parallel for
            for(int i=0;i<glb_splt_count;i++){
              disp[i]=std::lower_bound(&arr[0], &arr[0]+nelem, glb_splitters[i])-&arr[0];
            }
          }
          std::vector<long long> glb_disp(glb_splt_count,0);
          MPI_Allreduce(&disp[0], &glb_disp[0], glb_splt_count, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);

          long long* split_disp=&glb_disp[0];
          for(int i=0;i<glb_splt_count;i++)
            if(labs(glb_disp[i]-totSize/2)<labs(*split_disp-totSize/2)) split_disp=&glb_disp[i];
          split_key=glb_splitters[split_disp-&glb_disp[0]];

          totSize_new=(myrank<=(npes-1)/2?*split_disp:totSize-*split_disp);
          //double err=(((double)*split_disp)/(totSize/2))-1.0;
          //if(fabs(err)<0.01 || npes<=16) break;
          //else if(!myrank) std::cout<<err<<'\n';
        }

        // Split problem into two. O( N/p )
        int split_id=(npes-1)/2;
        {
          int new_p0=(myrank<=split_id?0:split_id+1);
          int cmp_p0=(myrank> split_id?0:split_id+1);

          int partner = myrank+cmp_p0-new_p0;
          if(partner>=npes) partner=npes-1;
          assert(partner>=0);

          bool extra_partner=( npes%2==1  && npes-1==myrank );

          // Exchange send sizes.
          char *sbuff, *lbuff;
          int     rsize=0,     ssize=0, lsize=0;
          int ext_rsize=0, ext_ssize=0;
          size_t split_indx=(nelem>0?std::lower_bound(&arr[0], &arr[0]+nelem, split_key)-&arr[0]:0);
          ssize=       (myrank> split_id? split_indx: nelem  -split_indx)*sizeof(T);
          sbuff=(char*)(myrank> split_id? &arr[0]   : &arr[0]+split_indx);
          lsize=       (myrank<=split_id? split_indx: nelem  -split_indx)*sizeof(T);
          lbuff=(char*)(myrank<=split_id? &arr[0]   : &arr[0]+split_indx);

          MPI_Status status;
          MPI_Sendrecv                  (&    ssize,1,MPI_INT, partner,0,   &    rsize,1,MPI_INT, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv(&ext_ssize,1,MPI_INT,split_id,0,   &ext_rsize,1,MPI_INT,split_id,   0,comm,&status);

          // Exchange data.
          rbuff    .Resize(    rsize/sizeof(T));
          rbuff_ext.Resize(ext_rsize/sizeof(T));
          MPI_Sendrecv                  (sbuff,ssize,MPI_BYTE, partner,0,   &rbuff    [0],    rsize,MPI_BYTE, partner,   0,comm,&status);
          if(extra_partner) MPI_Sendrecv( NULL,    0,MPI_BYTE,split_id,0,   &rbuff_ext[0],ext_rsize,MPI_BYTE,split_id,   0,comm,&status);

          int nbuff_size=lsize+rsize+ext_rsize;
          nbuff.Resize(nbuff_size/sizeof(T));
          omp_par::merge<T*>((T*)lbuff, (T*)(lbuff+lsize), &rbuff[0], &rbuff[0]+(rsize/sizeof(T)), &nbuff[0], omp_p, std::less<T>());
          if(ext_rsize>0 && nbuff.Dim()>0){
            nbuff_ext.Resize(nbuff_size/sizeof(T));
            omp_par::merge<T*>(&nbuff[0], &nbuff[0]+((lsize+rsize)/sizeof(T)), &rbuff_ext[0], &rbuff_ext[0]+(ext_rsize/sizeof(T)), &nbuff_ext[0], omp_p, std::less<T>());
            nbuff.Swap(nbuff_ext);
            nbuff_ext.Resize(0);
          }

          // Copy new data.
          totSize=totSize_new;
          nelem = nbuff_size/sizeof(T);
          arr.Swap(nbuff);
          nbuff.Resize(0);
        }

        {// Split comm.  O( log(p) ) ??
          MPI_Comm scomm;
          MPI_Comm_split(comm, myrank<=split_id, myrank, &scomm );
          comm=scomm;
          npes  =(myrank<=split_id? split_id+1: npes  -split_id-1);
          myrank=(myrank<=split_id? myrank    : myrank-split_id-1);
        }
      }

      SortedElem.Resize(nelem);
      memcpy(&SortedElem[0], &arr[0], nelem*sizeof(T));

      par::partitionW<T>(SortedElem, NULL , comm_);
      return 0;
    }//end function

  template<typename T>
    int HyperQuickSort(const std::vector<T>& arr_, std::vector<T>& SortedElem_, const MPI_Comm& comm_){
      Vector<T> SortedElem;
      const Vector<T> arr(arr_.size(),(T*)&arr_[0],false);

      int ret = HyperQuickSort(arr, SortedElem, comm_);
      SortedElem_.assign(&SortedElem[0],&SortedElem[0]+SortedElem.Dim());
      return ret;
    }


  template<typename T>
    int SortScatterIndex(const Vector<T>& key, Vector<size_t>& scatter_index, const MPI_Comm& comm, const T* split_key_){
      typedef SortPair<T,size_t> Pair_t;

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      long long npesLong = npes;

      //Vector<char> buff;
      //if(buff_!=NULL && buff_->Dim()>0){
      //  buff.ReInit(buff_->Dim(),&(*buff_)[0],false);
      //}

      Vector<Pair_t> parray;
      { // Allocate memory
        //size_t parray_size=key.Dim()*sizeof(Pair_t);
        //if(buff.Dim()>parray_size){
        //  parray.ReInit(key.Dim(),(Pair_t*)&buff[0],false);
        //  buff.ReInit(buff.Dim()-parray_size,&buff[0]+parray_size,false);
        //}else
        parray.Resize(key.Dim());
      }
      { // Build global index.
        long long glb_dsp=0;
        long long loc_size=key.Dim();
        MPI_Scan(&loc_size, &glb_dsp, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
        glb_dsp-=loc_size;
        #pragma omp parallel for
        for(size_t i=0;i<loc_size;i++){
          parray[i].key=key[i];
          parray[i].data=glb_dsp+i;
        }
      }

      Vector<Pair_t> psorted;
      { // Allocate memory
        //if(buff.Dim()>0){
        //  psorted.ReInit(buff.Dim()/sizeof(Pair_t), (Pair_t*)&buff[0], false);
        //}
      }
      HyperQuickSort(parray, psorted, comm);

      if(split_key_!=NULL){ // Partition data
        Vector<T> split_key(npesLong);
        MPI_Allgather((void*)split_key_  , 1, par::Mpi_datatype<T>::value(),
                            &split_key[0], 1, par::Mpi_datatype<T>::value(), comm);

        Vector<int> int_buff(npesLong*4);
        Vector<int> sendSz (npesLong,&int_buff[0]+npesLong*0,false);
        Vector<int> recvSz (npesLong,&int_buff[0]+npesLong*1,false);
        Vector<int> sendOff(npesLong,&int_buff[0]+npesLong*2,false);
        Vector<int> recvOff(npesLong,&int_buff[0]+npesLong*3,false);
        long long nlSize = psorted.Dim();

        // compute the partition offsets and sizes so that All2Allv can be performed.
        // initialize ...

        #pragma omp parallel for
        for (size_t i = 0; i < npesLong; i++) {
          sendSz[i] = 0;
        }

        //The Heart of the algorithm....
        if(nlSize>0) {
          // Determine processor range.
          long long pid1=std::lower_bound(&split_key[0], &split_key[0]+npesLong, psorted[       0].key)-&split_key[0]-1;
          long long pid2=std::upper_bound(&split_key[0], &split_key[0]+npesLong, psorted[nlSize-1].key)-&split_key[0]+0;
          pid1=(pid1<       0?       0:pid1);
          pid2=(pid2>npesLong?npesLong:pid2);

          #pragma omp parallel for
          for(int i=pid1;i<pid2;i++){
            Pair_t p1; p1.key=split_key[                 i];
            Pair_t p2; p2.key=split_key[i+1<npesLong?i+1:i];
            long long start = std::lower_bound(&psorted[0], &psorted[0]+nlSize, p1, std::less<Pair_t>())-&psorted[0];
            long long end   = std::lower_bound(&psorted[0], &psorted[0]+nlSize, p2, std::less<Pair_t>())-&psorted[0];
            if(i==         0) start=0     ;
            if(i==npesLong-1) end  =nlSize;
            sendSz[i]=end-start;
          }
        }

        // communicate with other procs how many you shall be sending and get how
        // many to recieve from whom.
        MPI_Alltoall(&sendSz[0], 1, par::Mpi_datatype<int>::value(),
            &recvSz[0], 1, par::Mpi_datatype<int>::value(), comm);

        // compute offsets ...
        sendOff[0] = 0; omp_par::scan(&sendSz[0],&sendOff[0],npesLong);
        recvOff[0] = 0; omp_par::scan(&recvSz[0],&recvOff[0],npesLong);

        // new value of nlSize, ie the local nodes.
        long long nn = recvSz[npesLong-1] + recvOff[npesLong-1];

        // allocate memory for the new arrays ...
        Vector<Pair_t> newNodes;
        {
          if(psorted.Capacity()>nn+std::max(nn,nlSize)){
            newNodes.ReInit(nn,&psorted[0]+std::max(nn,nlSize),false);
          }else newNodes.Resize(nn);
        }

        // perform All2All  ...
        par::Mpi_Alltoallv_sparse<Pair_t>(&psorted[0], &sendSz[0], &sendOff[0],
            &newNodes[0], &recvSz[0], &recvOff[0], comm);

        // reset the pointer ...
        psorted=newNodes;
      }

      scatter_index.Resize(psorted.Dim());
      #pragma omp parallel for
      for(size_t i=0;i<psorted.Dim();i++){
        scatter_index[i]=psorted[i].data;
      }

      return 0;
    }

  template<typename T>
    int ScatterForward(Vector<T>& data_, const Vector<size_t>& scatter_index, const MPI_Comm& comm){
      typedef SortPair<size_t,size_t> Pair_t;

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      long long npesLong = npes;

      size_t data_dim=0;
      long long send_size=0;
      long long recv_size=0;
      {
        recv_size=scatter_index.Dim();

        long long glb_size[2]={0,0};
        long long loc_size[2]={data_.Dim()*sizeof(T), recv_size};
        MPI_Allreduce(&loc_size, &glb_size, 2, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
        if(glb_size[0]==0 || glb_size[1]==0) return 0; //Nothing to be done.
        data_dim=glb_size[0]/glb_size[1];
        assert(glb_size[0]==data_dim*glb_size[1]);

        send_size=(data_.Dim()*sizeof(T))/data_dim;
      }

      Vector<char> recv_buff(recv_size*data_dim);
      Vector<char> send_buff(send_size*data_dim);

      // Global scan of data size.
      Vector<long long> glb_scan(npesLong);
      {
        long long glb_rank=0;
        MPI_Scan(&send_size, &glb_rank, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
        glb_rank-=send_size;

        MPI_Allgather(&glb_rank   , 1, par::Mpi_datatype<long long>::value(),
                      &glb_scan[0], 1, par::Mpi_datatype<long long>::value(), comm);
      }

      // Sort scatter_index.
      Vector<Pair_t> psorted(recv_size);
      {
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          psorted[i].key=scatter_index[i];
          psorted[i].data=i;
        }
        omp_par::merge_sort(&psorted[0], &psorted[0]+recv_size);
      }

      // Exchange send, recv indices.
      Vector<size_t> recv_indx(recv_size);
      Vector<size_t> send_indx(send_size);
      Vector<int> sendSz(npesLong);
      Vector<int> sendOff(npesLong);
      Vector<int> recvSz(npesLong);
      Vector<int> recvOff(npesLong);
      {
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          recv_indx[i]=psorted[i].key;
        }

        #pragma omp parallel for
        for(size_t i=0;i<npesLong;i++){
          size_t start=              std::lower_bound(&recv_indx[0], &recv_indx[0]+recv_size, glb_scan[  i])-&recv_indx[0];
          size_t end  =(i+1<npesLong?std::lower_bound(&recv_indx[0], &recv_indx[0]+recv_size, glb_scan[i+1])-&recv_indx[0]:recv_size);
          recvSz[i]=end-start;
          recvOff[i]=start;
        }

        MPI_Alltoall(&recvSz[0], 1, par::Mpi_datatype<int>::value(),
                     &sendSz[0], 1, par::Mpi_datatype<int>::value(), comm);
        sendOff[0] = 0; omp_par::scan(&sendSz[0],&sendOff[0],npesLong);
        assert(sendOff[npesLong-1]+sendSz[npesLong-1]==send_size);

        par::Mpi_Alltoallv_dense<size_t>(&recv_indx[0], &recvSz[0], &recvOff[0],
                                         &send_indx[0], &sendSz[0], &sendOff[0], comm);
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          assert(send_indx[i]>=glb_scan[rank]);
          send_indx[i]-=glb_scan[rank];
          assert(send_indx[i]<send_size);
        }
      }

      // Prepare send buffer
      {
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          size_t src_indx=send_indx[i]*data_dim;
          size_t trg_indx=i*data_dim;
          for(size_t j=0;j<data_dim;j++)
            send_buff[trg_indx+j]=data[src_indx+j];
        }
      }

      // All2Allv
      {
        #pragma omp parallel for
        for(size_t i=0;i<npesLong;i++){
          sendSz [i]*=data_dim;
          sendOff[i]*=data_dim;
          recvSz [i]*=data_dim;
          recvOff[i]*=data_dim;
        }
        par::Mpi_Alltoallv_dense<char>(&send_buff[0], &sendSz[0], &sendOff[0],
                                       &recv_buff[0], &recvSz[0], &recvOff[0], comm);
      }

      // Build output data.
      {
        data_.Resize(recv_size*data_dim/sizeof(T));
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          size_t src_indx=i*data_dim;
          size_t trg_indx=psorted[i].data*data_dim;
          for(size_t j=0;j<data_dim;j++)
            data[trg_indx+j]=recv_buff[src_indx+j];
        }
      }

      return 0;
    }

  template<typename T>
    int ScatterReverse(Vector<T>& data_, const Vector<size_t>& scatter_index, const MPI_Comm& comm, size_t loc_size){
      typedef SortPair<size_t,size_t> Pair_t;

      int npes, rank;
      MPI_Comm_size(comm, &npes);
      MPI_Comm_rank(comm, &rank);
      long long npesLong = npes;

      size_t data_dim=0;
      long long send_size=0;
      long long recv_size=0;
      {
        send_size=scatter_index.Dim();
        recv_size=loc_size;

        long long glb_size[3]={0,0};
        long long loc_size[3]={data_.Dim()*sizeof(T), send_size, recv_size};
        MPI_Allreduce(&loc_size, &glb_size, 3, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
        if(glb_size[0]==0 || glb_size[1]==0) return 0; //Nothing to be done.
        data_dim=glb_size[0]/glb_size[1];
        assert(glb_size[0]==data_dim*glb_size[1]);

        if(glb_size[1]!=glb_size[2]){
          recv_size=(((rank+1)*glb_size[1])/npesLong)-
                    (( rank   *glb_size[1])/npesLong);
        }
      }

      Vector<char> recv_buff(recv_size*data_dim);
      Vector<char> send_buff(send_size*data_dim);

      // Global data size.
      Vector<long long> glb_scan(npesLong);
      {
        long long glb_rank=0;
        MPI_Scan(&recv_size, &glb_rank, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
        glb_rank-=recv_size;

        MPI_Allgather(&glb_rank   , 1, par::Mpi_datatype<long long>::value(),
                      &glb_scan[0], 1, par::Mpi_datatype<long long>::value(), comm);
      }

      // Sort scatter_index.
      Vector<Pair_t> psorted(send_size);
      {
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          psorted[i].key=scatter_index[i];
          psorted[i].data=i;
        }
        omp_par::merge_sort(&psorted[0], &psorted[0]+send_size);
      }

      // Exchange send, recv indices.
      Vector<size_t> recv_indx(recv_size);
      Vector<size_t> send_indx(send_size);
      Vector<int> sendSz(npesLong);
      Vector<int> sendOff(npesLong);
      Vector<int> recvSz(npesLong);
      Vector<int> recvOff(npesLong);
      {
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          send_indx[i]=psorted[i].key;
        }

        #pragma omp parallel for
        for(size_t i=0;i<npesLong;i++){
          size_t start=              std::lower_bound(&send_indx[0], &send_indx[0]+send_size, glb_scan[  i])-&send_indx[0];
          size_t end  =(i+1<npesLong?std::lower_bound(&send_indx[0], &send_indx[0]+send_size, glb_scan[i+1])-&send_indx[0]:send_size);
          sendSz[i]=end-start;
          sendOff[i]=start;
        }

        MPI_Alltoall(&sendSz[0], 1, par::Mpi_datatype<int>::value(),
                     &recvSz[0], 1, par::Mpi_datatype<int>::value(), comm);
        recvOff[0] = 0; omp_par::scan(&recvSz[0],&recvOff[0],npesLong);
        assert(recvOff[npesLong-1]+recvSz[npesLong-1]==recv_size);

        par::Mpi_Alltoallv_dense<size_t>(&send_indx[0], &sendSz[0], &sendOff[0],
                                         &recv_indx[0], &recvSz[0], &recvOff[0], comm);
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          assert(recv_indx[i]>=glb_scan[rank]);
          recv_indx[i]-=glb_scan[rank];
          assert(recv_indx[i]<recv_size);
        }
      }

      // Prepare send buffer
      {
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<send_size;i++){
          size_t src_indx=psorted[i].data*data_dim;
          size_t trg_indx=i*data_dim;
          for(size_t j=0;j<data_dim;j++)
            send_buff[trg_indx+j]=data[src_indx+j];
        }
      }

      // All2Allv
      {
        #pragma omp parallel for
        for(size_t i=0;i<npesLong;i++){
          sendSz [i]*=data_dim;
          sendOff[i]*=data_dim;
          recvSz [i]*=data_dim;
          recvOff[i]*=data_dim;
        }
        par::Mpi_Alltoallv_dense<char>(&send_buff[0], &sendSz[0], &sendOff[0],
                                       &recv_buff[0], &recvSz[0], &recvOff[0], comm);
      }

      // Build output data.
      {
        data_.Resize(recv_size*data_dim/sizeof(T));
        char* data=(char*)&data_[0];
        #pragma omp parallel for
        for(size_t i=0;i<recv_size;i++){
          size_t src_indx=i*data_dim;
          size_t trg_indx=recv_indx[i]*data_dim;
          for(size_t j=0;j<data_dim;j++)
            data[trg_indx+j]=recv_buff[src_indx+j];
        }
      }

      return 0;
    }

}//end namespace
}//end namespace
