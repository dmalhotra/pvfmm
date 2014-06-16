/**
 * \file fmm_tree.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class FMM_Tree.
 */

#include <assert.h>
#include <fmm_node.hpp>
#include <profile.hpp>

namespace pvfmm{

template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::Initialize(typename FMM_Node_t::NodeData* init_data) {
  Profile::Tic("InitTree",this->Comm(),true);{

  //Build octree from points.
  MPI_Tree<FMM_Node_t>::Initialize(init_data);

  Profile::Tic("InitFMMData",this->Comm(),true,5);
  { //Initialize FMM data.
    std::vector<FMM_Node_t*>& nodes=this->GetNodeList();
    #pragma omp parallel for
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->FMMData()==NULL) nodes[i]->FMMData()=new typename FMM_Mat_t::FMMData;
    }
  }
  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::InitFMM_Tree(bool refine, BoundaryType bndry_) {
  Profile::Tic("InitFMM_Tree",this->Comm(),true);{

  interac_list.Initialize(this->Dim());
  bndry=bndry_;

  if(refine){
    //RefineTree
    Profile::Tic("RefineTree",this->Comm(),true,2);
    this->RefineTree();
    Profile::Toc();
  }

  //2:1 Balancing
  Profile::Tic("2:1Balance",this->Comm(),true,2);
  this->Balance21(bndry);
  Profile::Toc();

  //Redistribute nodes.
  Profile::Tic("Redistribute",this->Comm(),true,3);
  this->RedistNodes();
  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::SetupFMM(FMM_Mat_t* fmm_mat_) {
  Profile::Tic("SetupFMM",this->Comm(),true);{
  bool device=true;

  #ifdef __INTEL_OFFLOAD
  Profile::Tic("InitLocks",this->Comm(),false,3);
  MIC_Lock::init();
  Profile::Toc();
  #endif

  int omp_p=omp_get_max_threads();
  fmm_mat=fmm_mat_;

  //Construct LET
  Profile::Tic("ConstructLET",this->Comm(),false,2);
  this->ConstructLET(bndry);
  Profile::Toc();

  //Set Colleagues (Needed to build U, V, W and X lists.)
  Profile::Tic("SetColleagues",this->Comm(),false,3);
  this->SetColleagues(bndry);
  Profile::Toc();

  Profile::Tic("BuildLists",this->Comm(),false,3);
  BuildInteracLists();
  Profile::Toc();

  Profile::Tic("CollectNodeData",this->Comm(),false,3);
  //Build node list.
  FMM_Node_t* n=dynamic_cast<FMM_Node_t*>(this->PostorderFirst());
  std::vector<FMM_Node_t*> all_nodes;
  while(n!=NULL){
    all_nodes.push_back(n);
    n=static_cast<FMM_Node_t*>(this->PostorderNxt(n));
  }
  //Collect node data into continuous array.
  std::vector<Vector<FMM_Node_t*> > node_lists; // TODO: Remove this parameter, not really needed
  fmm_mat->CollectNodeData(all_nodes, node_data_buff, node_lists);
  Profile::Toc();

  setup_data.clear();
  precomp_lst.clear();
  setup_data.resize(8*MAX_DEPTH);
  precomp_lst.resize(8);

  Profile::Tic("UListSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*0].precomp_data=&precomp_lst[0];
    fmm_mat->U_ListSetup(setup_data[i+MAX_DEPTH*0],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, device);
  }
  Profile::Toc();
  Profile::Tic("WListSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*1].precomp_data=&precomp_lst[1];
    fmm_mat->W_ListSetup(setup_data[i+MAX_DEPTH*1],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, device);
  }
  Profile::Toc();
  Profile::Tic("XListSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*2].precomp_data=&precomp_lst[2];
    fmm_mat->X_ListSetup(setup_data[i+MAX_DEPTH*2],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, device);
  }
  Profile::Toc();
  Profile::Tic("VListSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*3].precomp_data=&precomp_lst[3];
    fmm_mat->V_ListSetup(setup_data[i+MAX_DEPTH*3],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, /*device*/ false);
  }
  Profile::Toc();
  Profile::Tic("D2DSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*4].precomp_data=&precomp_lst[4];
    fmm_mat->Down2DownSetup(setup_data[i+MAX_DEPTH*4],node_data_buff,node_lists,i, /*device*/ false);
  }
  Profile::Toc();
  Profile::Tic("D2TSetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*5].precomp_data=&precomp_lst[5];
    fmm_mat->Down2TargetSetup(setup_data[i+MAX_DEPTH*5],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, /*device*/ false);
  }
  Profile::Toc();

  Profile::Tic("S2USetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*6].precomp_data=&precomp_lst[6];
    fmm_mat->Source2UpSetup(setup_data[i+MAX_DEPTH*6],node_data_buff,node_lists,fmm_mat->Homogen()?(i==0?-1:MAX_DEPTH+1):i, /*device*/ false);
  }
  Profile::Toc();
  Profile::Tic("U2USetup",this->Comm(),false,3);
  for(size_t i=0;i<MAX_DEPTH;i++){
    setup_data[i+MAX_DEPTH*7].precomp_data=&precomp_lst[7];
    fmm_mat->Up2UpSetup(setup_data[i+MAX_DEPTH*7],node_data_buff,node_lists,i, /*device*/ false);
  }
  Profile::Toc();

  #ifdef __INTEL_OFFLOAD
  int wait_lock_idx=-1;
  wait_lock_idx=MIC_Lock::curr_lock();
  #pragma offload target(mic:0)
  {MIC_Lock::wait_lock(wait_lock_idx);}
  #endif

  }Profile::Toc();
}

template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::ClearFMMData() {
  Profile::Tic("ClearFMMData",this->Comm(),true);{

  bool device=true;
  if(setup_data[0+MAX_DEPTH*1]. input_data!=NULL) setup_data[0+MAX_DEPTH*1]. input_data->SetZero();
  if(setup_data[0+MAX_DEPTH*2].output_data!=NULL) setup_data[0+MAX_DEPTH*2].output_data->SetZero();
  if(setup_data[0+MAX_DEPTH*0].output_data!=NULL) setup_data[0+MAX_DEPTH*0].output_data->SetZero();

  if(device){ // Host2Device
    if(setup_data[0+MAX_DEPTH*1]. input_data!=NULL) setup_data[0+MAX_DEPTH*1]. input_data->AllocDevice(true);
    if(setup_data[0+MAX_DEPTH*2].output_data!=NULL) setup_data[0+MAX_DEPTH*2].output_data->AllocDevice(true);
    if(setup_data[0+MAX_DEPTH*0].output_data!=NULL) setup_data[0+MAX_DEPTH*0].output_data->AllocDevice(true);

    #ifdef __INTEL_OFFLOAD
    if(!fmm_mat->Homogen()){ // Wait
      int wait_lock_idx=-1;
      wait_lock_idx=MIC_Lock::curr_lock();
      #pragma offload target(mic:0)
      {MIC_Lock::wait_lock(wait_lock_idx);}
    }
    MIC_Lock::init();
    #endif
  }

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::RunFMM() {
  Profile::Tic("RunFMM",this->Comm(),true);{

  //Upward Pass
  Profile::Tic("UpwardPass",this->Comm(),false,2);
  UpwardPass();
  Profile::Toc();

  //Multipole Reduce Broadcast.
  Profile::Tic("ReduceBcast",this->Comm(),true,2);
  MultipoleReduceBcast();
  Profile::Toc();

  //Local 2:1 Balancing.
  //This can cause load imbalance, always use global 2:1 balance instead.
  //Profile::Tic("2:1Balance(local)",this->Comm(),false,3);
  //this->Balance21_local(bndry);
  //UpwardPass(true);
  //Profile::Toc();

  //Downward Pass
  Profile::Tic("DownwardPass",this->Comm(),true,2);
  DownwardPass();
  Profile::Toc();

  }Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::UpwardPass() {
  bool device=true;

  //Upward Pass (initialize all leaf nodes)
  Profile::Tic("S2U",this->Comm(),false,5);
  for(int i=0; i<(fmm_mat->Homogen()?1:MAX_DEPTH); i++){ // Source2Up
    if(!fmm_mat->Homogen()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*6],/*device*/ false);
    fmm_mat->Source2Up(setup_data[i+MAX_DEPTH*6]);
  }
  Profile::Toc();

  //Upward Pass (level by level)
  Profile::Tic("U2U",this->Comm(),false,5);
  for(int i=MAX_DEPTH-1; i>=0; i--){ // Up2Up
    if(!fmm_mat->Homogen()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*7],/*device*/ false);
    fmm_mat->Up2Up(setup_data[i+MAX_DEPTH*7]);
  }
  Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::BuildInteracLists() {
  std::vector<FMM_Node_t*>& n_list=this->GetNodeList();

  // Build interaction lists.
  int omp_p=omp_get_max_threads();
  {
    size_t k=n_list.size();
    #pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      size_t a=(k*j)/omp_p;
      size_t b=(k*(j+1))/omp_p;
      for(size_t i=a;i<b;i++){
        FMM_Node_t* n=n_list[i];
        n->interac_list.resize(Type_Count);
        n->interac_list[S2U_Type]=interac_list.BuildList(n,S2U_Type);
        n->interac_list[U2U_Type]=interac_list.BuildList(n,U2U_Type);
        n->interac_list[D2D_Type]=interac_list.BuildList(n,D2D_Type);
        n->interac_list[D2T_Type]=interac_list.BuildList(n,D2T_Type);
        n->interac_list[U0_Type]=interac_list.BuildList(n,U0_Type);
        n->interac_list[U1_Type]=interac_list.BuildList(n,U1_Type);
        n->interac_list[U2_Type]=interac_list.BuildList(n,U2_Type);
        n->interac_list[V_Type]=interac_list.BuildList(n,V_Type);
        n->interac_list[V1_Type]=interac_list.BuildList(n,V1_Type);
        n->interac_list[W_Type]=interac_list.BuildList(n,W_Type);
        n->interac_list[X_Type]=interac_list.BuildList(n,X_Type);
      }
    }
  }
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::MultipoleReduceBcast() {
  int num_p,rank;
  MPI_Comm_size(*this->Comm(),&num_p);
  MPI_Comm_rank(*this->Comm(),&rank );
  if(num_p==1) return;

  Profile::Tic("Reduce",this->Comm(),true,3);
  std::vector<MortonId> mins=this->GetMins();

  size_t bit_mask=1;
  size_t max_child=(1UL<<this->Dim());

  //Initialize initial send nodes.
  std::vector<FMM_Node_t*> send_nodes[2];

  //Initialize send_node[0]
  FMM_Node_t* tmp_node=static_cast<FMM_Node_t*>(this->RootNode());
  assert(!tmp_node->IsGhost());
  while(!tmp_node->IsLeaf()){
    FMM_Node_t* tmp_node_=NULL;
    for(size_t i=0;i<max_child;i++){
      tmp_node_=static_cast<FMM_Node_t*>(tmp_node->Child(i));
      if(tmp_node_!=NULL) if(!tmp_node_->IsGhost()) break;
    }
    tmp_node=tmp_node_; assert(tmp_node!=NULL);
  }
  int n[2];
  n[0]=tmp_node->Depth()+1;
  send_nodes[0].resize(n[0]);
  send_nodes[0][n[0]-1]=tmp_node;
  for(int i=n[0]-1;i>0;i--)
    send_nodes[0][i-1]=static_cast<FMM_Node_t*>(send_nodes[0][i]->Parent());

  //Initialize send_node[1]
  tmp_node=static_cast<FMM_Node_t*>(this->RootNode());
  while(!tmp_node->IsLeaf()){
    FMM_Node_t* tmp_node_=NULL;
    for(int i=max_child-1;i>=0;i--){
      tmp_node_=static_cast<FMM_Node_t*>(tmp_node->Child(i));
      if(tmp_node_!=NULL) if(!tmp_node_->IsGhost()) break;
    }
    tmp_node=tmp_node_; assert(tmp_node!=NULL);
  }
  n[1]=tmp_node->Depth()+1;
  send_nodes[1].resize(n[1]);
  send_nodes[1][n[1]-1]=tmp_node;
  for(int i=n[1]-1;i>0;i--)
    send_nodes[1][i-1]=static_cast<FMM_Node_t*>(send_nodes[1][i]->Parent());

  //Hypercube reduction.
  while(bit_mask<(size_t)num_p){
    size_t partner=rank^bit_mask; //Partner process id
    int merge_indx=((bit_mask & rank)==0?1:0);
    bit_mask=bit_mask<<1;
    //if(rank >= num_p - (num_p % bit_mask)) break;

    //Initialize send data.
    size_t s_node_cnt[2]={send_nodes[0].size(),send_nodes[1].size()};
    int send_size=2*sizeof(size_t)+(s_node_cnt[0]+s_node_cnt[1])*sizeof(MortonId);
    std::vector<PackedData> send_data(s_node_cnt[0]+s_node_cnt[1]);

    size_t s_iter=0;
    for(int i=0;i<2;i++)
    for(size_t j=0;j<s_node_cnt[i];j++){
      assert(send_nodes[i][j]!=NULL);
      send_data[s_iter]=send_nodes[i][j]->PackMultipole();
      send_size+=send_data[s_iter].length+sizeof(size_t);
      s_iter++;
    }

    char* send_buff=new char[send_size];
    char* buff_iter=send_buff;
    ((size_t*)buff_iter)[0]=s_node_cnt[0];
    ((size_t*)buff_iter)[1]=s_node_cnt[1];
    buff_iter+=2*sizeof(size_t);

    s_iter=0;
    for(int i=0;i<2;i++)
    for(size_t j=0;j<s_node_cnt[i];j++){
      ((MortonId*)buff_iter)[0]=send_nodes[i][j]->GetMortonId();
      buff_iter+=sizeof(MortonId);

      ((size_t*)buff_iter)[0]=send_data[s_iter].length;
      buff_iter+=sizeof(size_t);

      mem::memcopy((void*)buff_iter,send_data[s_iter].data,send_data[s_iter].length);
      buff_iter+=send_data[s_iter].length;

      s_iter++;
    }

    //Exchange send and recv sizes
    int recv_size=0;
    MPI_Status status;
    char* recv_buff=NULL;
    if(partner<(size_t)num_p){
      MPI_Sendrecv(&send_size,        1,  MPI_INT, partner, 0, &recv_size,         1,  MPI_INT, partner, 0, *this->Comm(), &status);
      recv_buff=new char[recv_size];
      MPI_Sendrecv(send_buff, send_size, MPI_BYTE, partner, 0,  recv_buff, recv_size, MPI_BYTE, partner, 0, *this->Comm(), &status);
    }

    //Need an extra broadcast for incomplete hypercubes.
    size_t p0_start=num_p - (num_p % (bit_mask   ));
    size_t p0_end  =num_p - (num_p % (bit_mask>>1));
    if(((size_t)rank >= p0_start) && ((size_t)num_p>p0_end) && ((size_t)rank < p0_end) ){
      size_t bit_mask0=1;
      size_t num_p0=p0_end-p0_start;
      while( bit_mask0 < num_p0 ){
        if( (bit_mask0<<1) > (num_p - p0_end) ){
          size_t partner0=rank^bit_mask0;
          if( rank-p0_start < bit_mask0 ){
            //Send
            MPI_Send(&recv_size,         1, MPI_INT , partner0, 0, *this->Comm());
            MPI_Send( recv_buff, recv_size, MPI_BYTE, partner0, 0, *this->Comm());
          }else if( rank-p0_start < (bit_mask0<<1) ){
            //Receive
            if(recv_size>0) delete[] recv_buff;
            MPI_Recv(&recv_size,         1, MPI_INT , partner0, 0, *this->Comm(), &status);
            recv_buff=new char[recv_size];
            MPI_Recv( recv_buff, recv_size, MPI_BYTE, partner0, 0, *this->Comm(), &status);
          }
        }
        bit_mask0=bit_mask0<<1;
      }
    }

    //Construct nodes from received data.
    if(recv_size>0){
      buff_iter=recv_buff;
      size_t r_node_cnt[2]={((size_t*)buff_iter)[0],((size_t*)buff_iter)[1]};
      buff_iter+=2*sizeof(size_t);
      std::vector<MortonId> r_mid[2];
      r_mid[0].resize(r_node_cnt[0]);
      r_mid[1].resize(r_node_cnt[1]);
      std::vector<FMM_Node_t*> recv_nodes[2];
      recv_nodes[0].resize(r_node_cnt[0]);
      recv_nodes[1].resize(r_node_cnt[1]);
      std::vector<PackedData> recv_data[2];
      recv_data[0].resize(r_node_cnt[0]);
      recv_data[1].resize(r_node_cnt[1]);
      for(int i=0;i<2;i++)
      for(size_t j=0;j<r_node_cnt[i];j++){
        r_mid[i][j]=((MortonId*)buff_iter)[0];
        buff_iter+=sizeof(MortonId);

        recv_data[i][j].length=((size_t*)buff_iter)[0];
        buff_iter+=sizeof(size_t);

        recv_data[i][j].data=(void*)buff_iter;
        buff_iter+=recv_data[i][j].length;
      }

      // Add multipole expansion to existing nodes.
      for(size_t i=0;i<r_node_cnt[1-merge_indx];i++){
        if(i<send_nodes[merge_indx].size()){
          if(r_mid[1-merge_indx][i]==send_nodes[merge_indx][i]->GetMortonId()){
            send_nodes[merge_indx][i]->AddMultipole(recv_data[1-merge_indx][i]);
          }else break;
        }else break;
      }

      bool new_branch=false;
      for(size_t i=0;i<r_node_cnt[merge_indx];i++){
        if(i<send_nodes[merge_indx].size() && !new_branch){
          if(r_mid[merge_indx][i]==send_nodes[merge_indx][i]->GetMortonId()){
            recv_nodes[merge_indx][i]=send_nodes[merge_indx][i];
          }else{
            new_branch=true;
            size_t n_=(i<(size_t)n[merge_indx]?n[merge_indx]:i);
            for(size_t j=n_;j<send_nodes[merge_indx].size();j++)
              delete send_nodes[merge_indx][j];
            if(i<(size_t)n[merge_indx]) n[merge_indx]=i;
          }
        }
        if(i>=send_nodes[merge_indx].size() || new_branch){
            recv_nodes[merge_indx][i]=static_cast<FMM_Node_t*>(this->NewNode());
            recv_nodes[merge_indx][i]->SetCoord(r_mid[merge_indx][i]);
            recv_nodes[merge_indx][i]->InitMultipole(recv_data[merge_indx][i]);
        }
      }
      send_nodes[merge_indx]=recv_nodes[merge_indx];
    }
    delete[] send_buff;
    delete[] recv_buff;
  }

  for(int i=0;i<2;i++)
  for(size_t j=n[i];j<send_nodes[i].size();j++)
    delete send_nodes[i][j];
  Profile::Toc();

  //Now Broadcast nodes to build LET.
  Profile::Tic("Broadcast",this->Comm(),true,4);
  this->ConstructLET(bndry);
  Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::DownwardPass() {
  bool device=true;

  Profile::Tic("Setup",this->Comm(),true,3);
  std::vector<FMM_Node_t*> leaf_nodes;
  int max_depth=0;
  { // Build leaf node list
    int max_depth_loc=0;
    std::vector<FMM_Node_t*>& nodes=this->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      FMM_Node_t* n=nodes[i];
      if(!n->IsGhost() && n->IsLeaf()) leaf_nodes.push_back(n);
      if(n->Depth()>max_depth_loc) max_depth_loc=n->Depth();
    }
    MPI_Allreduce(&max_depth_loc, &max_depth, 1, MPI_INT, MPI_MAX, *this->Comm());
  }
  Profile::Toc();

  #ifdef __INTEL_OFFLOAD
  if(device){ // Host2Device:Src
    Profile::Tic("Host2Device:Src",this->Comm(),false,5);
    if(setup_data[0+MAX_DEPTH*2]. coord_data!=NULL) setup_data[0+MAX_DEPTH*2]. coord_data->AllocDevice(true);
    if(setup_data[0+MAX_DEPTH*2]. input_data!=NULL) setup_data[0+MAX_DEPTH*2]. input_data->AllocDevice(true);
    Profile::Toc();
  }
  #endif

  if(bndry==Periodic){ //Add contribution from periodic infinite tiling.
    Profile::Tic("BoundaryCondition",this->Comm(),false,5);
    fmm_mat->PeriodicBC(dynamic_cast<FMM_Node_t*>(this->RootNode()));
    Profile::Toc();
  }

  for(size_t i=0; i<=(fmm_mat->Homogen()?0:max_depth); i++){ // U,V,W,X-lists

    if(!fmm_mat->Homogen()){ // Precomp
      std::stringstream level_str;
      level_str<<"Level-"<<std::setfill('0')<<std::setw(2)<<i<<"\0";
      Profile::Tic(level_str.str().c_str(),this->Comm(),false,5);

      Profile::Tic("Precomp",this->Comm(),false,5);
      {// Precomp U
        Profile::Tic("Precomp-U",this->Comm(),false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*0],device);
        Profile::Toc();
      }
      {// Precomp W
        Profile::Tic("Precomp-W",this->Comm(),false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*1],device);
        Profile::Toc();
      }
      {// Precomp X
        Profile::Tic("Precomp-X",this->Comm(),false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*2],device);
        Profile::Toc();
      }
      {// Precomp V
        Profile::Tic("Precomp-V",this->Comm(),false,10);
        fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*3], /*device*/ false);
        Profile::Toc();
      }
      Profile::Toc();
    }

    {// X-List
      Profile::Tic("X-List",this->Comm(),false,5);
      fmm_mat->X_List(setup_data[i+MAX_DEPTH*2], device);
      Profile::Toc();
    }

    #ifdef __INTEL_OFFLOAD
    if(i==0 && device){ // Host2Device:Mult
      Profile::Tic("Host2Device:Mult",this->Comm(),false,5);
      if(setup_data[0+MAX_DEPTH*1]. input_data!=NULL) setup_data[0+MAX_DEPTH*1]. input_data->AllocDevice(true);
      Profile::Toc();
    }

    if(device) if(i==(fmm_mat->Homogen()?0:max_depth)){ // Device2Host: LocalExp
      Profile::Tic("Device2Host:LocExp",this->Comm(),false,5);
      if(setup_data[0+MAX_DEPTH*2].output_data!=NULL){
        Matrix<Real_t>& output_data=*setup_data[0+MAX_DEPTH*2].output_data;
        assert(fmm_mat->dev_buffer.Dim()>=output_data.Dim(0)*output_data.Dim(1));
        output_data.Device2Host((Real_t*)&fmm_mat->dev_buffer[0]);
      }
      Profile::Toc();
    }
    #endif

    {// W-List
      Profile::Tic("W-List",this->Comm(),false,5);
      fmm_mat->W_List(setup_data[i+MAX_DEPTH*1], device);
      Profile::Toc();
    }

    {// U-List
      Profile::Tic("U-List",this->Comm(),false,5);
      fmm_mat->U_List(setup_data[i+MAX_DEPTH*0], device);
      Profile::Toc();
    }

    {// V-List
      Profile::Tic("V-List",this->Comm(),false,5);
      fmm_mat->V_List(setup_data[i+MAX_DEPTH*3], /*device*/ false);
      Profile::Toc();
    }

    if(!fmm_mat->Homogen()){ // Wait
      #ifdef __INTEL_OFFLOAD
      int wait_lock_idx=-1;
      if(device) wait_lock_idx=MIC_Lock::curr_lock();
      #pragma offload if(device) target(mic:0)
      {if(device) MIC_Lock::wait_lock(wait_lock_idx);}
      #endif
      Profile::Toc();
    }
  }

  #ifdef __INTEL_OFFLOAD
  Profile::Tic("D2H_Wait:LocExp",this->Comm(),false,5);
  if(device) if(setup_data[0+MAX_DEPTH*2].output_data!=NULL){
    Real_t* dev_ptr=(Real_t*)&fmm_mat->dev_buffer[0];
    Matrix<Real_t>& output_data=*setup_data[0+MAX_DEPTH*2].output_data;
    size_t n=output_data.Dim(0)*output_data.Dim(1);
    Real_t* host_ptr=output_data[0];
    output_data.Device2HostWait();

    #pragma omp parallel for
    for(size_t i=0;i<n;i++){
      host_ptr[i]+=dev_ptr[i];
    }
  }
  Profile::Toc();

  Profile::Tic("Device2Host:Trg",this->Comm(),false,5);
  if(device) if(setup_data[0+MAX_DEPTH*0].output_data!=NULL){ // Device2Host: Target
    Matrix<Real_t>& output_data=*setup_data[0+MAX_DEPTH*0].output_data;
    assert(fmm_mat->dev_buffer.Dim()>=output_data.Dim(0)*output_data.Dim(1));
    output_data.Device2Host((Real_t*)&fmm_mat->dev_buffer[0]);
  }
  Profile::Toc();
  #endif

  Profile::Tic("D2D",this->Comm(),false,5);
  for(size_t i=0; i<=max_depth; i++){ // Down2Down
    if(!fmm_mat->Homogen()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*4],/*device*/ false);
    fmm_mat->Down2Down(setup_data[i+MAX_DEPTH*4]);
  }
  Profile::Toc();

  Profile::Tic("D2T",this->Comm(),false,5);
  for(int i=0; i<=(fmm_mat->Homogen()?0:max_depth); i++){ // Down2Target
    if(!fmm_mat->Homogen()) fmm_mat->SetupPrecomp(setup_data[i+MAX_DEPTH*5],/*device*/ false);
    fmm_mat->Down2Target(setup_data[i+MAX_DEPTH*5]);
  }
  Profile::Toc();

  #ifdef __INTEL_OFFLOAD
  Profile::Tic("D2H_Wait:Trg",this->Comm(),false,5);
  if(device) if(setup_data[0+MAX_DEPTH*0].output_data!=NULL){
    Real_t* dev_ptr=(Real_t*)&fmm_mat->dev_buffer[0];
    Matrix<Real_t>& output_data=*setup_data[0+MAX_DEPTH*0].output_data;
    size_t n=output_data.Dim(0)*output_data.Dim(1);
    Real_t* host_ptr=output_data[0];
    output_data.Device2HostWait();

    #pragma omp parallel for
    for(size_t i=0;i<n;i++){
      host_ptr[i]+=dev_ptr[i];
    }
  }
  Profile::Toc();
  #endif

  Profile::Tic("PostProc",this->Comm(),false,5);
  fmm_mat->PostProcessing(leaf_nodes);
  Profile::Toc();
}


template <class FMM_Mat_t>
void FMM_Tree<FMM_Mat_t>::Copy_FMMOutput() {
  std::vector<FMM_Node_t*>& all_nodes=this->GetNodeList();
  int omp_p=omp_get_max_threads();

  // Copy output to the tree.
  {
    size_t k=all_nodes.size();
    #pragma omp parallel for
    for(int j=0;j<omp_p;j++){
      size_t a=(k*j)/omp_p;
      size_t b=(k*(j+1))/omp_p;
      fmm_mat->CopyOutput(&(all_nodes[a]),b-a);
    }
  }
}

}//end namespace
