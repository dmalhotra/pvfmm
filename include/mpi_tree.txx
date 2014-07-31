/**
 * \file mpi_tree.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class MPI_Tree.
 */

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <stdint.h>
#include <set>

#include <dtypes.h>
#include <ompUtils.h>
#include <parUtils.h>
#include <mem_utils.hpp>
#include <mpi_node.hpp>
#include <profile.hpp>

namespace pvfmm{

/**
 * @author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * @date 08 Feb 2011
 */
inline int p2oLocal(Vector<MortonId> & nodes, Vector<MortonId>& leaves,
    unsigned int maxNumPts, unsigned int maxDepth, bool complete) {
  assert(maxDepth<=MAX_DEPTH);

  std::vector<MortonId> leaves_lst;
  unsigned int init_size=leaves.Dim();
  unsigned int num_pts=nodes.Dim();

  MortonId curr_node=leaves[0];
  MortonId last_node=leaves[init_size-1].NextId();
  MortonId next_node;

  unsigned int curr_pt=0;
  unsigned int next_pt=curr_pt+maxNumPts;

  while(next_pt <= num_pts){
    next_node = curr_node.NextId();
    while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ){
      curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
      next_node = curr_node.NextId();
    }
    leaves_lst.push_back(curr_node);
    curr_node = next_node;

    unsigned int inc=maxNumPts;
    while(next_pt < num_pts && curr_node > nodes[next_pt]){
      // We have more than maxNumPts points per octant because the node can
      // not be refined any further.
      inc=inc<<1;
      next_pt+=inc;
      if(next_pt > num_pts){
        next_pt = num_pts;
        break;
      }
    }

    curr_pt = std::lower_bound(&nodes[0]+curr_pt,&nodes[0]+next_pt,curr_node,std::less<MortonId>())-&nodes[0];
    if(curr_pt >= num_pts) break;
    next_pt = curr_pt + maxNumPts;
    if(next_pt > num_pts) next_pt = num_pts;
  }
#ifndef NDEBUG
  for(size_t i=0;i<leaves_lst.size();i++){
    size_t a=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i],std::less<MortonId>())-&nodes[0];
    size_t b=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i].NextId(),std::less<MortonId>())-&nodes[0];
    assert(b-a<=maxNumPts || leaves_lst[i].GetDepth()==maxDepth-1);
    if(i==leaves_lst.size()-1) assert(b==nodes.Dim() && a<nodes.Dim());
    if(i==0) assert(a==0);
  }
#endif
  if(complete)
  while(curr_node<last_node){
    while( curr_node.NextId() > last_node && curr_node.GetDepth() < maxDepth-1 )
      curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
    leaves_lst.push_back(curr_node);
    curr_node = curr_node.NextId();
  }

  leaves=leaves_lst;
  return 0;
}

inline int points2Octree(const Vector<MortonId>& pt_mid, Vector<MortonId>& nodes,
          unsigned int maxDepth, unsigned int maxNumPts, const MPI_Comm& comm ) {

  int myrank, np;
  MPI_Comm_rank(comm, &myrank);
  MPI_Comm_size(comm, &np);

  // Sort morton id of points.
  Profile::Tic("SortMortonId", &comm, true, 5);
  Vector<MortonId> pt_sorted;
  //par::partitionW<MortonId>(pt_mid, NULL, comm);
  par::HyperQuickSort(pt_mid, pt_sorted, comm);
  size_t pt_cnt=pt_sorted.Dim();
  Profile::Toc();

  // Add last few points from next process, to get the boundary octant right.
  Profile::Tic("Comm", &comm, true, 5);
  {
    { // Adjust maxNumPts
      size_t glb_pt_cnt=0;
      MPI_Allreduce(&pt_cnt, &glb_pt_cnt, 1, par::Mpi_datatype<size_t>::value(), par::Mpi_datatype<size_t>::sum(), comm);
      if(glb_pt_cnt<maxNumPts*np) maxNumPts=glb_pt_cnt/np;
    }

    size_t recv_size=0;
    size_t send_size=(2*maxNumPts<pt_cnt?2*maxNumPts:pt_cnt);
    {
      MPI_Request recvRequest;
      MPI_Request sendRequest;
      MPI_Status statusWait;
      if(myrank < (np-1)) MPI_Irecv (&recv_size, 1, par::Mpi_datatype<size_t>::value(), myrank+1, 1, comm, &recvRequest);
      if(myrank >     0 ) MPI_Issend(&send_size, 1, par::Mpi_datatype<size_t>::value(), myrank-1, 1, comm, &sendRequest);
      if(myrank < (np-1)) MPI_Wait(&recvRequest, &statusWait);
      if(myrank >     0 ) MPI_Wait(&sendRequest, &statusWait); //This can be done later.
    }
    if(recv_size>0){// Resize pt_sorted.
      Vector<MortonId> pt_sorted_(pt_cnt+recv_size);
      mem::memcopy(&pt_sorted_[0], &pt_sorted[0], pt_cnt*sizeof(MortonId));
      pt_sorted.Swap(pt_sorted_);
    }
    {// Exchange data.
      MPI_Request recvRequest;
      MPI_Request sendRequest;
      MPI_Status statusWait;
      if(myrank < (np-1)) MPI_Irecv (&pt_sorted[0]+pt_cnt, recv_size, par::Mpi_datatype<MortonId>::value(), myrank+1, 1, comm, &recvRequest);
      if(myrank >     0 ) MPI_Issend(&pt_sorted[0]       , send_size, par::Mpi_datatype<MortonId>::value(), myrank-1, 1, comm, &sendRequest);
      if(myrank < (np-1)) MPI_Wait(&recvRequest, &statusWait);
      if(myrank >     0 ) MPI_Wait(&sendRequest, &statusWait); //This can be done later.
    }
  }
  Profile::Toc();

  // Construct local octree.
  Profile::Tic("p2o_local", &comm, false, 5);
  Vector<MortonId> nodes_local(1); nodes_local[0]=MortonId();
  p2oLocal(pt_sorted, nodes_local, maxNumPts, maxDepth, myrank==np-1);
  Profile::Toc();

  // Remove duplicate nodes on adjacent processors.
  Profile::Tic("RemoveDuplicates", &comm, true, 5);
  {
    size_t node_cnt=nodes_local.Dim();
    MortonId first_node;
    MortonId  last_node=nodes_local[node_cnt-1];
    { // Send last_node to next process and get first_node from previous process.
      MPI_Request recvRequest;
      MPI_Request sendRequest;
      MPI_Status statusWait;
      if(myrank < (np-1)) MPI_Issend(& last_node, 1, par::Mpi_datatype<MortonId>::value(), myrank+1, 1, comm, &recvRequest);
      if(myrank >     0 ) MPI_Irecv (&first_node, 1, par::Mpi_datatype<MortonId>::value(), myrank-1, 1, comm, &sendRequest);
      if(myrank < (np-1)) MPI_Wait(&recvRequest, &statusWait);
      if(myrank >     0 ) MPI_Wait(&sendRequest, &statusWait); //This can be done later.
    }

    size_t i=0;
    std::vector<MortonId> node_lst;
    if(myrank){
      while(i<node_cnt && nodes_local[i].getDFD(maxDepth)<first_node) i++; assert(i);
      last_node=nodes_local[i>0?i-1:0].NextId(); // Next MortonId in the tree after first_node.

      while(first_node<last_node){ // Complete nodes between first_node and last_node.
        while(first_node.isAncestor(last_node))
          first_node=first_node.getDFD(first_node.GetDepth()+1);
        if(first_node==last_node) break;
        node_lst.push_back(first_node);
        first_node=first_node.NextId();
      }
    }
    for(;i<node_cnt-(myrank==np-1?0:1);i++) node_lst.push_back(nodes_local[i]);
    nodes=node_lst;
  }
  Profile::Toc();

  // Repartition nodes.
  Profile::Tic("partitionW", &comm, false, 5);
  par::partitionW<MortonId>(nodes, NULL , comm);
  Profile::Toc();

  return 0;
}

template <class TreeNode>
void MPI_Tree<TreeNode>::Initialize(typename Node_t::NodeData* init_data){
  //Initialize root node.
  Profile::Tic("InitRoot",Comm(),false,3);
  Tree<TreeNode>::Initialize(init_data);
  TreeNode* rnode=this->RootNode();
  assert(this->dim==COORD_DIM);
  Profile::Toc();

  Profile::Tic("Points2Octee",Comm(),true,3);
  Vector<MortonId> lin_oct;
  { //Get the linear tree.
    // Compute MortonId from pt_coord.
    Vector<MortonId> pt_mid;
    Vector<Real_t>& pt_coord=rnode->pt_coord;
    size_t pt_cnt=pt_coord.Dim()/this->dim;
    pt_mid.Resize(pt_cnt);
    #pragma omp parallel for
    for(size_t i=0;i<pt_cnt;i++){
      pt_mid[i]=MortonId(pt_coord[i*COORD_DIM+0],pt_coord[i*COORD_DIM+1],pt_coord[i*COORD_DIM+2],this->max_depth);
    }

    //Get the linear tree.
    points2Octree(pt_mid,lin_oct,this->max_depth,init_data->max_pts,*Comm());
  }
  Profile::Toc();

  Profile::Tic("ScatterPoints",Comm(),true,3);
  { // Sort and partition point coordinates and values.
    std::vector<Vector<Real_t>*> coord_lst;
    std::vector<Vector<Real_t>*> value_lst;
    std::vector<Vector<size_t>*> scatter_lst;
    rnode->NodeDataVec(coord_lst, value_lst, scatter_lst);
    assert(coord_lst.size()==value_lst.size());
    assert(coord_lst.size()==scatter_lst.size());

    Vector<MortonId> pt_mid;
    Vector<size_t> scatter_index;
    for(size_t i=0;i<coord_lst.size();i++){
      if(!coord_lst[i]) continue;
      Vector<Real_t>& pt_coord=*coord_lst[i];
      { // Compute MortonId from pt_coord.
        size_t pt_cnt=pt_coord.Dim()/this->dim;
        pt_mid.Resize(pt_cnt);
        #pragma omp parallel for
        for(size_t i=0;i<pt_cnt;i++){
          pt_mid[i]=MortonId(pt_coord[i*COORD_DIM+0],pt_coord[i*COORD_DIM+1],pt_coord[i*COORD_DIM+2],this->max_depth);
        }
      }
      par::SortScatterIndex(pt_mid  , scatter_index, comm, &lin_oct[0]);
      par::ScatterForward  (pt_coord, scatter_index, comm);
      if(value_lst[i]!=NULL){
        Vector<Real_t>& pt_value=*value_lst[i];
        par::ScatterForward(pt_value, scatter_index, comm);
      }
      if(scatter_lst[i]!=NULL){
        Vector<size_t>& pt_scatter=*scatter_lst[i];
        pt_scatter=scatter_index;
      }
    }
  }
  Profile::Toc();

  //Initialize the pointer based tree from the linear tree.
  Profile::Tic("PointerTree",Comm(),false,3);
  { // Construct the pointer tree from lin_oct
    int omp_p=omp_get_max_threads();

    // Partition nodes between threads
    rnode->SetGhost(false);
    for(int i=0;i<omp_p;i++){
      size_t idx=(lin_oct.Dim()*i)/omp_p;
      Node_t* n=FindNode(lin_oct[idx], true);
      assert(n->GetMortonId()==lin_oct[idx]);
      UNUSED(n);
    }

    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t a=(lin_oct.Dim()* i   )/omp_p;
      size_t b=(lin_oct.Dim()*(i+1))/omp_p;

      size_t idx=a;
      Node_t* n=FindNode(lin_oct[idx], false);
      if(a==0) n=rnode;
      while(n!=NULL && (idx<b || i==omp_p-1)){
        n->SetGhost(false);
        MortonId dn=n->GetMortonId();
        if(idx<b && dn.isAncestor(lin_oct[idx])){
          if(n->IsLeaf()) n->Subdivide();
        }else if(idx<b && dn==lin_oct[idx]){
          if(!n->IsLeaf()) n->Truncate();
          assert(n->IsLeaf());
          idx++;
        }else{
          n->Truncate();
          n->SetGhost(true);
        }
        n=this->PreorderNxt(n);
      }
      assert(idx==b);
    }
  }
  Profile::Toc();

#ifndef NDEBUG
  CheckTree();
#endif
}


template <class TreeNode>
void MPI_Tree<TreeNode>::CoarsenTree(){
  int myrank;
  MPI_Comm_rank(*Comm(),&myrank);

  //Redistribute.
  {
    Node_t* n=this->PostorderFirst();
    while(n){
      if(n->IsLeaf() && !n->IsGhost()) break;
      n=this->PostorderNxt(n);
    }
    while(myrank){
      Node_t* n_parent=(Node_t*)n->Parent();
      Node_t* n_      =         n_parent;
      while(n_ && !n_->IsLeaf()){
        n_=this->PostorderNxt(n_);
        if(!n_) break;
      }
      if(!n_ || n_->IsGhost()) break;
      if(n->Depth()<=n_->Depth()) break;
      if(n_->Depth()<=1) break;
      n=n_;
    }
    MortonId loc_min=n->GetMortonId();
    RedistNodes(&loc_min);
  }

  //Truncate ghost nodes and build node list
  std::vector<Node_t*> leaf_nodes;
  {
    Node_t* n=this->PostorderFirst();
    while(n!=NULL){
      if(n->IsLeaf() && !n->IsGhost()) break;
      n->Truncate();
      n->SetGhost(true);
      n->ClearData();
      n=this->PostorderNxt(n);
    }
    while(n!=NULL){
      if(n->IsLeaf() && n->IsGhost()) break;
      if(n->IsLeaf()) leaf_nodes.push_back(n);
      n=this->PreorderNxt(n);
    }
    while(n!=NULL){
      n->Truncate();
      n->SetGhost(true);
      n->ClearData();
      n=this->PreorderNxt(n);
    }
  }
  size_t node_cnt=leaf_nodes.size();

  //Partition nodes between OpenMP threads.
  int omp_p=omp_get_max_threads();
  std::vector<MortonId> mid(omp_p);
  std::vector<MortonId> split_key(omp_p);
  #pragma omp parallel for
  for(int i=0;i<omp_p;i++){
    mid[i]=leaf_nodes[(i*node_cnt)/omp_p]->GetMortonId();
  }

  //Coarsen Tree.
  #pragma omp parallel for
  for(int i=0;i<omp_p;i++){
    Node_t* n_=leaf_nodes[i*node_cnt/omp_p];
    if(i*node_cnt/omp_p<(i+1)*node_cnt/omp_p)
    while(n_!=NULL){
      MortonId n_mid=n_->GetMortonId();
      if(!n_->IsLeaf() && !n_mid.isAncestor(mid[i].getDFD()))
        if(i<omp_p-1? !n_mid.isAncestor(mid[i+1].getDFD()):true)
          if(!n_->SubdivCond()) n_->Truncate();
      if(i<omp_p-1? n_mid==mid[i+1]: false) break;
      n_=this->PostorderNxt(n_);
    }
  }

  //Truncate nodes along ancestors of splitters.
  for(int i=0;i<omp_p;i++){
    Node_t* n_=FindNode(mid[i], false, this->RootNode());
    while(n_->Depth()>0){
      n_=(Node_t*)n_->Parent();
      if(!n_->SubdivCond()) n_->Truncate();
      else break;
    }
  }
}


template <class TreeNode>
void MPI_Tree<TreeNode>::RefineTree(){
  int np, myrank;
  MPI_Comm_size(*Comm(),&np);
  MPI_Comm_rank(*Comm(),&myrank);
  int omp_p=omp_get_max_threads();
  int n_child=1UL<<this->Dim();

  //Coarsen tree.
  MPI_Tree<TreeNode>::CoarsenTree();

  //Build node list.
  std::vector<Node_t*> leaf_nodes;
  {
    Node_t* n=this->PostorderFirst();
    while(n!=NULL){
      if(n->IsLeaf() && !n->IsGhost())
        leaf_nodes.push_back(n);
      n=this->PostorderNxt(n);
    }
  }
  size_t tree_node_cnt=leaf_nodes.size();

  //Adaptive subdivision of leaf nodes with load balancing.
  for(int l=0;l<this->max_depth;l++){
    //Subdivide nodes.
    std::vector<std::vector<Node_t*> > leaf_nodes_(omp_p);
    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t a=(leaf_nodes.size()* i   )/omp_p;
      size_t b=(leaf_nodes.size()*(i+1))/omp_p;
      for(size_t j=a;j<b;j++){
        if(leaf_nodes[j]->IsLeaf() && !leaf_nodes[j]->IsGhost()){
          if(leaf_nodes[j]->SubdivCond()) leaf_nodes[j]->Subdivide();
          if(!leaf_nodes[j]->IsLeaf())
            for(int k=0;k<n_child;k++)
              leaf_nodes_[i].push_back((Node_t*)leaf_nodes[j]->Child(k));
        }
      }
    }
    for(int i=0;i<omp_p;i++)
      tree_node_cnt+=(leaf_nodes_[i].size()/n_child)*(n_child-1);

    //Determine load imbalance.
    size_t global_max, global_sum;
    MPI_Allreduce(&tree_node_cnt, &global_max, 1, par::Mpi_datatype<size_t>::value(), par::Mpi_datatype<size_t>::max(), *Comm());
    MPI_Allreduce(&tree_node_cnt, &global_sum, 1, par::Mpi_datatype<size_t>::value(), par::Mpi_datatype<size_t>::sum(), *Comm());

    //RedistNodes if needed.
    if(global_max*np>4*global_sum){
      #ifndef NDEBUG
      Profile::Tic("RedistNodes",Comm(),true,4);
      #endif
      RedistNodes();
      #ifndef NDEBUG
      Profile::Toc();
      #endif

      //Rebuild node list.
      leaf_nodes.clear();
      Node_t* n=this->PostorderFirst();
      while(n!=NULL){
        if(n->IsLeaf() && !n->IsGhost())
          leaf_nodes.push_back(n);
        n=this->PostorderNxt(n);
      }
      tree_node_cnt=leaf_nodes.size();
    }else{
      //Combine partial list of nodes.
      int node_cnt=0;
      for(int j=0;j<omp_p;j++)
        node_cnt+=leaf_nodes_[j].size();
      leaf_nodes.resize(node_cnt);
      #pragma omp parallel for
      for(int i=0;i<omp_p;i++){
        int offset=0;
        for(int j=0;j<i;j++)
          offset+=leaf_nodes_[j].size();
        for(size_t j=0;j<leaf_nodes_[i].size();j++)
          leaf_nodes[offset+j]=leaf_nodes_[i][j];
      }
    }
  }

  RedistNodes();
  MPI_Tree<TreeNode>::CoarsenTree();
  RedistNodes();
  MPI_Tree<TreeNode>::CoarsenTree();
  RedistNodes();
}


template <class TreeNode>
void MPI_Tree<TreeNode>::RedistNodes(MortonId* loc_min) {
  int np, myrank;
  MPI_Comm_size(*Comm(),&np);
  MPI_Comm_rank(*Comm(),&myrank);
  if(np==1)return;

  //Create a linear tree in dendro format.
  Node_t* curr_node=this->PreorderFirst();
  std::vector<MortonId> in;
  std::vector<Node_t*> node_lst;
  while(curr_node!=NULL){
    if(curr_node->IsLeaf() && !curr_node->IsGhost()){
      node_lst.push_back(curr_node);
      in.push_back(curr_node->GetMortonId());
    }
    curr_node=this->PreorderNxt(curr_node);
  }
  size_t leaf_cnt=in.size();

  //Get new mins.
  std::vector<MortonId> new_mins(np);
  if(loc_min==NULL){
    //Partition vector of MortonIds using par::partitionW
    std::vector<MortonId> in_=in;
    std::vector<long long> wts(in_.size());
    #pragma omp parallel for
    for(size_t i=0;i<wts.size();i++){
      wts[i]=node_lst[i]->NodeCost();
    }
    par::partitionW<MortonId>(in_,&wts[0],*Comm());
    MPI_Allgather(&in_[0]     , 1, par::Mpi_datatype<MortonId>::value(),
                  &new_mins[0], 1, par::Mpi_datatype<MortonId>::value(), *Comm());
  }else{
    MPI_Allgather(loc_min     , 1, par::Mpi_datatype<MortonId>::value(),
                  &new_mins[0], 1, par::Mpi_datatype<MortonId>::value(), *Comm());
  }

  //Now exchange nodes according to new mins
  std::vector<PackedData> data(leaf_cnt);
  std::vector<int> send_cnts; send_cnts.assign(np,0);
  std::vector<int> send_size; send_size.assign(np,0);

  size_t sbuff_size=0;
  int omp_p=omp_get_max_threads();
  #pragma omp parallel for reduction(+:sbuff_size)
  for(int i=0;i<omp_p;i++){
    size_t a=( i   *np)/omp_p;
    size_t b=((i+1)*np)/omp_p;
    if(b>a){
      size_t p_iter=a;
      size_t node_iter=std::lower_bound(&in[0], &in[in.size()], new_mins[a])-&in[0];
      for( ;node_iter<node_lst.size();node_iter++){
        while(p_iter+1u<(size_t)np? in[node_iter]>=new_mins[p_iter+1]: false) p_iter++;
        if(p_iter>=b) break;
        send_cnts[p_iter]++;
        data[node_iter]=node_lst[node_iter]->Pack();
        send_size[p_iter]+=data[node_iter].length+sizeof(size_t)+sizeof(MortonId);
        sbuff_size       +=data[node_iter].length+sizeof(size_t)+sizeof(MortonId);
      }
    }
  }

  std::vector<int> recv_cnts(np);
  std::vector<int> recv_size(np);
  MPI_Alltoall(&send_cnts[0], 1, par::Mpi_datatype<int>::value(),
               &recv_cnts[0], 1, par::Mpi_datatype<int>::value(), *Comm());
  MPI_Alltoall(&send_size[0], 1, par::Mpi_datatype<int>::value(),
               &recv_size[0], 1, par::Mpi_datatype<int>::value(), *Comm());

  size_t recv_cnt=0;
  #pragma omp parallel for reduction(+:recv_cnt)
  for(int i=0;i<np;i++) recv_cnt+=recv_cnts[i];
  std::vector<MortonId> out(recv_cnt);

  std::vector<int> sdisp; sdisp.assign(np,0);
  std::vector<int> rdisp; rdisp.assign(np,0);
  omp_par::scan(&send_size[0],&sdisp[0],np); //TODO Don't need to do a full scan
  omp_par::scan(&recv_size[0],&rdisp[0],np); //     as most entries will be 0.
  size_t rbuff_size=rdisp[np-1]+recv_size[np-1];

  char* send_buff=new char[sbuff_size];
  char* recv_buff=new char[rbuff_size];
  std::vector<char*> data_ptr(leaf_cnt);
  char* s_ptr=send_buff;
  for(size_t i=0;i<leaf_cnt;i++){
    *((MortonId*)s_ptr)=in  [i]       ; s_ptr+=sizeof(MortonId);
    *((  size_t*)s_ptr)=data[i].length; s_ptr+=sizeof(size_t);
    data_ptr[i]=s_ptr                 ; s_ptr+=data[i].length;
  }
  #pragma omp parallel for
  for(int p=0;p<omp_p;p++){
    size_t a=( p   *leaf_cnt)/omp_p;
    size_t b=((p+1)*leaf_cnt)/omp_p;
    for(size_t i=a;i<b;i++)
      mem::memcopy(data_ptr[i], data[i].data, data[i].length);
  }

  par::Mpi_Alltoallv_sparse<char>(&send_buff[0], &send_size[0], &sdisp[0],
    &recv_buff[0], &recv_size[0], &rdisp[0], *Comm());

  char* r_ptr=recv_buff;
  std::vector<PackedData> r_data(recv_cnt);
  for(size_t i=0;i<recv_cnt;i++){
    out   [i]       =*(MortonId*)r_ptr; r_ptr+=sizeof(MortonId);
    r_data[i].length=*(  size_t*)r_ptr; r_ptr+=sizeof(size_t);
    r_data[i].data  =            r_ptr; r_ptr+=r_data[i].length;
  }

  //Initialize all new nodes.
  int nchld=1UL<<this->Dim();
  size_t node_iter=0;
  MortonId dn;
  node_lst.resize(recv_cnt);
  Node_t* n=this->PreorderFirst();
  while(n!=NULL && node_iter<recv_cnt){
    n->SetGhost(false);
    dn=n->GetMortonId();
    if(dn.isAncestor(out[node_iter]) && dn!=out[node_iter]){
      if(n->IsLeaf()){
        {
          n->SetGhost(true);
          n->Subdivide();
          n->SetGhost(false);
          for(int j=0;j<nchld;j++){
            Node_t* ch_node=(Node_t*)n->Child(j);
            ch_node->SetGhost(false);
          }
        }
      }
    }else if(dn==out[node_iter]){
      if(!n->IsLeaf()){
        n->Truncate();
        n->SetGhost(false);
      }
      node_lst[node_iter]=n;
      node_iter++;
    }else{
      n->Truncate(); //This node does not belong to this process.
      n->SetGhost(true);
    }
    n=this->PreorderNxt(n);
  }
  while(n!=NULL){
    n->Truncate();
    n->SetGhost(true);
    n=this->PreorderNxt(n);
  }
  #pragma omp parallel for
  for(int p=0;p<omp_p;p++){
    size_t a=( p   *recv_cnt)/omp_p;
    size_t b=((p+1)*recv_cnt)/omp_p;
    for(size_t i=a;i<b;i++)
      node_lst[i]->Unpack(r_data[i]);
  }

  //Free memory buffers.
  delete[] recv_buff;
  delete[] send_buff;
}


template <class TreeNode>
TreeNode* MPI_Tree<TreeNode>::FindNode(MortonId& key, bool subdiv,  TreeNode* start){
  int num_child=1UL<<this->Dim();
  Node_t* n=start;
  if(n==NULL) n=this->RootNode();
  while(n->GetMortonId()<key && (!n->IsLeaf()||subdiv)){
    if(n->IsLeaf() && !n->IsGhost()) n->Subdivide();
    if(n->IsLeaf()) break;
    for(int j=0;j<num_child;j++){
      if(((Node_t*)n->Child(j))->GetMortonId().NextId()>key){
        n=(Node_t*)n->Child(j);
        break;
      }
    }
  }
  assert(!subdiv || n->IsGhost() || n->GetMortonId()==key);
  return n;
}


//list must be sorted.
inline int lineariseList(std::vector<MortonId> & list, MPI_Comm comm) {
  int rank,size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  //Remove empty processors...
  int new_rank, new_size;
  MPI_Comm   new_comm;
  MPI_Comm_split(comm, (list.empty()?0:1), rank, &new_comm);

  MPI_Comm_rank (new_comm, &new_rank);
  MPI_Comm_size (new_comm, &new_size);
  if(!list.empty()) {
    //Send the last octant to the next processor.
    MortonId lastOctant = list[list.size()-1];
    MortonId lastOnPrev;

    MPI_Request recvRequest;
    MPI_Request sendRequest;

    if(new_rank > 0) {
      MPI_Irecv(&lastOnPrev, 1, par::Mpi_datatype<MortonId>::value(), new_rank-1, 1, new_comm, &recvRequest);
    }
    if(new_rank < (new_size-1)) {
      MPI_Issend( &lastOctant, 1, par::Mpi_datatype<MortonId>::value(), new_rank+1, 1, new_comm,  &sendRequest);
    }

    if(new_rank > 0) {
      std::vector<MortonId> tmp(list.size()+1);
      for(size_t i = 0; i < list.size(); i++) {
        tmp[i+1] = list[i];
      }

      MPI_Status statusWait;
      MPI_Wait(&recvRequest, &statusWait);
      tmp[0] = lastOnPrev;

      list.swap(tmp);
    }

    {// Remove duplicates and ancestors.
      std::vector<MortonId> tmp;
      if(!(list.empty())) {
        for(unsigned int i = 0; i < (list.size()-1); i++) {
          if( (!(list[i].isAncestor(list[i+1]))) && (list[i] != list[i+1]) ) {
            tmp.push_back(list[i]);
          }
        }
        if(new_rank == (new_size-1)) {
          tmp.push_back(list[list.size()-1]);
        }
      }
      list.swap(tmp);
    }

    if(new_rank < (new_size-1)) {
      MPI_Status statusWait;
      MPI_Wait(&sendRequest, &statusWait);
    }
  }//not empty procs only

  return 1;
}//end fn.

inline int balanceOctree (std::vector<MortonId > &in, std::vector<MortonId > &out,
    unsigned int dim, unsigned int maxDepth, bool periodic, MPI_Comm comm) {

  int omp_p=omp_get_max_threads();

  int rank, size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  if(size==1 && in.size()==1){
    out=in;
    return 0;
  }

#ifdef __VERBOSE__
  long long locInSize = in.size();
#endif

  //////////////////////////////////////////////////////////////////////////////////////////////////

  { //Redistribute.
    //Vector<long long> balance_wt(size);
    //#pragma omp parallel for
    //for(size_t i=0;i<size;i++){
    //  balance_wt[i]=in[i].GetDepth();
    //}
    //par::partitionW<MortonId>(in, &balance_wt[0], comm);
    par::partitionW<MortonId>(in, NULL, comm);
  }

  //Build level-by-level set of nodes.
  std::vector<std::set<MortonId> > nodes((maxDepth+1)*omp_p);
  #pragma omp parallel for
  for(int p=0;p<omp_p;p++){
    size_t a=( p   *in.size())/omp_p;
    size_t b=((p+1)*in.size())/omp_p;
    for(size_t i=a;i<b;){
      size_t d=in[i].GetDepth();
      if(d==0){i++; continue;}
      MortonId pnode=in[i].getAncestor(d-1);
      nodes[d-1+(maxDepth+1)*p].insert(pnode);
      while(i<b && d==in[i].GetDepth() && pnode==in[i].getAncestor(d-1)) i++;
    }

    //Add new nodes level-by-level.
    std::vector<MortonId> nbrs;
    unsigned int num_chld=1UL<<dim;
    for(unsigned int l=maxDepth;l>=1;l--){
      //Build set of parents of balancing nodes.
      std::set<MortonId> nbrs_parent;
      std::set<MortonId>::iterator start=nodes[l+(maxDepth+1)*p].begin();
      std::set<MortonId>::iterator end  =nodes[l+(maxDepth+1)*p].end();
      for(std::set<MortonId>::iterator node=start; node != end;){
        node->NbrList(nbrs, l, periodic);
        int nbr_cnt=nbrs.size();
        for(int i=0;i<nbr_cnt;i++)
          nbrs_parent.insert(nbrs[i].getAncestor(l-1));
        node++;
      }
      //Get the balancing nodes.
      std::set<MortonId>& ancestor_nodes=nodes[l-1+(maxDepth+1)*p];
      start=nbrs_parent.begin();
      end  =nbrs_parent.end();
      ancestor_nodes.insert(start,end);
    }

    //Remove non-leaf nodes. (optional)
    for(unsigned int l=1;l<=maxDepth;l++){
      std::set<MortonId>::iterator start=nodes[l  +(maxDepth+1)*p].begin();
      std::set<MortonId>::iterator end  =nodes[l  +(maxDepth+1)*p].end();
      std::set<MortonId>& ancestor_nodes=nodes[l-1+(maxDepth+1)*p];
      for(std::set<MortonId>::iterator node=start; node != end; node++){
        MortonId parent=node->getAncestor(node->GetDepth()-1);
        ancestor_nodes.erase(parent);
      }
    }
  }

  //Resize in.
  std::vector<size_t> node_cnt(omp_p,0);
  std::vector<size_t> node_dsp(omp_p,0);
  #pragma omp parallel for
  for(int i=0;i<omp_p;i++){
    for(unsigned int j=0;j<=maxDepth;j++)
      node_cnt[i]+=nodes[j+i*(maxDepth+1)].size();
  }
  omp_par::scan(&node_cnt[0],&node_dsp[0], omp_p);
  in.resize(node_cnt[omp_p-1]+node_dsp[omp_p-1]);

  //Copy leaf nodes to in.
  #pragma omp parallel for
  for(int p=0;p<omp_p;p++){
    size_t node_iter=node_dsp[p];
    for(unsigned int l=0;l<=maxDepth;l++){
      std::set<MortonId>::iterator start=nodes[l  +(maxDepth+1)*p].begin();
      std::set<MortonId>::iterator end  =nodes[l  +(maxDepth+1)*p].end();
      for(std::set<MortonId>::iterator node=start; node != end; node++)
        in[node_iter++]=*node;
    }
  }

#ifdef __VERBOSE__
  //Local size before removing duplicates and ancestors (linearise).
  long long locTmpSize = in.size();
#endif

  //Sort, Linearise, Redistribute.
  //TODO The following might work better as it reduces the comm bandwidth:
  //Split comm into sqrt(np) processes and sort, linearise for each comm group.
  //Then do the global sort, linearise with the original comm.
  par::HyperQuickSort(in, out, comm);
  lineariseList(out, comm);
  par::partitionW<MortonId>(out, NULL , comm);
  { // Add children

    //Remove empty processors...
    int new_rank, new_size;
    MPI_Comm   new_comm;
    MPI_Comm_split(comm, (out.empty()?0:1), rank, &new_comm);

    MPI_Comm_rank (new_comm, &new_rank);
    MPI_Comm_size (new_comm, &new_size);
    if(!out.empty()) {
      MortonId nxt_mid(0,0,0,0);
      { // Get last octant from previous process.
        assert(out.size());

        //Send the last octant to the next processor.
        MortonId lastOctant = out.back();
        MortonId lastOnPrev;

        MPI_Request recvRequest;
        MPI_Request sendRequest;

        if(rank > 0) {
          MPI_Irecv(&lastOnPrev, 1, par::Mpi_datatype<MortonId>::value(), rank-1, 1, comm, &recvRequest);
        }
        if(rank < (size-1)) {
          MPI_Issend( &lastOctant, 1, par::Mpi_datatype<MortonId>::value(), rank+1, 1, comm,  &sendRequest);
        }

        if(rank > 0) {
          MPI_Status statusWait;
          MPI_Wait(&recvRequest, &statusWait);
          nxt_mid = lastOnPrev.NextId();
        }

        if(rank < (size-1)) {
          MPI_Status statusWait;
          MPI_Wait(&sendRequest, &statusWait);
        }
      }

      std::vector<MortonId> out1;
      std::vector<MortonId> children;
      for(size_t i=0;i<out.size();i++){
        while(nxt_mid.getDFD()<out[i]){
          while(nxt_mid.isAncestor(out[i])){
            nxt_mid=nxt_mid.getAncestor(nxt_mid.GetDepth()+1);
          }
          out1.push_back(nxt_mid);
          nxt_mid=nxt_mid.NextId();
        }

        children=out[i].Children();
        for(size_t j=0;j<8;j++){
          out1.push_back(children[j]);
        }
        nxt_mid=out[i].NextId();
      }
      if(rank==size-1){
        while(nxt_mid.GetDepth()>0){
          out1.push_back(nxt_mid);
          nxt_mid=nxt_mid.NextId();
        }
      }
      out.swap(out1);
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __VERBOSE__
  long long locOutSize = out.size();
  long long globInSize, globTmpSize, globOutSize;
  MPI_Allreduce(&locInSize , &globInSize , 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
  MPI_Allreduce(&locTmpSize, &globTmpSize, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
  MPI_Allreduce(&locOutSize, &globOutSize, 1, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
  if(!rank) std::cout<<"Balance Octree. inpSize: "<<globInSize
                                    <<" tmpSize: "<<globTmpSize
                                    <<" outSize: "<<globOutSize
                                 <<" activeNpes: "<<size<<std::endl;
#endif
  return 0;
}//end function

template <class TreeNode>
void MPI_Tree<TreeNode>::Balance21(BoundaryType bndry) {
  int num_proc,myrank;
  MPI_Comm_rank(*Comm(),&myrank);
  MPI_Comm_size(*Comm(),&num_proc);

  //Using Dendro for balancing
  //Create a linear tree in dendro format.
  Node_t* curr_node=this->PreorderFirst();
  std::vector<MortonId> in;
  while(curr_node!=NULL){
    if(curr_node->IsLeaf() && !curr_node->IsGhost()){
      in.push_back(curr_node->GetMortonId());
    }
    curr_node=this->PreorderNxt(curr_node);
  }

  //2:1 balance
  Profile::Tic("ot::balanceOctree",Comm(),true,3);
  std::vector<MortonId> out;
  balanceOctree(in, out, this->Dim(), this->max_depth, (bndry==Periodic), *Comm());
  Profile::Toc();

  //Get new_mins.
  std::vector<MortonId> new_mins(num_proc);
  MPI_Allgather(&out[0]     , 1, par::Mpi_datatype<MortonId>::value(),
                &new_mins[0], 1, par::Mpi_datatype<MortonId>::value(), *Comm());


  // Refine to new_mins in my range of octants
  // or else RedistNodes(...) will not work correctly.
  {
    int i=0;
    std::vector<MortonId> mins=GetMins();
    while(new_mins[i]<mins[myrank] && i<num_proc) i++; //TODO: Use binary search.
    for(;i<num_proc;i++){
      Node_t* n=FindNode(new_mins[i], true);
      if(n->IsGhost()) break;
      else assert(n->GetMortonId()==new_mins[i]);
    }
  }

  //Redist nodes using new_mins.
  Profile::Tic("RedistNodes",Comm(),true,3);
  RedistNodes(&out[0]);
  #ifndef NDEBUG
  std::vector<MortonId> mins=GetMins();
  assert(mins[myrank].getDFD()==out[0].getDFD());
  #endif
  Profile::Toc();

  //Now subdivide the current tree as necessary to make it balanced.
  Profile::Tic("LocalSubdivide",Comm(),false,3);
  int omp_p=omp_get_max_threads();
  for(int i=0;i<omp_p;i++){
    size_t a=(out.size()*i)/omp_p;
    Node_t* n=FindNode(out[a], true);
    assert(n->GetMortonId()==out[a]);
    UNUSED(n);
  }
  #pragma omp parallel for
  for(int i=0;i<omp_p;i++){
    size_t a=(out.size()* i   )/omp_p;
    size_t b=(out.size()*(i+1))/omp_p;

    MortonId dn;
    size_t node_iter=a;
    Node_t* n=FindNode(out[node_iter], false);
    while(n!=NULL && node_iter<b){
      n->SetGhost(false);
      dn=n->GetMortonId();
      if(dn.isAncestor(out[node_iter]) && dn!=out[node_iter]){
        if(n->IsLeaf()) n->Subdivide();
      }else if(dn==out[node_iter]){
        assert(n->IsLeaf());
        //if(!n->IsLeaf()){ //This should never happen
        //  n->Truncate();
        //  n->SetGhost(false);
        //}
        assert(n->IsLeaf());
        node_iter++;
      }else{
        n->Truncate(); //This node does not belong to this process.
        n->SetGhost(true);
      }
      n=this->PreorderNxt(n);
    }
    if(i==omp_p-1){
      while(n!=NULL){
        n->Truncate();
        n->SetGhost(true);
        n=this->PreorderNxt(n);
      }
    }
  }
  Profile::Toc();
}


template <class TreeNode>
void MPI_Tree<TreeNode>::Balance21_local(BoundaryType bndry){
  //SetColleagues(bndry);

  std::vector<std::vector<Node_t*> > node_lst(this->max_depth+1);
  Node_t* curr_node=this->PreorderFirst();
  while(curr_node!=NULL){
    node_lst[curr_node->Depth()].push_back(curr_node);
    curr_node=this->PreorderNxt(curr_node);
  }

  int n1=pow(3.0,this->Dim());
  int n2=pow(2.0,this->Dim());
  for(int i=this->max_depth;i>0;i--){
    Real_t s=pow(0.5,i);
    for(size_t j=0;j<node_lst[i].size();j++){
      curr_node=node_lst[i][j];
      Real_t* coord=curr_node->Coord();
      if(!curr_node->IsLeaf()) for(int k=0;k<n1;k++){
        if(curr_node->Colleague(k)==NULL){
          Real_t c0[3]={coord[0]+((k/1)%3-1)*s+s*0.5,
                        coord[1]+((k/3)%3-1)*s+s*0.5,
                        coord[2]+((k/9)%3-1)*s+s*0.5};
          if(bndry==Periodic){
            c0[0]=c0[0]-floor(c0[0]);
            c0[1]=c0[1]-floor(c0[1]);
            c0[2]=c0[2]-floor(c0[2]);
          }

          if(c0[0]>0 && c0[0]<1)
          if(c0[1]>0 && c0[1]<1)
          if(c0[2]>0 && c0[2]<1){
            Node_t* node=this->RootNode();
            while(node->Depth()<i){
              if(node->IsLeaf()){
                node->Subdivide();
                for(int l=0;l<n2;l++){
                  node_lst[node->Depth()+1].push_back((Node_t*)node->Child(l));
                  /*
                  SetColleagues(bndry,(Node_t*)node->Child(l));
                  for(int i_=0;i_<n1;i_++){
                    Node_t* coll=(Node_t*)((Node_t*)node->Child(l))->Colleague(i_);
                    if(coll!=NULL) SetColleagues(bndry,coll);
                  }// */
                }
              }
              Real_t s1=pow(0.5,node->Depth()+1);
              Real_t* c1=node->Coord();
              int c_id=((c0[0]-c1[0])>s1?1:0)+
                       ((c0[1]-c1[1])>s1?2:0)+
                       ((c0[2]-c1[2])>s1?4:0);
              node=(Node_t*)node->Child(c_id);
              /*if(node->Depth()==i){
                c1=node->Coord();
                std::cout<<(c0[0]-c1[0])-s1/2<<' '
                std::cout<<(c0[1]-c1[1])-s1/2<<' '
                std::cout<<(c0[2]-c1[2])-s1/2<<'\n';
              }// */
            }
          }
        }
      }
    }
  }
}


template <class TreeNode>
void MPI_Tree<TreeNode>::SetColleagues(BoundaryType bndry, Node_t* node){
  int n1=(int)pow(3.0,this->Dim());
  int n2=(int)pow(2.0,this->Dim());

  if(node==NULL){
    Node_t* curr_node=this->PreorderFirst();
    if(curr_node!=NULL){
      if(bndry==Periodic){
        for(int i=0;i<n1;i++)
          curr_node->SetColleague(curr_node,i);
      }else{
        curr_node->SetColleague(curr_node,(n1-1)/2);
      }
      curr_node=this->PreorderNxt(curr_node);
    }

    Vector<std::vector<Node_t*> > nodes(MAX_DEPTH);
    while(curr_node!=NULL){
      nodes[curr_node->Depth()].push_back(curr_node);
      curr_node=this->PreorderNxt(curr_node);
    }
    for(size_t i=0;i<MAX_DEPTH;i++){
      size_t j0=nodes[i].size();
      Node_t** nodes_=&nodes[i][0];
      #pragma omp parallel for
      for(size_t j=0;j<j0;j++){
        SetColleagues(bndry, nodes_[j]);
      }
    }

  }else{
    /* //This is slower
    Node_t* root_node=this->RootNode();
    int d=node->Depth();
    Real_t* c0=node->Coord();
    Real_t s=pow(0.5,d);
    Real_t c[COORD_DIM];
    int idx=0;
    for(int i=-1;i<=1;i++)
    for(int j=-1;j<=1;j++)
    for(int k=-1;k<=1;k++){
      c[0]=c0[0]+s*0.5+s*k;
      c[1]=c0[1]+s*0.5+s*j;
      c[2]=c0[2]+s*0.5+s*i;
      if(bndry==Periodic){
        if(c[0]<0.0) c[0]+=1.0;
        if(c[0]>1.0) c[0]-=1.0;
        if(c[1]<1.0) c[1]+=1.0;
        if(c[1]>1.0) c[1]-=1.0;
        if(c[2]<1.0) c[2]+=1.0;
        if(c[2]>1.0) c[2]-=1.0;
      }
      node->SetColleague(NULL,idx);
      if(c[0]<1.0 && c[0]>0.0)
      if(c[1]<1.0 && c[1]>0.0)
      if(c[2]<1.0 && c[2]>0.0){
        MortonId m(c,d);
        Node_t* nbr=FindNode(m,false,root_node);
        while(nbr->Depth()>d) nbr=(Node_t*)nbr->Parent();
        if(nbr->Depth()==d) node->SetColleague(nbr,idx);
      }
      idx++;
    }
    /*/
    Node_t* parent_node;
    Node_t* tmp_node1;
    Node_t* tmp_node2;

    for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
    parent_node=(Node_t*)node->Parent();
    if(parent_node==NULL) return;

    int l=node->Path2Node();
    for(int i=0;i<n1;i++){ //For each coll of the parent
      tmp_node1=(Node_t*)parent_node->Colleague(i);
      if(tmp_node1!=NULL)
      if(!tmp_node1->IsLeaf()){
        for(int j=0;j<n2;j++){ //For each child
          tmp_node2=(Node_t*)tmp_node1->Child(j);
          if(tmp_node2!=NULL){

            bool flag=true;
            int a=1,b=1,new_indx=0;
            for(int k=0;k<this->Dim();k++){
              int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
              if(-1>indx_diff || indx_diff>1) flag=false;
              new_indx+=(indx_diff+1)*b;
              a*=2;b*=3;
            }
            if(flag){
              node->SetColleague(tmp_node2,new_indx);
            }
          }
        }
      }
    }// */
  }

}


template <class TreeNode>
bool MPI_Tree<TreeNode>::CheckTree(){
  int myrank,np;
  MPI_Comm_rank(*Comm(),&myrank);
  MPI_Comm_size(*Comm(),&np);
  std::vector<MortonId> mins=GetMins();

  std::stringstream st;
  st<<"PID_"<<myrank<<" : ";
  std::string str;

  Node_t* n=this->PostorderFirst();
  while(n!=NULL){
    if(myrank<np-1) if(n->GetMortonId().getDFD()>=mins[myrank+1])break;
    if(n->GetMortonId()>=mins[myrank] && n->IsLeaf() && n->IsGhost()){
      std::cout<<n->GetMortonId()<<'\n';
      std::cout<<mins[myrank]<<'\n';
      if(myrank+1<np) std::cout<<mins[myrank+1]<<'\n';
      std::cout<<myrank<<'\n';
      assert(false);
    }
    if(n->GetMortonId()<mins[myrank] && n->IsLeaf() && !n->IsGhost()){
      assert(false);
    }
    if(!n->IsGhost() && n->Depth()>0)
      assert(!((Node_t*)n->Parent())->IsGhost());
    n=this->PostorderNxt(n);
  }
  while(n!=NULL){
    if(n->IsLeaf() && !n->IsGhost()){
      st<<"non-ghost leaf node "<<n->GetMortonId()<<"; after last node.";
      str=st.str(); ASSERT_WITH_MSG(false,str.c_str());
    }
    n=this->PostorderNxt(n);
  }
  return true;
};


/**
 * \brief Determines if node is used in the region between Morton Ids m1 and m2
 * ( m1 <= m2 ).
 */
template <class TreeNode>
void IsShared(std::vector<TreeNode*>& nodes, MortonId* m1, MortonId* m2, BoundaryType bndry, std::vector<char>& shared_flag){
  MortonId mm1, mm2;
  if(m1!=NULL) mm1=m1->getDFD();
  if(m2!=NULL) mm2=m2->getDFD();
  shared_flag.resize(nodes.size());
  int omp_p=omp_get_max_threads();

  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    size_t a=((j  )*nodes.size())/omp_p;
    size_t b=((j+1)*nodes.size())/omp_p;
    std::vector<MortonId> nbr_lst;
    for(size_t i=a;i<b;i++){
      shared_flag[i]=false;
      TreeNode* node=nodes[i];
      assert(node!=NULL);
      if(node->Depth()<2){
        shared_flag[i]=true;
        continue;
      }
      node->GetMortonId().NbrList(nbr_lst, node->Depth()-1, bndry==Periodic);
      for(size_t k=0;k<nbr_lst.size();k++){
        MortonId n1=nbr_lst[k]         .getDFD();
        MortonId n2=nbr_lst[k].NextId().getDFD();
        if(m1==NULL || n2>mm1)
          if(m2==NULL || n1<mm2){
            shared_flag[i]=true;
            break;
          }
      }
    }
  }
}

inline void IsShared(std::vector<PackedData>& nodes, MortonId* m1, MortonId* m2, BoundaryType bndry, std::vector<char>& shared_flag){
  MortonId mm1, mm2;
  if(m1!=NULL) mm1=m1->getDFD();
  if(m2!=NULL) mm2=m2->getDFD();
  shared_flag.resize(nodes.size());
  int omp_p=omp_get_max_threads();

  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    size_t a=((j  )*nodes.size())/omp_p;
    size_t b=((j+1)*nodes.size())/omp_p;
    std::vector<MortonId> nbr_lst;
    for(size_t i=a;i<b;i++){
      shared_flag[i]=false;
      MortonId* node=(MortonId*)nodes[i].data;
      assert(node!=NULL);
      if(node->GetDepth()<2){
        shared_flag[i]=true;
        continue;
      }
      node->NbrList(nbr_lst, node->GetDepth()-1, bndry==Periodic);
      for(size_t k=0;k<nbr_lst.size();k++){
        MortonId n1=nbr_lst[k]         .getDFD();
        MortonId n2=nbr_lst[k].NextId().getDFD();
        if(m1==NULL || n2>mm1)
          if(m2==NULL || n1<mm2){
            shared_flag[i]=true;
            break;
          }
      }
    }
  }
}


/**
 * \brief Construct Locally Essential Tree by exchanging Ghost octants.
 */
template <class TreeNode>
void MPI_Tree<TreeNode>::ConstructLET(BoundaryType bndry){
  //Profile::Tic("LET_Hypercube", &comm, true, 5);
  //ConstructLET_Hypercube(bndry);
  //Profile::Toc();

  //Profile::Tic("LET_Sparse", &comm, true, 5);
  ConstructLET_Sparse(bndry);
  //Profile::Toc();

#ifndef NDEBUG
  CheckTree();
#endif
}

/**
 * \brief Hypercube based scheme to exchange Ghost octants.
 */
//#define PREFETCH_T0(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T0)
template <class TreeNode>
void MPI_Tree<TreeNode>::ConstructLET_Hypercube(BoundaryType bndry){
  int num_p,rank;
  MPI_Comm_size(*Comm(),&num_p);
  MPI_Comm_rank(*Comm(),&rank );
  if(num_p==1) return;
  int omp_p=omp_get_max_threads();
  std::vector<MortonId> mins=GetMins();

  // Build list of shared nodes.
  std::vector<Node_t*> shared_nodes; shared_nodes.clear();
  std::vector<Node_t*> node_lst; node_lst.clear();
  Node_t* curr_node=this->PreorderFirst();
  while(curr_node!=NULL){
    if(curr_node->GetMortonId().getDFD()>=mins[rank]) break;
    curr_node=this->PreorderNxt(curr_node);
  }
  while(curr_node!=NULL){
    if(curr_node->IsGhost()) break;
    node_lst.push_back(curr_node);
    curr_node=this->PreorderNxt(curr_node);
  }
  std::vector<char> node_flag0; node_flag0.clear();
  std::vector<char> node_flag1; node_flag1.clear();
  IsShared(node_lst,&mins[0],&mins[rank],bndry,node_flag0);
  if(rank<num_p-1) IsShared(node_lst,&mins[rank+1],NULL,bndry,node_flag1);
  for(size_t i=0;i<node_lst.size();i++){
    if(node_flag0[i] || (rank<num_p-1 && node_flag1[i]))
      shared_nodes.push_back(node_lst[i]);
  }
  //std::cout<<"Shared = "<<shared_nodes.size()<<'\n';

  // Pack shared nodes.
  static std::vector<char> shrd_buff_vec0(omp_p*64l*1024l*1024l);
  static std::vector<char> shrd_buff_vec1(omp_p*128l*1024l*1024l);
  static std::vector<char> send_buff_vec(omp_p*64l*1024l*1024l); char* send_buff;
  static std::vector<char> recv_buff_vec(omp_p*64l*1024l*1024l); char* recv_buff;

  std::vector<PackedData> shrd_data;
  size_t max_data_size=0;
  {
    long max_data_size_lcl=0;
    long max_data_size_glb=0;
    char* data_ptr=&shrd_buff_vec0[0];
    for(size_t i=0;i<shared_nodes.size();i++){
      PackedData p=shared_nodes[i]->Pack(true,data_ptr,sizeof(MortonId));
      ((MortonId*)data_ptr)[0]=shared_nodes[i]->GetMortonId();
      p.length+=sizeof(MortonId);
      shrd_data.push_back(p);
      data_ptr+=p.length;
      if(max_data_size_lcl<(long)p.length) max_data_size_lcl=p.length;
      assert(data_ptr<=&(*shrd_buff_vec0.end())); //TODO: resize if needed.
    }
    MPI_Allreduce(&max_data_size_lcl, &max_data_size_glb, 1, MPI_LONG, MPI_MAX, *Comm());
    max_data_size=max_data_size_glb;
  }

  // Memory slots for storing node data.
  std::set<void*> mem_set;
  size_t mem_set_size=0;

  size_t range[2]={0,(size_t)num_p-1};
  while(range[1]-range[0]>0){
    size_t split_p=(range[0]+range[1])/2;
    size_t new_range[2]={(size_t)rank<=split_p?range[0]:split_p+1,(size_t)rank<=split_p?split_p:range[1]};
    size_t com_range[2]={(size_t)rank> split_p?range[0]:split_p+1,(size_t)rank> split_p?split_p:range[1]};
    size_t partner=rank-new_range[0]+com_range[0];
    if(partner>range[1]) partner--;
    bool extra_partner=((size_t)rank==range[1] && ((range[1]-range[0])%2)==0);

    int send_length=0;
    std::vector<PackedData> shrd_data_new;
    IsShared(shrd_data, &mins[com_range[0]], (com_range[1]==(size_t)num_p-1?NULL:&mins[com_range[1]+1]),bndry, node_flag0);
    IsShared(shrd_data, &mins[new_range[0]], (new_range[1]==(size_t)num_p-1?NULL:&mins[new_range[1]+1]),bndry, node_flag1);
    {
      std::vector<void*> srctrg_ptr;
      std::vector<size_t> mem_size;
      for(size_t i=0;i<shrd_data.size();i++){
        PackedData& p=shrd_data[i];
        if( node_flag0[i]){ // Copy data to send buffer.
          char* data_ptr=(char*)&send_buff_vec[send_length];
          ((size_t*)data_ptr)[0]=p.length; data_ptr+=sizeof(size_t);
          //mem::memcopy(data_ptr,p.data,p.length);
          mem_size.push_back(p.length);
          srctrg_ptr.push_back(p.data);
          srctrg_ptr.push_back(data_ptr);
          send_length+=p.length+sizeof(size_t);
          assert((size_t)send_length<=send_buff_vec.size()); //TODO: resize if needed.
        }
        if(!node_flag1[i]){ // Free memory slot.
          //assert(node_flag0[0]);
          if(p.data>=&shrd_buff_vec1[0] && p.data<&shrd_buff_vec1[0]+shrd_buff_vec1.size())
            mem_set.insert(p.data);
        } else shrd_data_new.push_back(p);
      }
      shrd_data=shrd_data_new;
      #pragma omp parallel for
      for(int k=0;k<omp_p;k++){
        size_t i0=((k+0)*mem_size.size())/omp_p;
        size_t i1=((k+1)*mem_size.size())/omp_p;
        for(size_t i=i0;i<i1;i++){
          mem::memcopy(srctrg_ptr[2*i+1],srctrg_ptr[2*i+0],mem_size[i]);
        }
      }
    }

    //Exchange send size.
    int recv_length=0;
    int extra_recv_length=0;
    int extra_send_length=0;
    MPI_Status status;
    MPI_Sendrecv                  (&      send_length,1,MPI_INT,partner,0,      &recv_length,1,MPI_INT,partner,0,*Comm(),&status);
    if(extra_partner) MPI_Sendrecv(&extra_send_length,1,MPI_INT,split_p,0,&extra_recv_length,1,MPI_INT,split_p,0,*Comm(),&status);

    //SendRecv data.
    assert((size_t)send_length                  <=send_buff_vec.size()); send_buff=&send_buff_vec[0];
    assert((size_t)recv_length+extra_recv_length<=recv_buff_vec.size()); recv_buff=&recv_buff_vec[0];
    MPI_Sendrecv                  (send_buff,send_length,MPI_BYTE,partner,0, recv_buff             ,      recv_length,MPI_BYTE,partner,0,*Comm(),&status);
    if(extra_partner) MPI_Sendrecv(     NULL,          0,MPI_BYTE,split_p,0,&recv_buff[recv_length],extra_recv_length,MPI_BYTE,split_p,0,*Comm(),&status);

    //Get nodes from received data.
    {
      std::vector<void*> srctrg_ptr;
      std::vector<size_t> mem_size;
      int buff_length=0;
      while(buff_length<recv_length+extra_recv_length){
        PackedData p0,p1;
        p0.length=((size_t*)&recv_buff_vec[buff_length])[0];
        p0.data=(char*)&recv_buff_vec[buff_length]+sizeof(size_t);
        buff_length+=p0.length+sizeof(size_t);

        p1.length=p0.length;
        if(mem_set.size()==0){
          assert(mem_set_size*max_data_size<shrd_buff_vec1.size());
          p1.data=&shrd_buff_vec1[mem_set_size*max_data_size];
          mem_set_size++;
        }else{
          p1.data=*mem_set.begin();
          mem_set.erase(mem_set.begin());
        }
        //mem::memcopy(p1.data,p0.data,p0.length);
        mem_size.push_back(p0.length);
        srctrg_ptr.push_back(p0.data);
        srctrg_ptr.push_back(p1.data);
        shrd_data.push_back(p1);
      }
      #pragma omp parallel for
      for(int k=0;k<omp_p;k++){
        size_t i0=((k+0)*mem_size.size())/omp_p;
        size_t i1=((k+1)*mem_size.size())/omp_p;
        for(size_t i=i0;i<i1;i++){
          mem::memcopy(srctrg_ptr[2*i+1],srctrg_ptr[2*i+0],mem_size[i]);
        }
      }
    }

    range[0]=new_range[0];
    range[1]=new_range[1];
  }

  //Add shared_nodes to the tree.
  //std::cout<<"Number of Ghost Nodes = "<<shrd_data.size()<<'\n';
  int nchld=(1UL<<this->Dim()); // Number of children.
  std::vector<Node_t*> shrd_nodes(shrd_data.size());
  for(size_t i=0;i<shrd_data.size();i++){ // Find shared nodes.
    MortonId& mid=*(MortonId*)shrd_data[i].data;
    Node_t* srch_node=this->RootNode();
    while(srch_node->GetMortonId()!=mid){
      Node_t* ch_node;
      if(srch_node->IsLeaf()){
        srch_node->SetGhost(true);
        srch_node->Subdivide();
      }
      for(int j=nchld-1;j>=0;j--){
        ch_node=(Node_t*)srch_node->Child(j);
        if(ch_node->GetMortonId()<=mid){
          srch_node=ch_node;
          break;
        }
      }
    }
    shrd_nodes[i]=srch_node;
  }
  #pragma omp parallel for
  for(size_t i=0;i<shrd_data.size();i++){
    if(shrd_nodes[i]->IsGhost()) { // Initialize ghost node.
      PackedData p=shrd_data[i];
      p.data=((char*)p.data)+sizeof(MortonId);
      p.length-=sizeof(MortonId);
      shrd_nodes[i]->Unpack(p);
    }
  }
  //Now LET is complete.

}

/**
 * \brief Sparse communication scheme to exchange Ghost octants.
 */
template <class TreeNode>
void MPI_Tree<TreeNode>::ConstructLET_Sparse(BoundaryType bndry){
  typedef int MPI_size_t;
  struct CommData{
    MortonId mid;
    TreeNode* node;
    size_t pkd_length;

    size_t   usr_cnt;
    MortonId usr_mid[COLLEAGUE_COUNT];
    size_t   usr_pid[COLLEAGUE_COUNT];
  };

  int num_p,rank;
  MPI_Comm_size(*Comm(),&num_p);
  MPI_Comm_rank(*Comm(),&rank );
  if(num_p==1) return;

  int omp_p=omp_get_max_threads();
  std::vector<MortonId> mins=GetMins();

  // Allocate Memory.
  static std::vector<char> send_buff;
  static std::vector<char> recv_buff;

  //Profile::Tic("SharedNodes", &comm, false, 5);
  CommData* node_comm_data=NULL; // CommData for all nodes.
  std::vector<void*> shared_data; // CommData for shared nodes.
  std::vector<par::SortPair<size_t,size_t> > pid_node_pair; // <pid, shared_data index> list
  { // Set node_comm_data
    MortonId mins_r0=mins[         rank+0         ].getDFD();
    MortonId mins_r1=mins[std::min(rank+1,num_p-1)].getDFD();

    std::vector<TreeNode*> nodes=this->GetNodeList();
    node_comm_data=(CommData*)this->memgr.malloc(sizeof(CommData)*nodes.size());
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      std::vector<MortonId> nbr_lst;
      size_t a=(nodes.size()* tid   )/omp_p;
      size_t b=(nodes.size()*(tid+1))/omp_p;
      for(size_t i=a;i<b;i++){
        bool shared=false;
        CommData& comm_data=node_comm_data[i];
        comm_data.node=nodes[i];
        comm_data.mid=comm_data.node->GetMortonId();
        comm_data.usr_cnt=0;

        if(comm_data.node->IsGhost()) continue;
        if(comm_data.node->Depth()==0) continue;
        if(comm_data.mid.getDFD()<mins_r0) continue;

        MortonId mid0=comm_data.mid.         getDFD();
        MortonId mid1=comm_data.mid.NextId().getDFD();

        comm_data.mid.NbrList(nbr_lst,comm_data.node->Depth()-1, bndry==Periodic);
        comm_data.usr_cnt=nbr_lst.size();
        for(size_t j=0;j<nbr_lst.size();j++){
          MortonId usr_mid=nbr_lst[j];
          MortonId usr_mid_dfd=usr_mid.getDFD();
          comm_data.usr_mid[j]=usr_mid;
          comm_data.usr_pid[j]=std::upper_bound(&mins[0],&mins[num_p],usr_mid_dfd)-&mins[0]-1;
//          if(usr_mid_dfd<mins_r0 || (rank+1<num_p && usr_mid_dfd>=mins_r1)){ // Find the user pid.
//            size_t usr_pid=std::upper_bound(&mins[0],&mins[num_p],usr_mid_dfd)-&mins[0]-1;
//            comm_data.usr_pid[j]=usr_pid;
//          }else comm_data.usr_pid[j]=rank;
          if(!shared){ // Check if this node needs to be transferred during broadcast.
            if(comm_data.usr_pid[j]!=rank || (rank+1<num_p && usr_mid.NextId()>mins_r1) ){
              shared=true;
            }
          }
        }
        #pragma omp critical (ADD_SHARED)
        if(shared){
          for(size_t j=0;j<comm_data.usr_cnt;j++)
          if(comm_data.usr_pid[j]!=rank){
            bool unique_pid=true;
            for(size_t k=0;k<j;k++){
              if(comm_data.usr_pid[j]==comm_data.usr_pid[k]){
                unique_pid=false;
                break;
              }
            }
            if(unique_pid){
              par::SortPair<size_t,size_t> p;
              p.key=comm_data.usr_pid[j];
              p.data=shared_data.size();
              pid_node_pair.push_back(p);
            }
          }
          shared_data.push_back(&comm_data);
        }
      }
    }
    omp_par::merge_sort(&pid_node_pair[0], &pid_node_pair[pid_node_pair.size()]);
    //std::cout<<rank<<' '<<shared_data.size()<<' '<<pid_node_pair.size()<<'\n';
  }
  //Profile::Toc();

  //Profile::Tic("PackNodes", &comm, false, 5);
  { // Pack shared nodes.
    #pragma omp parallel for
    for(size_t tid=0;tid<omp_p;tid++){
      size_t buff_length=10l*1024l*1024l; // 10MB buffer per thread.
      char* buff=(char*)this->memgr.malloc(buff_length);

      size_t a=( tid   *shared_data.size())/omp_p;
      size_t b=((tid+1)*shared_data.size())/omp_p;
      for(size_t i=a;i<b;i++){
        CommData& comm_data=*(CommData*)shared_data[i];
        PackedData p0=comm_data.node->Pack(true,buff);
        assert(p0.length<buff_length);

        shared_data[i]=this->memgr.malloc(sizeof(CommData)+p0.length);
        CommData& new_comm_data=*(CommData*)shared_data[i];
        new_comm_data=comm_data;

        new_comm_data.pkd_length=sizeof(CommData)+p0.length;
        mem::memcopy(((char*)shared_data[i])+sizeof(CommData),buff,p0.length);
      }
      this->memgr.free(buff);
    }

    // now CommData is stored in shared_data
    this->memgr.free(node_comm_data);
    node_comm_data=NULL;
  }
  //Profile::Toc();

  //Profile::Tic("SendBuff", &comm, false, 5);
  std::vector<MPI_size_t> send_size(num_p,0);
  std::vector<MPI_size_t> send_disp(num_p,0);
  if(pid_node_pair.size()){ // Build send_buff.
    std::vector<size_t> size(pid_node_pair.size(),0);
    std::vector<size_t> disp(pid_node_pair.size(),0);
    #pragma omp parallel for
    for(size_t i=0;i<pid_node_pair.size();i++){
      size[i]=((CommData*)shared_data[pid_node_pair[i].data])->pkd_length;
    }
    omp_par::scan(&size[0],&disp[0],pid_node_pair.size());

    // Resize send_buff.
    if(send_buff.size()<size[pid_node_pair.size()-1]+disp[pid_node_pair.size()-1]){
      send_buff.resize(size[pid_node_pair.size()-1]+disp[pid_node_pair.size()-1]);
    }

    // Copy data to send_buff.
    #pragma omp parallel for
    for(size_t i=0;i<pid_node_pair.size();i++){
      size_t shrd_idx=pid_node_pair[i].data;
      mem::memcopy(&send_buff[disp[i]], shared_data[shrd_idx], size[i]);
    }

    // Compute send_size, send_disp.
    {
      // Compute send_size.
      #pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        size_t a=(pid_node_pair.size()* tid   )/omp_p;
        size_t b=(pid_node_pair.size()*(tid+1))/omp_p;
        if(a>0 && a<pid_node_pair.size()){
          size_t p0=pid_node_pair[a].key;
          while(a<pid_node_pair.size() && p0==pid_node_pair[a].key) a++;
        }
        if(b>0 && b<pid_node_pair.size()){
          size_t p1=pid_node_pair[b].key;
          while(b<pid_node_pair.size() && p1==pid_node_pair[b].key) b++;
        }
        for(size_t i=a;i<b;i++){
          send_size[pid_node_pair[i].key]+=size[i];
        }
      }

      // Compute send_disp.
      omp_par::scan(&send_size[0],&send_disp[0],num_p);
    }
  }
  //Profile::Toc();

  //Profile::Tic("A2A_Sparse", &comm, true, 5);
  size_t recv_length=0;
  { // Allocate recv_buff.
    std::vector<MPI_size_t> recv_size(num_p,0);
    std::vector<MPI_size_t> recv_disp(num_p,0);
    MPI_Alltoall(&send_size[0], 1, par::Mpi_datatype<MPI_size_t>::value(),
                 &recv_size[0], 1, par::Mpi_datatype<MPI_size_t>::value(), *Comm());
    omp_par::scan(&recv_size[0],&recv_disp[0],num_p);
    recv_length=recv_size[num_p-1]+recv_disp[num_p-1];
    if(recv_buff.size()<recv_length){
      recv_buff.resize(recv_length);
    }
    par::Mpi_Alltoallv_sparse(&send_buff[0], &send_size[0], &send_disp[0],
                              &recv_buff[0], &recv_size[0], &recv_disp[0], *Comm());
  }
  //Profile::Toc();

  //Profile::Tic("Unpack", &comm, false, 5);
  std::vector<void*> recv_data; // CommData for received nodes.
  { // Unpack received octants.
    std::vector<par::SortPair<MortonId,size_t> > mid_indx_pair;
    for(size_t i=0; i<recv_length;){
      CommData& comm_data=*(CommData*)&recv_buff[i];
      recv_data.push_back(&comm_data);
      { // Add mid_indx_pair
        par::SortPair<MortonId,size_t> p;
        p.key=comm_data.mid;
        p.data=mid_indx_pair.size();
        mid_indx_pair.push_back(p);
      }
      i+=comm_data.pkd_length;
      assert(comm_data.pkd_length>0);
    }

    std::vector<Node_t*> recv_nodes(recv_data.size());
    { // Find received octants in tree.
      omp_par::merge_sort(&mid_indx_pair[0], &mid_indx_pair[0]+mid_indx_pair.size());
      std::vector<size_t> indx(omp_p+1);
      for(size_t i=0;i<=omp_p;i++){
        size_t j=(mid_indx_pair.size()*i)/omp_p;
        if(j>0) while(j<mid_indx_pair.size()-1){
          if(mid_indx_pair[j+1].key.GetDepth()<=
             mid_indx_pair[j].key.GetDepth()) break;
          j++;
        }
        indx[i]=j;
      }

      int nchld=(1UL<<this->Dim()); // Number of children.
      if(mid_indx_pair.size()>0)
      for(size_t tid=1;tid<omp_p;tid++){
        size_t j=indx[tid];
        MortonId& mid=mid_indx_pair[j].key;
        Node_t* srch_node=this->RootNode();
        while(srch_node->GetMortonId()!=mid){
          Node_t* ch_node;
          if(srch_node->IsLeaf()){
            srch_node->SetGhost(true);
            srch_node->Subdivide();
          }
          for(int j=nchld-1;j>=0;j--){
            ch_node=(Node_t*)srch_node->Child(j);
            if(ch_node->GetMortonId()<=mid){
              srch_node=ch_node;
              break;
            }
          }
        }
      }

      #pragma omp parallel for
      for(size_t tid=0;tid<omp_p;tid++){
        size_t a=indx[tid  ];
        size_t b=indx[tid+1];
        for(size_t j=a;j<b;j++){ // Find shared nodes.
          size_t i=mid_indx_pair[j].data;
          MortonId& mid=mid_indx_pair[j].key;
          Node_t* srch_node=this->RootNode();
          while(srch_node->GetMortonId()!=mid){
            Node_t* ch_node;
            if(srch_node->IsLeaf()){
              srch_node->SetGhost(true);
              srch_node->Subdivide();
            }
            for(int j=nchld-1;j>=0;j--){
              ch_node=(Node_t*)srch_node->Child(j);
              if(ch_node->GetMortonId()<=mid){
                srch_node=ch_node;
                break;
              }
            }
          }
          recv_nodes[i]=srch_node;
        }
      }
    }
    #pragma omp parallel for
    for(size_t i=0;i<recv_data.size();i++){ // Unpack
      if(!recv_nodes[i]->IsGhost()) continue;
      assert(recv_nodes[i]->IsGhost());
      CommData& comm_data=*(CommData*)recv_data[i];

      PackedData p;
      p.data=((char*)recv_data[i])+sizeof(CommData);
      p.length=comm_data.pkd_length-sizeof(CommData);
      recv_nodes[i]->Unpack(p);
    }
  }
  //Profile::Toc();

  //Profile::Tic("Broadcast", &comm, true, 5);
  { // Broadcast octants.
    std::vector<MortonId> shrd_mid;
    if(rank+1<num_p){ // Set shrd_mid.
      MortonId m=mins[rank+1];
      while(m.GetDepth()>0 && m.getDFD()>=mins[rank+1]){
        m=m.getAncestor(m.GetDepth()-1);
      }

      size_t d=m.GetDepth()+1;
      shrd_mid.resize(d);
      for(size_t i=0;i<d;i++){
        shrd_mid[i]=m.getAncestor(i);
      }
    }

    std::vector<void*> shrd_data; // CommData for shared nodes.
    { // Set shrd_data
      for(size_t i=0;i<shared_data.size();i++){
        CommData& comm_data=*(CommData*)shared_data[i];
        assert(comm_data.mid.GetDepth()>0);
        size_t d=comm_data.mid.GetDepth()-1;
        if(d<shrd_mid.size() && shrd_mid[d].getDFD()>=mins[rank])
        for(size_t j=0;j<comm_data.usr_cnt;j++){
          if(comm_data.usr_mid[j]==shrd_mid[d]){
            shrd_data.push_back(&comm_data);
            break;
          }
        }
        if(shrd_data.size()==0 || shrd_data.back()!=&comm_data) this->memgr.free(&comm_data);
      }
      for(size_t i=0;i<recv_data.size();i++){
        CommData& comm_data=*(CommData*)recv_data[i];
        assert(comm_data.mid.GetDepth()>0);
        size_t d=comm_data.mid.GetDepth()-1;
        if(d<shrd_mid.size() && shrd_mid[d].getDFD()>=mins[rank])
        for(size_t j=0;j<comm_data.usr_cnt;j++){
          if(comm_data.usr_mid[j]==shrd_mid[d]){
            char* data_ptr=(char*)this->memgr.malloc(comm_data.pkd_length);
            mem::memcopy(data_ptr, &comm_data, comm_data.pkd_length);
            shrd_data.push_back(data_ptr);
            break;
          }
        }
      }
    }

    size_t pid_shift=1;
    while(pid_shift<num_p){
      MPI_size_t recv_pid=(rank>=pid_shift?rank-pid_shift:rank);
      MPI_size_t send_pid=(rank+pid_shift<num_p?rank+pid_shift:rank);

      MPI_size_t send_length=0;
      if(send_pid!=rank){ // Send data for send_pid
        std::vector<void*> send_data;
        std::vector<size_t> send_size;
        for(size_t i=0; i<shrd_data.size();i++){
          CommData& comm_data=*(CommData*)shrd_data[i];
          size_t d=comm_data.mid.GetDepth()-1;
          bool shared=(d<shrd_mid.size() && shrd_mid[d].NextId().getDFD()>mins[send_pid].getDFD());
          if(shared) for(size_t j=0;j<comm_data.usr_cnt;j++){ // if send_pid already has this node then skip
            if(comm_data.usr_pid[j]==send_pid){
              shared=false;
              break;
            }
          }
          if(!shared) continue;

          send_data.push_back(&comm_data);
          send_size.push_back(comm_data.pkd_length);
        }
        std::vector<size_t> send_disp(send_data.size(),0);
        omp_par::scan(&send_size[0],&send_disp[0],send_data.size());
        if(send_data.size()>0) send_length=send_size.back()+send_disp.back();

        // Resize send_buff.
        if(send_buff.size()<send_length){
          send_buff.resize(send_length);
        }

        // Copy data to send_buff.
        #pragma omp parallel for
        for(size_t i=0;i<send_data.size();i++){
          CommData& comm_data=*(CommData*)send_data[i];
          mem::memcopy(&send_buff[send_disp[i]], &comm_data, comm_data.pkd_length);
        }
      }

      MPI_size_t recv_length=0;
      { // Send-Recv data
        MPI_Request request;
        MPI_Status status;
        if(recv_pid!=rank) MPI_Irecv(&recv_length, 1, par::Mpi_datatype<MPI_size_t>::value(),recv_pid, 1, *Comm(), &request);
        if(send_pid!=rank) MPI_Send (&send_length, 1, par::Mpi_datatype<MPI_size_t>::value(),send_pid, 1, *Comm());
        if(recv_pid!=rank) MPI_Wait(&request, &status);

        // Resize recv_buff
        if(recv_buff.size()<recv_length){
          recv_buff.resize(recv_length);
        }

        if(recv_length>0) MPI_Irecv(&recv_buff[0], recv_length, par::Mpi_datatype<char>::value(),recv_pid, 1, *Comm(), &request);
        if(send_length>0) MPI_Send (&send_buff[0], send_length, par::Mpi_datatype<char>::value(),send_pid, 1, *Comm());
        if(recv_length>0) MPI_Wait(&request, &status);
      }

      std::vector<void*> recv_data; // CommData for received nodes.
      { // Unpack received octants.
        std::vector<par::SortPair<MortonId,size_t> > mid_indx_pair;
        for(size_t i=0; i<recv_length;){
          CommData& comm_data=*(CommData*)&recv_buff[i];
          recv_data.push_back(&comm_data);
          { // Add mid_indx_pair
            par::SortPair<MortonId,size_t> p;
            p.key=comm_data.mid;
            p.data=mid_indx_pair.size();
            mid_indx_pair.push_back(p);
          }
          i+=comm_data.pkd_length;
          assert(comm_data.pkd_length>0);
        }

        std::vector<Node_t*> recv_nodes(recv_data.size());
        int nchld=(1UL<<this->Dim()); // Number of children.
//        for(size_t i=0;i<recv_data.size();i++){ // Find received octants in tree.
//          CommData& comm_data=*(CommData*)recv_data[i];
//          MortonId& mid=comm_data.mid;
//          Node_t* srch_node=this->RootNode();
//          while(srch_node->GetMortonId()!=mid){
//            Node_t* ch_node;
//            if(srch_node->IsLeaf()){
//              srch_node->SetGhost(true);
//              srch_node->Subdivide();
//            }
//            for(int j=nchld-1;j>=0;j--){
//              ch_node=(Node_t*)srch_node->Child(j);
//              if(ch_node->GetMortonId()<=mid){
//                srch_node=ch_node;
//                break;
//              }
//            }
//          }
//          recv_nodes[i]=srch_node;
//        }
        { // Find received octants in tree.
          omp_par::merge_sort(&mid_indx_pair[0], &mid_indx_pair[0]+mid_indx_pair.size());
          std::vector<size_t> indx(omp_p+1);
          for(size_t i=0;i<=omp_p;i++){
            size_t j=(mid_indx_pair.size()*i)/omp_p;
            if(j>0) while(j<mid_indx_pair.size()-1){
              if(mid_indx_pair[j+1].key.GetDepth()<=
                 mid_indx_pair[j].key.GetDepth()) break;
              j++;
            }
            indx[i]=j;
          }

          int nchld=(1UL<<this->Dim()); // Number of children.
          if(mid_indx_pair.size()>0)
          for(size_t tid=1;tid<omp_p;tid++){
            size_t j=indx[tid];
            MortonId& mid=mid_indx_pair[j].key;
            Node_t* srch_node=this->RootNode();
            while(srch_node->GetMortonId()!=mid){
              Node_t* ch_node;
              if(srch_node->IsLeaf()){
                srch_node->SetGhost(true);
                srch_node->Subdivide();
              }
              for(int j=nchld-1;j>=0;j--){
                ch_node=(Node_t*)srch_node->Child(j);
                if(ch_node->GetMortonId()<=mid){
                  srch_node=ch_node;
                  break;
                }
              }
            }
          }

          #pragma omp parallel for
          for(size_t tid=0;tid<omp_p;tid++){
            size_t a=indx[tid  ];
            size_t b=indx[tid+1];
            for(size_t j=a;j<b;j++){ // Find shared nodes.
              size_t i=mid_indx_pair[j].data;
              MortonId& mid=mid_indx_pair[j].key;
              Node_t* srch_node=this->RootNode();
              while(srch_node->GetMortonId()!=mid){
                Node_t* ch_node;
                if(srch_node->IsLeaf()){
                  srch_node->SetGhost(true);
                  srch_node->Subdivide();
                }
                for(int j=nchld-1;j>=0;j--){
                  ch_node=(Node_t*)srch_node->Child(j);
                  if(ch_node->GetMortonId()<=mid){
                    srch_node=ch_node;
                    break;
                  }
                }
              }
              recv_nodes[i]=srch_node;
            }
          }
        }
        #pragma omp parallel for
        for(size_t i=0;i<recv_data.size();i++){
          if(!recv_nodes[i]->IsGhost()) continue;
          assert(recv_nodes[i]->IsGhost());
          CommData& comm_data=*(CommData*)recv_data[i];

          PackedData p;
          p.data=((char*)recv_data[i])+sizeof(CommData);
          p.length=comm_data.pkd_length-sizeof(CommData);
          recv_nodes[i]->Unpack(p);
        }
      }

      pid_shift<<=1;
      send_pid=(rank+pid_shift<num_p?rank+pid_shift:rank);
      if(send_pid!=rank){ // Set shrd_data
        for(size_t i=0;i<recv_data.size();i++){
          CommData& comm_data=*(CommData*)recv_data[i];

          //{ // Skip if this node already exists.
          //  bool skip=false;
          //  for(size_t k=0;k<shrd_data.size();k++){
          //    CommData& comm_data_=*(CommData*)shrd_data[k];
          //    if(comm_data_.mid==comm_data.mid){
          //      assert(false);
          //      skip=true;
          //      break;
          //    }
          //  }
          //  if(skip) continue;
          //}

          assert(comm_data.mid.GetDepth()>0);
          size_t d=comm_data.mid.GetDepth()-1;
          if(d<shrd_mid.size() && shrd_mid[d].isAncestor(mins[rank]) && shrd_mid[d].NextId().getDFD()>mins[send_pid].getDFD())
          for(size_t j=0;j<comm_data.usr_cnt;j++){
            if(comm_data.usr_mid[j]==shrd_mid[d]){
              char* data_ptr=(char*)this->memgr.malloc(comm_data.pkd_length);
              mem::memcopy(data_ptr, &comm_data, comm_data.pkd_length);
              shrd_data.push_back(data_ptr);
              break;
            }
          }
        }
      }
    }

    // Free data
    //Profile::Tic("Free", &comm, false, 5);
    for(size_t i=0;i<shrd_data.size();i++) this->memgr.free(shrd_data[i]);
    //Profile::Toc();
  }
  //Profile::Toc();
}


inline bool isLittleEndian(){
  uint16_t number = 0x1;
  uint8_t *numPtr = (uint8_t*)&number;
  return (numPtr[0] == 1);
}

template <class TreeNode>
void MPI_Tree<TreeNode>::Write2File(const char* fname, int lod){
  typedef double VTKData_t;
  int myrank, np;
  MPI_Comm_size(*Comm(),&np);
  MPI_Comm_rank(*Comm(),&myrank);

  std::vector<Real_t> coord_;  //Coordinates of octant corners.
  std::vector<Real_t> value_;  //Data value at points.
  std::vector<VTKData_t> coord;  //Coordinates of octant corners.
  std::vector<VTKData_t> value;  //Data value at points.
  std::vector<int32_t> mpi_rank;  //MPI_Rank at points.
  std::vector<int32_t> connect;  //Cell connectivity.
  std::vector<int32_t> offset ;  //Cell offset.
  std::vector<uint8_t> types  ;  //Cell types.

  //Build list of octant corner points.
  Node_t* n=this->PreorderFirst();
  while(n!=NULL){
    if(!n->IsGhost() && n->IsLeaf())
      n->VTU_Data(coord_, value_, connect, offset, types, lod);
    n=this->PreorderNxt(n);
  }
  for(size_t i=0;i<coord_.size();i++) coord.push_back(coord_[i]);
  for(size_t i=0;i<value_.size();i++) value.push_back(value_[i]);
  int pt_cnt=coord.size()/COORD_DIM;
  int dof=(pt_cnt?value.size()/pt_cnt:0);
  assert(value.size()==(size_t)pt_cnt*dof);
  int cell_cnt=types.size();

  mpi_rank.resize(pt_cnt);
  int new_myrank=myrank;//rand();
  for(int i=0;i<pt_cnt;i++) mpi_rank[i]=new_myrank;

  //Open file for writing.
  std::stringstream vtufname;
  vtufname<<fname<<std::setfill('0')<<std::setw(6)<<myrank<<".vtu";
  std::ofstream vtufile;
  vtufile.open(vtufname.str().c_str());
  if(vtufile.fail()) return;

  //Proceed to write to file.
  size_t data_size=0;
  vtufile<<"<?xml version=\"1.0\"?>\n";
  if(isLittleEndian()) vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  else                 vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  //===========================================================================
  vtufile<<"  <UnstructuredGrid>\n";
  vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfCells=\""<<cell_cnt<<"\">\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Points>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKData_t)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKData_t);
  vtufile<<"      </Points>\n";
  //---------------------------------------------------------------------------
  vtufile<<"      <PointData>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKData_t)*8<<"\" NumberOfComponents=\""<<dof<<"\" Name=\"value\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+value.size()*sizeof(VTKData_t);
  vtufile<<"        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+mpi_rank.size()*sizeof(int32_t);
  vtufile<<"      </PointData>\n";
  //---------------------------------------------------------------------------
  //---------------------------------------------------------------------------
  vtufile<<"      <Cells>\n";
  vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
  vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
  vtufile<<"        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+types.size()  *sizeof(uint8_t);
  vtufile<<"      </Cells>\n";
  //---------------------------------------------------------------------------
  //vtufile<<"      <CellData>\n";
  //vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKData_t)*8<<"\" Name=\"Velocity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  //vtufile<<"      </CellData>\n";
  //---------------------------------------------------------------------------

  vtufile<<"    </Piece>\n";
  vtufile<<"  </UnstructuredGrid>\n";
  //===========================================================================
  vtufile<<"  <AppendedData encoding=\"raw\">\n";
  vtufile<<"    _";

  int32_t block_size;
  block_size=coord   .size()*sizeof(VTKData_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord   [0], coord   .size()*sizeof(VTKData_t));
  block_size=value   .size()*sizeof(VTKData_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value   [0], value   .size()*sizeof(VTKData_t));
  block_size=mpi_rank.size()*sizeof(  int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&mpi_rank[0], mpi_rank.size()*sizeof(  int32_t));

  block_size=connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect[0], connect.size()*sizeof(int32_t));
  block_size=offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset [0], offset .size()*sizeof(int32_t));
  block_size=types  .size()*sizeof(uint8_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&types  [0], types  .size()*sizeof(uint8_t));

  vtufile<<"\n";
  vtufile<<"  </AppendedData>\n";
  //===========================================================================
  vtufile<<"</VTKFile>\n";
  vtufile.close();


  if(myrank) return;
  std::stringstream pvtufname;
  pvtufname<<fname<<".pvtu";
  std::ofstream pvtufile;
  pvtufile.open(pvtufname.str().c_str());
  if(pvtufile.fail()) return;
  pvtufile<<"<?xml version=\"1.0\"?>\n";
  pvtufile<<"<VTKFile type=\"PUnstructuredGrid\">\n";
  pvtufile<<"  <PUnstructuredGrid GhostLevel=\"0\">\n";
  pvtufile<<"      <PPoints>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKData_t)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\"/>\n";
  pvtufile<<"      </PPoints>\n";
  pvtufile<<"      <PPointData>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKData_t)*8<<"\" NumberOfComponents=\""<<dof<<"\" Name=\"value\"/>\n";
  pvtufile<<"        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
  pvtufile<<"      </PPointData>\n";
  {
    // Extract filename from path.
    std::stringstream vtupath;
    vtupath<<'/'<<fname<<'\0';
    char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
    //std::string fname_ = boost::filesystem::path(fname).filename().string().
    for(int i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<std::setfill('0')<<std::setw(6)<<i<<".vtu\"/>\n";
  }
  pvtufile<<"  </PUnstructuredGrid>\n";
  pvtufile<<"</VTKFile>\n";
  pvtufile.close();
}


template <class TreeNode>
const std::vector<MortonId>& MPI_Tree<TreeNode>::GetMins(){
  Node_t* n=this->PreorderFirst();
  while(n!=NULL){
    if(!n->IsGhost() && n->IsLeaf()) break;
    n=this->PreorderNxt(n);
  }
  ASSERT_WITH_MSG(n!=NULL,"No non-ghost nodes found on this process.");

  MortonId my_min;
  my_min=n->GetMortonId();

  int np;
  MPI_Comm_size(*Comm(),&np);
  mins.resize(np);

  MPI_Allgather(&my_min , 1, par::Mpi_datatype<MortonId>::value(),
                &mins[0], 1, par::Mpi_datatype<MortonId>::value(), *Comm());

  return mins;
}

}//end namespace
