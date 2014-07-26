/**
 * \file pvfmm.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 1-2-2014
 * \brief This file contains wrapper functions for PvFMM.
 */

#ifndef _PVFMM_HPP_
#define _PVFMM_HPP_

#include <mpi.h>
#include <cstdlib>
#include <iostream>

#include <pvfmm_common.hpp>
#include <fmm_cheb.hpp>
#include <fmm_node.hpp>
#include <fmm_tree.hpp>

namespace pvfmm{

typedef FMM_Node<Cheb_Node<double> > ChebFMM_Node;
typedef FMM_Cheb<ChebFMM_Node>       ChebFMM;
typedef FMM_Tree<ChebFMM>            ChebFMM_Tree;
typedef ChebFMM_Node::NodeData       ChebFMM_Data;
typedef void (*ChebFn)(double* , int , double*);

ChebFMM_Tree* ChebFMM_CreateTree(int cheb_deg, int data_dim, ChebFn fn_ptr, std::vector<double>& trg_coord, MPI_Comm& comm,
                                 double tol=1e-6, int max_pts=100, BoundaryType bndry=FreeSpace, int init_depth=0){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  ChebFMM_Data tree_data;
  tree_data.cheb_deg=cheb_deg;
  tree_data.data_dof=data_dim;
  tree_data.input_fn=fn_ptr;
  tree_data.tol=tol;
  bool adap=true;

  tree_data.dim=COORD_DIM;
  tree_data.max_depth=MAX_DEPTH;
  tree_data.max_pts=max_pts;

  { // Set points for initial tree.
    std::vector<double> coord;
    size_t N=pow(8.0,init_depth);
    N=(N<np?np:N)*max_pts;
    size_t NN=ceil(pow((double)N,1.0/3.0));
    size_t N_total=NN*NN*NN;
    size_t start= myrank   *N_total/np;
    size_t end  =(myrank+1)*N_total/np;
    for(size_t i=start;i<end;i++){
      coord.push_back(((double)((i/  1    )%NN)+0.5)/NN);
      coord.push_back(((double)((i/ NN    )%NN)+0.5)/NN);
      coord.push_back(((double)((i/(NN*NN))%NN)+0.5)/NN);
    }
    tree_data.pt_coord=coord;
  }

  // Set target points.
  tree_data.trg_coord=trg_coord;

  ChebFMM_Tree* tree=new ChebFMM_Tree(comm);
  tree->Initialize(&tree_data);
  tree->InitFMM_Tree(adap,bndry);
  return tree;
}

void ChebFMM_Evaluate(ChebFMM_Tree* tree, std::vector<double>& trg_val, size_t loc_size=0){
  tree->RunFMM();
  Vector<double> trg_value;
  Vector<size_t> trg_scatter;
  {// Collect data from each node to trg_value and trg_scatter.
    std::vector<double> trg_value_;
    std::vector<size_t> trg_scatter_;
    std::vector<ChebFMM_Node*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<double>& trg_value=nodes[i]->trg_value;
        Vector<size_t>& trg_scatter=nodes[i]->trg_scatter;
        for(size_t j=0;j<trg_value.Dim();j++) trg_value_.push_back(trg_value[j]);
        for(size_t j=0;j<trg_scatter.Dim();j++) trg_scatter_.push_back(trg_scatter[j]);
      }
    }
    trg_value=trg_value_;
    trg_scatter=trg_scatter_;
  }
  par::ScatterReverse(trg_value,trg_scatter,*tree->Comm(),loc_size);
  trg_val.assign(&trg_value[0],&trg_value[0]+trg_value.Dim());;
}




typedef FMM_Node<MPI_Node<double> > PtFMM_Node;
typedef FMM_Pts<PtFMM_Node>         PtFMM;
typedef FMM_Tree<PtFMM>             PtFMM_Tree;
typedef PtFMM_Node::NodeData        PtFMM_Data;

PtFMM_Tree* PtFMM_CreateTree(std::vector<double>&  src_coord, std::vector<double>&  src_value,
                             std::vector<double>& surf_coord, std::vector<double>& surf_value,
                             std::vector<double>& trg_coord, MPI_Comm& comm, int max_pts=100,
                             BoundaryType bndry=FreeSpace, int init_depth=0){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  PtFMM_Data tree_data;
  bool adap=true;

  tree_data.dim=COORD_DIM;
  tree_data.max_depth=MAX_DEPTH;
  tree_data.max_pts=max_pts;

  // Set source points.
  tree_data. src_coord= src_coord;
  tree_data. src_value= src_value;
  tree_data.surf_coord=surf_coord;
  tree_data.surf_value=surf_value;

  // Set target points.
  tree_data.trg_coord=trg_coord;
  tree_data. pt_coord=trg_coord;

  PtFMM_Tree* tree=new PtFMM_Tree(comm);
  tree->Initialize(&tree_data);
  tree->InitFMM_Tree(adap,bndry);
  return tree;
}

PtFMM_Tree* PtFMM_CreateTree(std::vector<double>&  src_coord, std::vector<double>&  src_value,
                             std::vector<double>& trg_coord, MPI_Comm& comm, int max_pts=100,
                             BoundaryType bndry=FreeSpace, int init_depth=0){
  std::vector<double> surf_coord;
  std::vector<double> surf_value;
  return PtFMM_CreateTree(src_coord, src_value, surf_coord,surf_value, trg_coord, comm, max_pts, bndry, init_depth);
}

void PtFMM_Evaluate(PtFMM_Tree* tree, std::vector<double>& trg_val, size_t loc_size=0, std::vector<double>* src_val=NULL, std::vector<double>* surf_val=NULL){
  if(src_val){
    std::vector<size_t> src_scatter_;
    std::vector<PtFMM_Node*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<size_t>& src_scatter=nodes[i]->src_scatter;
        for(size_t j=0;j<src_scatter.Dim();j++) src_scatter_.push_back(src_scatter[j]);
      }
    }

    Vector<double> src_value=*src_val;
    Vector<size_t> src_scatter=src_scatter_;
    par::ScatterForward(src_value,src_scatter,*tree->Comm());

    size_t indx=0;
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<double>& src_value_=nodes[i]->src_value;
        for(size_t j=0;j<src_value_.Dim();j++){
          src_value_[j]=src_value[indx];
          indx++;
        }
      }
    }
  }
  if(surf_val){
    std::vector<size_t> surf_scatter_;
    std::vector<PtFMM_Node*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<size_t>& surf_scatter=nodes[i]->surf_scatter;
        for(size_t j=0;j<surf_scatter.Dim();j++) surf_scatter_.push_back(surf_scatter[j]);
      }
    }

    Vector<double> surf_value=*surf_val;
    Vector<size_t> surf_scatter=surf_scatter_;
    par::ScatterForward(surf_value,surf_scatter,*tree->Comm());

    size_t indx=0;
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<double>& surf_value_=nodes[i]->surf_value;
        for(size_t j=0;j<surf_value_.Dim();j++){
          surf_value_[j]=surf_value[indx];
          indx++;
        }
      }
    }
  }
  tree->RunFMM();
  Vector<double> trg_value;
  Vector<size_t> trg_scatter;
  {
    std::vector<double> trg_value_;
    std::vector<size_t> trg_scatter_;
    std::vector<PtFMM_Node*>& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<double>& trg_value=nodes[i]->trg_value;
        Vector<size_t>& trg_scatter=nodes[i]->trg_scatter;
        for(size_t j=0;j<trg_value.Dim();j++) trg_value_.push_back(trg_value[j]);
        for(size_t j=0;j<trg_scatter.Dim();j++) trg_scatter_.push_back(trg_scatter[j]);
      }
    }
    trg_value=trg_value_;
    trg_scatter=trg_scatter_;
  }
  par::ScatterReverse(trg_value,trg_scatter,*tree->Comm(),loc_size);
  trg_val.assign(&trg_value[0],&trg_value[0]+trg_value.Dim());;
}

}//end namespace

#endif //_PVFMM_HPP_
