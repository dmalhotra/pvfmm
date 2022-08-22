/**
 * \file pvfmm.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 26-10-2018
 * \brief This file contains the definitions of the wrapper functions for PVFMM.
 */

#include <mpi_node.hpp>
#include <fmm_tree.hpp>
#include <fmm_pts.hpp>
#include <vector.hpp>
#include <parUtils.h>

namespace pvfmm{

template <class Real>
inline ChebFMM_Tree<Real>* ChebFMM_CreateTree(int cheb_deg, int data_dim, ChebFn<Real> fn_ptr, std::vector<Real>& trg_coord, MPI_Comm& comm,
                                              Real tol, int max_pts, BoundaryType bndry, int init_depth){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  ChebFMM_Data<Real> tree_data;
  tree_data.cheb_deg=cheb_deg;
  tree_data.data_dof=data_dim;
  tree_data.input_fn=fn_ptr;
  tree_data.tol=tol;
  bool adap=true;

  tree_data.dim=PVFMM_COORD_DIM;
  tree_data.max_depth=PVFMM_MAX_DEPTH;
  tree_data.max_pts=max_pts;

  { // Set points for initial tree.
    std::vector<Real> coord;
    size_t N=sctl::pow<unsigned int>(8,init_depth);
    N=(N<(size_t)np?np:N)*max_pts;
    size_t NN=(size_t)ceil(sctl::pow<Real>(N,1.0/3.0));
    size_t N_total=NN*NN*NN;
    size_t start= myrank   *N_total/np;
    size_t end  =(myrank+1)*N_total/np;
    for(size_t i=start;i<end;i++){
      coord.push_back((((i/  1    )%NN)+(Real)0.5)/NN);
      coord.push_back((((i/ NN    )%NN)+(Real)0.5)/NN);
      coord.push_back((((i/(NN*NN))%NN)+(Real)0.5)/NN);
    }
    tree_data.pt_coord=coord;
  }

  // Set target points.
  tree_data.trg_coord=trg_coord;

  auto* tree=new ChebFMM_Tree<Real>(comm);
  tree->Initialize(&tree_data);
  tree->InitFMM_Tree(adap,bndry);
  return tree;
}

template <class Real>
inline ChebFMM_Tree<Real>* ChebFMM_CreateTree(int cheb_deg, std::vector<Real>& node_coord, std::vector<Real>& fn_coeff, std::vector<Real>& trg_coord, MPI_Comm& comm, BoundaryType bndry){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  ChebFMM_Data<Real> tree_data;
  tree_data.input_fn=ChebFn<Real>();
  tree_data.tol=0;
  bool adap=false;

  tree_data.dim=PVFMM_COORD_DIM;
  tree_data.max_depth=PVFMM_MAX_DEPTH;
  tree_data.max_pts=1;

  tree_data.cheb_deg=cheb_deg;
  tree_data.pt_value=fn_coeff;
  tree_data.pt_coord=node_coord;
  const long Ncoeff = (cheb_deg+1)*(cheb_deg+2)*(cheb_deg+3)/6;
  { // Set data_dof
    long long glb_size[2], loc_size[2] = {(long long)node_coord.size()/PVFMM_COORD_DIM, (long long)fn_coeff.size()/Ncoeff};
    MPI_Allreduce(&loc_size, &glb_size, 2, par::Mpi_datatype<long long>::value(), par::Mpi_datatype<long long>::sum(), comm);
    tree_data.data_dof = glb_size[1]/glb_size[0];
  }
  assert(node_coord.size() && (node_coord.size()/PVFMM_COORD_DIM)*PVFMM_COORD_DIM == node_coord.size());
  assert(fn_coeff.size() == node_coord.size()/PVFMM_COORD_DIM * tree_data.data_dof * Ncoeff);

  // Set target points.
  tree_data.trg_coord=trg_coord;

  auto* tree=new ChebFMM_Tree<Real>(comm);
  tree->Initialize(&tree_data);

  for (auto& node : tree->GetNodeList()) {
    if(node->IsLeaf() && !node->IsGhost()){
      node->ChebData() = node->pt_value;
      node->pt_value.ReInit(0);
    } else {
      node->ChebData().ReInit(0);
    }
  }

  tree->InitFMM_Tree(adap,bndry);
  return tree;
}

template <class Real>
inline void ChebFMM_Evaluate(std::vector<Real>& trg_val, ChebFMM_Tree<Real>* tree, size_t loc_size){
  tree->RunFMM();

  Vector<Real> trg_value;
  Vector<size_t> trg_scatter;
  {// Collect data from each node to trg_value and trg_scatter.
    std::vector<Real> trg_value_;
    std::vector<size_t> trg_scatter_;
    const auto& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        const auto& trg_value=nodes[i]->trg_value;
        const auto& trg_scatter=nodes[i]->trg_scatter;
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

template <class Real>
inline void ChebFMM_GetPotentialCoeff(std::vector<Real>& coeff, ChebFMM_Tree<Real>* tree){
  coeff.resize(0);
  const auto& nodes=tree->GetNodeList();
  for(size_t i=0;i<nodes.size();i++){
    if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
      Vector<Real>& cheb_out =((typename ChebFMM<Real>::FMMData*)nodes[i]->FMMData())->cheb_out;
      for (size_t k = 0; k < cheb_out.Dim(); k++) {
        coeff.push_back(cheb_out[k]);
      }
    }
  }
}

template <class Real>
inline void ChebFMM_GetLeafCoord(std::vector<Real>& node_coord, ChebFMM_Tree<Real>* tree){
  node_coord.resize(0);
  const auto& nodes=tree->GetNodeList();
  for(size_t i=0;i<nodes.size();i++){
    if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
      const auto& coord = nodes[i]->Coord();
      for (int j = 0; j < 3; j++) node_coord.push_back(coord[j]);
    }
  }
}

template <class Real>
void ChebFMM_Coeff2Nodes(std::vector<Real>& node_val, int ChebDeg, int dof, const std::vector<Real>& coeff){
  const long M0 = (ChebDeg+1)*(ChebDeg+2)*(ChebDeg+3)/6;
  const long M1 = (ChebDeg+1)*(ChebDeg+1)*(ChebDeg+1);
  const long N = coeff.size() / M0 / dof;
  assert(coeff.size() == (size_t)N*M0*dof);

  if (node_val.size() != (size_t)N*M1*dof) node_val.resize(N*M1*dof);

  std::vector<Real> cheb_nds(ChebDeg+1);
  for (long i = 0; i < ChebDeg+1; i++) cheb_nds[i] = -(Real)cos(M_PI*(i*2+1)/(ChebDeg+1)/2);

  #pragma omp parallel
  {
    Vector<Real> buff(dof*M1);

    int np = omp_get_num_threads();
    int tid = omp_get_thread_num();
    long a = N*(tid+0)/np;
    long b = N*(tid+1)/np;
    for (long i = a; i < b; i++) {
      const Vector<Real> coeff_(dof*M0, (Real*)coeff.data() + i * dof*M0, false);
      cheb_eval(coeff_, ChebDeg, cheb_nds, cheb_nds, cheb_nds, buff);

      const Matrix<Real> buff_(dof,M1, buff.Begin(), false);
      Matrix<Real> node_val_(M1,dof, (Real*)node_val.data() + i * M1*dof, false);
      Matrix<Real>::Transpose(node_val_, buff_);
    }
  }
}

template <class Real>
void ChebFMM_Nodes2Coeff(std::vector<Real>& coeff, int ChebDeg, int dof, const std::vector<Real>& node_val) {
  const long M0 = (ChebDeg+1)*(ChebDeg+2)*(ChebDeg+3)/6;
  const long M1 = (ChebDeg+1)*(ChebDeg+1)*(ChebDeg+1);
  const long N = node_val.size() / M1 / dof;
  assert(node_val.size() == (size_t)N*M1*dof);

  if (coeff.size() != (size_t)N*M0*dof) coeff.resize(N*M0*dof);

  #pragma omp parallel
  {
    Vector<Real> buff(dof*M1);

    int np = omp_get_num_threads();
    int tid = omp_get_thread_num();
    long a = N*(tid+0)/np;
    long b = N*(tid+1)/np;
    for (long i = a; i < b; i++) {
      const Matrix<Real> node_val_(M1,dof, (Real*)node_val.data() + i * M1*dof, false);
      Matrix<Real> buff_(dof,M1, buff.Begin(), false);
      Matrix<Real>::Transpose(buff_, node_val_);

      cheb_approx<Real,Real>(buff.Begin(), ChebDeg, dof, coeff.data()+i*dof*M0);
    }
  }
}


template <class Real>
inline PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>&  src_coord, const std::vector<Real>&  src_value,
                                          const std::vector<Real>& surf_coord, const std::vector<Real>& surf_value,
                                          const std::vector<Real>& trg_coord, const MPI_Comm& comm, int max_pts,
                                          BoundaryType bndry, int init_depth){
  int np, myrank;
  MPI_Comm_size(comm, &np);
  MPI_Comm_rank(comm, &myrank);

  PtFMM_Data<Real> tree_data;
  bool adap=true;

  tree_data.dim=PVFMM_COORD_DIM;
  tree_data.max_depth=PVFMM_MAX_DEPTH;
  tree_data.max_pts=max_pts;

  // Set source points.
  tree_data. src_coord= src_coord;
  tree_data. src_value= src_value;
  tree_data.surf_coord=surf_coord;
  tree_data.surf_value=surf_value;

  // Set target points.
  tree_data.trg_coord=trg_coord;
  tree_data. pt_coord=trg_coord;

  auto* tree=new PtFMM_Tree<Real>(comm);
  tree->Initialize(&tree_data);
  tree->InitFMM_Tree(adap,bndry);
  return tree;
}

template <class Real>
inline PtFMM_Tree<Real>* PtFMM_CreateTree(const std::vector<Real>& src_coord, const std::vector<Real>&  src_value,
                                          const std::vector<Real>& trg_coord, const MPI_Comm& comm, int max_pts,
                                          BoundaryType bndry, int init_depth){
  std::vector<Real> surf_coord;
  std::vector<Real> surf_value;
  return PtFMM_CreateTree(src_coord, src_value, surf_coord,surf_value, trg_coord, comm, max_pts, bndry, init_depth);
}

template <class Real>
inline void PtFMM_Evaluate(PtFMM_Tree<Real>* tree, std::vector<Real>& trg_val, size_t loc_size, const std::vector<Real>* src_val, const std::vector<Real>* surf_val){
  if(src_val){
    std::vector<size_t> src_scatter_;
    const auto& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        const auto& src_scatter=nodes[i]->src_scatter;
        for(size_t j=0;j<src_scatter.Dim();j++) src_scatter_.push_back(src_scatter[j]);
      }
    }

    Vector<Real> src_value=*src_val;
    Vector<size_t> src_scatter=src_scatter_;
    par::ScatterForward(src_value,src_scatter,*tree->Comm());

    size_t indx=0;
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<Real>& src_value_=nodes[i]->src_value;
        for(size_t j=0;j<src_value_.Dim();j++){
          src_value_[j]=src_value[indx];
          indx++;
        }
      }
    }
  }
  if(surf_val){
    std::vector<size_t> surf_scatter_;
    const auto& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        const auto& surf_scatter=nodes[i]->surf_scatter;
        for(size_t j=0;j<surf_scatter.Dim();j++) surf_scatter_.push_back(surf_scatter[j]);
      }
    }

    Vector<Real> surf_value=*surf_val;
    Vector<size_t> surf_scatter=surf_scatter_;
    par::ScatterForward(surf_value,surf_scatter,*tree->Comm());

    size_t indx=0;
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        Vector<Real>& surf_value_=nodes[i]->surf_value;
        for(size_t j=0;j<surf_value_.Dim();j++){
          surf_value_[j]=surf_value[indx];
          indx++;
        }
      }
    }
  }
  tree->RunFMM();
  Vector<Real> trg_value;
  Vector<size_t> trg_scatter;
  {
    std::vector<Real> trg_value_;
    std::vector<size_t> trg_scatter_;
    const auto& nodes=tree->GetNodeList();
    for(size_t i=0;i<nodes.size();i++){
      if(nodes[i]->IsLeaf() && !nodes[i]->IsGhost()){
        const auto& trg_value=nodes[i]->trg_value;
        const auto& trg_scatter=nodes[i]->trg_scatter;
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

