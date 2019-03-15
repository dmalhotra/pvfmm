/**
 * \file pvfmm.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 26-10-2018
 * \brief This file contains the definitions of the wrapper functions for PVFMM.
 */

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
    size_t N=pvfmm::pow<unsigned int>(8,init_depth);
    N=(N<np?np:N)*max_pts;
    size_t NN=ceil(pvfmm::pow<Real>(N,1.0/3.0));
    size_t N_total=NN*NN*NN;
    size_t start= myrank   *N_total/np;
    size_t end  =(myrank+1)*N_total/np;
    for(size_t i=start;i<end;i++){
      coord.push_back(((Real)((i/  1    )%NN)+0.5)/NN);
      coord.push_back(((Real)((i/ NN    )%NN)+0.5)/NN);
      coord.push_back(((Real)((i/(NN*NN))%NN)+0.5)/NN);
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
inline void ChebFMM_Evaluate(ChebFMM_Tree<Real>* tree, std::vector<Real>& trg_val, size_t loc_size){
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

