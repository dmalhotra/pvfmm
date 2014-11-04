/**
 * \file fmm_tree.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the definition of the FMM_Tree class.
 */

#include <mpi.h>
#include <vector>

#include <pvfmm_common.hpp>
#include <interac_list.hpp>
#include <fmm_node.hpp>
#include <mpi_tree.hpp>
#include <matrix.hpp>

#ifndef _PVFMM_FMM_TREE_HPP_
#define _PVFMM_FMM_TREE_HPP_

namespace pvfmm{

/**
 * \brief Base class for FMM tree.
 */
template <class FMM_Mat_t>
class FMM_Tree: public MPI_Tree<typename FMM_Mat_t::FMMNode_t>{
  friend FMM_Mat_t;

 public:

  typedef typename FMM_Mat_t::FMMNode_t Node_t;
  typedef typename FMM_Mat_t::Real_t Real_t;

  /**
   * \brief Constructor.
   */
  FMM_Tree(MPI_Comm c): MPI_Tree<Node_t>(c), fmm_mat(NULL), bndry(FreeSpace) { };

  /**
   * \brief Virtual destructor.
   */
  virtual ~FMM_Tree(){
  }

  /**
   * \brief Initialize the distributed MPI tree.
   */
  virtual void Initialize(typename Node_t::NodeData* data_) ;

  /**
   * \brief Initialize FMM_Tree.
   */
  void InitFMM_Tree(bool refine, BoundaryType bndry=FreeSpace);

  /**
   * \brief Run FMM
   */
  void SetupFMM(FMM_Mat_t* fmm_mat_);

  /**
   * \brief Run FMM
   */
  void RunFMM();

  /**
   * \brief Clear FMM data: multipole, local expansions and target potential.
   */
  void ClearFMMData();

  /**
   * \brief Build interaction lists for all nodes.
   */
  void BuildInteracLists();

  /**
   * \brief Upward FMM pass (Including MultipoleReduceBcast).
   */
  void UpwardPass();

  /**
   * \brief Reduction and broadcast of multipole expansions.
   */
  void MultipoleReduceBcast() ;

  /**
   * \brief Downward FMM pass.
   */
  void DownwardPass();

  /**
   * \brief Copy FMM output to the tree.
   */
  void Copy_FMMOutput();

 protected:

  std::vector<Matrix<Real_t> > node_data_buff;
  InteracList<Node_t> interac_list;
  FMM_Mat_t* fmm_mat; //Computes all FMM translations.
  BoundaryType bndry;

  std::vector<Matrix<char> > precomp_lst; //Precomputed data for each interaction type.
  std::vector<SetupData<Real_t> > setup_data;

  std::vector<Vector<Real_t> > upwd_check_surf;
  std::vector<Vector<Real_t> > upwd_equiv_surf;
  std::vector<Vector<Real_t> > dnwd_check_surf;
  std::vector<Vector<Real_t> > dnwd_equiv_surf;
};

}//end namespace

#include <fmm_tree.txx>

#endif //_PVFMM_FMM_TREE_HPP_

