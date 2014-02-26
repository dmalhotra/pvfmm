/**
 * \file fmm_pts.hpp
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 3-07-2011
 * \brief This file contains the definition of the FMM_Pts class.
 * This handles all the translations for point sources and targets.
 */

#ifndef _PVFMM_FMM_PTS_HPP_
#define _PVFMM_FMM_PTS_HPP_

#include <pvfmm_common.hpp>
#include <mpi.h>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <interac_list.hpp>
#include <kernel.hpp>
#include <mpi_node.hpp>

#if defined(PVFMM_HAVE_CUDA)
#include <cuda_func.hpp>
#endif

namespace pvfmm{

/**
 * \brief This class contains FMM specific data that each node contains
 * along with the functions for manipulating the data.
 */
template <class Real_t>
class FMM_Data{

 public:

  virtual ~FMM_Data(){}

  virtual FMM_Data* NewData(){return new FMM_Data;}

  /**
   * \brief Clear all data.
   */
  virtual void Clear();

  /**
   * \brief Pack multipole expansion.
   */
  virtual PackedData PackMultipole(void* buff_ptr=NULL);

  /**
   * \brief Add the multipole expansion from p0 to the current multipole
   * expansion.
   */
  virtual void AddMultipole(PackedData p0);

  /**
   * \brief Initialize multipole expansion using p0.
   */
  virtual void InitMultipole(PackedData p0, bool own_data=true);

  //FMM specific node data.
  Vector<Real_t> upward_equiv;
  Vector<Real_t> dnward_equiv;
};


template <class Real_t>
struct SetupData{
  int level;
  Kernel<Real_t>* kernel;
  std::vector<Mat_Type> interac_type;

  std::vector<void*> nodes_in ;
  std::vector<void*> nodes_out;
  std::vector<Vector<Real_t>*>  input_vector;
  std::vector<Vector<Real_t>*> output_vector;

  //#####################################################

  Matrix< char>  interac_data;
  Matrix< char>* precomp_data;
  Matrix<Real_t>*  coord_data;
  Matrix<Real_t>*  input_data;
  Matrix<Real_t>* output_data;
};


template <class FMMNode>
class FMM_Pts{

 public:

  typedef typename FMMNode::Real_t Real_t;
  typedef FMMNode FMMNode_t;

  class FMMData: public FMM_Data<Real_t>{

   public:

    virtual ~FMMData(){}

    virtual FMM_Data<Real_t>* NewData(){return new FMMData;}
  };

  /**
   * \brief Constructor.
   */
  FMM_Pts(): vprecomp_fft_flag(false), vlist_fft_flag(false),
               vlist_ifft_flag(false), mat(NULL){};

  /**
   * \brief Virtual destructor.
   */
  virtual ~FMM_Pts();

  /**
   * \brief Initialize all the translation matrices (or load from file).
   * \param[in] mult_order Order of multipole expansion.
   * \param[in] kernel Kernel functions and related data.
   */
  void Initialize(int mult_order, const MPI_Comm& comm, const Kernel<Real_t>* kernel, const Kernel<Real_t>* aux_kernel=NULL);

  /**
   * \brief Order for the multipole expansion.
   */
  int& MultipoleOrder(){return multipole_order;}

  /**
   * \brief Whether using homogeneous kernel?
   */
  bool& Homogen(){return kernel.homogen;}

  virtual void CollectNodeData(std::vector<FMMNode*>& nodes, std::vector<Matrix<Real_t> >& buff, std::vector<Vector<FMMNode_t*> >& n_list, std::vector<size_t> extra_size = std::vector<size_t>(0));

  void SetupPrecomp(SetupData<Real_t>& setup_data, bool device=false);
  void SetupInterac(SetupData<Real_t>& setup_data, bool device=false);
  void EvalList    (SetupData<Real_t>& setup_data, bool device=false); // Run on CPU by default.

  void SetupInteracPts(SetupData<Real_t>& setup_data, bool shift_src, bool shift_trg, Matrix<Real_t>* M, bool device);
  void EvalListPts    (SetupData<Real_t>& setup_data, bool device=false); // Run on CPU by default.

  /**
   * \brief Initialize multipole expansions for the given array of leaf nodes
   * at a given level.
   */
  virtual void InitMultipole(FMMNode**, size_t n, int level);

  /**
   * \brief Initialize multipole expansions for the given array of non-leaf
   * nodes from that of its children.
   */
  virtual void Up2Up(FMMNode**, size_t n, int level);

  virtual void PeriodicBC(FMMNode* node);

  /**
   * \brief Compute V-List intractions.
   */
  virtual void V_ListSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void V_List     (SetupData<Real_t>&  setup_data, bool device=false);

  /**
   * \brief Compute X-List intractions.
   */
  virtual void X_ListSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void X_List     (SetupData<Real_t>&  setup_data, bool device=false);

  /**
   * \brief Compute contribution of local expansion from the parent.
   */
  virtual void Down2DownSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void Down2Down     (SetupData<Real_t>&  setup_data, bool device=false);

  /**
   * \brief Compute target potential from the local expansion.
   */
  virtual void Down2TargetSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void Down2Target     (SetupData<Real_t>&  setup_data, bool device=false);

  /**
   * \brief Compute W-List intractions.
   */
  virtual void W_ListSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void W_List     (SetupData<Real_t>&  setup_data, bool device=false);

  /**
   * \brief Compute U-List intractions.
   */
  virtual void U_ListSetup(SetupData<Real_t>&  setup_data, std::vector<Matrix<Real_t> >& node_data, std::vector<Vector<FMMNode_t*> >& n_list, int level, bool device);
  virtual void U_List     (SetupData<Real_t>&  setup_data, bool device=false);

  virtual void PostProcessing(std::vector<FMMNode_t*>& nodes);

  /**
   * \brief For each node, copy FMM output from FMM_Data to the node.
   */
  virtual void CopyOutput(FMMNode** nodes, size_t n);

  Vector<char> dev_buffer;
  Vector<char> cpu_buffer;

 protected:

  virtual void PrecompAll(Mat_Type type, int level=-1);

  virtual Permutation<Real_t>& PrecompPerm(Mat_Type type, Perm_Type perm_indx);

  virtual Matrix<Real_t>& Precomp(int level, Mat_Type type, size_t mat_indx);
  typename FFTW_t<Real_t>::plan vprecomp_fftplan; bool vprecomp_fft_flag;

  void FFT_UpEquiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& fft_vec,
      Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_);
  typename FFTW_t<Real_t>::plan vlist_fftplan; bool vlist_fft_flag;

  void FFT_Check2Equiv(size_t dof, size_t m, size_t ker_dim0, Vector<size_t>& ifft_vec,
      Vector<Real_t>& input_data, Vector<Real_t>& output_data, Vector<Real_t>& buffer_, Matrix<Real_t>& M);
  typename FFTW_t<Real_t>::plan vlist_ifftplan; bool vlist_ifft_flag;

  std::vector<Vector<Real_t> > upwd_check_surf;
  std::vector<Vector<Real_t> > upwd_equiv_surf;
  std::vector<Vector<Real_t> > dnwd_check_surf;
  std::vector<Vector<Real_t> > dnwd_equiv_surf;

  InteracList<FMMNode> interac_list;
  Kernel<Real_t> kernel;     //The kernel function.
  Kernel<Real_t> aux_kernel; //Auxiliary kernel for source-to-source translations.
  PrecompMat<Real_t>* mat;   //Handles storage of matrices.
  std::string mat_fname;
  int multipole_order;       //Order of multipole expansion.
  MPI_Comm comm;

};

}//end namespace

#include <fmm_pts.txx>

#endif //_PVFMM_FMM_PTS_HPP_

